---
title: GEMM Tiling
status: draft
tags: [gemm, tiling]
difficulty: 3
order: 2
related: [/gpu/gpu-architecture, /gpu/compute-units, /gpu/special-instructions, /inference/prefill-decode-roofline, /handson/cuda-tiled-matmul, /gpu/triton, /inference/quantization]
stack: [k-gemm, hw-gpu]
train: [tr-k-gemm]
---

# GEMM Tiling

> 把矩阵切成块，让每个读进来的字节被用很多次

## 一句话结论

GEMM 本身的算术强度很高（4096³ 约 1365 FLOP/B），但前提是 $A$、$B$ 只读一次，而片上放不下整个矩阵。tiling 的做法是把 $C$ 切成 $BM \times BN$ 的块，每块只需要 $A$ 的一条行条带和 $B$ 的一条列条带，沿 $K$ 一段一段搬进片上反复用，算术强度是 $\frac{BM \cdot BN}{BM + BN}$。同样的思路在 shared memory → 寄存器这一级再做一次。tile 越大复用越多，但受 shared memory 和寄存器容量限制，而且 tile 越大块数越少，SM 可能喂不满。decode 的 GEMV 的 $M$ 只有 batch 那么大，没有东西可复用，tiling 救不了。

## 约定

GEMM 写作 $C = A B$，$A$ 是 $M \times K$，$B$ 是 $K \times N$，$C$ 是 $M \times N$，全部行主序，元素 bf16（2 字节），累加用 fp32。推理里线性层 $y = xW$ 对应 $A = x$（$M$ = 这一步的 token 数），$B = W$（$K$ = 输入维，$N$ = 输出维）。

计数规则和 [roofline](/inference/prefill-decode-roofline#符号和计数规则) 一致：先数乘加（MAC），再 ×2 换成 FLOP。GEMM 共 $MKN$ 次 MAC，即 $2MKN$ FLOP。

H100 SXM 的数字（[H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)）：

| 项 | 值 |
|---|---|
| bf16 稠密峰值 | 989 TFLOP/s（产品页的 1979 是 with sparsity，除以 2） |
| HBM 带宽 | 3.35 TB/s |
| ridge point | $989 / 3.35 \approx 295$ FLOP/B |
| SM 数 | 132 |
| 每 SM shared memory | 最多 228 KB |
| L2 | 50 MB |

989 这个峰值来自 Tensor Core，它怎么算、为什么比 CUDA core 快 16 倍，见 [计算单元](/gpu/compute-units)。本页只关心一件事：怎么把数据喂到它够用。

## 起点：GEMM 本身是 compute-bound

最理想情况下 $A$、$B$ 各读一次、$C$ 写一次：

$$
\text{AI}_{\text{ideal}} = \frac{2MKN}{2\,\text{B} \cdot (MK + KN + MN)}
$$

$M = N = K = 4096$：FLOP $= 2 \times 4096^3 \approx 1.37 \times 10^{11}$，字节 $= 3 \times 4096^2 \times 2 \approx 100$ MB，AI ≈ 1365，是 ridge 的 4.6 倍。算要 $1.37 \times 10^{11} / 989 \times 10^{12} \approx 0.14$ ms，搬只要 $100\ \text{MB} / 3.35\ \text{TB/s} \approx 0.03$ ms。

但「各读一次」要求整个 $A$、$B$ 都待在片上：两个 4096 × 4096 的 bf16 矩阵共 64 MB，一个 SM 只有 228 KB shared memory。下面每一步都是在问：实际能做到的强度离 1365 有多远。

## 朴素版：没有复用

每个线程算 $C$ 的一个元素，从 global 读 $A$ 的一行和 $B$ 的一列：

```python
# 线程 (i, j)
acc = 0
for k in range(K):
    acc += A[i, k] * B[k, j]    # 每次 MAC 从 global 读 2 个元素
C[i, j] = acc
```

读 $2K$ 个元素（$4K$ 字节），做 $K$ 次 MAC（$2K$ FLOP）：

$$
\text{AI}_{\text{naive}} = \frac{2K}{4K} = 0.5\ \text{FLOP/B}
$$

比 ridge 低 600 倍。问题在于 $A[i, k]$ 被第 $i$ 行的 $N$ 个线程各读了一遍，$B[k, j]$ 被第 $j$ 列的 $M$ 个线程各读了一遍。同一个数读了几千次，每次只用一下。

## 第一级：block tile 放进 shared memory

让一个 block 负责 $C$ 的一个 $BM \times BN$ 块。这个块要用到 $A$ 的 $BM$ 行和 $B$ 的 $BN$ 列，沿 $K$ 每次搬 $BK$ 那么宽一段进 shared memory，block 内所有线程从 shared memory 反复取：

```python
# block (bm, bn) 负责 C[m0:m0+BM, n0:n0+BN]
acc = zeros(BM, BN)
for k0 in range(0, K, BK):
    As = A[m0:m0+BM, k0:k0+BK]    # 从 global 读 BM*BK 个元素 -> smem
    Bs = B[k0:k0+BK, n0:n0+BN]    # 从 global 读 BK*BN 个元素 -> smem
    acc += As @ Bs                # BM*BN*BK 次 MAC，操作数全在片上
C[m0:m0+BM, n0:n0+BN] = acc
```

$As$ 里的每个元素被这个块的 $BN$ 列各用一次，$Bs$ 里的每个元素被 $BM$ 行各用一次。每轮读 $(BM + BN) \cdot BK$ 个元素，做 $BM \cdot BN \cdot BK$ 次 MAC，$BK$ 约掉：

$$
\text{AI}_{\text{tile}} = \frac{2 \cdot BM \cdot BN \cdot BK}{2\,\text{B} \cdot (BM + BN) \cdot BK} = \frac{BM \cdot BN}{BM + BN}\ \text{FLOP/B}
$$

| BM × BN | AI（相对 global） |
|---|---|
| 1 × 1（朴素） | 0.5 |
| 32 × 32 | 16 |
| 128 × 128 | 64 |
| 128 × 256 | 85 |
| 256 × 256 | 128 |

两点：

1. **AI 只取决于 $BM$、$BN$，和 $BK$ 无关**。$BK$ 只决定每轮搬多少、shared memory 占多少。
2. **即使 256 × 256 也低于 295**。只看单个 block 的话，大 tile GEMM 仍是 memory-bound。剩下的差距靠 L2 补。

### L2：同一波 block 共享条带

同时在跑的 block 读的是同几条 $A$ 行条带和 $B$ 列条带。第一个 block 把条带从 HBM 读进 L2，后面的 block 命中 L2，真正到 HBM 的字节少得多。

粗算 $M = N = K = 4096$、128 × 128 tile：$C$ 有 32 × 32 = 1024 个 tile，一波 132 个。一条 $A$ 行条带是 $128 \times 4096 \times 2\ \text{B} = 1$ MiB，$B$ 列条带一样。一波的 FLOP 是 $132 \times 2 \times 128 \times 128 \times 4096 \approx 1.77 \times 10^{10}$。

| block 发射顺序 | 一波覆盖的 tile | 到 HBM 的条带 | 字节 | 有效 AI |
|---|---|---|---|---|
| 不靠 L2（每个 block 自己读） | — | 132 × 2 = 264 条 | 264 MiB | 64 |
| 行优先：先排满一行 32 个 | 4 行多一点 × 32 列 | 5 + 32 = 37 条 | 37 MiB | ≈ 460 |
| 分组：每 8 行一组，组内按列排 | 8 行 × 17 列 | 8 + 17 = 25 条 | 25 MiB | ≈ 680 |

两种顺序都过了 295，分组的 HBM 字节再少三分之一。分组就是 Triton 教程里的 `GROUP_SIZE_M`，把一维的 program id 映射成二维 tile 坐标时，让相邻的 id 落在一个矮宽的方块里，而不是一整行（见 [Triton](/gpu/triton#一个完整的-matmul-kernel)）：

```python
pid = program_id(0)
num_pid_n = cdiv(N, BN)
num_pid_in_group = GROUP_SIZE_M * num_pid_n        # 一组 = GROUP_SIZE_M 行 tile
group = pid // num_pid_in_group
first_m = group * GROUP_SIZE_M
group_rows = min(cdiv(M, BM) - first_m, GROUP_SIZE_M)   # 最后一组可能不满
local = pid % num_pid_in_group                     # 组内编号
pid_m = first_m + local % group_rows               # 组内先往下走
pid_n = local // group_rows                        # 走满 group_rows 行再换列
```

## 第二级：每个线程算一小块（寄存器 tile）

数据进了 shared memory，问题挪到下一层：从 shared memory 读到寄存器也要发指令。如果 block 里每个线程只算 $C$ tile 的 1 个元素，每次 MAC 要从 shared memory 读 2 个数，load 指令比 FMA 还多。

办法和上一级一样：每个线程负责 $TM \times TN$ 个输出，读进寄存器的一个数用很多次：

```cpp
// 每个线程负责 C tile 里 TM×TN 个元素；As、Bs 已在 shared memory
float acc[TM][TN] = {0};
for (int kk = 0; kk < BK; ++kk) {
    float a[TM], b[TN];
    for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][kk];   // 读 TM 个
    for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];   // 读 TN 个
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] += a[i] * b[j];                          // 做 TM*TN 次 MAC
}
```

每个 `kk` 读 $TM + TN$ 个数，做 $TM \cdot TN$ 次 MAC。数一下每个 `kk` 里的指令：

| 每线程输出 | shared memory load | FMA | FMA 占指令的比例 |
|---|---|---|---|
| 1 × 1 | 2 | 1 | 33% |
| 8 × 8，标量 load | 16 | 64 | 80% |
| 8 × 8，`float4` 向量 load | 4 | 64 | 94% |

一个 warp 调度器每周期只发 1 条指令，load 占掉的发射槽 FMA 就用不上（[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) 的 kernel 4、5 有实测）。代价是累加器：8 × 8 个 fp32 就是每线程 64 个寄存器，下面「tile 选多大」会算到。

CUTLASS 在线程之上还有一层 warp tile：一个 warp 负责 $WM \times WN$ 的子块，道理相同，只是复用的单位从一个线程换成 32 个线程。

## 三级 tiling

把上面两级和最内层的矩阵乘指令串起来，就是 CUTLASS 的三级结构（[CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)）：

```python
# block 级：一个 block 算 C[BM, BN]，A/B 的 tile 放 shared memory
for k0 in range(0, K, BK):
    load A[BM, BK], B[BK, BN] -> smem                  # 异步，多 stage 流水
    # warp 级：每个 warp 算 C tile 里的 [WM, WN] 子块，累加器在寄存器
    for kk in range(0, BK, 16):
        a_frag = smem A[warp_m : warp_m+WM, kk:kk+16]  # smem -> 寄存器
        b_frag = smem B[kk:kk+16, warp_n : warp_n+WN]
        # 指令级：WM×WN 子块再拆成 16×8 的小块，每块一条 mma.m16n8k16
        for i, j in tiles_of(WM, 16) x tiles_of(WN, 8):
            acc[i][j] = mma(a_frag[i], b_frag[j], acc[i][j])
store acc -> C   # epilogue
```

| 级 | 数据从哪搬到哪 | 典型 tile | 一个 A 元素被复用 | 一个 B 元素被复用 |
|---|---|---|---|---|
| block | global → shared memory | 128 × 128 × 64 | $BN$ = 128 次 | $BM$ = 128 次 |
| warp | shared memory → 寄存器 | 64 × 64 | $WN$ = 64 次 | $WM$ = 64 次 |
| 指令 | 寄存器 → Tensor Core | 16 × 8 × 16（`mma`） | $n$ = 8 次 | $m$ = 16 次 |

每一级都是同一件事：一块数据搬到更近的存储里，被这一级 tile 的另一个维度复用。指令级的 tile 形状由硬件定，各代指令的形状见 [普通 SIMT 之外的指令](/gpu/special-instructions) 第 3 节。

最后的 **epilogue** 是 kernel fusion 最自然的落点：$C$ 的 tile 还在寄存器里，把 bias、激活、量化做完再写回，省掉一次完整的读写（见 [CUDA Graph 与 Kernel Fusion](/gpu/cuda-graph-fusion)）。

### Hopper 上的变化

三级结构不变，变的是每一级由谁来搬（[Hopper Tuning Guide §1.4.1](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)，指令细节见 [普通 SIMT 之外的指令](/gpu/special-instructions)）：

| 级 | Ampere | Hopper |
|---|---|---|
| global → smem | 每个线程发 `cp.async`，自己算地址 | **TMA**：一个线程发一条指令，硬件搬整块 tile，顺手按 swizzle 写入避开 bank conflict |
| smem → 寄存器 | `ldmatrix` 搬进寄存器再喂 `mma` | **`wgmma`** 直接从 smem 读 B（A 可选），这一步省掉 |
| 搬和算怎么重叠 | 同一批 warp 轮流发 load 和 mma | **warp specialization**：producer warp 只发 TMA，consumer warp 只算 `wgmma`，用 barrier 交接 |

FlashAttention-3 靠这一套把 H100 上 FP16 attention 做到 740 TFLOP/s（75% 利用率，作者自测；FlashAttention-2 在 H100 上只有 35%）（[Shah et al., 2024, arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）。

## tile 选多大

tile 越大复用越多，但有三个上限。

### shared memory 和寄存器卡住上限

1. **shared memory**：$BM = BN = 128$，$BK = 64$，一份 $A$、$B$ tile 是 $(128 \times 64 + 64 \times 128) \times 2\ \text{B} = 32$ KB。为了让搬下一块和算当前块重叠，通常开 3–4 份（多级流水，stage），就是 96–128 KB，已经占掉 228 KB 的一半。
2. **寄存器**：$C$ tile 的 fp32 累加器 $128 \times 128 \times 4\ \text{B} = 64$ KB = 16384 个寄存器。block 有 256 个线程的话，每个线程光累加器就占 64 个寄存器。按 [occupancy 的算法](/gpu/gpu-architecture#基本功三-occupancy-挂够多的-warp-来藏延迟)，一个 SM 只能放一两个这样的 block。GEMM 常年低 occupancy，靠大 tile 的复用和异步流水藏延迟。

### wave quantization：tile 数和 SM 数对不齐

tile 越大，tile 数越少。grid 里的 tile 按「波」上 SM，设每个 SM 同时只跑 1 个 tile（大 tile GEMM 常见），132 个 SM 一波跑 132 个 tile：

$$
\text{效率} = \frac{\text{tile 数}}{\lceil \text{tile 数} / 132 \rceil \times 132}
$$

| $M = N$ | tile | tile 数 | 波数 | 效率 | AI（单 block） |
|---|---|---|---|---|---|
| 4096 | 128 × 128 | 32 × 32 = 1024 | 8 | 1024 / 1056 = 97% | 64 |
| 1536 | 128 × 128 | 12 × 12 = 144 | 2 | 144 / 264 = 55% | 64 |
| 1536 | 64 × 64 | 24 × 24 = 576 | 5 | 576 / 660 = 87% | 32 |

1536 用 128 × 128 时，第二波只有 12 个 tile，120 个 SM 闲着。换成 64 × 64 填满了 SM，但复用减半。cuBLAS 的 heuristic 和 Triton 的 autotune 就是在这些 kernel 变体里挑。

另一条路是不再以 tile 为单位分活。Stream-K 把所有 tile 的 K 循环迭代排成一条长队，平均切给每个 SM，一个 SM 可能算完一个 tile 的后半段再接另一个 tile 的前半段，被切开的 tile 最后再合并部分和（[Osama et al., 2023, arXiv:2301.03598](https://arxiv.org/abs/2301.03598)）。

## 为什么 decode 的 GEMV 打不满

decode 一步只处理 $B$ 个 token（每个请求 1 个），线性层的 $M = B$。tiling 靠的是「$A$ 的一个元素被 $BN$ 列复用、$B$ 的一个元素被 $BM$ 行复用」；$M = 1$ 时 $BM$ 最多是 1，权重的每个元素只被用一次，复用不存在。

以 Llama-3-70B 的 `o_proj`（$K = N = 8192$）为例，bf16：

1. **算术强度**：权重 $KN \times 2$ B = 128 MB；FLOP $= 2BKN$。$B \ll K, N$ 时激活的字节可以忽略：

$$
\text{AI}_{\text{GEMV}} \approx \frac{2BKN}{2KN} = B
$$

batch 1 时 AI = 1，比 ridge 低 295 倍。这和 [roofline 页](/inference/prefill-decode-roofline#decode) 对整个模型推的 $\text{AI}_{\text{decode}} \approx B$ 是同一件事：每层都是这样。

2. **耗时**：搬 128 MB 要 $128\ \text{MB} / 3.35\ \text{TB/s} \approx 40\ \mu s$；算的部分即使把 $M$ pad 到矩阵乘指令要求的 16 行，也只要 $2 \times 16 \times 8192^2 / 989\ \text{TFLOP/s} \approx 2.2\ \mu s$。

3. **并行度也不够**：$M = 1$ 只有一行 tile，按 $BN = 128$ 切 $N$ 只有 $8192 / 128 = 64$ 个 block，132 个 SM 里一半没活干，连带宽都不一定打得满（每个 SM 要承担的在途字节翻倍，见 [Little 定律](/gpu/gpu-architecture#基本功三-occupancy-挂够多的-warp-来藏延迟)）。

对策，按「改分子还是改分母」分：

| 对策 | 改了什么 | 效果 |
|---|---|---|
| 加大 batch（continuous batching） | 分子：一次读权重服务 $B$ 个 token | AI ≈ $B$ 线性上涨，直到 KV cache 的字节开始主导 |
| weight-only 量化（W4A16） | 分母：权重从 2 B 变 0.5 B | 字节 ÷ 4，GEMV 时间近似 ÷ 4；FLOP 不变，还多了反量化 |
| split-K | 并行度：把 $K$ 切 $S$ 段，每段一个 block，最后把 $S$ 个部分和加起来 | $S = 4$ 时 64 → 256 个 block，SM 都有活干；代价是多一次合并 |
| speculative decoding | 分子：一次验证多个 token，$M$ 从 1 变 $k$ | 相当于给单请求凑 batch |

## 面试追问

::: details Q：block tile 的算术强度为什么和 BK 无关？
每轮读 $(BM + BN) \cdot BK$ 个元素、做 $BM \cdot BN \cdot BK$ 次 MAC，两边都正比于 $BK$。$BK$ 决定的是每轮占多少 shared memory、K 循环转几圈：$BK$ 太小，每轮的固定开销（同步、发 load）摊不开；$BK$ 太大，shared memory 放不下几个 stage。
:::

::: details Q：tile 为什么不能一直加大？
shared memory（每 SM 228 KB，要放多个 stage）和寄存器（累加器 $BM \times BN \times 4$ B）有上限；tile 越大 block 越少，小矩阵上 wave quantization 越严重；$M$、$N$ 不整除 tile 时 pad 的浪费也越大。cuBLAS 的 heuristic 和 Triton 的 autotune 都是在这几项之间找甜点。
:::

::: details Q：寄存器 tile 为什么常取 8 × 8？
每个 `kk` 读 $TM + TN$ 个数、做 $TM \cdot TN$ 次 MAC。8 × 8 时 16 次 load 换 64 次 FMA，再用 `float4` 向量 load 就只剩 4 条 load 指令，FMA 占 94% 的发射槽。再加大收益很小，累加器却按 $TM \cdot TN$ 涨：16 × 16 要 256 个寄存器，超过每线程 255 的上限。
:::

::: details Q：batch 从 1 涨到 64，decode 每步的时间涨多少？
权重部分几乎不涨：仍是读一遍 128 MB（每层每个线性层同理），AI 从 1 涨到 64 仍低于 295，还是 memory-bound。涨的是 KV cache 的读取（和 batch × 上下文长度成正比）和激活。所以小 batch 时加 batch 几乎是免费的吞吐，这就是 continuous batching 的出发点。
:::

::: details Q：为什么量化到 W4A16 对 decode 有效，对 prefill 几乎没用？
decode 是 memory-bound，时间由读权重的字节数决定，权重从 16 bit 变 4 bit 读取量减四分之三，直接提速。prefill 是 compute-bound，时间由 FLOPs 决定，W4A16 在计算前还要反量化回 bf16，FLOPs 不减反增。要加速 prefill 得用 W8A8 / FP8 这种低精度计算（H100 的 FP8 稠密峰值是 bf16 的 2 倍）。两条路线的代码见 [量化](/inference/quantization#路线二-先-dequantize-再算)。
:::

::: details Q：split-K 和 Stream-K 有什么区别？
split-K 固定把 $K$ 切成 $S$ 段，每段的部分和写到临时 buffer（或 atomic 加），再合并，解决的是「tile 太少」。Stream-K 把所有 tile 的所有 K 迭代排成一条长队，平均切给每个 SM，跨 SM 的部分 tile 再做修正，解决的是「tile 数不是 SM 数的整数倍」（wave quantization）。
:::

## 手撕

常见题：手写 shared memory tiled matmul，并解释每级 tile 的复用。从朴素版到 block tile 再到寄存器 tile 的完整代码见 [CUDA Tiled Matmul](/handson/cuda-tiled-matmul)。Simon Boehm 的 worklog 把这条路一路做到接近 cuBLAS，每一步都有实测（[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)）。

## 参考

- [CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)：三级 tiling、流水、epilogue
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [Triton 教程：Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：`GROUP_SIZE_M` 和 L2 命中
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：TMA、cluster、distributed shared memory
- [FlashAttention-3 (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608)：Hopper 上的 TMA、wgmma、warp specialization
- [Stream-K (arXiv:2301.03598)](https://arxiv.org/abs/2301.03598)

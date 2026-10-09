---
title: Tensor Core 与 GEMM Tiling
status: draft
tags: [tensor-core, gemm]
difficulty: 3
order: 2
related: [/gpu/gpu-architecture, /inference/prefill-decode-roofline, /handson/cuda-tiled-matmul, /gpu/triton, /inference/quantization]
stack: [k-gemm, hw-gpu]
---

# Tensor Core 与 GEMM Tiling

> 为什么 GEMV 打不满

## 一句话结论

Tensor Core 一条指令算一个小矩阵乘（warp 级的 `mma` 是 16×8×16），H100 上 bf16 稠密峰值 989 TFLOP/s，约是 CUDA core FP32 的 15 倍。但它只有在数据复用够多时才喂得饱：GEMM 靠 tiling 把 A、B 的块留在 shared memory 和寄存器里反复用；decode 的 GEMV（M = batch 很小）每个权重读进来只用一次，算术强度约等于 batch，带宽先打满，Tensor Core 大部分时间在等数据。

## 推导

### 约定和符号

GEMM 写作 $C = A B$，$A$ 是 $M \times K$，$B$ 是 $K \times N$，$C$ 是 $M \times N$，全部行主序，元素 bf16（2 字节），累加用 fp32。推理里线性层 $y = xW$ 对应 $A = x$（$M$ = 这一步的 token 数），$B = W$（$K$ = 输入维，$N$ = 输出维）。

计数规则和 [roofline](/inference/prefill-decode-roofline#符号和计数规则) 一致：先数乘加（MAC），再 ×2 换成 FLOP。GEMM 共 $MKN$ 次 MAC，即 $2MKN$ FLOP。

H100 SXM 的规格：bf16 Tensor Core 稠密 989 TFLOP/s（产品页标 1979 是 with sparsity，除以 2），CUDA core FP32 是 67 TFLOP/s，HBM 3.35 TB/s，ridge point $989 / 3.35 \approx 295$ FLOP/B（[H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)）。

### Tensor Core 是什么

CUDA core 一条 FMA 指令每个线程算 1 次乘加。Tensor Core 一条指令由一组线程合作算一个小矩阵乘：

| 指令 | 谁发 | 形状（bf16） | 每条指令的 MAC | 操作数从哪来 |
|---|---|---|---|---|
| `mma.sync`（Ampere 起） | 1 个 warp | m16n8k16（也有 m16n8k8） | 16 × 8 × 16 = 2048 | 寄存器 |
| `wgmma.mma_async`（Hopper） | 1 个 warpgroup = 4 个连续 warp | m64nNk16，N = 8, 16, …, 256 | 最大 64 × 256 × 16 = 262144 | A 在寄存器或 shared memory，B 在 shared memory |

形状出自 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)（§9.7.16 warp 级 mma、§9.7.17 wgmma，warpgroup 的定义也在 §9.7.17.1）。`wgmma` 是异步的：发出去之后 warp 可以接着干别的（比如发下一批 load），再用 `wgmma.wait_group` 等结果。

两个推论：

1. **矩阵太小时 Tensor Core 会浪费**。`mma` 的 M 最小是 16，batch = 1 的 GEMV 只有 1 行有效，要 pad 到 16 行，15/16 的乘法白做。不过下面会算到，GEMV 是 memory-bound，这点浪费不影响耗时。
2. **峰值只有 Tensor Core 能给**。没走上 Tensor Core 的 matmul（比如 fp32 或形状不对齐退化到 CUDA core）最多只有 67 TFLOP/s。ncu 的 Compute Workload Analysis 里看到 FMA pipe 高、Tensor pipe 低，就是这种情况（[Profiling](/gpu/profiling#_3-定了-bound-之后看哪几节)）。

### GEMM 本身够不够 compute-bound

最理想情况下 $A$、$B$ 各读一次、$C$ 写一次：

$$
\text{AI}_{\text{ideal}} = \frac{2MKN}{2(MK + KN + MN)}
$$

$M = N = K = 4096$：FLOP $= 2 \times 4096^3 \approx 1.37 \times 10^{11}$，字节 $= 3 \times 4096^2 \times 2 \approx 100$ MB，AI ≈ 1365，远高于 295。算的时间 $1.37 \times 10^{11} / 989 \times 10^{12} \approx 0.14$ ms，搬的时间 $100\ \text{MB} / 3.35\ \text{TB/s} \approx 0.03$ ms，是 compute-bound。

但「各读一次」要求整个 $A$、$B$ 都放在片上，H100 每个 SM 只有 228 KB shared memory，放不下。kernel 怎么切数据，决定了实际的强度离这个上限有多远。

### 为什么要 tiling：从朴素到分块

**朴素版**：每个线程算 $C$ 的一个元素，从 global 读 $A$ 的一行和 $B$ 的一列，共 $2K$ 个元素，做 $K$ 次 MAC。整体：

$$
\text{AI}_{\text{naive}} = \frac{2K\ \text{FLOP}}{2K \times 2\ \text{B}} = 0.5\ \text{FLOP/B}
$$

比 ridge 低 600 倍。问题是同一个 $A$ 元素被 $N$ 个线程各读一遍，没有复用。

**block tile**：一个 block 负责 $C$ 的一个 $BM \times BN$ 块，沿 $K$ 每次搬 $A$ 的 $BM \times BK$ 和 $B$ 的 $BK \times BN$ 进 shared memory，block 内所有线程从 shared memory 反复取用：

```text
for k0 in range(0, K, BK):
    As = A[m0:m0+BM, k0:k0+BK]    # 从 global 读 BM*BK 个元素
    Bs = B[k0:k0+BK, n0:n0+BN]    # 从 global 读 BK*BN 个元素
    C_tile += As @ Bs             # BM*BN*BK 次 MAC，全部在片上
```

每次迭代读 $(BM + BN) \cdot BK$ 个元素，做 $BM \cdot BN \cdot BK$ 次 MAC：

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

注意即使 256 × 256，这个数也低于 295。剩下的差距靠 **L2** 补：同时在跑的几十个 block 读的是同几条 $A$ 行条带和 $B$ 列条带，这些读大部分命中 50 MB 的 L2，真正到 HBM 的字节比上表算的少得多。这也是为什么 block 的发射顺序会影响性能：让同一波 block 挤在几行几列里，L2 命中率更高（Triton 教程里的 `GROUP_SIZE_M`，见 [Triton](/gpu/triton#一个完整的-matmul-kernel)）。

**tile 不能无限大**，两个上限：

1. **shared memory**：$BM = BN = 128$，$BK = 64$，一份 $A$、$B$ tile 是 $(128 \times 64 + 64 \times 128) \times 2\ \text{B} = 32$ KB。为了让搬下一块和算当前块重叠，通常开 3–4 份（多级流水，stage），就是 96–128 KB，已经占掉 228 KB 的一半。
2. **寄存器**：$C$ tile 的 fp32 累加器 $128 \times 128 \times 4\ \text{B} = 64$ KB = 16384 个寄存器。block 有 256 个线程的话，每个线程光累加器就占 64 个寄存器。按 [occupancy 的算法](/gpu/gpu-architecture#occupancy-多少个-warp-才藏得住延迟)，一个 SM 只能放一两个这样的 block。GEMM 常年低 occupancy，靠大 tile 的复用和异步流水藏延迟。

### 三级 tiling

CUTLASS 把一个 GEMM 分成三级，每级对应一层存储（[CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)）：

```text
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
store acc -> C （epilogue：可顺手加 bias、激活、量化，见 kernel fusion）
```

每往下一级，数据被复用的次数乘上去：global 的一个元素被 block 内 $BN$（或 $BM$）个输出复用，shared memory 的一个元素被 warp 内 $WN$（或 $WM$）个输出复用。最后的 epilogue 是 kernel fusion 最自然的落点：$C$ 的 tile 还在寄存器里，逐元素操作做完再写回，省掉一次读写（见 [CUDA Graph 与 Kernel Fusion](/gpu/cuda-graph-fusion)）。

**Hopper 上的变化**（[Hopper Tuning Guide §1.4.1](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)）：

1. **TMA**（Tensor Memory Accelerator）：一个线程发一条指令，硬件把 1D–5D 的 tile 在 global 和 shared memory 之间搬，不占其他线程的寄存器和指令槽，还能直接按 swizzle 布局写进 shared memory 避开 bank conflict。
2. **wgmma**：B（和可选的 A）直接从 shared memory 读，不必先搬进寄存器，warp 级那一层的 smem → 寄存器搬运省掉了。
3. **warp specialization**：block 里一部分 warp 只负责发 TMA 搬数据（producer），另一部分只算 wgmma（consumer），两边用 barrier 交接，搬和算完全重叠。FlashAttention-3 就是靠 TMA + wgmma 的异步和 warp specialization，把 H100 上 FP16 attention 做到 740 TFLOP/s（75% 利用率，作者自测；FlashAttention-2 在 H100 上只有 35%）（[Shah et al., 2024, arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）。
4. **thread block cluster**：相邻几个 block 组成 cluster，可以读写彼此的 shared memory（distributed shared memory），TMA 也能一次把同一块数据 multicast 给 cluster 里的多个 block。

### wave quantization：tile 数和 SM 数对不齐

grid 里的 tile 按「波」上 SM。设每个 SM 同时只跑 1 个 tile（大 tile GEMM 常见），132 个 SM 一波跑 132 个 tile：

$$
\text{效率} = \frac{\text{tile 数}}{\lceil \text{tile 数} / 132 \rceil \times 132}
$$

- $M = N = 4096$，128 × 128 tile：$32 \times 32 = 1024$ 个 tile，8 波，效率 $1024 / 1056 = 97\%$。
- $M = N = 1536$：$12 \times 12 = 144$ 个 tile，第二波只有 12 个 tile，效率 $144 / 264 = 55\%$，后半段 120 个 SM 闲着。

cuBLAS 靠在一堆 kernel 变体里挑 tile 大小缓解这个问题；Stream-K 改成按 K 循环的迭代数平均分给 SM，不再以 tile 为单位分（[Osama et al., 2023, arXiv:2301.03598](https://arxiv.org/abs/2301.03598)）。

### 为什么 decode 的 GEMV 打不满

decode 一步只处理 $B$ 个 token（每个请求 1 个），线性层的 $M = B$。以 Llama-3-70B 的 `o_proj`（$K = N = 8192$）为例，bf16：

1. **算术强度**：权重 $KN \times 2$ B = 128 MB；FLOP $= 2BKN$。$B \ll K, N$ 时激活的字节可以忽略：

$$
\text{AI}_{\text{GEMV}} \approx \frac{2BKN}{2KN} = B
$$

batch 1 时 AI = 1，比 ridge 低 295 倍。这和 [roofline 页](/inference/prefill-decode-roofline#decode) 对整个模型推的 $\text{AI}_{\text{decode}} \approx B$ 是同一件事：每层都是这样。

2. **耗时**：搬 128 MB 要 $128\ \text{MB} / 3.35\ \text{TB/s} \approx 40\ \mu s$；算（按 pad 到 16 行算）只要 $2 \times 16 \times 8192^2 / 989\ \text{TFLOP/s} \approx 2.2\ \mu s$。Tensor Core 的利用率不到 6%，而且即使 pad 浪费了 15/16 也无所谓。

3. **并行度也不够**：$M = 1$ 只有一行 tile，按 $BN = 128$ 切 $N$ 只有 $8192 / 128 = 64$ 个 block，132 个 SM 里一半没活干，连带宽都不一定打得满（每个 SM 要承担的在途字节翻倍，见 [Little 定律](/gpu/gpu-architecture#occupancy-多少个-warp-才藏得住延迟)）。

对策，按「改分子还是改分母」分：

| 对策 | 改了什么 | 效果 |
|---|---|---|
| 加大 batch（continuous batching） | 分子：一次读权重服务 $B$ 个 token | AI ≈ $B$ 线性上涨，直到 KV cache 的字节开始主导 |
| weight-only 量化（W4A16） | 分母：权重从 2 B 变 0.5 B | 字节 ÷ 4，GEMV 时间近似 ÷ 4；FLOP 不变，还多了反量化 |
| split-K | 并行度：把 $K$ 切 $S$ 段，每段一个 block，最后把 $S$ 个部分和加起来 | $S = 4$ 时 64 → 256 个 block，SM 都有活干；代价是多一次合并 |
| speculative decoding | 分子：一次验证多个 token，$M$ 从 1 变 $k$ | 相当于给单请求凑 batch |

## 面试追问

::: details Q：为什么量化到 W4A16 对 decode 有效，对 prefill 几乎没用？
decode 是 memory-bound，时间由读权重的字节数决定，权重从 16 bit 变 4 bit 读取量减四分之三，直接提速。prefill 是 compute-bound，时间由 FLOPs 决定，W4A16 在计算前还要反量化回 bf16，FLOPs 不减反增。要加速 prefill 得用 W8A8 / FP8 这种 Tensor Core 原生支持的低精度计算（H100 的 FP8 稠密峰值是 bf16 的 2 倍）。两条路线的代码见 [量化](/inference/quantization#路线二-先-dequantize-再算)。
:::

::: details Q：tile 为什么不能一直加大？
shared memory（每 SM 228 KB，要放多个 stage）和寄存器（累加器 $BM \times BN \times 4$ B）有上限；tile 越大 block 越少，小矩阵上 wave quantization 越严重；$M$、$N$ 不整除 tile 时 pad 的浪费也越大。cuBLAS 的 heuristic 和 Triton 的 autotune 都是在这几项之间找甜点。
:::

::: details Q：为什么累加器用 fp32，不用 bf16？
bf16 只有 8 位有效位（7 位尾数加隐含位），$K = 8192$ 个乘积连加时，累加值变大后再加小的乘积会被舍掉。Tensor Core 的 bf16 `mma` 本身就以 fp32 累加（PTX 里 bf16 的 `mma` 累加器类型是 `.f32`），最后写回时再转 bf16。
:::

::: details Q：batch 从 1 涨到 64，decode 每步的时间涨多少？
权重部分几乎不涨：仍是读一遍 128 MB（每层每个线性层同理），AI 从 1 涨到 64 仍低于 295，还是 memory-bound。涨的是 KV cache 的读取（和 batch × 上下文长度成正比）和激活。所以小 batch 时加 batch 几乎是免费的吞吐，这就是 continuous batching 的出发点。
:::

::: details Q：split-K 和 Stream-K 有什么区别？
split-K 固定把 $K$ 切成 $S$ 段，每段的部分和写到临时 buffer（或 atomic 加），再合并，解决的是「tile 太少」。Stream-K 把所有 tile 的所有 K 迭代排成一条长队，平均切给每个 SM，一个 SM 可能算完一个 tile 的后半段再接另一个 tile 的前半段，跨 SM 的部分 tile 再做修正，解决的是「tile 数不是 SM 数的整数倍」（wave quantization）。
:::

## 手撕

常见题：手写 shared memory tiled matmul，并解释每级 tile 的复用。框架见 [CUDA Tiled Matmul](/handson/cuda-tiled-matmul)；再进一步是每个线程算 $TM \times TN$ 个输出（寄存器 tile）：

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

每个 `kk` 从 shared memory 读 $TM + TN$ 个数、做 $TM \cdot TN$ 次 MAC。$TM = TN = 8$ 时每读 1 个数做 4 次 MAC，比每线程 1 个输出（读 2 个做 1 次）的 shared memory 流量少 8 倍。Simon Boehm 的 worklog 把这条路从朴素版一路做到接近 cuBLAS，每一步都有实测（[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)）。

## 参考

- [CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)：三级 tiling、流水、epilogue
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`mma` 和 `wgmma` 的形状
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：TMA、cluster、distributed shared memory
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [FlashAttention-3 (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608)：Hopper 上的 TMA、wgmma、warp specialization
- [Stream-K (arXiv:2301.03598)](https://arxiv.org/abs/2301.03598)

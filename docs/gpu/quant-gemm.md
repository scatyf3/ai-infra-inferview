---
title: 量化 GEMM Kernel：W4A16 / FP8
status: draft
tags: [quantization, gemm, kernel]
difficulty: 4
order: 3
related: [/gpu/tensor-core-gemm, /inference/quantization, /gpu/compute-units, /inference/prefill-decode-roofline, /gpu/moe-grouped-gemm, /gpu/memory-bound-kernels, /gpu/special-instructions]
stack: [k-gemm, ld-repack]
train: [tr-k-gemm]
---

# 量化 GEMM Kernel

> 低比特的数在哪一步变回能算的格式，scale 在哪一步乘回去

## 一句话结论

量化 GEMM 就是在 [GEMM 的三级 tiling](/gpu/tensor-core-gemm) 里多做两件事：把低比特的数变成 Tensor Core 能吃的格式，把 scale 乘回去。这两件事放在哪一级，决定了省下的是什么：

- **W4A16**：权重以 int4 从 HBM 一路搬到寄存器，最后一步才转成 bf16，喂给 bf16 Tensor Core。省的是 HBM 字节（÷ 4），只在 memory-bound 时有用，也就是 decode、小 batch。
- **FP8 W8A8**：两边都是 FP8，直接进 FP8 Tensor Core，不用转换。字节 ÷ 2，算力 × 2，batch 大小都有用。scale 不随 $k$ 变就在 epilogue 乘一次；随 $k$ 变（block scaling）就得在主循环里每段 $K$ 乘一次。

| | W4A16（Marlin 一类） | FP8，per-tensor / per-channel | FP8，block scaling（DeepSeek-V3） |
|---|---|---|---|
| 进 Tensor Core 的格式 | bf16 / fp16 | FP8 | FP8 |
| 格式转换在哪 | smem → 寄存器之后、`mma` 之前 | 不用转 | 不用转 |
| scale 在哪乘 | per-group：转换时乘进权重；per-channel：epilogue | epilogue，一次 | 主循环里，每 128 个 $K$ 一次，在 CUDA core 上 |
| 省什么 | 权重字节 ÷ 4 | 字节 ÷ 2、算力 × 2 | 同左，累加精度更好 |
| 适合 | decode、小 batch | prefill、大 batch | 训练、大模型推理 |

量化格式和粒度本身（int4 / FP8、per-group / per-block 是什么）见 [量化](/inference/quantization)，本页只讲 kernel 怎么实现。

## 约定

GEMM 写作 $Y = XW$，$X$ 是激活（$B \times K$，$B$ 是这一步的 token 数），$W$ 是权重（$K \times N$）。H100 SXM：bf16 稠密峰值 989 TFLOP/s，FP8 稠密峰值 1979 TFLOP/s，HBM 3.35 TB/s。下面的例子用 Llama-3-70B 的 `o_proj`，$K = N = 8192$。

## 先算账：省字节和省算力各在哪有用

权重有 $KN$ 个元素，FLOP 是 $2BKN$。$B \ll K, N$ 时激活的字节可以忽略，三种格式只差在权重每个元素几个字节、Tensor Core 多快：

| 格式 | 权重字节 | 算力 | 算术强度 | 从 memory-bound 变 compute-bound 的 $B$ |
|---|---|---|---|---|
| bf16 | $2KN$ | 989 TFLOP/s | $\frac{2BKN}{2KN} = B$ | $B = 295$ |
| W4A16 | $0.5KN$ | 989 TFLOP/s（还是 bf16 Tensor Core） | $4B$ | $4B = 295$，$B \approx 74$ |
| FP8 W8A8 | $KN$ | 1979 TFLOP/s | $2B$ | $2B = 1979 / 3.35 = 591$，$B \approx 295$ |

每种格式的时间是 $\max(\text{字节} / \text{带宽},\ \text{FLOP} / \text{算力})$。代入 `o_proj`：

| $B$ | bf16 | W4A16 | FP8 | W4A16 比 bf16 快 | FP8 比 bf16 快 |
|---|---|---|---|---|---|
| 1 | 40 μs（读权重） | 10 μs（读权重） | 20 μs（读权重） | 4× | 2× |
| 64 | 40 μs（读权重） | 10 μs（读权重） | 20 μs（读权重） | 4× | 2× |
| 256 | 40 μs（读权重） | 35 μs（算） | 20 μs（读权重） | 1.15× | 2× |
| 1024 | 139 μs（算） | 139 μs（算） | 69 μs（算） | 1× | 2× |

（权重字节：bf16 134 MB、int4 34 MB、FP8 67 MB；计算：$2B \cdot 8192^2$ 除以对应的峰值。）

两个结论：

1. **W4A16 的加速随 batch 掉下去**。$B \le 74$ 时是 4 倍；$74 < B < 295$ 时 W4A16 已经在算、bf16 还在读，加速是 $295 / B$；$B \ge 295$ 两边都在算，而且都用 bf16 Tensor Core，W4A16 还多了转换的开销，反而可能更慢。Marlin 论文实测 batch 16–32 以内接近 4 倍，64–128 逐渐下降（[Frantar et al., 2024, arXiv:2408.11743](https://arxiv.org/abs/2408.11743)），比上面理想的 74 掉得早，因为实际 kernel 跑不到峰值算力，转换也要占发射槽。
2. **FP8 在任何 batch 下都是 2 倍**。字节和算力都折半，两边的拐点重合在 $B = 295$。代价是激活也要量化：每一步多一个把 $X$ 转成 FP8 的 kernel，算 scale 要扫一遍 $X$（这是个 [memory-bound kernel](/gpu/memory-bound-kernels)，通常和前面的 RMSNorm 融合）。

## W4A16：格式转换放到最后一步

### 放在哪一级

int4 权重要变成 bf16 才能进 bf16 Tensor Core。在三级 tiling 里有三个地方可以放这一步：

| 放在哪 | 每个权重从 HBM 搬多少 | shared memory 里存的是 | 问题 |
|---|---|---|---|
| 单独一个 kernel 先反量化，写回 HBM，再调 bf16 GEMM | 读 0.5 B + 写 2 B + 读 2 B = 4.5 B | bf16 | 比不量化（2 B）还多，白做 |
| global → smem 时转 | 0.5 B | bf16，占 4 倍空间 | `cp.async` / TMA 只能原样搬字节，转换得先读进寄存器再写 smem，异步搬运用不上 |
| **smem → 寄存器之后、`mma` 之前** | 0.5 B | int4 | 无，Marlin 的做法 |

结论是越晚越好：转换之前的每一级（HBM、L2、smem、smem → 寄存器）搬的都是 int4，都只要 bf16 的四分之一。反正 Tensor Core 之前数据一定在寄存器里，转换放在这里不多走一步。

```python
# W4A16 的主循环（一个 block，示意）
for k0 in range(0, K, BK):
    cp_async(Xs, X[m0:m0+BM, k0:k0+BK])        # 激活：bf16，小，靠 L2 反复用
    cp_async(Wq_s, Wq[k0:k0+BK, n0:n0+BN],     # 权重：int4，只用一次，evict_first
             hint="evict_first")
    s = scales[k0 // g, n0:n0+BN]               # per-group：BK 整除 g，一个 tile 只落在一组里
    for kk in range(0, BK, 16):
        a = ldmatrix(Xs[..., kk:kk+16])         # bf16 fragment
        q = lds128(Wq_s[...])                   # 一次 16 B，正好是这个线程要的 int4
        b = int4_to_bf16(q) * s                 # 在寄存器里转换、乘 scale
        acc = mma(a, b, acc)                    # bf16 Tensor Core
```

### 转换要比 HBM 送数据快

decode 时每 SM 每周期要转约 28 个权重才跟得上 HBM。朴素的 `int → float → half` 卡在 float → half 这条转换指令上，每 SM 每周期只有 16 个，kernel 会被转换拖住而不是被带宽。通用的绕法是把 4 bit 整数直接塞进 fp16 `1024.0` 的尾数里再减掉 1024，一条 LOP3 加一条 HSUB2 处理两个权重，推导见 [计算单元：类型转换也走 XU](/gpu/compute-units#类型转换也走-xu)。

非对称量化还有 zero point $z$：$w = s \cdot (q - z)$。上面的技巧得到 $h = 1024 + q$，于是

$$
w = s \cdot h - s \cdot (1024 + z)
$$

$s \cdot (1024 + z)$ 每组预先算好，转换加乘 scale 合成一条 HFMA2，一次两个权重。

### 权重离线重排

`mma` 要求每个线程拿着 B 片段里特定位置的几个元素，这些位置在行主序的权重里并不连续。如果直接按行主序存 int4，每个线程要发好几次小的 load 再拼起来。

Marlin 在加载模型时把权重重排一次：切成 16 × 64 的块连续存放，块内的顺序正好是 fragment layout，每个线程一次 16 B 的向量读，拿到的就是它要的那几个量化权重，不用 `ldmatrix`、不用 shuffle；顺序还照顾了上面的转换技巧，让一条指令能同时抠出两个权重（[Frantar et al., 2024](https://arxiv.org/abs/2408.11743)）。

代价是权重不再是普通的行主序张量：每种 kernel 有自己的格式。推理框架在加载 GPTQ / AWQ 的 checkpoint 时做这一步重排，比如 vLLM 的 `gptq_marlin_repack`。

### scale 乘在哪

| 粒度 | scale 随 $k$ 变吗 | 在哪乘 |
|---|---|---|
| per-channel | 不变 | epilogue，写回前乘一次（Marlin 就这么做） |
| per-group（沿 $K$ 每 $g$ 个） | 每 $g$ 个变一次 | 主循环里，转换时乘进权重 |

per-group 不能挪到 epilogue：$\sum_k x_k \cdot s_{\lfloor k/g \rfloor} q_k$ 里的 $s$ 跟着 $k$ 变，提不到求和外面。让 $BK$ 整除 $g$（比如 $g = BK = 128$），一个 K-tile 就只落在一组里，只需要读一行 scale，放进 shared memory 给 tile 里的所有 $kk$ 共用。

### 小 batch 时的并行度

decode 时 $B$ 只有几到几十，$M$ 方向只有一个 tile。按 $BN = 128$ 切 $N = 8192$ 只有 64 个 tile，132 个 SM 喂不满，和 [GEMV 的问题](/gpu/tensor-core-gemm#面试追问) 一样。Marlin 用 striped partitioning：把所有 (K 段, N 段) 的工作平均切给每个 SM，一个 SM 的那一条可以跨好几列，同一列的部分和最后再合并，思路和 Stream-K 相同。

其余两点也是为 memory-bound 服务的：权重每个字节只用一次，用 `cp.async` 带 `evict_first` 提示，读完就让它离开 L2，把 L2 留给要反复读的激活；流水深度 4，寄存器里再做一层双缓冲，让搬运和 `mma` 完全重叠。

## FP8：直接进低精度 Tensor Core

### scale 不随 $k$ 变：放进 epilogue

per-tensor，或者激活 per-token、权重 per-channel 时：

$$
Y_{ij} \approx s_x[i] \cdot s_w[j] \cdot \sum_k X_{q,ik} W_{q,kj}
$$

$s_x[i] s_w[j]$ 提到求和外面，主循环和 bf16 GEMM 一模一样，只是 `wgmma` 换成 FP8 版本（每条 $k = 32$，bf16 是 16）。epilogue 里 $C$ 还在寄存器，乘一次 $s_x[i] s_w[j]$ 再转 bf16 写回，几乎免费。推导见 [量化：路线一](/inference/quantization#路线一-低精度原生计算)。

### block scaling：scale 随 $k$ 变，只能在主循环里乘

DeepSeek-V3 的 FP8 GEMM 用更细的粒度：激活每 $1 \times 128$ 一个 scale，权重每 $128 \times 128$ 一个 scale（[DeepSeek-AI, 2024, arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）。$K$ 按 128 分段，每段 $t$ 的 scale 不同：

$$
Y_{ij} \approx \sum_{t} s_x[i, t] \cdot s_w[t, j] \cdot \sum_{k \in t} X_{q,ik} W_{q,kj}
$$

取 $BK = 128$，一个 K-tile 正好是一段：

```python
acc = zeros(BM, BN, fp32)                      # 真正的累加器，CUDA core 上的 FP32
for t in range(K // 128):
    part = zeros(BM, BN)                       # Tensor Core 的累加器
    for kk in range(0, 128, 32):
        part = wgmma_fp8(Xq[:, kk], Wq[kk, :], part)   # 4 条 wgmma，k = 32
    acc += part * s_x[:, t][:, None] * s_w[t, n_block] # CUDA core：每元素 1 次 FMA
```

`BN` 取 128 的倍数且和权重的块对齐时，$s_w[t, j]$ 在整个 tile 里是一个数，每段只要读 $BM$ 个 $s_x$ 和 1 个 $s_w$。

这样做顺带解决了 Hopper 的一个精度问题：FP8 `wgmma` 内部累加只保留约 14 bit，$K$ 很长时误差会积起来。DeepSeek-V3 每 128 个元素就把部分和提升到 CUDA core 上用 FP32 加，Tensor Core 只负责短的求和。代价有两个：

1. **CUDA core 的活**：每段每个输出元素 1 次 FMA，是 Tensor Core 乘加数的 $1 / 128$。但 FP8 Tensor Core 比 FP32 CUDA core 快约 30 倍（1979 vs 67 TFLOP/s），不重叠的话要多花约 $30 / 128 \approx 23\%$ 的时间。所以要让两个 warpgroup 交替：一个在做提升，另一个在发 `wgmma`。
2. **寄存器**：`part` 和 `acc` 两份累加器同时在寄存器里，tile 大小受 [寄存器上限](/gpu/tensor-core-gemm#shared-memory-和寄存器卡住上限) 卡得更紧。

开源实现是 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。

### Blackwell：scale 进了硬件

Blackwell 的 `tcgen05.mma` 直接支持 block-scaled 格式：MXFP8 / MXFP4 每 32 个元素一个 E8M0 scale，NVFP4 每 16 个元素一个 E4M3 scale。Tensor Core 自己乘 scale，累加器在 TMEM 里，上面那套 CUDA core 提升不再需要。指令细节见 [普通 SIMT 之外的指令](/gpu/special-instructions)。

## 面试追问

::: details Q：为什么不先用一个 kernel 把 int4 反量化成 bf16，再直接调 cuBLAS？
反量化 kernel 要读 0.5 B、写 2 B，GEMM 再读 2 B，每个权重共 4.5 B，比直接存 bf16 的 2 B 还多。量化省字节的前提是 int4 一路搬到寄存器，所以转换必须融合进 GEMM 的主循环。
:::

::: details Q：W4A16 在 prefill 时为什么可能比 bf16 还慢？
prefill 的 $B$ 是几千，早过了 $B = 295$ 的拐点，两边都是 compute-bound，而且算的都是 bf16 Tensor Core，算力一样。W4A16 还要在主循环里转换格式、乘 scale，占掉发射槽，所以只会更慢。要加速 prefill 得用 FP8 这种真正更快的 Tensor Core。
:::

::: details Q：per-group 的 scale 为什么不能像 per-channel 一样放到 epilogue？
per-channel 的 $s_w[j]$ 不随 $k$ 变，可以提到 $\sum_k$ 外面，等累加完乘一次。per-group 的 scale 沿 $K$ 每 $g$ 个变一次，提不出去，只能在主循环里乘：要么转换时乘进权重（W4A16），要么每段累加完乘进部分和（FP8 block scaling）。
:::

::: details Q：FP8 block scaling 为什么要两个 warpgroup 交替？
每 128 个 $K$ 要在 CUDA core 上做一次「部分和 × scale 加进 FP32 累加器」。这部分 FMA 只有 Tensor Core 乘加数的 1/128，但 CUDA core 慢约 30 倍，串行做会多花约 23% 的时间。两个 warpgroup 交替，一个提升时另一个发 `wgmma`，Tensor Core 就不会停。
:::

::: details Q：权重为什么要离线重排，不能在 kernel 里重排吗？
`mma` 要求每个线程拿着片段里特定位置的元素。在 kernel 里重排要么多发 load 再拼，要么用 shuffle，都占发射槽；decode 时 kernel 本来就在和带宽赛跑。离线重排只在加载模型时做一次，之后每个线程一次 16 B 的读就拿到要的数据。
:::

## 参考

- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models (arXiv:2408.11743)](https://arxiv.org/abs/2408.11743)：W4A16 kernel，权重重排、striped partitioning、evict_first
- [Kim et al., 2022 (arXiv:2211.10017)](https://arxiv.org/abs/2211.10017)：int4 → fp16 的 LOP3 转换技巧
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：FP8 block scaling、累加精度和 CUDA core 提升
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)：FP8 block scaling GEMM 和 MoE grouped GEMM 的实现

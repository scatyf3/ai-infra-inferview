---
title: 量化 GEMM Kernel：W4A16 / FP8
status: draft
tags: [quantization, gemm, kernel]
difficulty: 4
order: 3
related: [/gpu/tensor-core-gemm, /gpu/split-k-stream-k, /inference/quantization, /gpu/compute-units, /inference/prefill-decode-roofline, /gpu/moe-grouped-gemm, /gpu/memory-bound-kernels, /gpu/special-instructions]
stack: [k-gemm, ld-repack]
train: [tr-k-gemm]
---

# 量化 GEMM Kernel

> 低比特的数在哪一步变回能算的格式，scale 在哪一步乘回去

TLDR。

首先要考虑硬件约束，从bf16到int8，fp4，硬件的计算单元支持越来越多，当然无论如何，只能同格式的表示才能进行运算，两个不同格式的目前还不能

从而我们有两种量化路线，一个是weight only，这是为了在decode里省load weight from hbm的带宽，代表精度是w4a16。这条路线最大的实现困难就是在计算之前把weight dequant回bf16，需要回答1. 在哪里dequant（寄存器，省smem） 2. 怎么高效dequant（有前辈写好的ptx指令，通过一些位操作实现） 3. weight排布和bf16不一样，怎么把bf16上的tiling在w4下实现？（ldmatrix 处理不了 int4，权重要离线重排成 fragment 顺序）

另外一个路线是weight和activation都量化，直接做原生低精度运算，这里最大的挑战就是activation online quantize，一般和rmsnorm fuse

最后，无论是哪个 x bar = scale times xq, scale乘在哪里？如果是per row/col（tensor-wise/channel-wise），即scale顺着求和的维度K，直接后面聚合的时候乘就行。如果per group，则在gemm的时候需要判断我们现在在哪个group，边乘scale边算。

<QuantGemmOverview />

## 一句话结论

tensorcore的input要求：$D = A \times B + C$。A 和 B 必须是同一类低精度格式，C/D 是更宽的累加器，其中每个硬件架构支持：

| 架构（sm） | 新增的输入格式 (A, B) | 累加器 (C, D) | 指令 |
|---|---|---|---|
| Volta (sm_70) | FP16 | FP16 / FP32 | `mma.sync` |
| Turing (sm_75) | INT8、INT4、INT1（二值） | INT32 | `mma.sync` |
| Ampere (sm_80) | BF16、TF32、FP64 | FP32（FP64 用 FP64） | `mma.sync`，另有 2:4 稀疏 |
| Ada (sm_89) | FP8（E4M3 / E5M2） | FP32 / FP16 | `mma.sync` |
| Hopper (sm_90) | FP8（E4M3 / E5M2），INT4 / INT1 被砍 | FP32 / FP16 | `wgmma` |
| Blackwell (sm_100 / sm_120) | FP6（E3M2 / E2M3）、FP4（E2M1），以及带 block scale 的 MXFP8 / MXFP6 / MXFP4（每 32 个元素一个 E8M0 scale）和 NVFP4（每 16 个元素一个 E4M3 scale） | FP32 | `tcgen05.mma`（sm_100）；sm_120 也就是 RTX 50 系，用带 block scale 的 `mma.sync` |


量化 GEMM 就是在 [GEMM 的三级 tiling](/gpu/tensor-core-gemm) 里多做两件事：把低比特的数变成 Tensor Core 能吃的格式，把 scale 乘回去。这两件事放在哪一级，决定了省下的是什么：

- **W4A16**：权重以 int4 从 HBM 一路搬到寄存器，最后一步才转成 bf16，喂给 bf16 Tensor Core。省的是 HBM 字节（÷ 4），只在 memory-bound 时有用，也就是 decode、小 batch。
- **FP8 W8A8**：两边都是 FP8，直接进 FP8 Tensor Core，不用转换。字节 ÷ 2，算力 × 2，batch 大小都有用。scale 不随 $k$ 变就在 epilogue 乘一次；随 $k$ 变（block scaling）就得在主循环里每段 $K$ 乘一次。

| | W4A16（Marlin 一类） | FP8，per-tensor / per-channel | FP8，block scaling（DeepSeek-V3） |
|---|---|---|---|
| 进 Tensor Core 的格式 | bf16 / fp16 | FP8 | FP8 |
| 格式转换在哪 | smem → 寄存器之后、`mma` 之前，[理由见下](#放在哪一级) | 不用转 | 不用转 |
| scale 在哪乘 | per-group：转换时乘进权重；per-channel：epilogue | epilogue，一次 | 主循环里，每 128 个 $K$ 一次，在 CUDA core 上 |
| 省什么 | 权重字节 ÷ 4 | 字节 ÷ 2、算力 × 2 | 同左，累加精度更好 |
| 适合 | decode、小 batch | prefill、大 batch | 训练、大模型推理 |

量化格式和粒度本身（int4 / FP8、per-group / per-block 是什么）见 [量化](/inference/quantization)，本页只讲 kernel 怎么实现。

## 约定

GEMM 写作 $Y = XW$，$X$ 是激活（$B \times K$，$B$ 是这一步的 token 数，decode 时等于 batch size），$W$ 是权重（$K \times N$）。H100 SXM：bf16 稠密峰值 989 TFLOP/s，FP8 稠密峰值 1979 TFLOP/s，HBM 3.35 TB/s。下面的例子用 Llama-3-70B 的 `o_proj`，$K = N = 8192$。

## 先算账：省字节和省算力各在哪有用

用 roofline 估一下：一个线性层的耗时取读权重和算两者中慢的那个，$t = \max(\text{权重字节} / \text{带宽},\ 2BKN / \text{峰值算力})$。三种格式只差在每个权重几个字节（bf16 2、int4 0.5、FP8 1），以及用哪种 Tensor Core（W4A16 还是 bf16 的）。

<QuantRooflineChart />

两个结论：

1. **W4A16 的加速随 batch 掉下去**。$B \le 74$ 时是 4 倍；$74 < B < 295$ 时 W4A16 已经在算、bf16 还在读，加速是 $295 / B$；$B \ge 295$ 两边都在算，而且都用 bf16 Tensor Core，W4A16 还多了转换的开销，反而可能更慢。Marlin 论文实测 batch 16–32 以内接近 4 倍，64–128 逐渐下降（[Frantar et al., 2024, arXiv:2408.11743](https://arxiv.org/abs/2408.11743)），比上面理想的 74 掉得早，因为实际 kernel 跑不到峰值算力，转换也要占发射槽。
2. **FP8 在任何 batch 下都是 2 倍**。字节和算力都折半，两边的拐点重合在 $B = 295$。代价是激活也要量化：每一步多一个把 $X$ 转成 FP8 的 kernel，算 scale 要扫一遍 $X$（这是个 [memory-bound kernel](/gpu/memory-bound-kernels)，通常和前面的 RMSNorm 融合）。

## scale 乘在哪：两种格式共用的规则

量化后 $w_k = s \cdot q_k$。一个输出 $y = \sum_k x_k w_k$，scale 能不能提到 $\sum_k$ 外面，只看它随不随 $k$ 变：

| 粒度 | scale 随 $k$ 变吗 | 在哪乘 | W4A16 | FP8 |
|---|---|---|---|---|
| per-tensor / per-channel（激活 per-token） | 不变 | epilogue，写回前乘一次 | Marlin 的 per-channel 模式 | 主循环和 bf16 GEMM 一样，见 [FP8](#fp8-直接进低精度-tensor-core) |
| per-group / per-block（沿 $K$ 每 $g$ 个一组） | 每 $g$ 个变一次 | 主循环里，求和之前 | 乘进操作数：dequant 时合进 `hfma2` | 乘进部分和：每 128 个 $K$ 在 CUDA core 上乘一次 |
| MX / NVFP4 block scale | 每 32 / 16 个变一次 | Tensor Core 自己乘 | — | [Blackwell](#blackwell-scale-进了硬件) |

举个例子：$K = 4$，$x = [1, 2, 3, 4]$，量化后 $q = [1, -1, 2, 0]$。

**per-channel**：这一列只有一个 $s = 0.5$，$w_k = s \cdot q_k$。

$$
y = \sum_k x_k \cdot s \, q_k = s \sum_k x_k q_k = 0.5 \times (1 - 2 + 6 + 0) = 2.5
$$

$s$ 提到求和外面：主循环只算整数 $q$ 解出来的 $\sum x_k q_k = 5$，epilogue 乘一次 0.5。

**per-group，$g = 2$**：$k = 0, 1$ 用 $s_0 = 0.5$，$k = 2, 3$ 用 $s_1 = 2$。

$$
y = s_0 (x_0 q_0 + x_1 q_1) + s_1 (x_2 q_2 + x_3 q_3) = 0.5 \times (-1) + 2 \times 6 = 11.5
$$

主循环要是照样只累加 $\sum x_k q_k = 5$，epilogue 得乘 $11.5 / 5 = 2.3$。这个数取决于 $x$，每个 token 都不一样，没法预先算好，所以 scale 只能在求和之前乘。

在主循环里乘有两种做法，取决于操作数进 Tensor Core 之前有没有经过 ALU：

- **乘进操作数（W4A16）**：int4 本来就要在寄存器里 dequant 成 bf16，scale 合进同一条 `hfma2`（见 [dequant 实现](#dequant-实现)），不多花指令。
- **乘进部分和（FP8 block scaling）**：FP8 不用转换，`wgmma` 的 B 直接从 smem 读，中途不经过 ALU。要把 scale 乘进操作数，就得先转成更宽的格式，FP8 Tensor Core 就白用了。所以 Tensor Core 只累加一组之内的 $\sum_{k \in t} x_k q_k$，每组结束后在 CUDA core 上乘 scale、加进 FP32 累加器，每个输出元素每组 1 次 FMA（见下文 FP8 一节）。

一个 K-tile 里要读几个 scale，取决于 $BK$ 和 $g$ 的关系。取 $BK = g$（W4A16 和 DeepSeek-V3 都常用 128），一个 tile 正好落在一组里，每列只读一个 scale。

## W4A16：格式转换放到最后一步

### 放在哪一级

int4 权重要变成 bf16 才能进 bf16 Tensor Core。转换是线程的 ALU 做的，只能对寄存器里的数做；HBM 和 smem 里的数据不能直接算。所以问题不是「从 HBM 读出来就转」，而是「数据哪一次经过寄存器时转」。权重从 HBM 到 Tensor Core 的路径是：

```
HBM ──(L2)──> smem ──> 寄存器 ──> mma
     cp.async / TMA    ldmatrix / lds
```

`cp.async` / TMA 把字节原样从 HBM 搬进 smem，不经过寄存器；smem → 寄存器这一步是每个线程自己发 load，之后数据就在寄存器里了。转换可以放在三个地方：

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

### dequant 实现

上面的 `int4_to_bf16` 不是一条指令。PTX 里没有 int4 → bf16 / fp16 的 `cvt`，int4 在寄存器里又是 8 个挤在一个 32 bit 里，本来就得先移位、掩码拆开。对比之下，FP8 和 FP4 有原生转换：

| 格式 | 转成 fp16 的指令 | 架构 |
|---|---|---|
| int4 / int8 | 没有，自己用位运算拼 | — |
| FP8（E4M3 / E5M2） | `cvt.rn.f16x2.e4m3x2`，一条转两个；CUDA 里是 `cuda_fp8.h` 的 `__nv_cvt_fp8x2_to_halfraw2` | sm_89+ |
| FP4（E2M1） | `cvt.rn.f16x2.e2m1x2` | sm_100a / sm_120a |

拆出来之后也不能走普通的 `int → float → half`，因为它太慢。decode 时 HBM 3.35 TB/s 按 int4 算每秒送进 6.7 × 10¹² 个权重，摊到 H100 每 SM 每周期约 28 个，转换至少要这么快才藏得住。而 float → half / bf16 这一步走 XU，每 SM 每周期只有 16 个，kernel 会卡在转换上而不是带宽上（推导见 [计算单元：类型转换也走 XU](/gpu/compute-units#类型转换也走-xu)）。

int4 的通用做法是把 4 bit 直接塞进一个浮点常数的尾数低位。常数选尾数最低位恰好等于 1 的那个数：fp16 尾数 10 bit，用 $2^{10} = 1024$（`0x6400`）；bf16 尾数 7 bit，用 $2^7 = 128$（`0x4300`）。`(q & 0xF) | 0x4300` 按 bf16 读出来就是 $128 + q$，再减掉 $128 + z$（$z$ 是零点，对称 int4 一般取 8）就得到 $q - z$。一个 32 bit 寄存器装两个 bf16，所以一条 `lop3`（任意三输入位运算，这里算 `(a & b) | c`）一次处理两个权重。Marlin 的 bf16 版本（[vLLM `csrc/quantization/gptq_marlin/dequant.h`](https://github.com/vllm-project/vllm)）：

```cuda
// q：32 bit，装 8 个 int4（离线重排过，相邻两个 bf16 要的半字节正好隔 16 bit）
lo = lop3(q,      0x000f000f, 0x43004300)  // 半字节 0、4 → (128 + q0, 128 + q4)
hi = lop3(q >> 4, 0x000f000f, 0x43004300)  // 半字节 1、5 → (128 + q1, 128 + q5)
b0 = hfma2(lo, 1.0, -136.0)                 // 减 128 + 8，得到 q - 8
b1 = hfma2(hi, 1.0, -136.0)
// q >> 8 再来一遍，拿到剩下 4 个
```

fp16 版本多一个小技巧：fp16 尾数有 10 bit，高半字节不用移位，直接用掩码 `0x00f000f0` 塞进去，得到 $1024 + 16q$，再用一条 `hfma2` 乘 $1/16$、减 $72 = (1024 + 16 \cdot 8) / 16$。bf16 尾数只有 7 bit，$16q$ 最大到 240，会溢出到指数位，所以 bf16 只能先移位。

`lop3` 和 `hfma2` 都走整宽单元，每条处理两个权重，每 SM 每周期分别能出 128 和 256 个，远高于 28。

scale 也能合进同一条 `hfma2`。记常数为 $c$（bf16 是 128，fp16 是 1024），塞完得到 $h = c + q$，非对称量化 $w = s \cdot (q - z)$，于是

$$
w = s \cdot h - s \cdot (c + z)
$$

$s \cdot (c + z)$ 每组预先算好，`hfma2(h, s, -s·(c+z))` 一条指令同时完成去偏移、减零点、乘 scale，一次两个权重。

不用自己从头写，现成的有：

- **CUDA**：Marlin 的 `dequant`（几行 inline PTX，`asm volatile("lop3.b32 ...")`）；CUTLASS 混合输入 GEMM 里的 `NumericArrayConverter<bfloat16_t, int4b_t, N>`；TensorRT-LLM 和 FasterTransformer 里的 int4 → fp16 / bf16 转换函数。字节重排可以用 `prmt`，对应 intrinsic `__byte_perm`。
- **Triton**：没有 int4 这种 dtype，一般把打包好的权重存成 `int32`，在 kernel 里自己拆：

  ```python
  w = tl.load(Wq_ptrs)                    # int32，每个装 8 个 int4
  q = (w >> shift) & 0xF                  # shift = 4 * j，拆出第 j 个
  b = (q.to(tl.bfloat16) - 8) * s         # 整数 → 浮点，减零点，乘 scale
  ```

  能跑，但 `.to(tl.bfloat16)` 编译出来是普通的整数 → 浮点转换，不是上面的 `lop3` 技巧，转换吞吐会成为瓶颈（16 < 28，见上）。想要同样的技巧，可以用 `tl.inline_asm_elementwise` 把那几条 PTX 嵌进去。FP8 和 MXFP4 不用这么麻烦：`.to()` 直接用原生 `cvt`，Blackwell 上 `tl.dot_scaled` 还能直接走 block-scaled MMA。

### 权重离线重排

TL;DR：`mma` 要求每个线程拿着 B 里几个分散位置的元素。bf16 由 `ldmatrix` 在运行时用硬件分发到位；int4 的元素比 `ldmatrix` 的最小单位（16 bit）还小，而且解包之后直接进 `mma`，中间没有机会再调位置，所以只能离线把权重排成「每个线程读进来的正好是自己要的、顺序也适合 `lop3`」。

**bf16 不需要操心。** `mma.m16n8k16` 的 B 是 16 × 8（K × N），每个线程 2 个 32 bit 寄存器、4 个元素。lane 号为 $\ell$ 的线程，记 $t = \ell \bmod 4$，它要的是

$$
n = \lfloor \ell / 4 \rfloor,\qquad k \in \{2t,\ 2t{+}1\}\ (\text{b0}),\quad \{2t{+}8,\ 2t{+}9\}\ (\text{b1})
$$

这 4 个元素在 W 里分属不同的行。bf16 GEMM 用 `ldmatrix.trans`：每个线程给一行 smem 地址，硬件按行读、再按上面的位置把元素发给各线程。这一步重排是硬件免费做的。

**int4 多出两个问题，`ldmatrix` 都解决不了：**

1. **一个 32 bit 装 8 个元素，分属不同线程。** GPTQ 的 checkpoint 沿 K 打包，`qweight` 形状 `[K/8, N]` int32，一个 word 是同一列的 $k = 0..7$。按上式，这 8 个值分给一个 quad 的 4 个线程，每个线程只要 2 个。`ldmatrix` 拆不开半字节，直接读的话每个线程要把整个 word 读进来、只用 1/4，再移位挑出自己那几个。
2. **解包后的顺序必须就是 fragment 顺序。** 转换在寄存器里做：1 个 int4 word 解出 4 个 bf16x2，直接喂给 `mma`。所以一个线程读到的 word 里必须正好是它 fragment 要的 8 个值。而且上面的 `lop3` 一次抠出相隔 16 bit 的第 0、4 个半字节组成一个 bf16x2，所以 fragment 里成对的 $(k, k{+}1)$ 也得事先放到第 0、4 个半字节上。

**Marlin 的做法是加载模型时重排一次**，让每个线程一条 16 B 向量读拿到的全是自己要的（[Frantar et al., 2024](https://arxiv.org/abs/2408.11743)）。块大小 16 × 64 就是这么算出来的：

- 16 B = 32 个 int4；每条 `m16n8k16` 每个线程要 4 个 B 值，32 个值够 8 条 `mma`，即 8 个 n8 tile
- 一个 warp：32 线程 × 32 个值 = 1024 = 16 × 64，正好一个 $k16 \times n64$ 的块

所以权重按 16 × 64 的块连续存放，块内按「线程 → fragment → `lop3` 半字节顺序」排好。运行时一个 warp 每线程一条 16 B 读覆盖一块，不用 `ldmatrix`、不用 shuffle：

```python
# 一个 warp 处理一个 16 × 64 的 B 块（重排后）
q4 = lds128(Wq_s[blk, lane])            # 16 B = 4 个 word，全是这个线程要的
for w in range(4):                      # 每个 word → 2 个 n8 tile 的 fragment
    b = dequant(q4[w])                  # 4 条 lop3 + 4 条 hfma2 → 4 个 bf16x2，已是 fragment 顺序
    acc[2*w]   = mma(a, b[0:2], acc[2*w])
    acc[2*w+1] = mma(a, b[2:4], acc[2*w+1])
```

代价是权重不再是普通的行主序张量：每种 kernel 有自己的格式。推理框架在加载 GPTQ / AWQ 的 checkpoint 时做这一步重排，比如 vLLM 的 `gptq_marlin_repack`。

### 小 batch 时的并行度

decode 时 $B$ 只有几到几十，$M$ 方向只有一个 tile。按 $BN = 128$ 切 $N = 8192$ 只有 64 个 tile，132 个 SM 喂不满，和 [GEMV 的问题](/gpu/tensor-core-gemm#面试追问) 一样。Marlin 用 striped partitioning：把所有 (K 段, N 段) 的工作平均切给每个 SM，一个 SM 的那一条可以跨好几列，同一列的部分和最后再合并，思路和 [Stream-K](/gpu/split-k-stream-k) 相同。

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

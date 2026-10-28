---
title: GEMM Tiling
status: draft
tags: [gemm, tiling]
difficulty: 3
order: 2
related: [/gpu/gpu-architecture, /gpu/compute-units, /gpu/special-instructions, /inference/prefill-decode-roofline, /handson/cuda-tiled-matmul, /gpu/triton, /inference/quantization, /gpu/quant-gemm, /gpu/moe-grouped-gemm, /gpu/memory-bound-kernels]
stack: [k-gemm, hw-gpu]
train: [tr-k-gemm]
---

# GEMM Tiling

> 把矩阵切成块，让每个读进来的字节被用很多次

## 一句话结论

GEMM 本身的算术强度很高（4096³ 约 1365 FLOP/B），但前提是 $A$、$B$ 只读一次，而片上放不下整个矩阵。tiling 的做法是把 $C$ 切成 $BM \times BN$ 的块，每块只需要 $A$ 的一条行条带和 $B$ 的一条列条带，沿 $K$ 一段一段搬进片上反复用，算术强度是 $\frac{BM \cdot BN}{BM + BN}$。同样的思路在 shared memory → 寄存器这一级再做一次。tile 越大复用越多，但受 shared memory 和寄存器容量限制，而且 tile 越大块数越少，SM 可能喂不满。decode 的 GEMV 的 $M$ 只有 batch 那么大，没有东西可复用，tiling 救不了。

整页的手段按存储层级排开：

<GemmOverview />

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

## 内积和外积：tiling 在两者之间折中

$C = AB$ 有两种基本算法，差别只在 $k$ 这层循环放在最内还是最外。

### 内积：一个输出一个输出地算

$$
C_{ij} = \sum_{k} A_{ik} B_{kj} = \mathbf{a}_i^\top \mathbf{b}_j
$$

$\mathbf{a}_i$ 是 $A$ 的第 $i$ 行，$\mathbf{b}_j$ 是 $B$ 的第 $j$ 列，$C$ 的每个元素是一次长度为 $K$ 的内积：

```python
for i in range(M):
    for j in range(N):
        acc = 0                        # 只要 1 个累加器
        for k in range(K):             # k 在最内层
            acc += A[i, k] * B[k, j]   # 每做 1 次 MAC 读 2 个数
        C[i, j] = acc
```

上面的朴素版就是它。

### 外积：一层一层地叠

$$
C = \sum_{k} A_{:,k}\, B_{k,:}
$$

$A_{:,k}$ 是 $A$ 的第 $k$ 列（$M \times 1$），$B_{k,:}$ 是 $B$ 的第 $k$ 行（$1 \times N$），两者的外积是一个 $M \times N$ 的秩 1 矩阵。$C$ 是 $K$ 个这样的矩阵叠起来：

```python
C = zeros(M, N)                        # M·N 个累加器同时活着
for k in range(K):                     # k 在最外层
    a, b = A[:, k], B[k, :]            # 读 M + N 个
    for i in range(M):
        for j in range(N):
            C[i, j] += a[i] * b[j]     # 做 M·N 次 MAC
```

| | 内积 | 外积 |
|---|---|---|
| $k$ 循环 | 最内层 | 最外层 |
| 同时活着的累加器 | 1 | $MN$ |
| 每一轮 | 读 2 个，做 1 次 MAC | 读 $M + N$ 个，做 $MN$ 次 MAC |
| 每读 1 个数做几次 MAC | $\frac{1}{2}$ | $\frac{MN}{M + N}$ |
| 省的是 | 空间 | load |

内积几乎不占空间，但读进来的数只用一次。外积读进来的 $a_i$ 被 $N$ 个输出各用一次、$b_j$ 被 $M$ 个输出各用一次，代价是整个 $C$ 的部分和要一直留在手边：4096 × 4096 的 fp32 是 64 MB，寄存器和 shared memory 都放不下。

### tiling：块间内积，块内外积

把 $C$ 切成 $T_M \times T_N$ 的块。块和块之间按内积走（一块沿 $k$ 累加完再换下一块），块内部按外积走：

$$
C_{IJ} = \sum_{k} A_{I,k}\, B_{k,J}
$$

$I$、$J$ 是一块的行、列下标集合，$A_{I,k}$ 是 $T_M$ 维的列向量，$B_{k,J}$ 是 $T_N$ 维的行向量：

```python
for I in tiles(M, TM):
    for J in tiles(N, TN):
        acc = zeros(TM, TN)                    # TM·TN 个累加器
        for k in range(K):
            acc += outer(A[I, k], B[k, J])     # 读 TM + TN 个，做 TM·TN 次 MAC
        C[I, J] = acc
```

$T_M = T_N = 1$ 就是内积，$T_M = M$、$T_N = N$ 就是外积，中间每一档都是两者的折中：

$$
\text{每读 1 个数做的 MAC} = \frac{T_M T_N}{T_M + T_N}, \qquad \text{空间} = T_M T_N
$$

空间固定为 $S = T_M T_N$ 时，方块 $T_M = T_N = \sqrt{S}$ 复用最高，每读 1 个数做 $\sqrt{S} / 2$ 次 MAC：**复用只随空间的平方根增长**。L2 那节「一波铺成方块最省」是同一个结论。

$k$ 这一维不占空间。它是归约维，按时间流过去，所以 $BK$ 只决定每轮搬多少，不影响复用（下面「AI 和 $BK$ 无关」那条）。累加器原地不动、$A$ 和 $B$ 从旁边流过的这种做法，在加速器文献里叫 output stationary。

<InnerOuterViz />

### 每一级都在做同一个折中

tiling 本质上是：针对每一级存储的容量，找内积和外积之间最好的折中点。空间放在哪、预算多大由硬件定；能换来多少复用，由 $\frac{T_M T_N}{T_M + T_N}$ 定。下面几节逐级展开：

| 级 | 外积的空间放在哪 | 预算（H100） | 典型块 | 每读 1 个数做几次 MAC |
|---|---|---|---|---|
| 一波 block（L2） | L2 里的 $A$、$B$ 条带 | 50 MB | 8 × 17 个 tile | 每条条带被 132 / 25 ≈ 5.3 个 block 用 |
| block（第 1 级） | 累加器在寄存器，$A$、$B$ 的 tile 在 shared memory | 每 SM 256 KB 寄存器、228 KB smem | 128 × 128 | $\frac{128 \cdot 128}{128 + 128} = 64$ |
| 线程（CUDA core 第 3 级） | 每线程的寄存器 | 每线程 ≤ 255 个 | 8 × 8 | $\frac{64}{16} = 4$ |
| 指令（Tensor Core 第 3 级） | `mma` 的 fragment | 硬件定 | 16 × 8（$k = 16$） | $\frac{16 \cdot 8}{16 + 8} \approx 5.3$ |

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

### L2：让同时跑的 block 读同一份数据

前文假设每个 tile 都自己从 HBM 读。实际上 block 之间有大量重复读，调整 tile 之间的计算顺序，可以让重复的那部分尽量命中 L2。

先看重复在哪。tile $(i, j)$ 要读 $A$ 的第 $i$ 条**行条带**（$BM$ 行 × 整个 $K$）和 $B$ 的第 $j$ 条**列条带**（整个 $K$ × $BN$ 列），K 循环里 `As`、`Bs` 一段段扫过的就是这两条：

```
        A (M×K)                    B (K×N)                 C (M×N)
  ┌───────────────────┐       ┌────┬────┬────┐       ┌────┬────┬────┐
  │                   │       │    │████│    │       │    │    │    │
  ├───────────────────┤       │    │████│    │       ├────┼────┼────┤
  │███████████████████│ BM 行 │    │████│    │   =   │    │ ▓▓ │    │  ← tile (i, j)
  ├───────────────────┤       │    │████│    │       ├────┼────┼────┤
  │                   │       │    │████│    │       │    │    │    │
  └───────────────────┘       └────┴────┴────┘       └────┴────┴────┘
     A 的第 i 条行条带           B 的第 j 条列条带
```

同一行的 tile 共用一条 $A$ 行条带，同一列的 tile 共用一条 $B$ 列条带。global 的读都经过 L2：同一波的 block 同时在跑，它们要的每条条带只需从 HBM 读一次进 L2，其余 block 读的都是 L2 里这一份。

我们无法手动控制tile的计算顺序，然而在硬件驱动里，硬件按 block 编号从小到大往 SM 上派，所以我们可以手动调每个tile的blockid.

最直接的做法是按行优先编号：一波的 block 排在 C 的同一行（或少数几行）里，大家共用 A 的行条带，但 B 的列条带几乎每个 block 各读各的。一波铺成又扁又长的形状，A 复用得多、B 复用得少。更好的办法是分组走：每 GROUP_SIZE_M 行一组，组内先往下走再往右，让每一波尽量铺成方块，A 和 B 两边都有复用。这个技巧在 Triton 里叫 grouped ordering（GROUP_SIZE_M），在 CUTLASS 里叫 threadblock swizzle / rasterization。


#### 小例子：4 × 4 个 tile，一波 4 个

假设只有 4 个 SM，一波跑 4 个 block。点「下一波」，看每一波点亮的 4 个 block 一共要从 HBM 读几条条带：

<L2WaveViz />

把第一波（编号 0–3）摊开看。格子里是 block 编号：

```
行优先映射                       分组映射（每 2 行一组，组内按列走）
      列0 列1 列2 列3                  列0 列1 列2 列3
行0  [ 0][ 1][ 2][ 3]           行0  [ 0][ 2][ 4][ 6]
行1  [ 4][ 5][ 6][ 7]           行1  [ 1][ 3][ 5][ 7]
行2  [ 8][ 9][10][11]           行2  [ 8][10][12][14]
行3  [12][13][14][15]           行3  [ 9][11][13][15]
```

第一波是 0–3：

| 映射 | 第一波覆盖 | 从 HBM 读的条带 |
|---|---|---|
| 不靠 L2（每个 block 各读各的） | — | 4 × 2 = 8 条 |
| 行优先 | 1 行 × 4 列 | 1 + 4 = 5 条 |
| 分组 | 2 行 × 2 列 | 2 + 2 = 4 条 |

一波的 tile 数固定，读的条带数 = 覆盖的行数 + 列数。和面积固定时方块周长最短是一回事：铺成方块比铺成一长条省。

#### 算一下省多少

换成 H100 上的真实规模：$M = N = K = 4096$，$BM = BN = 128$，$BK = 64$。

**1. 有多少 tile，一波几个。** $C$ 切成 $4096 / 128 = 32$ 行 × 32 列 = 1024 个 tile。132 个 SM 各跑 1 个 block，一波 132 个 tile。

**2. 一条条带多大。**

- $A$ 的一条行条带：$A$ 的 128 行 × 全部 4096 列，$128 \times 4096$ 个 bf16，$128 \times 4096 \times 2\ \text{B} = 1$ MiB。
- $B$ 的一条列条带：全部 4096 行 × $B$ 的 128 列，也是 1 MiB。

对上前面的代码：K 循环转 $4096 / 64 = 64$ 轮，每轮搬进来的 `As` 是 $128 \times 64 \times 2\ \text{B} = 16$ KB，64 轮加起来正好 1 MiB，就是一整条 $A$ 行条带；`Bs` 同理。所以**一个 block 跑完，正好把一条 $A$ 行条带和一条 $B$ 列条带各读一遍，共 2 MiB**。

**3. 一波算多少。** 每个 block 算 $128 \times 128$ 个输出，每个输出 $4096$ 次 MAC：

$$
132 \times 2 \times 128 \times 128 \times 4096 \approx 1.77 \times 10^{10}\ \text{FLOP}
$$

**4. 一波要从 HBM 读几条条带。** 字节 = 条带数 × 1 MiB，有效 AI = FLOP ÷ 字节：

| 映射 | 一波覆盖 | 到 HBM 的条带 | 字节 | 有效 AI（FLOP/B） |
|---|---|---|---|---|
| 不靠 L2 | 各读各的 | 132 × 2 = 264 | 264 MiB | 64 |
| 行优先 | 5 行 × 32 列 | 5 + 32 = 37 | 37 MiB | ≈ 457 |
| 分组，`GROUP_SIZE_M = 8` | 8 行 × 17 列 | 8 + 17 = 25 | 25 MiB | ≈ 676 |

三行的 HBM traffic 怎么来的（每条条带 1 MiB）：

- **不靠 L2**：132 个 block 各读自己的 1 条 $A$ + 1 条 $B$，同一条被几个 block 读就算几次：$132 \times 2 \times 1\ \text{MiB} = 264$ MiB。
- **行优先**：编号 0–131 一行行排，132 = 4 × 32 + 4，占满第 0–3 行，再占第 4 行前 4 格。碰到的 $A$ 行条带是第 0–4 行，5 条；前 4 行是整行，32 条 $B$ 列条带全碰到。每条只从 HBM 读一次，其余命中 L2：$(5 + 32) \times 1\ \text{MiB} = 37$ MiB。
- **分组**：每 8 行一组，组内一列列往下走，每列 8 个。132 = 16 × 8 + 4，占满前 16 列，再占第 17 列 4 格，碰到 8 条 $A$、17 条 $B$：$(8 + 17) \times 1\ \text{MiB} = 25$ MiB。

1. **光靠 L2 命中**，HBM 字节从 264 MiB 降到 37 MiB，有效 AI 从 64 涨到 457，越过 ridge 295，GEMM 回到 compute-bound。
2. **分组映射**再把 37 降到 25，HBM 字节少三分之一。
3. 前提是这一波的条带在 L2（50 MB）里待得住。37 MiB 放得下；而且同一波的 block 沿 $K$ 大致同步推进，同一时刻真正要留住的只是每条条带的当前一段，要求比整条放下宽松。

分组就是 Triton 教程里的 `GROUP_SIZE_M`（CUTLASS 叫 threadblock swizzle / rasterization，和 shared memory 里避 bank conflict 的 swizzle 不是一回事），见 [Triton](/gpu/triton#一个完整的-matmul-kernel)：

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

这一节先用 CUDA core 的写法（每个线程自己发 FMA）讲寄存器这一级的复用。换成 Tensor Core 以后，「每个线程算一小块」由 `mma` 指令的片段取代，见下面的 [三级 tiling](#三级-tiling)。

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

用一个缩小的 block tile（16 × 16，$BK = 8$）看一个线程在每个 `kk` 里读什么、算什么。切换 $TM \times TN$，看 load 和 FMA 的比例怎么变：

<RegTileViz />

每个 `kk` 读 $TM + TN$ 个数，做 $TM \cdot TN$ 次 MAC。数一下每个 `kk` 里的指令：

| 每线程输出 | shared memory load | FMA | FMA 占指令的比例 |
|---|---|---|---|
| 1 × 1 | 2 | 1 | 33% |
| 8 × 8，标量 load | 16 | 64 | 80% |
| 8 × 8，`float4` 向量 load | 4 | 64 | 94% |

一个 warp 调度器每周期只发 1 条指令，load 占掉的发射槽 FMA 就用不上（[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) 的 kernel 4、5 有实测）。代价是累加器：8 × 8 个 fp32 就是每线程 64 个寄存器，下面「tile 选多大」会算到。

CUTLASS 在线程之上还有一层 warp tile：一个 warp 负责 $WM \times WN$ 的子块，道理相同，只是复用的单位从一个线程换成 32 个线程。

## 三级 tiling

把上面两级和最内层的矩阵乘指令串起来，就是 CUTLASS 的三级结构（[CUTLASS：Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)）。第三级是什么，取决于用哪种计算单元：

| | CUDA core（SIMT FMA） | Tensor Core |
|---|---|---|
| 第 1 级 | block tile：global → shared memory | block tile：global → shared memory |
| 第 2 级 | warp tile：shared memory → 寄存器 | warp tile：shared memory → 寄存器 |
| 第 3 级 | 线程 tile：每线程 $TM \times TN$（上一节） | **指令 tile**：一条 `mma` 的形状，硬件定 |

上一节的线程 tile 是 CUDA core 版的第三级。下面按 Tensor Core 版写：

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

### 第三级：指令 tile

最内层的 tile 是一条 Tensor Core 指令一次算的块。Ampere 上是 `mma.sync.m16n8k16`：一个 warp 一起发，算 $C[16, 8] \mathrel{+}= A[16, 16] \cdot B[16, 8]$，共 16 × 8 × 16 = 2048 次 MAC。A 片段的每个元素被 $n = 8$ 列复用，B 片段的每个元素被 $m = 16$ 行复用，道理和前两级一样。

和前两级不同的有两点：

1. **形状不能选**。能选的只是一个 warp tile 里排多少个指令块：64 × 64 的 warp tile 是 $(64 / 16) \times (64 / 8) = 32$ 个块，每个 $k = 16$ 的步长发 32 条 `mma`。所以 $WM$ 得是 16 的倍数，$WN$ 得是 8 的倍数。
2. **没有「每个线程算一小块」了**。A 片段、B 片段和累加器摊在 warp 32 个线程的寄存器里，每个线程拿哪几个元素由硬件规定（fragment layout）：fp16 下每线程 A 有 8 个、B 有 4 个、累加器 4 个 fp32。从 shared memory 读片段用 `ldmatrix`，一条指令就按这个 layout 把数据分到 32 个线程。

指令一代比一代大：Hopper 的 `wgmma` 由一个 warpgroup（4 个 warp）发，形状 m64nNk16，$N$ 最大 256；Blackwell 的 `tcgen05.mma` 一个线程就能发，最大 m128，两个 SM 配对能到 m256，累加器放进 TMEM。指令大到这个程度，warp tile 和指令 tile 基本重合：Hopper 上一个 warpgroup 负责的 64 × 256 往往就是一条 `wgmma`。为什么要越做越大、一条指令要跑多少个周期，见 [普通 SIMT 之外的指令](/gpu/special-instructions) 第 3 节。

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

## 离最优还有多远

「最优」分两层看：搬运的字节能不能再少，以及算的时间能不能更接近峰值。这里的最优都指经典的 $O(MNK)$ 算法；Strassen 一类算法能减少 FLOP，但数值稳定性差、访存不规则，GPU 上的 GEMM 库不用。

### 搬运：tiling 贴着 I/O 下界

快存储容量为 $S$ 个数时，任何经典矩阵乘法在快慢存储之间搬的数据量都有下界。Hong 和 Kung 在 1981 年证明了渐近形式 $\Omega(MNK / \sqrt{S})$，后来的工作把领头项的常数也定了下来（[Smith et al., arXiv:1702.02017](https://arxiv.org/abs/1702.02017)；[Kwasniewski et al., 2019, arXiv:1908.09606](https://arxiv.org/abs/1908.09606)）：

$$
Q \ge \frac{2MNK}{\sqrt{S}} + \text{低阶项}
$$

方块 tiling 取 $T_M = T_N = \sqrt{S}$，搬的量是 $\frac{2MNK}{T} = \frac{2MNK}{\sqrt{S}}$，正好是领头项。所以 [内积和外积](#内积和外积-tiling-在两者之间折中) 那节的「每读 1 个数最多做 $\sqrt{S}/2$ 次 MAC」不只是 tiling 这个算法的性质，而是所有经典算法的上限：**在搬运上，方块 tiling 已经是最优的**。

实际 kernel 离下界还差一个常数，因为快存储不能全给累加器。拿 block 这一级算：H100 每 SM 的寄存器是 256 KB，即 $S = 65536$ 个 fp32，下界对应 256 × 256 的 tile，每读 1 个数做 128 次 MAC。但寄存器还要放地址、片段和流水用的变量，实际最大常用 128 × 256，每读 1 个数做 $\frac{128 \cdot 256}{128 + 256} \approx 85$ 次，global → smem 的流量是下界的 1.5 倍。这点差距不影响结果：有了 L2，有效 AI 早就越过了 ridge。

### 计算：compute-bound 之后，最优就是峰值

搬运不再是瓶颈，时间的下界就是 FLOP ÷ 峰值。4096³ 是 $1.37 \times 10^{11} / 989\ \text{TFLOP/s} \approx 0.14$ ms。到不了这个数的原因：

| 损失 | 4096³、128 × 128 tile 上粗算 | 怎么补 |
|---|---|---|
| wave quantization | 1024 个 tile 排 8 波，最后一波空 32 个 SM：约 3% | 选 tile 大小、Stream-K |
| epilogue 写回不重叠 | 一个 tile 主循环 $\frac{2 \cdot 128 \cdot 128 \cdot 4096}{989\ \text{TFLOP/s} / 132} \approx 18\ \mu s$；写回 32 KB 的 bf16，按每 SM 分到的 $3.35\ \text{TB/s} / 132 \approx 25\ \text{GB/s}$ 要约 1.3 μs：约 7% | persistent kernel + TMA store；CUTLASS 的 ping-pong kernel 让两个 consumer warpgroup 算不同的 tile、交替进 epilogue，一个写回时另一个在算 |
| 流水填充和排空 | 每个 tile 开头要等前几个 stage 搬完才能算 | persistent kernel 提前搬下一个 tile 的数据 |
| 频率 | 989 按 1.83 GHz 算；满载撞功耗墙时频率会掉，掉多少和输入数据有关 | 无法靠 kernel 补；测量时锁频并注明配置 |

前三项是 kernel 设计能补的，Hopper 上的 persistent kernel、warp specialization 主要就在补这几项；最后一项是硬件的事。实测的百分比随卡型（SXM / PCIe）、频率、矩阵形状和输入分布变化很大，引用别人的数字前先看配置。作为参照，FlashAttention-3 在 H100 上做到 75%（作者自测）。

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

::: details Q：为什么 decode 的 GEMV 打不满？
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
- [A Tight I/O Lower Bound for Matrix Multiplication (arXiv:1702.02017)](https://arxiv.org/abs/1702.02017)：经典矩阵乘法的 I/O 下界 $2MNK/\sqrt{S}$
- [Red-blue pebbling revisited (arXiv:1908.09606)](https://arxiv.org/abs/1908.09606)：下界的低阶项和并行版本

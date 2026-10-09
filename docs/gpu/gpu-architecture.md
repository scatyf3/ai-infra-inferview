---
title: GPU 执行模型：SM / Warp / Shared Memory
status: draft
tags: [cuda, sm, warp]
difficulty: 3
order: 1
related: [/gpu/tensor-core-gemm, /gpu/profiling, /inference/prefill-decode-roofline, /handson/kernel-mindset, /handson/cuda-tiled-matmul]
stack: [hw-gpu, hw-mem]
---

# GPU 执行模型：SM / Warp / Shared Memory

> bank conflict、coalescing、occupancy

## 一句话结论

GPU 是「很多 SM，每个 SM 同时挂着很多 warp，用 warp 之间的切换来藏访存延迟」。一个 warp 是 32 个线程，执行同一条指令；shared memory 是每个 SM 上由程序员自己管理的片上存储。写 kernel 的三个基本功都从这里推出来：

1. **coalescing**：一个 warp 的 32 个线程访问连续地址，global memory 的读写才不浪费。
2. **避免 bank conflict**：一个 warp 访问 shared memory 时别让多个线程打到同一个 bank。
3. **occupancy 够用**：SM 上驻留的 warp 够多（或者每个 warp 发出的独立访存够多），才藏得住几百个时钟周期的访存延迟。

## 推导

### 硬件：SM 和内存层次

本页的数字都用 H100 SXM（compute capability 9.0），出处见每行：

| 部件 | H100 SXM | 出处 |
|---|---|---|
| SM 数 | 132 | [Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| 每 SM 的 Tensor Core | 4 个（全卡 528） | 同上 |
| 每 SM 的 FP32 CUDA core | 128 | 同上 |
| 寄存器 | 每 SM 65536 个 32-bit（256 KB） | [Hopper Tuning Guide §1.4.1.1](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| 每 SM 最多驻留 | 64 个 warp（2048 线程）、32 个 block | 同上 |
| shared memory | 每 SM 最多 228 KB，单个 block 最多 227 KB；和 L1 共用 256 KB | 同上；Hopper In-Depth |
| L2 | 50 MB，全卡共享 | 同上 |
| HBM3 | 80 GB，3.35 TB/s | [H100 产品页](https://www.nvidia.com/en-us/data-center/h100/) |

从上到下越来越大、越来越慢：寄存器（每个线程私有）→ shared memory / L1（同一 SM 上的 block 内共享）→ L2（全卡共享）→ HBM。CUDA Best Practices Guide 给的量级是 global memory 访问有「hundreds of clock cycles」的延迟（[Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)）。后面所有优化都在回答一件事：怎么少跑 HBM，跑的时候怎么不让 SM 干等。

### 软件层次：grid / block / warp / thread

写 kernel 时程序员看到四层：

1. **grid**：一次 launch 的全部 block。block 之间互不通信、执行顺序不保证，所以没有跨 block 的 `__syncthreads`；要全局同步，就切成两个 kernel（或用 atomic）。
2. **block**：整体调度到**一个** SM 上，跑完才释放。block 内线程共享 shared memory，可以用 `__syncthreads()` 互相等。
3. **warp**：硬件把 block 里的线程按线性编号每 32 个切成一个 warp，warp 是真正的调度和执行单位，32 个线程执行同一条指令（SIMT，见 [CUDA Programming Guide §1.2.2.2 Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)）。
4. **thread**：有自己的寄存器和下标。

下标约定（行主序：二维数组 `A[r][c]` 存在地址 `r * ncols + c`）：

```cpp
// 1D：每个线程负责一个元素
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] = a * x[i] + y[i];

// 2D block：线性编号先走 x 再走 y，每 32 个线性编号组成一个 warp
int tid  = threadIdx.y * blockDim.x + threadIdx.x;
int warp = tid / 32, lane = tid % 32;

// 2D 矩阵：x 方向对列，y 方向对行，这样同一 warp 里 threadIdx.x 连续 → 访问同一行的连续列
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

最后两行的映射不是随意的：让 `threadIdx.x`（warp 内变化最快的那一维）对应内存里连续的那一维，就是下一节的 coalescing。

### coalescing：global memory 按 32 B sector 读

compute capability 6.0 以上，一个 warp 的一次访存会被合并成若干个 32 字节的事务，事务数等于覆盖这 32 个线程地址所需的 32 B sector 数（[Best Practices Guide：Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)）。设 `A` 是行主序 float（4 B）矩阵，warp 内线程 `t = 0..31`：

1. **按行读 `A[row][t]`**：32 个地址连续，共 32 × 4 = 128 B，正好 **4 个 sector**，搬进来的每个字节都有用。
2. **按列读 `A[t][col]`**：相邻线程差一整行（`ncols × 4` 字节），每个线程落在不同 sector，要 **32 个 sector** = 1024 B，只用到其中 128 B，利用率 4 / 32 = **12.5%**。

一般式：warp 内相邻线程地址相差 $s$ 个元素，每元素 $b$ 字节，起始地址 32 B 对齐。32 个地址横跨 $32 \cdot s \cdot b$ 字节：

$$
\text{sector 数} = \min\left(32,\ \left\lceil \frac{32 \cdot s \cdot b}{32} \right\rceil\right) = \min(32,\ s \cdot b), \qquad
\text{利用率} = \frac{32 \cdot b}{32 \cdot \text{sector 数}}
$$

float（$b = 4$）连续读（$s = 1$）是 4 个 sector、利用率 100%；$s = 2$ 是 8 个、50%；$s \ge 8$ 后每个线程独占一个 sector，封顶 32 个、12.5%。不对齐时多一个 sector。ncu 的 Memory Workload Analysis 里会给出每个请求实际用了几个 sector，见 [Profiling](/gpu/profiling#_3-定了-bound-之后看哪几节)。

矩阵转置是逃不开的例子：读 `in` 按行、写 `out` 就按列，总有一边跨行。解法是借 shared memory 把跨行那一步挪到片上：block 按行读一个 32×32 tile 进 shared memory，再从 shared memory 按列取、按行写回 global。两次 global 访问都合并了（Mark Harris，[An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)，文中有各版本的带宽实测）。代价是 shared memory 那一步按列读，引出 bank conflict。

### shared memory 和 bank conflict

shared memory 分成 32 个 bank，连续的 4 字节字依次落在连续的 bank 上，所以 float 下标 `i` 落在 bank `i % 32`。一次访存里不同线程访问**不同 bank** 可以同时完成；多个线程访问**同一 bank 的不同地址**要串行；多个线程访问**同一地址**是广播，不冲突（[Best Practices Guide：Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)；Mark Harris，[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)）。

转置里的 tile，一个 warp 读一列 `tile[t][c]`，`t = 0..31`：

1. **`__shared__ float tile[32][32]`**：`tile[t][c]` 的下标是 `32t + c`，bank 是 `(32t + c) % 32 = c`，32 个线程全打在 bank `c` 上，**32 路冲突**，这次读要分 32 次完成。
2. **`__shared__ float tile[32][33]`**：每行多 1 个 float，下标变成 `33t + c`，bank 是 `(33t + c) % 32 = (t + c) % 32`，`t` 从 0 到 31 各不相同，**无冲突**。代价是多 32 个 float，约 3%。

按行读 `tile[r][t]` 时两种声明都不冲突（`r*32 + t` 或 `r*33 + t` 对 32 取模都覆盖 32 个不同 bank），所以 padding 专门解决按列访问 tile 的情况。GEMM 里同样的问题用 swizzle（按位异或重排地址）解决，不浪费空间，CUTLASS 和 Hopper 的 TMA 都内建了这种布局。

### occupancy：多少个 warp 才藏得住延迟

**定义**：occupancy = SM 上实际驻留的 warp 数 ÷ 上限（H100 是 64）。为什么要驻留很多 warp：warp scheduler 每个周期从「已经准备好」的 warp 里挑一个发指令；一个 warp 在等访存时，别的 warp 顶上，延迟就被藏住了。

**算法**：一个 block 能不能上 SM，要同时满足四个上限，取最紧的那个：

$$
\text{blocks/SM} = \min\left(
\left\lfloor \frac{65536}{\text{regs/thread} \times \text{threads/block}} \right\rfloor,\
\left\lfloor \frac{228\,\text{KB}}{\text{smem/block} + 1\,\text{KB}} \right\rfloor,\
\left\lfloor \frac{64}{\text{warps/block}} \right\rfloor,\
32
\right)
$$

smem 那一项的 1 KB 是 CUDA 给每个 block 保留的（[Hopper Tuning Guide §1.4.2.4](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)）。实际寄存器按 warp 以固定粒度分配，会有取整，精确值用 ncu 的 Occupancy 一节或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查。

**例 1（只看寄存器）**：每线程 128 个寄存器。每个 warp 要 128 × 32 = 4096 个，65536 / 4096 = **16 个 warp**，occupancy = 16 / 64 = **25%**。要 100% 就得把寄存器压到 65536 / 2048 = **32 个/线程**。

**例 2（四个上限一起）**：block 256 线程（8 warp）、每线程 64 寄存器、每 block 48 KB shared memory。

1. 寄存器：65536 / (64 × 256) = 4 个 block。
2. shared memory：228 / (48 + 1) = 4.65 → 4 个 block。
3. warp：64 / 8 = 8 个 block。
4. block 上限：32。

取最小值 4 个 block = 32 个 warp，occupancy 50%，寄存器和 shared memory 同时卡住。

**要多少才够**：用 Little 定律算「为了打满带宽，路上要同时有多少字节」：

$$
\text{在途字节} = \text{带宽} \times \text{延迟}
$$

延迟取 500 ns 做量级估算（假设值：Best Practices Guide 只说几百个时钟周期，H100 的实测值要用 microbenchmark 量）。全卡在途字节 = 3.35 TB/s × 500 ns ≈ 1.7 MB，摊到 132 个 SM 是每 SM 约 **12.7 KB** 在路上。

- 每个线程一次只发一个 4 B 的 load：需要 12.7 KB / 4 B ≈ 3200 个线程同时在等，超过每 SM 2048 线程的上限，**occupancy 100% 也打不满带宽**。
- 每个线程发 `float4`（16 B）的向量 load，或者连续发 4 个互不依赖的 load（ILP）：需要约 800 个线程 ≈ 25 个 warp，occupancy 40% 就够。

这就是 Volkov「Better Performance at Lower Occupancy」的论点：藏延迟靠的是在途的独立操作数，warp 多（TLP）和每个 warp 发得多（ILP）可以互换（[Volkov, GTC 2010](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)）。GEMM 和 FlashAttention 常年 occupancy 很低：它们用大量寄存器存 tile 和累加器，换来的数据复用比多驻留几个 warp 值钱得多（Simon Boehm 的 [matmul worklog](https://siboehm.com/articles/22/CUDA-MMM) 在真实 kernel 上按寄存器和 shared memory 算过 occupancy）。

结论：occupancy 是手段不是目标。先在 ncu 里确认是延迟没藏住（DRAM 和 SM 吞吐都低，stall 原因以 long scoreboard 为主），再考虑用 `__launch_bounds__` 限制寄存器或减少 shared memory；寄存器压得太狠会溢出（spill）到 local memory，local memory 实际在 HBM 上，反而更慢。

### warp divergence

warp 的 32 个线程共用一条指令流。`if` 两边都有线程要走时，硬件先执行一边（另一边的线程被屏蔽、空等），再执行另一边，两条路径的时间相加（[CUDA Programming Guide §3.2](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：「the warp executes each branch path taken, disabling threads that are not on that path」）。

```cpp
// 有 divergence：同一个 warp 里奇数、偶数线程走不同分支
if (threadIdx.x % 2 == 0) a(); else b();

// 没有 divergence：分支条件按 warp 为单位一致
if ((threadIdx.x / 32) % 2 == 0) a(); else b();
```

判断方法：看分支条件在同一个 warp 的 32 个线程上是不是相同。边界判断 `if (i < n)` 只会在最后一个 warp 里分叉，可以忽略。

## 面试追问

::: details Q：warp divergence 是什么，为什么 attention 的 causal mask 不会严重 divergence？
同一 warp 内线程走不同分支时，硬件串行执行两条路径。causal mask 按 tile 处理：设 Q tile 下标 `qi`、K tile 下标 `kj`，`kj < qi` 的 tile 全部可见，`kj > qi` 的 tile 全部被 mask、整块跳过，这个判断按 tile 做，同一个 block 的线程走同一路。只有 `kj == qi` 的对角 tile 要逐元素 mask，而且用 `where(allowed, s, -inf)` 这样的选择指令（`allowed = q_idx[:, None] >= k_idx[None, :]`），不产生分支。FlashAttention-2 的 causal 实现就是跳过全 mask 的 tile（[Dao, 2023, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）。
:::

::: details Q：行主序 float 矩阵，一个 warp 读 `A[row][t]` 和 `A[t][col]` 各要几个 sector？
按行 4 个 sector（128 B 全有用）；按列 32 个 sector（1024 B 里只用 128 B，利用率 12.5%）。见上面 coalescing 一节。
:::

::: details Q：`tile[32][32]` 按列读几路 bank conflict？为什么 `[32][33]` 就没了？
`[32][32]` 时 `tile[t][c]` 的 bank 恒为 `c`，32 路冲突；`[32][33]` 时 bank 是 `(t + c) % 32`，32 个线程各占一个 bank。
:::

::: details Q：每线程 128 个寄存器，occupancy 多少？要不要提到 100%？
65536 / (128 × 32) = 16 个 warp，16 / 64 = 25%。不一定要提：按 Little 定律，在途的独立访存够多就行，ILP 可以替代 warp 数。确认是延迟没藏住再动寄存器。
:::

::: details Q：`__syncthreads()` 写在 `if` 里会怎样？
只有当条件在整个 block 上取值一致时才允许；否则一部分线程到不了这个 barrier，行为未定义，常见表现是挂死（[CUDA Programming Guide：C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)）。边界判断要写在 barrier 前后的读写上，不要把 barrier 包进去。
:::

::: details Q：L2 有 50 MB，decode 的权重能放进 L2 吗？
放不进。70B 模型 bf16 权重 140 GB，8B 模型也有 16 GB，比 L2 大两到三个数量级，每步 decode 都要从 HBM 重新读一遍，这就是 decode memory-bound 的来源（[roofline](/inference/prefill-decode-roofline)）。L2 有用的地方是 GEMM 里相邻 block 复用同一条 A / B 条带（见 [Tensor Core 与 GEMM](/gpu/tensor-core-gemm)），以及小的激活、KV 的局部复用。
:::

## 手撕

常见题：

1. 解释 `threadIdx` / `blockIdx` 到全局下标的映射（上面的代码块）。
2. 给一段 kernel，指出哪里访存不合并。
3. 写合并访存、无 bank conflict 的矩阵转置：

```cpp
#define TILE 32
#define ROWS 8   // block 是 32 × 8 个线程，每个线程搬 4 行
// in: n × n 行主序；launch: grid(cdiv(n,32), cdiv(n,32)), block(32, 8)
__global__ void transpose(float* out, const float* in, int n) {
    __shared__ float tile[TILE][TILE + 1];          // +1：按列读时错开 bank

    int x = blockIdx.x * TILE + threadIdx.x;        // in 的列
    int y = blockIdx.y * TILE + threadIdx.y;        // in 的行
    for (int j = 0; j < TILE; j += ROWS)            // 按行读 in：warp 内 x 连续，合并
        if (x < n && y + j < n)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;            // 交换 block 坐标：out 的列
    y = blockIdx.x * TILE + threadIdx.y;            // out 的行
    for (int j = 0; j < TILE; j += ROWS)            // 按行写 out：合并；按列读 tile：靠 padding 避开冲突
        if (x < n && y + j < n)
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}
```

这是 Mark Harris 转置博客里的 `transposeNoBankConflicts`，博客里有朴素版、加 shared memory 版、再加 padding 版的带宽对比。入门见 [kernel mindset](/handson/kernel-mindset)，tiled matmul 见 [CUDA Tiled Matmul](/handson/cuda-tiled-matmul)。

## 参考

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)：[Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)、[Advanced Kernel Programming（divergence）](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)、[Compute Capabilities 表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：32 B 事务、shared memory bank、occupancy
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：228 KB smem、65536 寄存器、64 warp、50 MB L2
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Mark Harris：[How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)、[Using Shared Memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)、[An Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Vasily Volkov, Better Performance at Lower Occupancy (GTC 2010)](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)

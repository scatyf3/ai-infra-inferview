---
title: 普通 SIMT 之外的指令
status: draft
tags: [cuda, ptx, tensor-core, tma]
difficulty: 3
order: 1.5
related: [/gpu/gpu-architecture, /gpu/tensor-core-gemm, /handson/cuda-reduce, /handson/cuda-tiled-matmul, /inference/flash-attention, /gpu/triton]
stack: [hw-gpu]
---

# 普通 SIMT 之外的指令

> 本页讲 naive SIMT 模型之外的指令。[GPU 执行模型](/gpu/gpu-architecture) 里的 `Warp.step` 只描述了一种指令：32 个线程各自读写自己的寄存器，做一次标量运算。真实的 SM 里还有几块专用硬件，对应的指令大致分三类：**搬数据**（warp 内的 shuffle，HBM 和 shared memory 之间的 `cp.async` / TMA）、**特化计算**（专做矩阵乘加的 Tensor Core，以及吞吐很低的 SFU）、**编排异步流水**（`mbarrier`、`setmaxnreg`）。其中不少在 CPU 上也有对应，比如 AVX 的 shuffle、AMX 的矩阵指令。本页只做一张地图，细节去对应的页面。数字都用 H100 SXM。

## 总览

naive SIMT 的指令长这样：

```python
# naive SIMT：每个线程只碰自己的寄存器，一次算一个标量
for t in warp.threads:                     # 同时执行，不是循环
    t.regs[d] = op(t.regs[a], t.regs[b])
```

下面六类各自在一点上不同于它：

| 大类 | 小节 | 打破了哪条假设 | 代表指令 | 最早架构 | CPU 上的对应 |
|---|---|---|---|---|---|
| 搬数据 | 1. 访存 | 每个线程自己搬 4 B，并且经过寄存器 | 向量化 load、`cp.async`、TMA、`ldmatrix` | sm_75 / 80 / 90 | AVX 向量 load、`prefetch` |
| 搬数据 | 2. warp 内通信 | 只能读自己的寄存器 | `__shfl_*_sync`、`__ballot_sync`、`__reduce_*_sync` | Kepler 起就有 | AVX 寄存器内 shuffle（`vpermps`、`vpshufb`） |
| 特化计算 | 3. 矩阵 | 一次算一个标量 | `mma.sync` → `wgmma` → `tcgen05.mma` | sm_80 / 90a / 100a | Intel AMX（`TDPBF16PS`）、ARM SME |
| 特化计算 | 4. 特殊函数 | 所有运算的吞吐一样 | `MUFU.EX2`、`RSQ`、`RCP` | 图形时代就有，不是为 AI 加的 | SSE `rcpps` / `rsqrtps`；exp 没有硬件指令，靠多项式 |
| 编排异步流水 | 5. 同步 / 原子 | 只有 `__syncthreads` | `mbarrier`、`atomicAdd`、cluster barrier | sm_80 / 90 | `lock` 前缀原子、`mfence` |
| 编排异步流水 | 6. 资源调配 | 寄存器数在编译时定死 | `setmaxnreg` | sm_90a | 无 |

注意 warp 内通信只发生在同一个 warp 的 32 个 lane 之间。不同 warp 之间不能直接交换寄存器，要经过 shared memory；不同 block 之间要经过 Hopper 的 distributed shared memory（第 5 节）。

只有 Tensor Core 是为 AI 专门加的。SFU 是老硬件，只是在 attention 里成了瓶颈。前两大类的趋势是让数据少经过寄存器、由少数线程代表整块发起，省下的寄存器和线程留给计算。第三大类则负责让搬运和计算重叠起来。

## 1. 访存：搬得更多，以及不经过寄存器

这一类指令改的是两件**互相独立**的事：

| 指令 | 一条指令搬多少 | 经不经过寄存器 | 架构 |
|---|---|---|---|
| `LDG.32`（读一个 `float`） | 每线程 4 B，一个 warp 128 B | 经过 | 一直有 |
| `LDG.128`（读一个 `float4`） | 每线程 16 B，一个 warp 512 B：**搬得更多** | 经过 | 一直有 |
| `cp.async` | 每线程 4 / 8 / 16 B | **不经过**，直接写进 shared memory | sm_80 |
| TMA | **一个线程**发一条，搬一整块 tile：**搬得更多** | **不经过** | sm_90 |
| `ldmatrix` | 一个 warp 读 1 / 2 / 4 个 8×8 的 16-bit 矩阵 | 方向反过来：smem → 寄存器，按 Tensor Core 要的布局分好 | sm_75 |

下面分开讲：1.1 讲「搬得更多」，1.2 讲「不经过寄存器」，1.3 讲把两件事都做到底的 TMA，1.4 讲数据从 smem 进寄存器的最后一程。

### 1.1 搬得更多：向量化 load

**一条 load 指令在做什么。** 第一部分讲过，一条指令由一个 warp 的 32 个线程同时执行。load 指令里每个线程给一个地址、要一个宽度，访存单元把 32 个请求合并成若干个 32 B 的 sector 去 L2 / HBM 取（这就是 [coalescing](/gpu/gpu-architecture)）。宽度可以是 1、2、4、8、16 B，16 B 就是 `LDG.128`，CUDA C++ 里写成读一个 `float4`。

**同样的数据，指令少 4 倍。** 一个 warp 读 512 个连续字节：

| 写法 | warp 级 load 指令数 | sector 数 | 有用比例 |
|---|---|---|---|
| 每线程读 4 次 `float`（`LDG.32` × 4） | 4 | 4 × 4 = 16 | 100% |
| 每线程读 1 次 `float4`（`LDG.128` × 1） | 1 | 16 | 100% |

总流量完全一样，coalescing 也一样好。省下的是：

1. **指令数**：load 指令少 4 倍，配套的地址计算也少 4 倍。每个 scheduler 每周期只发一条指令，elementwise kernel 里 load / store 加地址计算往往比真正的运算还多，省下来的发射槽是实打实的。
2. **在途字节**：一条指令就有 512 B 在路上。按 [GPU 执行模型](/gpu/gpu-architecture) 里 Little 定律的算法，每个 SM 要有 7–13 KB 在途才能把 HBM 带宽用满。16 个 warp 每个读一个 `float` 只有 16 × 128 B = 2 KB，换成 `float4` 就是 8 KB，occupancy 不变而在途字节翻 4 倍。

**一个完整的例子**（`y = a * x`）：

```cuda
__global__ void scale(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
    int i  = blockIdx.x * blockDim.x + threadIdx.x;      // 第 i 个 float4
    int n4 = n / 4;
    if (i < n4) {
        float4 v = reinterpret_cast<const float4*>(x)[i];   // 一条 LDG.128
        v.x *= a; v.y *= a; v.z *= a; v.w *= a;
        reinterpret_cast<float4*>(y)[i] = v;                // 一条 STG.128
    }
    if (i < n % 4) y[n4 * 4 + i] = a * x[n4 * 4 + i];       // 尾巴：最后不足 4 个的元素
}
```

**代价和条件**：

- **地址必须 16 B 对齐**，否则直接报 misaligned address。`cudaMalloc` 和 PyTorch 分配的首地址满足这个条件，但切片之后的 tensor（比如 `x[1:]`）不一定满足。
- **每个线程要 4 个寄存器接这 16 B**。读得越宽，寄存器占得越多。
- **单线程最多 16 B**。想一条指令搬更多，只能换一种指令，也就是 1.3 的 TMA。
- bf16 / fp16 时 16 B 是 8 个元素，常用 `uint4` 或 4 个 `__nv_bfloat162` 来读。

### 1.2 不经过寄存器：`cp.async`

**普通写法怎么把数据搬进 shared memory。** GEMM、attention 都要先把一个 tile 从 HBM 搬进 smem，再在片上反复使用。在 Ampere 之前只能这样写：

```cuda
// 普通写法：两条指令，数据在寄存器里中转
float4 r = A4[g];        // LDG：HBM → 寄存器
smem4[s] = r;            // STS：寄存器 → shared memory
```

数据本来只需要从 HBM 到 smem，却在寄存器里停了一下。原因是普通 load 指令的语义只有一种：「读回来，写进某个寄存器」。这个中转有两个代价。

**代价一：线程卡在 `STS` 上。** `STS` 要用 `r`，`r` 要等 `LDG` 回来，HBM 的延迟约 479 个周期（第一部分的实测值）。这段时间里这个 warp 什么也做不了。

**代价二：想不卡住，就要拿寄存器来换。** Ampere 之前的 double buffering 是这样写的：先把下一个 tile load 进寄存器，然后去算当前 tile，算完再把寄存器里的数据 `STS` 进 smem。线程确实不卡了，但下一个 tile 在路上的这段时间，它一直占着寄存器。算一下有多少：

- GEMM 常见的 tile 是 BM = BN = 128、BK = 32，bf16。A 的 tile 是 128 × 32 × 2 B = 8 KB，B 也是 8 KB，一级共 16 KB。
- block 有 256 个线程，每个线程要暂存 16 KB / 256 = 64 B，也就是 **16 个寄存器**。
- 想让更多级同时在路上，每多一级就再多 16 个。而这时累加器已经占了每线程 64 个以上的寄存器（128 × 128 个 fp32 累加器分给 256 个线程）。

**`cp.async` 换了一种语义**：指令里只给源地址和目的地址，数据由访存单元直接写进 smem，不进寄存器。

```cuda
// cp.async（PTX）：只给两个地址
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;   // 发出就不管，线程继续往下走
cp.async.commit_group;                                     // 把之前发出的拷贝打成一组
// ... 先去算别的 ...
cp.async.wait_group 0;                                     // 等到本线程的所有组都完成
__syncthreads();                                           // 再等其他线程的那份也到
```

CUDA C++ 里可以用 `__pipeline_memcpy_async` 或 `cuda::memcpy_async`，编译出来是同一条指令（[CUDA Programming Guide：Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)）。

**硬件为什么做得到。** H100 每个 SM 的 L1 和 shared memory 是同一块 256 KB 的 SRAM，按配置在两者之间划分，smem 最多 228 KB（[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)）。数据从 L2 回来本来就要进这块 SRAM，直接落在 smem 那一侧即可，不必先送到寄存器文件再写回来。`.cg` 表示不在 L1 里留副本，只写 smem（只支持 16 B）；`.ca` 会在 L1 里也留一份（支持 4 / 8 / 16 B）（[PTX ISA：cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）。

**两个代价都没了**：中转寄存器从每级 16 个变成 0 个；线程发完拷贝就继续往下走，到真正要用数据时才在 `wait_group` 处等。

**多级流水就这么写**。`wait_group N` 的意思是「等到本线程最多还剩 N 组没完成」，所以保持 S − 1 组在路上：

```python
# S 级流水（S 份 smem buffer 轮流用）
for k in range(S - 1):                 # 先把前 S-1 个 tile 的拷贝发出去
    cp_async(buf[k % S], tile[k]); commit_group()

for k in range(num_tiles):
    cp_async(buf[(k + S - 1) % S], tile[k + S - 1]); commit_group()   # 再发一块
    wait_group(S - 1)                  # tile[k] 那一组到了（后面还有 S-1 组在路上）
    syncthreads()
    compute(buf[k % S])                # 算 tile[k]，同时 S-1 块在路上
    syncthreads()                      # 算完才能让下一轮覆盖这块 buffer
```

S = 3 时，每个 block 有 2 × 16 KB = 32 KB 在路上，远超 7–13 KB 的需求，而且一个寄存器都不占。

**注意 `wait_group` 只管本线程**：每个线程只能确认自己发的那几份拷贝已经到了，所以后面还要一个 `__syncthreads()`，等所有线程的那份都到齐。这一点 TMA 用 `mbarrier` 解决得更干净。

### 1.3 TMA：一个线程搬整块

`cp.async` 解决了中转，但还剩一件事：**每个线程仍然要自己算地址**。搬一个 128 × 32 的 bf16 tile 要 512 个 16 B 的拷贝，256 个线程每个算 2 次「第几行、第几列、有没有越界、指针是多少」，每次都要好几条整数指令。

TMA（Tensor Memory Accelerator，Hopper 新加的硬件单元）把这些也交给了硬件：

1. **host 端**用 `cuTensorMapEncodeTiled` 把整个张量描述好：基地址、各维大小、各维 stride、一次搬的 box 大小、swizzle 方式。这个描述叫 tensor map，作为 kernel 参数传进去（[CUDA Driver API：Tensor Map](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)）。
2. **kernel 里**只要一个线程发一条指令，给出 tile 的坐标（下面的函数名是示意，真实写法是 inline PTX 或 CUTLASS 的封装）：

```cuda
__shared__ alignas(128) bf16 As[128][32];
__shared__ uint64_t bar;                          // mbarrier，见第 5 节

if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(&bar, 128 * 32 * 2);   // 告诉 barrier：要等 8 KB 到齐
    // cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    tma_load_2d(&As, &tmap_A, /*col=*/k * 32, /*row=*/m * 128, &bar);
}
mbarrier_wait(&bar, phase);                       // 所有线程等字节数到齐
```

**硬件替你做了什么**：

- **地址生成**：从坐标和 tensor map 算出每一行的地址。
- **越界补零**：tile 超出张量边界的部分自动填 0，kernel 里不用写边界判断。
- **swizzle**：写进 smem 时按 32 / 64 / 128 B 的模式把地址打乱，让之后按列读的时候不产生 [bank conflict](/gpu/gpu-architecture)，省掉手动 padding。
- **完成通知**：搬完自动在 `mbarrier` 上扣掉对应的字节数。所有线程等同一个 barrier，不需要 `cp.async` 那样「各等各的、再 `__syncthreads`」。
- **multicast**：在 thread block cluster 里，一条 TMA 可以把同一块数据同时写进多个 SM 的 smem。GEMM 里相邻 block 要用同一条 A 或 B，这样就只从 L2 读一次。

**限制**：全局内存的基地址要 16 B 对齐，各维的 stride 要是 16 B 的倍数，box 每一维最多 256 个元素。

**这对 kernel 结构的影响**：搬数据只要一个线程，于是可以专门分出一个 warp（甚至只用其中一个线程）来发 TMA，其余 warp 全部用来算。这就是 warp specialization：producer 只发 TMA，consumer 只算 `wgmma`。producer 几乎不需要寄存器，第 6 节的 `setmaxnreg` 就是把它的寄存器让给 consumer。FlashAttention-3 和 CUTLASS 的 Hopper GEMM 都是这个结构（[Shah et al., 2024, arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）。

### 1.4 最后一程：`ldmatrix` 从 smem 进寄存器

计算终究要用寄存器里的操作数（Hopper 的 `wgmma` 可以直接读 smem，是例外，见第 3 节）。在 Ampere 上，`mma.sync` 要求操作数按一种固定的、分散的布局放在 32 个线程的寄存器里。以 m16n8k16 的 A（16 × 16 bf16）为例，设 `g = lane / 4`、`c = (lane % 4) * 2`，每个线程持有 8 个元素（[PTX ISA：mma.m16n8k16 fragment](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）：

```python
# lane 号为 lane 的线程，在 4 个 32-bit 寄存器里各存 2 个 bf16
a[0:2] = A[g    ][c   : c+2]
a[2:4] = A[g + 8][c   : c+2]
a[4:6] = A[g    ][c+8 : c+10]
a[6:8] = A[g + 8][c+8 : c+10]
```

用普通的 `LDS` 读，每个线程要算 4 个地址、发 4 条 `LDS.32`。`ldmatrix.x4` 把这个 16 × 16 的块看成 4 个 8 × 8 的小矩阵：32 个线程各提供一行（16 B）的地址，**一个 warp 一条指令**就把 4 个小矩阵读进来，并且直接按上面的布局分到每个线程的 4 个寄存器里。

- 指令数少 4 倍，也不用记 fragment 布局。
- `.trans` 变体在读的同时转置，B 矩阵按列主序存放时用它。
- 8 行地址如果落在同一个 bank 上会有 bank conflict，所以 smem 的布局要做 swizzle。这和 TMA 写入时的 swizzle 正好配套。

### 1.5 小结

| | `LDG` → `STS` | `cp.async` | TMA |
|---|---|---|---|
| 一条指令搬多少 | 每线程最多 16 B | 每线程最多 16 B | 一整块 tile，只要 1 个线程发 |
| 中转寄存器 | 每级流水每线程约 16 个（上面的 GEMM 例子） | 0 | 0 |
| 地址谁算 | 每个线程 | 每个线程 | 硬件，按 tensor map |
| 越界 / swizzle | 自己写 | 自己写 | 硬件做 |
| 线程在哪等 | `STS` 处（等 HBM 回来） | `wait_group` 处，之前可以去算别的 | `mbarrier` 处，之前可以去算别的 |
| 怎么知道全到齐了 | `__syncthreads` | `wait_group` + `__syncthreads` | `mbarrier` 字节计数 |

演进的方向很一致：搬运的发起者从「每个线程」变成「一个线程代表整块」，数据不再经过寄存器。寄存器省下来留给计算，搬数据的线程也可以很少。

## 2. warp 内通信：直接读别人的寄存器

```python
# __shfl_down_sync(mask, v, off)：lane i 读 lane i+off 的 v
vals = [t.regs[v] for t in warp.threads]          # 先把 32 个值都取出来
for i, t in enumerate(warp.threads):
    t.regs[d] = vals[i + off] if i + off < 32 else vals[i]
```

一个 warp 本来就在同一条指令里执行，所以交换数据既**不经过 shared memory**，也**不需要 `__syncthreads()`**。warp 内求和做 log₂32 = 5 轮 shuffle 就够了，见 [cuda-reduce](/handson/cuda-reduce)。

同一类的还有：

- `__ballot_sync(mask, pred)`：每个 lane 给一个 bool，返回一个 32-bit 掩码，每个 lane 都能拿到。`__any_sync` 和 `__all_sync` 是它的简化版本。
- `__reduce_add_sync`（还有 min / max / and / or / xor，sm_80）：一条指令完成 warp 内归约，**只支持整数**，浮点仍然要用 shuffle。

## 3. 矩阵：Tensor Core 的发射粒度越来越大

| 指令 | 架构 | 谁发 | 一条指令的形状 | A / B 在哪 | 累加器在哪 |
|---|---|---|---|---|---|
| `mma.sync` | sm_80 | 1 个 warp | m16n8k16 | 寄存器 | 寄存器 |
| `wgmma.mma_async` | sm_90a | 4 个 warp（warpgroup） | m64nNk16，N 最大 256 | A 在寄存器或 smem，B 在 smem | 寄存器 |
| `tcgen05.mma` | sm_100a | **1 个线程** | 最大 m128 / m256（两个 SM 配对） | A 在 smem 或 TMEM，B 在 smem | TMEM |

TMEM（Tensor Memory）是 Blackwell 每个 SM 新加的一块存储，128 行 × 512 列 × 32 bit = **256 KB**，和寄存器文件一样大（[PTX ISA §9.7.18 Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）。累加器从寄存器搬进 TMEM 以后，寄存器压力大大减轻，这和第 1 节「不经过寄存器」是同一个方向。

### 一条指令为什么能算一整个 tile

这条指令不是在一个周期里算完的。每个 SM 有 4 个 Tensor Core（每个 sub-partition 一个），每个都是固定大小的乘加阵列。用峰值吞吐 ÷（SM 数 × 频率）就能反推出它每周期能做多少 FMA：

| 卡 | dense FP16 峰值 | SM 数 × boost 频率 | 每 SM 每周期 FMA | 每个 TC 每周期 FMA |
|---|---|---|---|---|
| A100 | 312 TFLOPS | 108 × 1.41 GHz | 1024 | 256 |
| H100 | 989 TFLOPS | 132 × 1.83 GHz | 2048 | 512 |

（A100 的数字见 [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)，H100 的数字和本页其他地方一致。）

拿它去除一条指令里的 FMA 数：

- A100 上的 `mma.sync m16n8k16`：16 × 8 × 16 = 2048 FMA，由 1 个 TC 执行，**约 8 个周期**。
- H100 上的 `wgmma m64n256k16`：64 × 256 × 16 = 262144 FMA，由 4 个 TC 一起执行，**约 128 个周期**。

所以一条指令更像交给协处理器的一个命令：TC 里有定序逻辑，按 k 和 n 方向把 tile 切成小块，一块一块地送进阵列。Volta 上还能看到这层拆分，一条 warp 级 MMA 在 SASS 里会展开成多条 `HMMA.884`（分好几个 STEP），见 [Jia et al., 2018, arXiv:1804.06826](https://arxiv.org/abs/1804.06826)。Ampere 起拆分挪进了硬件，`mma.sync m16n8k16` 在 SASS 里只剩一条 `HMMA.16816`。NVIDIA 没公开定序器的细节，以上是根据指令形状和吞吐推出来的。

指令越大，就越需要配套硬件：

| 架构 | 加了什么 | 用途 |
|---|---|---|
| sm_80 | TC 内部定序 | 一条 warp 级指令跑好几个周期 |
| sm_90 `wgmma` | TC 按 descriptor 直接读 shared memory；4 个 sub-partition 的 TC 协同执行；异步执行，用 `wgmma.commit_group` / `wait_group` 等待结果 | B 不用先搬进寄存器；指令发出去马上返回，上百个周期后再等结果 |
| sm_100 `tcgen05` | TMEM；TC 变成 SM 级的单元，一个线程就能发起；完成时由 `tcgen05.commit` 通知 mbarrier | 累加器不占寄存器；发起方式和 TMA 一样，填好描述符就交出去 |

一条指令要跑上百个周期，所以只能做成异步的。这和第 1 节的 TMA 是同一个思路：线程负责下命令，真正干活的是专用硬件。

### 为什么要越做越大：喂数据的通路跟不上

TC 的吞吐每代翻倍，但每个 sub-partition 的调度器每周期仍然只能发 1 条指令，寄存器文件的带宽也没有同比增长。指令太小的话，这两样会先被用满。

**发射槽。** 在 H100 上，一条 `m16n8k16` 只够 TC 算 2048 / 512 = 4 个周期。同一个 warp 还要发 `ldmatrix`、地址计算、循环控制，这些都和 MMA 抢发射槽，TC 就喂不饱。一条 `wgmma m64n256k16` 能让 TC 忙 128 个周期，发射开销就分摊掉了。

**A / B 的读带宽。** 每次 FMA 要读的输入字节数是 A、B 的总字节数除以 FMA 数（FP16，每个元素 2 B）：

- m16n8k16：(16×16 + 16×8) × 2 B / 2048 ≈ **0.375 B/FMA**
- m64n256k16：(64×16 + 16×256) × 2 B / 262144 ≈ **0.039 B/FMA**

m、n 越大，A 的每个元素被 n 列复用、B 的每个元素被 m 行复用，每次 FMA 需要的输入带宽降了约 10 倍。

**累加器的读写带宽。** 每条指令要读一次 C、写一次 D，都是 $m \times n$ 个 FP32：

$$\frac{m n \times 4\,\text{B} \times 2}{m n k} = \frac{8}{k}\ \text{B/FMA}$$

m、n 在式子里约掉了，只剩 k。k = 16 时固定是 **0.5 B/FMA**，比上面 A、B 加起来还多一个数量级，把 tile 做宽也省不下来。这就是 Blackwell 要把累加器从寄存器挪进 TMEM 的原因：累加器只在 TC 旁边读写，不再占用寄存器文件的带宽。

这些指令在 GEMM 里处在哪一级 tile，见 [GEMM Tiling](/gpu/tensor-core-gemm#三级-tiling)。

## 4. 特殊函数：SFU 很窄

`exp2`、`log2`、`rsqrt`、`rcp`、`sin`、`cos` 不走 FP32 单元，而是走 SFU（SASS 里显示为 `MUFU.*`）。两者的吞吐差很多：

| | 每 SM 每周期 | H100 全卡 |
|---|---|---|
| FP32 FMA | 128 | 128 × 2 × 132 × 1.83 GHz ≈ 62 TFLOPS |
| SFU（如 `exp2`） | 16 | 16 × 132 × 1.83 GHz ≈ **3.9 T 次/s** |
| Tensor Core FP16 / BF16 dense | — | ≈ 989 TFLOPS |

这会让 attention 的 softmax 成为瓶颈。设 head dim 为 $d$，每个 (q, k) 对：

- matmul：$QK^\top$ 用 $2d$ FLOPs，$PV$ 用 $2d$ FLOPs，共 $4d$。$d = 128$ 时是 **512**。
- exp：1 次。

matmul 的量是 exp 的 512 倍，但吞吐也高了 989 / 3.9 ≈ 256 倍，所以 **exp 要花掉 matmul 一半的时间**。FlashAttention-3 因此用 pingpong 调度，让一个 warpgroup 做 softmax 时另一个做 GEMM，把这一半时间藏起来（[Shah et al., 2024, arXiv:2407.08608](https://arxiv.org/abs/2407.08608)，3.9 TFLOPS 由论文脚注按 16 / SM / 周期算出）。

硬件只有 `ex2`，没有 `exp`。`__expf(x)`（或者 `-use_fast_math` 下的 `expf`）会编译成 `ex2(x · log₂e)`。所以 FlashAttention 把 softmax scale 和 log₂e 合成一个常数，一条 `FFMA` 加一条 `MUFU.EX2` 就能完成（FA3 附录里的 SASS 就是这个模式）。

### SFU 内部怎么算：查表 + 二次插值

纯查表不行，FP32 精度要的表太大；纯多项式也不行，要做好几次乘法。SFU 把两者合起来：先用输入的高位查一张系数表，再用低位做一次二次插值。这个设计来自 NVIDIA 工程师 [Oberman & Siu, 2005, ARITH-17](https://ieeexplore.ieee.org/document/1467649/)，方法本身见 [Piñeiro, Oberman, Muller & Bruguera, 2005, IEEE TC](https://perso.ens-lyon.fr/jean-michel.muller/QuadraticIEEETC0305.pdf)。同一个单元当年还负责图形里的像素属性插值，所以说它是图形时代的硬件。

以 `ex2(x)` 为例：

```python
def mufu_ex2(x):
    # 1. 范围归约：整数部分不用算，直接写进 FP32 的指数位
    i = floor(x)
    f = x - i                            # f ∈ [0, 1)
    # 2. 把 f 拆成高位和低位
    hi = f 的前几位                       # 决定落在哪个小区间
    lo = f 的剩余位                       # 区间内的偏移
    # 3. 查这个区间的 minimax 二次系数，插值
    c0, c1, c2 = TABLE_EX2[hi]
    mant = c0 + c1 * lo + c2 * lo * lo   # 一个平方器 + 两个乘法器 + 加法
    return mant * 2**i                   # 拼上指数位
```

切多少位、表有多大，NVIDIA 没有公开。

- **第 1 步每个函数不一样。** `ex2` 拆成整数加小数；`lg2` 直接取 FP32 的指数位当整数部分，只对 [1, 2) 里的尾数查表；`rsqrt` 和 `rcp` 也只处理尾数，指数位单独取反或减半。`sin` / `cos` 要先把角度归约到一个周期内，这一步在 SFU 外面用普通 FP32 指令做（Maxwell / Pascal 上有一条专门的 `RRO` 指令，见 [CUDA Binary Utilities 指令集参考](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference)）。
- **第 2、3 步所有函数共用**，只换系数表，所以一个单元能支持六种函数。
- **结果是近似值，不做正确舍入。** PTX 里这些指令都带 `.approx`（`ex2.approx.f32`、`rsqrt.approx.f32`），误差界见 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 各条指令的说明。不开 fast math 的 `expf` 会被展开成一串 `FFMA`，在 FP32 单元上做归约和多项式修正，精度更高，但要十几条指令。

为什么只给 16 个：每个 lane 都要一套查表、平方器和乘法器，面积比一个 FMA 单元大得多，而图形负载里超越函数本来就少，所以数量只有 FP32 单元的 1/8。每个 sub-partition 每周期只能算 4 个，一个 warp 的 32 个 lane 要 8 个周期才能发完一条 `MUFU`。到了 attention，每个 (q, k) 对都要做一次 exp，这个比例就成了上面算出来的瓶颈。

## 5. 同步与原子

| 指令 | 范围 | 做什么 |
|---|---|---|
| `__syncthreads()`（`bar.sync`） | block | 等所有线程都到达这一行 |
| `mbarrier`（sm_80；sm_90 加了字节计数） | block，放在 shared memory 里 | 一个计数器，可以等「N 个线程 arrive」加上「X 个字节到齐」。TMA 搬完会自动扣字节数，consumer 只等这个 barrier，不用等整个 block，这是生产者/消费者流水线的基础 |
| `atomicAdd` 等 | 全卡 | global 上的原子操作在 L2 里执行。浮点原子的累加顺序不固定，所以结果不确定（见下面的追问） |
| cluster barrier、distributed shared memory（sm_90） | cluster（相邻几个 block，可移植上限 8 个，H100 可开到 16 个） | cluster 内的 block 可以互相同步，也可以直接读写对方的 shared memory。这是 [GPU 执行模型](/gpu/gpu-architecture) 第 5 节「没有跨 block 同步」 唯一放宽的地方 |

::: details Q：为什么浮点 `atomicAdd` 的结果不确定？warp reduce 呢？
**两件事叠在一起。** 一是浮点加法不满足结合律，每加一次都要舍入，先加谁会影响结果。fp32 在 1e8 附近的 ulp 是 2^(26−23) = 8，所以：

```
(1e8 + 1) - 1e8 = 1e8 - 1e8 = 0      ← 1e8 + 1 舍入回 1e8，这个 1 丢了
(1e8 - 1e8) + 1 = 1
```

二是 `atomicAdd` 的顺序就是请求到达 L2 的先后：同一地址上的原子操作由 L2 逐个串行执行，谁先到谁先加。CUDA 只保证每次「读、加、写回」是原子的，不保证多个 atomic 之间的顺序。谁先到，取决于 block 什么时候被派上 SM、当时 L2 和访存有多拥挤，每次运行都可能不同。所以同样的输入跑两次，最后几位可能不同，fp32 下相对差约 1e-7。整数 `atomicAdd` 满足结合律，没有这个问题。

PyTorch 的复现性文档把它列为不确定性的来源，FlashAttention 一行写着 "The backward pass uses non-deterministic atomic operations by default"（[PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)）：FA 的 backward 用 atomic 把多个 block 的 dQ 加在一起，`flash_attn_func` 因此提供了 `deterministic=True` 选项。

**warp reduce 是确定的。** shuffle 的配对由代码写死：第一轮 lane i 加 lane i+16，然后是 i+8、i+4……是一棵形状固定的树。32 个线程在同一条指令里执行，没有「谁先到」，所以同样的输入每次结果都逐位相同。block 内用 shared memory 做树形归约也一样。要分清两点：

- **确定，不等于和串行求和一样。** 树形的顺序是 `(v0+v16)+(v8+v24)+…`，CPU 循环是 `((v0+v1)+v2)+…`，最后几位可以不同，但每次都是同一个值。
- **换了配置，树就变了。** block 大小、grid 大小、每个线程先串行累加几个元素，都会改变树的形状和结果。

所以一个 reduce 里，线程内串行累加、warp 内 shuffle、block 内归约这三层都是确定的，不确定性只来自跨 block 合并时用了 atomic。改成每个 block 写 partial，再按下标顺序合并（第二个 kernel，或者「最后完成的 block 收尾」），结果就确定了（[cuda-reduce](/handson/cuda-reduce)）。

**补充一个推理场景**：LLM 推理的 forward 基本不用 atomic，同一个 batch 重复跑，结果是一样的。线上结果不一致，主要原因是负载变化导致 batch size 变化，而很多 kernel 在不同 batch size 下会选不同的归约切法，也就是上面的「换了配置，树就变了」。解决办法是让 kernel 满足 batch invariance（[He / Thinking Machines, 2025, Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）。
:::

## 6. 资源调配：`setmaxnreg`

普通 kernel 每个线程用多少寄存器在编译时就定死了，所有 warp 都一样。`setmaxnreg`（sm_90a）允许一个 warpgroup 在运行中改自己的上限，取值 24 到 256，必须是 8 的倍数，并且整个 warpgroup 的 4 个 warp 都要执行这条指令（[PTX ISA §9.7.21 setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）。

### 先说 warp specialization

**TL;DR**：一个 block 里的 warp 分成两种角色，一部分只搬数据（producer），另一部分只做计算（consumer），中间用 `mbarrier` 交接。搬运和计算各跑各的，就能重叠起来，不用再靠编译器在同一个 warp 的指令流里把两者交错排好。

Ampere 上的做法是每个 warp 既搬又算，用多级 stage 的软件流水把两者错开：

```python
# 所有 warp 跑同一段代码
for k in range(K_tiles):
    cp_async(smem[(k + S - 1) % S], tile(k + S - 1))  # 每个线程自己算地址、发 load
    cp_async_wait(S - 2); __syncthreads()             # 等最老的那一级到齐
    mma(acc, smem[k % S])
```

这种写法有三个问题：

1. **搬和算挤在同一条指令流里。** 一旦等数据，这个 warp 的 `mma` 也发不出去。stage 数和指令顺序都是编译时定的，访存延迟一抖动，等待就暴露出来。
2. **寄存器两头都要留。** 每个 warp 既要放地址计算用的寄存器，又要放累加器。
3. **同步太粗。** `__syncthreads()` 要整个 block 一起等。

Hopper 上分成两种角色：

```python
if warpgroup == 0:                        # producer：1 个线程发 TMA 就够了
    for k in range(K_tiles):
        wait(empty[k % S])                # 等 consumer 用完这个槽
        if lane == 0: tma_load(smem[k % S], tile(k), bar=full[k % S])
else:                                     # consumer：只管算
    for k in range(K_tiles):
        wait(full[k % S])                 # TMA 的字节数到齐就放行
        wgmma(acc, smem[k % S])
        arrive(empty[k % S])
```

这个想法很早就有了，最早见于 Fermi 时代的 CudaDMA（[Bauer, Cook & Khailany, SC'11](https://research.nvidia.com/publication/2011-11_cudadma-optimizing-gpu-memory-bandwidth-warp-specialization)）。但当时 producer 也得一个线程一个线程地经过寄存器去搬数据，要占不少线程和寄存器，代价高。到 Hopper 上这件事才变得划算，靠的是本页前面几节的硬件：

- TMA 只要一个线程就能发起，并且是异步的，producer 几乎不占资源（第 1 节）；
- `mbarrier` 带字节计数，consumer 只等自己要用的那一级 stage，不用等整个 block（第 5 节）；
- `wgmma` 是异步的，consumer 发出去以后还能接着做别的事。FA3 就让两个 consumer warpgroup 轮流做 softmax 和 GEMM（第 4 节）；
- 剩下的问题是 producer 用不完的寄存器，`setmaxnreg` 就是解决这个的。

效果数字见 [GEMM Tiling](/gpu/tensor-core-gemm#hopper-上的变化)。

### 寄存器怎么分

它是为 warp specialization 准备的：producer 只负责发 TMA，用不了几个寄存器；consumer 跑 `wgmma`，累加器要占大量寄存器。CUTLASS 的 Hopper cooperative GEMM 是 1 个 producer warpgroup 加 2 个 consumer warpgroup，producer 降到 40、consumer 升到 232（[sm90_gemm_tma_warpspecialized_cooperative.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp)）：

$$
128 \times (40 + 2 \times 232) = 64512 \le 65536
$$

正好用满一个 SM 的寄存器文件。如果所有 warp 用同一个上限，每个线程最多只能分到 168 个寄存器（65536 / 384 ≈ 170，向下取到 8 的倍数）。

## 怎么用到这些指令

| 写法 | 能用到的 |
|---|---|
| CUDA C++ intrinsic | `__shfl_*_sync`、`__ballot_sync`、`atomicAdd`、`__expf`、`float4` |
| inline PTX：`asm volatile("cp.async.cg.shared.global ...")` | 全部，但 `wgmma` 和 `setmaxnreg` 要用 `-arch=sm_90a` 编译，`tcgen05` 要用 `sm_100a`。`a` 表示架构专属特性，不向后兼容 |
| CUTLASS / CuTe | 把 TMA、`wgmma`、`mbarrier`、`setmaxnreg` 封装成 copy atom 和 MMA atom |
| Triton | `tl.dot` 编译成 `mma` / `wgmma`，`tl.exp2` 编译成 `MUFU.EX2`，Hopper 上的 tensor descriptor 编译成 TMA。warp specialization 需要较新的版本（见 [Triton](/gpu/triton)） |

想确认最终生成了哪条指令，就看 SASS：`cuobjdump -sass` 或 ncu 的 Source 页（[Profiling](/gpu/profiling)）。

## 面试追问

::: details Q：warp 内求和，为什么 shuffle 比 shared memory 树快？
shuffle 直接在寄存器之间交换：每轮只要一条指令，不用先 store 到 smem 再 load 回来，也不用 `__syncthreads()`，因为同一个 warp 本来就在同一条指令里执行。32 个数做 5 轮就够了。
:::

::: details Q：H100 上 attention 的 softmax 为什么会成为瓶颈？
exp 走的是 SFU，每 SM 每周期只能做 16 次，全卡约 3.9 T 次/s，而 Tensor Core 是 989 TFLOPS，差 256 倍。head dim 128 时，matmul FLOPs 只是 exp 次数的 512 倍，所以 exp 要花掉 matmul 一半的时间。FA3 用两个 warpgroup pingpong，让 softmax 和 GEMM 重叠执行。
:::

::: details Q：`cp.async` 和 TMA 有什么区别？
两者都是异步地从 HBM 搬到 shared memory、不经过寄存器。`cp.async` 仍然是每个线程搬 4 到 16 B，地址由每个线程自己算。TMA 由一个线程发一条指令搬一整块多维 tile，地址和越界由硬件处理，完成时通过 `mbarrier` 的字节计数通知。所以 TMA 能让 producer 只占很少的线程和寄存器。
:::

## 参考

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`cp.async`、`cp.async.bulk.tensor`、`ldmatrix`、`mma`、`wgmma`、`tcgen05`（§9.7.18 Tensor Memory）、`mbarrier`、`setmaxnreg`
- [CUDA C++ Best Practices Guide：Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions)：各类指令每 SM 每周期的吞吐
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：TMA、thread block cluster、distributed shared memory
- [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)：单线程发射、TMEM 存累加器
- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)、He / Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)：atomic 与 batch invariance
- Shah et al., *FlashAttention-3*, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)：SFU 吞吐、pingpong 调度、SASS 中 `FFMA` + `MUFU.EX2` 的交错

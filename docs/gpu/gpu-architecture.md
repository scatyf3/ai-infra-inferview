---
title: GPU 执行模型：SM / Warp / Shared Memory
status: draft
tags: [cuda, sm, warp]
difficulty: 3
order: 1
related: [/gpu/tensor-core-gemm, /gpu/profiling, /inference/prefill-decode-roofline, /handson/kernel-mindset, /handson/cuda-tiled-matmul]
stack: [hw-gpu, hw-mem]
train: [tr-hw-gpu]
---

# GPU 执行模型：SM / Warp / Shared Memory

> 先讲硬件长什么样，再从硬件推出写 kernel 的三个基本功：coalescing、bank conflict、occupancy。数字都用 H100 SXM（compute capability 9.0）。

## 第一部分：硬件 overview

### 0. GPU vs CPU：跑的程序不一样

CPU 和 GPU 的区别要从它们跑的程序说起：

| | CPU | GPU |
|---|---|---|
| 典型程序 | 操作系统、数据库、业务逻辑：串行，分支多，前后依赖强 | matmul、attention、逐元素运算：同一个操作作用在几百万个元素上 |
| 互不依赖的活 | 少。比如遍历链表，下一个地址要等上一个读回来才知道 | 极多。每个元素都是一个独立的活 |
| 数据复用 | 多。反复访问的数据（工作集）通常能放进几十 MB 的缓存 | 少。常常是把一大块数据从头扫到尾、每个数只用一次。比如 decode 每步读 35 GB 权重，L2 只有 50 MB |
| 优化目标 | 延迟：单个任务尽快完成 | 吞吐：单位时间处理尽量多的元素 |

下一节会看到，正是这张表决定了两者怎么应对同一个问题。

### 1. 共同的问题：内存墙

现代计算单元的共同问题是 memory 跟不上 compute，也就是内存墙，CPU 和 GPU 都有。对 GPU 来说，从 HBM 读一个数要等约 **479 个时钟周期**，按 1.8 GHz 算约 270 ns（Luo et al. 在同为 Hopper 架构的 H800 上测得，[arXiv:2402.13499](https://arxiv.org/abs/2402.13499)，作者自测）。在这段时间里，计算单元本来可以执行几百条指令。

两者的解法不同：

- **CPU**：用大缓存、预取和乱序执行，尽量让**一个线程**少等。
- **GPU**：不想办法让单个线程少等，而是在芯片上**同时挂着大量线程**，一个在等内存，就换另一个来执行。H100 一共可以同时驻留 132 个 SM × 2048 = **27 万个线程**。

**为什么解法不同**，有 workload 和架构两方面的原因。

1. **workload**（上一节的表）：
   - CPU 的程序数据复用多，缓存能兜住大部分访存，所以加大缓存有效。但程序里互不依赖的活少，等内存时没有别的事可做，只能想办法少等。
   - GPU 的程序是把一大块数据扫一遍，缓存兜不住，等内存躲不掉。但互不依赖的活多到用不完，所以可以靠切换把等待藏起来。
2. **架构**：只有切换足够便宜，靠切换藏延迟才可行。GPU 的切换便宜，不是因为 context 小，而是因为**根本不交换 context**。
   - **GPU**：SM 上驻留的每个 warp，寄存器和程序计数器都同时放在片上的寄存器文件里，各占一块，一直占到 warp 结束。所谓切换，只是 scheduler 这个周期发 warp 3 的指令、下个周期改发 warp 7 的指令，不搬任何数据，所以每个周期都可以换。context 本身并不小：每线程 32 个寄存器时，一个 warp 是 32 × 32 × 4 B = 4 KB，64 个 warp 正好占满一个 SM 的 256 KB 寄存器。所以寄存器文件的总量决定了能同时挂多少线程，第二部分的 occupancy 就是在算这件事。
   - **CPU**：一个核只有一套硬件寄存器。换线程要经过操作系统，先把当前线程的寄存器写回内存、再读入另一个线程的，加上换完之后缓存被冲掉，比等一次内存还慢。x86 的超线程是在一个核里放 2 套寄存器，两个线程之间就能免费切换，这和 GPU 是同一个思路，只是 GPU 每个 SM 放了 64 套。
   - **片上存储花在哪**：CPU 每个核的寄存器只够一两个线程用，片上面积主要给了缓存和乱序执行的控制逻辑。H100 全卡寄存器 132 × 256 KB ≈ 33 MB，和 50 MB 的 L2 是同一量级，都用来同时存几十万个线程的 context。

下面所有的硬件结构，都是围绕「挂很多线程、轮流执行」这一点设计的。

### 2. 计算单元：从小到大五层

把每一层写成一个类：字段是 H100 的数字，方法是这一层能做的事。这是示意，不是真实的 API。

```python
### conceptual
class Thread:
    regs: list[u32]            # 私有寄存器，每线程最多 255 个
    threadIdx, blockIdx        # 自己的下标，用来算该处理哪个元素（见第 5 节）

class Warp:                    # 硬件的执行单位，程序员管不着
    threads: list[Thread]      # 固定 32 个，硬件按线程编号每 32 个切一刀
    pc: int                    # 32 个线程共用一个程序计数器

    def step(self):            # 取一条指令，32 个线程同时执行，各用各的寄存器（SIMT）
        inst = fetch(self.pc)
        for t in self.threads:          # 同时执行，不是循环
            inst.run(t.regs)

class Block:                   # 程序员的协作单位，大小在 launch 时自己定
    warps: list[Warp]          # 线程数 / 32 个；每 block 最多 1024 线程 = 32 warp
    smem: bytes                # shared memory，组内所有线程可见；每 block 最多 227 KB

    def syncthreads(self): ... # 等组内所有线程都到这一行。不同 block 之间没有这个操作

### hardware
class SM:                      # Streaming Multiprocessor，一个「小处理器」
    reg_file  = 65536          # 32-bit 寄存器，共 256 KB；所有驻留 warp 的寄存器同时放在这里
    smem_size = 228 * KB       # 驻留的 block 分着用
    MAX_WARPS, MAX_BLOCKS = 64, 32
    n_schedulers = 4           # 4 个子分区，每个周期各发一条指令
    resident: list[Block] = []

    def can_admit(self, blk) -> bool:
        # 寄存器、shared memory、warp 数、block 数四项都够才放得下。
        # 资源按 block 申请，所以 occupancy 是按 block 数「一个 SM 放得下几个」
        ...

    def admit(self, blk):      # 整个 block 放上来，不会拆到两个 SM；跑完才释放资源
        self.resident.append(blk)

    def cycle(self): ...       # 每个周期挑 warp 发指令，见第 4 节

class GPU:
    sms  = [SM() for _ in range(132)]
    l2   = 50 * MB             # 所有 SM 共享
    hbm  = 80 * GB             # 所有 SM 共享，3.35 TB/s

    def launch(self, kernel, grid, block_dim):
        pending = [Block(kernel, b, block_dim) for b in grid]   # 所有 block 排队
        while pending:         # 哪个 SM 放得下就放上去；顺序不保证，block 之间不能互相等
            for sm in self.sms:
                if pending and sm.can_admit(pending[0]):
                    sm.admit(pending.pop(0))
```

**warp 是理解 GPU 的关键。** 程序员写的是单个线程的代码，但硬件每次取一条指令，交给 32 个线程同时执行，也就是上面的 `Warp.step`。这种方式叫 SIMT（Single Instruction, Multiple Threads，见 [CUDA Programming Guide §1.2.2.2 Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)）。所以后面判断访存好坏、分支好坏，都要看**同一条指令下 32 个线程合起来**在做什么，而不是看单个线程。

SM 的内部结构（4 个子分区、128 个 FP32 单元、16 个 SFU、4 个 Tensor Core，各自每周期算多少）见 [SM 里的计算单元](/gpu/compute-units)。程序员怎么写 block、怎么算下标，以及和 Triton 怎么对应，见第 5 节。`Warp.step` 只描述了普通指令，shuffle、TMA、Tensor Core、SFU 这些专用指令见 [普通 SIMT 之外的指令](/gpu/special-instructions)。

### 3. 存储：每一层谁能看见、多大、多慢

| 层 | 谁能看见 | 容量 | 延迟（周期） |
|---|---|---|---|
| 寄存器 | 单个线程 | 每 SM 65536 个 32-bit = 256 KB | 指令直接读 |
| shared memory | 同一个 block 的线程 | 每 SM 最多 228 KB，单个 block 最多 227 KB | ≈ 29 |
| L2 | 整张卡 | 50 MB | ≈ 263 |
| HBM | 整张卡 | 80 GB，3.35 TB/s | ≈ 479 |

容量来自 [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) 和 [H100 产品页](https://www.nvidia.com/en-us/data-center/h100/)，延迟来自上面 Luo et al. 的实测。每一层的带宽和推理时各层存什么，见 [显存层次](/stack/hw-mem)。

要记住两点：

1. **shared memory 比 HBM 快约 16 倍**（29 周期对 479 周期），但它要**程序员自己管理**：自己写代码把数据从 HBM 搬进来，再自己从里面读。常见用法是把一个 tile 从 HBM 读进来一次，然后在片上反复使用。
2. **寄存器和 shared memory 都是每个 SM 一份、大小固定**。一个 block 用得越多，同一个 SM 上能同时挂的 block 就越少。

### 4. 一个 SM 是怎么跑的

第 2 节里 `GPU.launch` 负责把 block 放上 SM，`SM.cycle` 负责每个周期发指令。展开 `SM.cycle`，132 个 SM 各自独立这样跑：

```python
def cycle(self):                          # SM 的一个时钟周期
    warps = [w for blk in self.resident for w in blk.warps]
    for s in range(self.n_schedulers):    # 4 个 scheduler 各自挑，各管一部分 warp
        w = 在 s 管的 warps 里任意挑一个「下一条指令的操作数已经就绪」的
        if w:
            w.step()                      # 32 个线程一起执行这条指令
        else:
            pass                          # 全都在等内存：空转一个周期（stall）

    for blk in 已经跑完的 block:            # 释放它的寄存器和 shared memory，
        self.resident.remove(blk)         # GPU.launch 才能把下一个 block 放上来
```

第二部分的规则都能从这段伪代码里读出来：

| 伪代码里的哪一句 | 推出的规则 |
|---|---|
| `w.step()` 执行的是一条访存指令 | 32 个地址合起来决定搬多少数据，这是 **coalescing**（读 HBM）和 **bank conflict**（读 shared memory） |
| `can_admit` 决定能挂几个 block；挑不到就 stall | 挂的 warp 越多，越不容易空转，这是 **occupancy** |
| `w.step()` 执行的是一条分支，32 个线程共用一个 `pc` | 线程想走不同分支时只能两边都走一遍，这是 **warp divergence** |

### 5. 程序员看到的：grid / block / thread

写 CUDA kernel 时，程序员只描述线程怎么分组，不直接管 SM 和 warp：

1. **grid**：一次 launch 的全部 block。block 之间互不通信，执行顺序也不保证，所以**没有跨 block 的同步**。需要全局同步时，就拆成两个 kernel，或者用 atomic。
2. **block**：会被整个放到**某一个** SM 上，跑完才释放。block 内的线程共享 shared memory，可以用 `__syncthreads()` 互相等待。
3. **thread**：每个线程用自己的下标算出该处理哪个元素。

warp 是硬件切的：block 里的线程按线性编号，每 32 个一组。

```cpp
// 1D：每个线程负责一个元素
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) y[i] = a * x[i] + y[i];

// 2D block：线性编号先走 x 再走 y，每 32 个线性编号组成一个 warp
int tid  = threadIdx.y * blockDim.x + threadIdx.x;
int warp = tid / 32, lane = tid % 32;

// 2D 矩阵（行主序：A[r][c] 存在地址 r * ncols + c）
int col = blockIdx.x * blockDim.x + threadIdx.x;   // x 方向对应列
int row = blockIdx.y * blockDim.y + threadIdx.y;   // y 方向对应行
```

最后两行的对应关系不是随便定的。同一个 warp 里变化最快的是 `threadIdx.x`，让它对应内存里连续的那一维（列），一个 warp 就会读同一行里连续的 32 个元素。原因见下面的基本功一。

::: tip TL;DR：1D block 和 2D block 有什么区别？
**基本上只是为了写下标方便，把数据的形状和线程的排布对应起来。** 硬件只认一维编号：先把 block 压平成 `tid = x + y * blockDim.x`（x 变化最快），再每 32 个切成一个 warp。所以 `block(256)` 和 `block(32, 8)` 的调度、资源占用、occupancy 完全一样，2D 只是省掉了自己用 `/` 和 `%` 拆行和列。

真正有影响的只有两点：

1. **形状决定哪些线程在同一个 warp，进而影响 coalescing。** x 方向要对应内存里连续的那一维，而且 `blockDim.x` 至少要凑满一个 32 B 的 sector（float 是 8 个），最好是 32。读行主序的 float 矩阵时：`block(32, 8)` 一个 warp 读 1 行，4 个 sector，利用率 100%；`block(4, 64)` 一个 warp 横跨 8 行、每行 16 B，要 8 个 sector，利用率 50%。
2. **各维度的上限不一样**：`gridDim` 的 x 最大 2³¹ − 1，y 和 z 只有 65535；`blockDim` 的 z 最大 64，三维乘积不超过 1024（[Compute Capabilities 表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)）。所以大的那一维要放在 grid 的 x 上。Triton 也一样，`tl.program_id(1)` 和 `tl.program_id(2)` 对应的 grid 维度最大 65535。
:::

**和 Triton 的对应**：Triton 的一个 program 就是 CUDA 的一个 block，`num_warps` 是这个 block 有几个 warp。Triton 不让你写单个线程，`BLOCK` 个元素怎么分给 `num_warps × 32` 个线程由编译器决定（见 [Triton 编程模型](/gpu/triton)）。所以下面三条规则在 Triton 里一样成立，只是编译器替你处理了其中一部分。

## 第二部分：写 kernel 的三个基本功

### 基本功一：coalescing，读 HBM 时让 32 个地址挨在一起

**问题**：一个 warp 执行一条 load 指令，32 个线程各要一个地址。内存系统不会按 4 字节一个个去取，而是以 **32 B 的 sector** 为单位搬运。这次访问要搬多少个 sector，取决于这 32 个地址覆盖了多少个不同的 sector（compute capability 6.0 起，见 [Best Practices Guide：Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)）。

**例子**：`A` 是行主序的 float（4 B）矩阵，warp 内线程 `t = 0..31`：

1. **按行读 `A[row][t]`**：32 个地址连续，共 32 × 4 = 128 B，正好 **4 个 sector**，搬进来的每个字节都有用。
2. **按列读 `A[t][col]`**：相邻线程差一整行，每个线程落在不同的 sector，要搬 **32 个 sector** = 1024 B，只用到其中 128 B，利用率 4 / 32 = **12.5%**。

**规律**：32 个线程的地址挨得越散，要搬的 sector 越多，浪费也越多。连续读最好（4 个 sector）；每个线程落在不同 sector 时最差（32 个）。ncu 的 Memory Workload Analysis 会显示每个请求实际用了几个 sector，见 [Profiling](/gpu/profiling#_3-定了-bound-之后看哪几节)。

**做法**：让 `threadIdx.x` 对应内存里连续的那一维。

**读和写总有一边跨行时怎么办（以矩阵转置为例）**

TLDR：把非合并访存放到快sram里处理

转置是 `out[c][r] = in[r][c]`。如果每个线程把读进来的值直接写出去，线程 t 读 `in[r][t]`，就只能写 `out[t][r]`。读是合并的，写就是跨行的；反过来让写合并，读就跨行。同一个线程读的元素和写的元素在两个矩阵里是转置关系，这一点绕不过去。

解决办法是**让写的线程使用别的线程读进来的值**。寄存器是每个线程私有的，同一个 block 里的线程只能通过 shared memory 交换数据。所以每个 block 负责一个 32×32 的块，分两个阶段：

```cpp
__shared__ float tile[32][32];          // 只放这个 block 负责的 32×32 块，4 KB
// tx = threadIdx.x，ty = threadIdx.y；(x0, y0) 是这个块在 in 里的左上角

tile[ty][tx] = in[y0 + ty][x0 + tx];    // 阶段 1：warp ty 读 in 的一行，合并
__syncthreads();                        // 等整个块都读进来
out[x0 + ty][y0 + tx] = tile[tx][ty];   // 阶段 2：warp ty 写 out 的一行，合并
```

下面可以单步走一遍，对比三种写法：直接转置、经 shared memory 中转、再加 padding。看每一步搬了几个 sector、shared memory 有几路冲突。

<TransposeViz />

阶段 2 里，线程 t 写的值是 `tile[t][ty]`，也就是别的 warp 在阶段 1 读进来的。所以需要 shared memory 来中转，也需要 `__syncthreads()` 来保证那个 warp 已经写进去了。

**这不是预处理。** 整个转置只有一次 kernel launch，每个 block 各搬各的块，整个矩阵在 HBM 上只读一遍、写一遍，和直接转置的读写次数一样，区别在每次搬多少个 sector。一个 warp 处理一行 32 个 float 时：

| | 读 | 写 | HBM 流量 |
|---|---|---|---|
| 直接转置 | 4 个 sector | 32 个 sector | 36 × 32 B = 1152 B |
| 经 shared memory 中转 | 4 个 sector | 4 个 sector | 8 × 32 B = 256 B |

跨行访问其实还在，变成了阶段 2 里按列读 `tile[t][ty]`，只是它发生在 shared memory 里。shared memory 没有 32 B sector 的浪费，但按列读会遇到下一个问题：bank conflict（Mark Harris，[An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)，文中有各版本的带宽实测）。


对于带转置的其他kernel，16 位的 GEMM 和 attention 里，转置是融合在 kernel 内部的，对fp8可能额外有麻烦，需要真的transpose，但一般和其他操作fuse


### 基本功二：bank conflict，读 shared memory 时别让 32 个线程挤到同一个 bank

TLDR: 这里shared memory的back conflict是比如说按非dim=-1维度访问，有可能全落到同一个bank

**bank 是什么**：shared memory 在硬件上不是一整块存储，而是切成了 **32 个能各自独立工作的小存储体**，每一个叫一个 bank。每个 bank 每个周期能读或写一个 4 字节的字。32 个 bank 同时工作，一个周期最多能交出 32 × 4 B = 128 B，正好够一个 warp 的 32 个线程每人拿一个 float。shared memory 的高带宽就是这么来的（实测每 SM 每周期约 128 B，见 [显存层次](/stack/hw-mem)）。

**地址怎么分到 bank 上**：按 4 字节一个字轮流分。第 0 个字在 bank 0，第 1 个字在 bank 1，……，第 31 个字在 bank 31，第 32 个字又回到 bank 0。所以 float 数组的下标 `i` 落在 bank `i % 32`：

```
下标:  0  1  2 … 31 | 32 33 34 … 63 | 64 …
bank:  0  1  2 … 31 |  0  1  2 … 31 |  0 …
```



**问题**：一个 bank 一个周期只能服务一个地址。所以一个 warp 访问 shared memory 时，要看这 32 个地址落在哪些 bank 上：

- 32 个线程访问 **32 个不同的 bank**：一次完成。
- 多个线程访问**同一个 bank 的不同地址**：只能排队，n 个线程挤在一个 bank 上就要分 n 次，叫 **n 路 bank conflict**。
- 多个线程访问**同一个地址**：广播，不算冲突。

（[Best Practices Guide：Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)；Mark Harris，[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)）

**例子**：转置里的 tile，一个 warp 读一列 `tile[t][c]`，`t = 0..31`：

1. **`__shared__ float tile[32][32]`**：`tile[t][c]` 的下标是 `32t + c`，bank 是 `(32t + c) % 32 = c`。32 个线程全挤在 bank `c` 上，**32 路冲突**，这次读要分 32 次完成。
2. **`__shared__ float tile[32][33]`**：每行多放 1 个 float，下标变成 `33t + c`，bank 是 `(33t + c) % 32 = (t + c) % 32`。`t` 从 0 到 31 各不相同，**没有冲突**。代价是多用 32 个 float，约 3%。

按行读 `tile[r][t]` 时，两种声明都不冲突：`r*32 + t` 和 `r*33 + t` 对 32 取模都正好覆盖 32 个不同的 bank。所以 padding 专门用来解决「按列读 tile」。GEMM 里同样的问题用 swizzle 解决，也就是用异或重排地址，不浪费空间。CUTLASS 和 Hopper 的 TMA 都内置了这种布局。

一般地，按列读时相邻线程差一整行。一行是 s 个字时，冲突路数是 gcd(s, 32)：s 是 32 的倍数就全挤在一个 bank，s 是奇数就没有冲突。所以会不会冲突要看行宽对 32 取模后的结果，跟「按哪一维读」没有直接关系。下图是真实规模，32 个 bank、32 个线程。可以换行宽、换 half、打开 swizzle，看 32 个 lane 落在 bank 表的哪里：

<BankConflictViz />

### 基本功三：occupancy，挂够多的 warp 来藏延迟

**问题**：回到第 4 节的伪代码，如果 SM 上挂的每个 warp 都在等内存，scheduler 就只能空转。挂的 warp 越多，越可能有 warp 已经准备好。这一节回答三个问题：挂了多少算多（定义），能挂多少（由资源决定），要挂多少才够（由 Little 定律决定）。

#### 1. 驻留和 occupancy

**驻留（resident）**：block 被 `GPU.launch` 放上 SM（`admit`）之后，它的所有 warp 就驻留在这个 SM 上了。寄存器和 shared memory 一直占着，直到整个 block 跑完才释放。驻留**不等于正在执行**。`SM.cycle` 里 4 个 scheduler 每个周期各挑 1 个 warp，所以整个 SM 每个周期最多只有 4 个 warp 在发指令。其余驻留的 warp 要么操作数已经就绪、只是没轮到（eligible），要么在等内存或等 `__syncthreads()`（stalled）。

**定义**：occupancy = SM 上实际驻留的 warp 数 ÷ 上限（H100 是 64）。

所以 occupancy 衡量的是 scheduler 的**候选池有多大**，不是有多少 warp 在同时计算。候选池越大，某个 warp 等内存时，越可能有别的 warp 顶上。

#### 2. 谁定什么

```cuda
__shared__ float buf[12288];   // 48 KB
kernel<<<4096, 256>>>(...);    // <<<gridDim, blockDim>>>
// nvcc -Xptxas -v 会打印：Used 64 registers, 49152 bytes smem
```

| | 谁定 | 这个例子 | 硬件上限（H100） |
|---|---|---|---|
| 每个 block 几个线程 | 你，`blockDim` | 256 = 8 个 warp | 每 block 1024；每 SM 共 2048 = 64 个 warp |
| 每个 block 多少 shared memory | 你，`__shared__` 数组或动态 smem | 48 KB | 每 block 227 KB；每 SM 共 228 KB |
| 每个线程多少寄存器 | 编译器（`-Xptxas -v` 查看，`__launch_bounds__` 限制） | 64 | 每线程 255；每 SM 共 65536 |
| 一共几个 block | 你，`gridDim` | 4096 | 不受 SM 限制 |
| **一个 SM 同时驻留几个 block** | **硬件**，`can_admit` 看资源够不够 | **4**（下面例 2 算） | 32 |

4096 个 block 不会同时跑。每个 SM 塞到第 4 个 block 时寄存器和 shared memory 都用完了，全卡同时驻留 132 × 4 = 528 个，剩下的 3568 个排队，等前面的 block 跑完、释放了资源再补上去。

#### 3. 能拿到多少：一个 SM 同时放得下几个 block

这就像装箱：SM 是一个箱子，里面有四种容量；block 是一件货，每种容量各占一份。block 只能整个放，不能拆半个，所以能装几件看哪种容量先用完：

$$
\text{blocks/SM} = \min\left(
\left\lfloor \frac{65536}{\text{regs/thread} \times \text{threads/block}} \right\rfloor,\
\left\lfloor \frac{228\,\text{KB}}{\text{smem/block} + 1\,\text{KB}} \right\rfloor,\
\left\lfloor \frac{64}{\text{warps/block}} \right\rfloor,\
32
\right)
$$

$$
\text{occupancy} = \frac{\text{blocks/SM} \times \text{warps/block}}{64}
$$

shared memory 那一项多出的 1 KB 是 CUDA 给每个 block 预留的（[Hopper Tuning Guide §1.4.2.4](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)）。寄存器实际是按 warp、以固定粒度分配的，会有取整。精确值可以看 ncu 的 Occupancy 一节，或者调用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。

**例 1（只看寄存器）**：block 256 个线程，每线程 128 个寄存器。65536 / (128 × 256) = 2 个 block = 16 个 warp，occupancy **25%**。

| 每线程寄存器 | 最多驻留 warp | occupancy 上限 |
|---|---|---|
| 32 | 64 | 100% |
| 64 | 32 | 50% |
| 128 | 16 | 25% |
| 255 | 8 | 12.5% |

这张表是**代价**，不是推荐值：kernel 用了这么多寄存器，occupancy 最多就到这里。另外 occupancy 是按台阶变的。从 128 压到 112，65536 / (112 × 256) = 2.28，还是 2 个 block，occupancy 没变。要放进第 3 个 block，按公式得压到 85 个；但寄存器按每 warp 256 个的粒度分配，85 × 32 = 2720 会取整到 2816，相当于 88 个/线程，还是放不下，实际要压到 80 个。

**例 2（四项一起看）**：block 有 256 个线程（8 个 warp），每线程 64 个寄存器，每 block 用 48 KB shared memory。

1. 寄存器：65536 / (64 × 256) = 4 个 block。
2. shared memory：228 / (48 + 1) = 4.65 → 4 个 block。
3. warp：64 / 8 = 8 个 block。
4. block 上限：32。

取最小值，4 个 block = 32 个 warp，occupancy 50%。寄存器和 shared memory 同时卡住了，想提高就得两个一起动：只把 shared memory 降到 36 KB（能放 6 个），或者只把寄存器降到 48 个（能放 5 个），结果都还是 4；两个都降，才到 5 个 block，occupancy 62.5%。

所以第 2 节开头的 `<<<4096, 256>>>` 要分 4096 / 528 ≈ 7.8 轮（wave）跑完。最后一轮只填满了 0.8，空着的 SM 就在等，这叫 tail effect。

#### 4. 需要多少：Little 定律

上面的约束只能告诉你 occupancy 能到多少，不能告诉你要到多少才够。设计时需要一个目标值：到了这个值就可以停，再往上加没有收益，反而要付出寄存器和 shared memory 的代价。

回到最开始的动机：多挂 warp 是为了在等内存的时候有别的 warp 能顶上。对 memory-bound 的 kernel，比如 elementwise、norm、decode attention，速度就等于 HBM 带宽用了多少。"延迟藏住了"换个说法就是"HBM 一直在忙，没有空闲"。

所以问题变成：要让 HBM 一直忙，同一时刻得有多少字节在路上（请求已经发出、数据还没回来）？这就是 Little 定律：

$$
\text{在途字节} = \text{带宽} \times \text{延迟}
$$

带宽是硬件参数。延迟指一条 load 指令从发出去，到数据回到寄存器、可以用了，中间等的时间。它由三件事决定：数据在哪一层命中、路上的物理开销（片上网络、地址翻译、DRAM 换行）、路上排了多久的队。前两项基本由硬件固定，第三项随负载变化，带宽越接近打满，队越长。

**先看一个 warp**：每个 SM 分到的带宽是 3.35 TB/s ÷ 132 ≈ 25 B/ns。一个 warp 每个线程读一个 `float`，合起来是 128 B。在 270 ns 的空载延迟里，HBM 本来能给这个 SM 送 25 × 270 ≈ 6.9 KB；只有这一个 warp 在等的话，它只要了 128 B，带宽用了不到 2%。要用满，得有几十个这样的请求同时在路上。

**延迟取多少**：270 ns 是第 1 节的空载实测值，是下限。带宽接近打满时要排队，延迟会更长。具体长多少取决于访存模式和负载，没有统一的公开数字，这里取 500 ns 作为保守的上限（这是估算值，不是实测值）。于是每个 SM 需要的在途字节是：

$$
25\,\text{B/ns} \times (270 \sim 500)\,\text{ns} \approx 7 \sim 13\,\text{KB}
$$

这个数可能差 2 倍，但数量级是确定的。它的用法是筛方案：差了几倍的设计，2 倍的误差不影响结论；落在这个范围里的临界情况，交给 ncu 实测。

**每个 SM 的在途字节从哪来**：

$$
\text{每 SM 在途字节} = \text{驻留 warp 数} \times \text{每个 warp 同时在等的字节}
$$

第一项就是第 3 部分算的 occupancy。第二项由代码决定：每个线程读一个 `float` 是 128 B，读一个 `float4` 是 512 B，ILP 连着发 4 个互不依赖的 `float` 也是 512 B。拿几种设计对照 7–13 KB：

| 设计 | 驻留 warp | 每个 warp 在等 | 每 SM 在途 | 对照 7–13 KB |
|---|---|---|---|---|
| 每线程 1 个 `float` | 16（25%） | 128 B | 2 KB | **肯定不够**，只能用到 15–30% 的带宽 |
| 每线程 1 个 `float` | 64（100%） | 128 B | 8 KB | 临界：延迟短时够，长时只有约 64% |
| 每线程 1 个 `float4` | 16（25%） | 512 B | 8 KB | 临界，同上 |
| 每线程 1 个 `float4` | 32（50%） | 512 B | 16 KB | **肯定够** |
| TMA 每个 SM 保持 2 块 64 × 64 的 bf16 tile 在路上 | 和 warp 数无关 | — | 16 KB | **肯定够** |

第二行和第三行的在途字节一样多，但一个 occupancy 是 100%，一个只有 25%。所以设计时只看 occupancy 不够，要看两项的乘积：

- 在途字节明显够了就停，不用为了 occupancy 去压寄存器；
- 不够时，先加大每个 warp 在等的字节，这通常比压资源换 occupancy 便宜。

这就是 Volkov「Better Performance at Lower Occupancy」的观点：藏延迟靠的是路上有多少个独立的访存。多挂 warp（TLP）和每个 warp 多发几个（ILP）可以互相替代（[Volkov, GTC 2010](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)）。提高每个线程在途字节的三种常见写法：

```cuda
// ① 向量化：一条指令搬 16 B（地址要 16 B 对齐）。elementwise、norm 类 kernel 的标准写法
float4 a = reinterpret_cast<const float4*>(x)[i];

// ② ILP：先把互不依赖的 load 全部发出去，再用。指针加 __restrict__，编译器才敢把 load 往前提
float v[4];
#pragma unroll
for (int k = 0; k < 4; k++) v[k] = x[i + k * stride];   // 4 个 load 同时在路上
#pragma unroll
for (int k = 0; k < 4; k++) out[i + k * stride] = f(v[k]);

// ③ 多级流水：GEMM / attention 用 cp.async 或 Hopper 的 TMA 预取后面的 tile
// 发出 tile[0..S-2] 的拷贝
// for k: 发出 tile[k+S-1]；等 tile[k] 到；算 tile[k]    ← 始终有 S-1 块在路上
```


GEMM 和 FlashAttention 走的是 ILP 和多级流水这条路。它们的 occupancy 一直很低：寄存器用来存 tile 和累加器，换来的数据复用比多挂几个 warp 值钱得多（Simon Boehm 的 [matmul worklog](https://siboehm.com/articles/22/CUDA-MMM) 在真实 kernel 上按寄存器和 shared memory 算过 occupancy）。在途字节靠预取 tile 来凑，跟驻留几个 warp 无关，见上表最后一行。

另一种在途字节不够的情况是 block 太少，很多 SM 根本没活干。decode 阶段 query 长度是 1，FlashAttention 只在 batch 和 head 上并行，batch 小时 block 数远少于 SM 数。Flash-Decoding 把 KV 沿序列长度切成多段，各段并行算，最后用 log-sum-exp 合并（[Dao et al., Flash-Decoding, 2023](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)，作者自测长序列端到端 decode 最多快 8 倍）。

#### 5. 结论：先诊断，再动手

occupancy 是手段，不是目标。先用 ncu 确认确实是延迟没藏住：

- DRAM 和 SM 吞吐都明显低于峰值；
- warp stall 的主要原因是 long scoreboard（在等 global memory）；
- Eligible Warps per Scheduler 经常小于 1，也就是 scheduler 经常挑不到 warp。

确认之后，按这个顺序改：

1. 向量化 load 和 ILP，最便宜；
2. 加流水级数；
3. block 太少就切更细，增加并行度；
4. 最后才压资源：用 `__launch_bounds__(256, 2)` 告诉编译器每个 SM 至少要放 2 个 256 线程的 block，它就会把寄存器限制在 65536 / (2 × 256) = 128 个/线程以内；或者减少 shared memory。

压寄存器时要用 `-Xptxas -v` 检查有没有 spill。寄存器压得太狠会溢出到 local memory，而 local memory 实际在 HBM 上，反而更慢。

### 附：warp divergence

**为什么放在附录**：前面三个基本功都在解决访存问题，divergence 浪费的是计算，不在这条主线上。而且对 AI kernel 基本不重要：GEMM、attention、elementwise、norm 的循环次数都由 shape 决定，同一个 warp 的线程走的路一样；mask 用 `where` 这类选择指令，不产生分支；causal mask 按 tile 整块跳过（见面试追问）。只有同一个 warp 里每个线程的工作量不一样时才需要关心，比如每个线程循环次数不同、二分查找、采样里依赖数据的提前退出。

**问题**：一个 warp 的 32 个线程共用一条指令流。遇到 `if` 时，如果两边都有线程要走，硬件只能先执行一边（另一边的线程被屏蔽、空等），再执行另一边，两条路径的时间加起来（[CUDA Programming Guide §3.2](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)：「the warp executes each branch path taken, disabling threads that are not on that path」）。

```cpp
// a() 和 b() 各要 100 个周期
// 有 divergence：同一个 warp 里奇数、偶数线程走不同分支 → 先 a 再 b，200 个周期
if (threadIdx.x % 2 == 0) a(); else b();

// 没有 divergence：条件按 warp 编号算，同一个 warp 走同一边 → 100 个周期
if ((threadIdx.x / 32) % 2 == 0) a(); else b();
```

**判断方法**：看分支条件在同一个 warp 的 32 个线程上是否相同。不同 warp 走不同分支没关系，warp 之间本来就各自调度。边界判断 `if (i < n)` 只会让最后一个 warp 分叉，可以忽略。

## 手撕：矩阵转置

coalescing 和 bank conflict 一起用上。读 `in` 和写 `out` 都按行，所以都是合并访存；跨行那一步放在 shared memory 里，用 padding 避开 bank conflict。

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

这是 Mark Harris 转置博客里的 `transposeNoBankConflicts`。博客里有朴素版、加 shared memory 版、再加 padding 版的带宽对比。入门见 [kernel mindset](/handson/kernel-mindset)，tiled matmul 见 [CUDA Tiled Matmul](/handson/cuda-tiled-matmul)。

## 面试追问

::: details Q：行主序 float 矩阵，一个 warp 读 `A[row][t]` 和 `A[t][col]` 各要几个 sector？
按行读 4 个 sector，128 B 全部有用；按列读 32 个 sector，1024 B 里只用到 128 B，利用率 12.5%。见基本功一。
:::

::: details Q：`tile[32][32]` 按列读是几路 bank conflict？为什么 `[32][33]` 就没有了？
`[32][32]` 时 `tile[t][c]` 的 bank 恒为 `c`，32 路冲突；`[32][33]` 时 bank 是 `(t + c) % 32`，32 个线程各占一个 bank。见基本功二。
:::

::: details Q：每线程 128 个寄存器，occupancy 多少？要不要提到 100%？
65536 / (128 × 32) = 16 个 warp，16 / 64 = 25%。不一定要提高：按 Little 定律，只要路上的独立访存够多就行，ILP 可以替代 warp 数。先确认是延迟没藏住，再去动寄存器。
:::

::: details Q：warp divergence 是什么？为什么 attention 的 causal mask 不会造成严重的 divergence？
同一个 warp 里的线程走不同分支时，硬件要把两条路径依次执行一遍。

causal mask 是按 tile 判断的：设 Q tile 下标为 `qi`、K tile 下标为 `kj`，`kj < qi` 的 tile 全部可见，`kj > qi` 的 tile 全部被 mask、整块跳过。这个判断对整个 block 都一样，不会分叉。只有 `kj == qi` 的对角 tile 需要逐元素 mask。S = 4096、tile 128 时一共 32 × 32 个 tile，要计算的 528 个里只有对角线上的 32 个需要逐元素 mask。逐元素 mask 也是用 `where(allowed, s, -inf)` 这样的选择指令（`allowed = q_idx[:, None] >= k_idx[None, :]`），所有线程执行同一条指令，不产生分支。FlashAttention-2 的 causal 实现就是这样跳过整块被 mask 的 tile（[Dao, 2023, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）。
:::

::: details Q：`__syncthreads()` 写在 `if` 里会怎样？
只有当条件在整个 block 上取值一致时才允许。否则一部分线程到不了这个 barrier，行为未定义，常见的表现是程序挂住（[CUDA Programming Guide：C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)）。边界判断应该写在 barrier 前后的读写语句上，不要把 barrier 包进 `if`。
:::

::: details Q：L2 有 50 MB，decode 的权重能放进 L2 吗？
放不进。70B 模型的 bf16 权重有 140 GB，8B 模型也有 16 GB，比 L2 大两到三个数量级，每步 decode 都要从 HBM 重新读一遍，这就是 decode 受带宽限制的原因（[roofline](/inference/prefill-decode-roofline)）。L2 有用的地方是 GEMM 里相邻 block 复用同一条 A / B 条带（见 [GEMM Tiling](/gpu/tensor-core-gemm#l2-同一波-block-共享条带)），以及小的激活、KV 的局部复用。
:::

## 参考

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)：[Warps and SIMT](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)、[Advanced Kernel Programming（divergence）](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)、[Compute Capabilities 表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：32 B 事务、shared memory bank、occupancy
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：228 KB smem、65536 寄存器、64 warp、50 MB L2
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, [arXiv:2402.13499](https://arxiv.org/abs/2402.13499)：各级存储的延迟实测
- Mark Harris：[How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)、[Using Shared Memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)、[An Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Vasily Volkov, Better Performance at Lower Occupancy (GTC 2010)](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf)

---
title: Triton 编程模型
status: draft
tags: [triton]
difficulty: 3
order: 3
related: [/handson/triton_primitives, /handson/triton-softmax, /handson/kernel-mindset, /gpu/tensor-core-gemm, /gpu/memory-bound-kernels, /framework/torch-compile]
stack: [k-lang]
---

# Triton 编程模型

> autotune；什么时候写 Triton、什么时候直接调 cuBLAS / CUTLASS

## 一句话结论

Triton 让你以「块」而不是「线程」为单位写 kernel：一个 program 处理一个 tile，`tl.load` / `tl.store` 带 mask 处理边界，块内怎么分给线程、怎么合并访存、什么时候进 shared memory、什么时候同步，由编译器决定。适合写 memory-bound 的融合算子（softmax、RMSNorm、各种 attention 变体）和形状特殊的 kernel；标准 dense GEMM 优先调 cuBLAS / CUTLASS，Triton 的 matmul 在常见形状上能接近 cuBLAS，但 Hopper 上最新的硬件技巧（warp specialization 等）一般先出现在 CUTLASS 和手写 CUDA 里。

## 推导

### 编程模型：program 而不是 thread

一个 Triton kernel 描述**一个 program 做什么**。launch 时给一个 grid，grid 里每个点是一个 program，对应 CUDA 的一个 block（Tillet et al., [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations, MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)）。

| 谁决定 | 内容 |
|---|---|
| **你** | grid 有多大、每个 program 负责哪块数据（`tl.program_id` + `tl.arange` 拼出下标向量）、块大小 `BLOCK`、`num_warps`、`num_stages` |
| **编译器** | 块内每个线程拿哪几个元素（layout）、合并访存、要不要经过 shared memory、何时插 barrier、`tl.dot` 怎么映射到 Tensor Core、多级流水怎么排 |

和 CUDA 逐项对照（`threadIdx` 没有对应物、`mask` 代替 `if (i < n)`、二维指针怎么用 stride 拼）见 [triton 原语](/handson/triton_primitives#映射-program-而不是-thread)，这里不重复。下面看一个把这些都用上的完整例子。

<TritonProgramViz mode="matmul" />

### 一个完整的 matmul kernel

约定：$C = AB$，$A$ 是 $M \times K$，$B$ 是 $K \times N$，任意 stride（所以转置后的视图也能直接传进来），fp32 累加，输出 bf16。改写自 [Triton 官方 matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：

```python
import torch, triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],          # 这三个值变了就重新挑配置
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # 1. 一维 pid -> 二维 (pid_m, pid_n)，按 GROUP_M 行一组排
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. 这个 program 负责 C[pid_m*BM : +BM, pid_n*BN : +BN]
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # (BM, BK)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # (BK, BN)

    # 3. 沿 K 循环，累加器在寄存器里
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k0 < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                 # 编译器映射到 Tensor Core
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    # 4. 写回一次（这里就是 epilogue：要融合激活、bias，就在 store 之前对 acc 做）
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

几个要点：

1. **M、N 进 grid，K 进循环**：输出的每个 tile 互不依赖（并行维），K 是归约维，在 program 内部累加。这是 [kernel mindset](/handson/kernel-mindset) 的「并行维 vs 归约维」。
2. **`tl.dot` 就是 Tensor Core**：块大小满足最小形状时，编译器生成 `mma` / `wgmma` 指令；你不用管 fragment 的寄存器布局。为什么要分块、tile 大小怎么影响算术强度，见 [GEMM Tiling](/gpu/tensor-core-gemm#第一级-block-tile-放进-shared-memory)。
3. **`GROUP_M` 只改 pid 的映射，不改计算量**。教程里的数字：$C$ 有 9 × 9 个 tile，按行主序发，算完第一行 9 个 tile 要读 $A$ 的 1 条行条带（9 个块）加 $B$ 的全部 9 条列条带（81 个块），共 90 个块；按 3 × 3 分组发，算完 9 个 tile 只要 $A$ 的 3 条（27 个块）加 $B$ 的 3 条（27 个块），共 54 个块。同时在跑的 program 读的数据更集中，更容易命中 L2。教程报告在 A100 上从约 220 提到 245 TFLOP/s（作者自测）。

### num_warps、num_stages 和 shared memory 预算

- **`num_warps`**：一个 program 用几个 warp 执行，即 `32 × num_warps` 个线程。块大、每个线程分到的元素多时加 warp，降低每线程的寄存器压力。
- **`num_stages`**：K 循环的软件流水级数。编译器在算第 $k$ 块时，提前发出第 $k+1, \dots, k+s-1$ 块的 load，每一级都要在 shared memory 里占一份 $A$、$B$ tile：

$$
\text{smem} \approx \text{num\_stages} \times (BM \cdot BK + BK \cdot BN) \times 2\ \text{B}
$$

| 配置 | 每级 | 总共 |
|---|---|---|
| 128 × 256 × 64，3 级 | $(128 \cdot 64 + 64 \cdot 256) \times 2 = 48$ KB | 144 KB |
| 128 × 128 × 64，4 级 | 32 KB | 128 KB |
| 64 × 128 × 32，4 级 | 12 KB | 48 KB |

H100 每个 block 最多 227 KB（[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)），超了会编译失败（`OutOfResources`）。占得越多，一个 SM 能驻留的 program 越少，下一节算。

### occupancy：编译器算用量，你定参数

occupancy 的定义和装箱公式见 [GPU 架构 · 基本功三](/gpu/gpu-architecture#基本功三-occupancy-挂够多的-warp-来藏延迟)，在 Triton 里不变：一个 program 就是一个 block，资源按 program 申请，一个 SM 放几个 program 看寄存器、shared memory、warp 数哪个先用完。变的是公式的输入由谁写：

| | CUDA | Triton |
|---|---|---|
| 每个 block 几个线程 | 你，`blockDim` | 你，`32 × num_warps` |
| 每个 block 多少 shared memory | 你，`__shared__` 数组 | 编译器算，主要由 `BM`、`BN`、`BK`、`num_stages` 决定（上一节的公式） |
| 每个线程多少寄存器 | 编译器，`__launch_bounds__` 限制 | 编译器，主要由块大小决定；`maxnreg` 限制 |
| 一共几个 block | 你，`gridDim` | 你，`grid` |

所以编译器只是把你选的参数**翻译**成资源用量，不会为了 occupancy 反过来改参数：它不会把 `BM` 调小好多放一个 program。shared memory 超过每 block 上限就报 `OutOfResources`，寄存器超过每线程 255 个就 spill 到 local memory（实际在 HBM 上）。autotune 也不看 occupancy，只看实测时间。

**寄存器看块大小，num_warps 只决定怎么分**。块里每个元素都得有一个线程拿着。matmul 的 fp32 累加器 `acc` 有 $BM \cdot BN$ 个元素，平摊到 $32 \cdot \text{num\_warps}$ 个线程：

$$
\text{累加器寄存器/线程} = \frac{BM \cdot BN}{32 \cdot \text{num\_warps}}
$$

这是下限，地址、mask、正在用的 $A$、$B$ 片段还要另算。乘回线程数，一个 program 至少占 $BM \cdot BN$ 个寄存器，和 `num_warps` 无关。所以加 `num_warps` 不会让一个 SM 多放 program，只是把同一堆寄存器分给更多线程：每线程的数降下来，不容易超过 255 而 spill。

**例：上面 autotune 的三个配置在 H100 上各放几个 program**（寄存器只算了累加器，所以那一列是上限；smem 一项按 228 / (smem + 1)）：

| 配置 | 累加器寄存器/线程 | 寄存器限 | smem | smem 限 | warp 限 | programs/SM | occupancy |
|---|---|---|---|---|---|---|---|
| 128×256×64，3 级，8 warps | 32768 / 256 = 128 | ≤ 65536 / (128 × 256) = 2 | 144 KB | 1 | 8 | **1** | 8 / 64 = 12.5% |
| 128×128×64，4 级，4 warps | 16384 / 128 = 128 | ≤ 4 | 128 KB | 1 | 16 | **1** | 4 / 64 = 6.25% |
| 64×128×32，4 级，4 warps | 8192 / 128 = 64 | ≤ 8 | 48 KB | 4 | 16 | **≤ 4** | ≤ 25% |

前两个 occupancy 只有 6–12%，在大方阵上却常常是 autotune 选出来最快的。按 [Little 定律](/gpu/gpu-architecture#_4-需要多少-little-定律) 看在途字节：`num_stages` = $s$ 时，算第 $k$ 块的同时有 $s - 1$ 块在路上，

- 128×256×64，3 级：2 × 48 KB = 96 KB / SM；
- 128×128×64，4 级：3 × 32 KB = 96 KB / SM。

都是每 SM 需要的 7–13 KB 的近十倍。延迟靠 `num_stages` 预取藏住，不靠多挂 warp；寄存器拿去放大 tile，换数据复用。CUDA 里 GEMM 走「ILP + 多级流水」这条路，在 Triton 里对应的旋钮就是 `num_stages`。

**memory-bound 的 kernel：寄存器 ≈ 一整行**。softmax、RMSNorm 这类一行一个 program 的写法没有 `tl.dot`，shared memory 只用来做跨 warp 归约的暂存，很少；限制几乎全在寄存器上，每线程至少拿 $\text{BLOCK} / (32 \cdot \text{num\_warps})$ 个元素。整个 SM 的寄存器文件是 65536 × 4 B = 256 KB，所以一行 fp32 有多长，就直接决定了能放几个 program：

| 行长 N（fp32） | 一行占寄存器 | programs/SM ≤ | 8 warps 时元素/线程 | 16 warps 时元素/线程 |
|---|---|---|---|---|
| 4096 | 16 KB（1/16 个寄存器文件） | 16，实际被 warp 限卡在 8（8 warps）或 4（16 warps） | 16 | 8 |
| 32768 | 128 KB（1/2） | 2 | 128，加上开销接近 255，容易 spill | 64 |
| 65536 | 256 KB（全部） | 放不下 | — | — |

第二行就是 [Triton 版 Softmax](/handson/triton-softmax#面试追问) 里「行长了加 `num_warps`」的原因：不是为了 occupancy（按寄存器还是最多 2 个 program），是为了不 spill。第三行说明一行一个 program 的写法有硬上限，再长就得分块、跨 program 合并。

**查实际用量，算 persistent grid**。编译后能拿到真实的寄存器数和 shared memory，[fused softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 用它算出每个 SM 放几个 program，然后只发正好一轮的 program，每个 program 循环处理多行（persistent kernel）：

```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties(torch.cuda.current_device())
NUM_SM, NUM_REGS = props["multiprocessor_count"], props["max_num_regs"]   # H100：132，65536
SIZE_SMEM, WARP_SIZE = props["max_shared_mem"], props["warpSize"]         # 每 block 上限，32

# 只编译不运行，拿到资源用量
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                               BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1,))
kernel._init_handles()
n_regs = kernel.n_regs                  # 每线程寄存器；kernel.n_spills 看有没有 spill
size_smem = kernel.metadata.shared      # 每 program 的 shared memory（字节）

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)   # 寄存器限
occupancy = min(occupancy, SIZE_SMEM // size_smem)         # smem 限
num_programs = min(NUM_SM * occupancy, n_rows)
kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)

# kernel 里：for row in tl.range(tl.program_id(0), n_rows, tl.num_programs(0), num_stages=num_stages)
```

这是估算：没算 warp 限（`64 // num_warps`）、寄存器按 warp 的分配粒度、每 block 预留的 1 KB。要精确值，ncu 的 Occupancy 一节对 Triton kernel 一样能用。

**想提高 occupancy 能动什么**：

| 旋钮 | 降什么 | 代价 |
|---|---|---|
| 减小 `BM`、`BN`（或 `BLOCK`） | 每 program 的寄存器和 smem | GEMM 的数据复用变差，算术强度下降 |
| 减小 `num_stages` | smem | 在途字节变少 |
| `triton.Config(..., maxnreg=128)` | 每线程寄存器封顶，对应 PTX 的 `.maxnreg`（[triton.Config 文档](https://triton-lang.org/main/python-api/generated/triton.Config.html)） | 可能 spill，看 `n_spills` |
| 加 `num_warps` | 每线程寄存器（按寄存器能放几个 program 不变） | warp 限 `64 // num_warps` 变紧 |

Triton 没有 `__launch_bounds__(256, 2)` 那种「每 SM 至少放 2 个」的写法，用 `maxnreg` 自己算：`num_warps=8` 时 65536 / (2 × 256) = 128，即 `maxnreg=128`。先诊断再动手的顺序和 CUDA 一样（[GPU 架构 · 结论](/gpu/gpu-architecture#_5-结论-先诊断-再动手)）；实际做法通常是把这些组合都放进 autotune 的 `configs`，让实测决定。

### autotune

`@triton.autotune(configs, key, ...)` 做的事（[triton.autotune 文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)）：

1. 第一次以某组 `key` 的值调用时，把 `configs` 里每个配置都编译、跑若干次计时，选最快的。
2. 结果按 `key` 的值缓存在进程内；`key` 相同的后续调用直接用缓存的配置。`cache_results=True` 会把计时结果存盘，下次启动不用重测。
3. **代价**：第一次调用要编译和测 $|\text{configs}|$ 个 kernel，可能几秒到几十秒。推理服务要在启动时 warmup 掉，或者让 `key` 只包含少数取值（比如 decode 的 batch 分桶后再做 key）。
4. **原地更新的坑**：调优时 kernel 会被跑很多遍。如果 kernel 原地累加输出（比如 split-K 用 `tl.atomic_add` 往 C 里加），每跑一遍都加一次，结果就错了。要把这类参数放进 `reset_to_zero`（每个配置跑之前清零），或用 `restore_value`（跑完恢复）。
5. `prune_configs_by` 可以按形状提前剪掉明显不合适的配置，减少首次调用的时间。

### 什么时候写 Triton，什么时候调库

| 场景 | 选什么 | 原因 |
|---|---|---|
| 标准 dense GEMM（bf16 / fp8） | cuBLAS / cuBLASLt / CUTLASS | 针对每代硬件手调，有大量 kernel 变体和选择 heuristic；Triton 教程里 fp16 matmul 和 cuBLAS 互有胜负（4096² 时约 221 vs 222 TFLOP/s，1024² 时约 95 vs 105，文档构建时的作者自测） |
| memory-bound 的融合算子：残差 + norm、SwiGLU、RoPE、量化 | **Triton** | 收益来自少一次 HBM 读写（见 [kernel fusion](/gpu/memory-bound-kernels#少搬-融合)），几十行就能写完；fused softmax 教程算过：PyTorch 逐算子的 softmax 读 $5MN + 2M$、写 $3MN + 2M$ 个元素，融合后约读写各 $MN$，理论约 4 倍 |
| 形状特殊、库里没有：GQA / paged KV 的 attention 变体、MoE 的 grouped GEMM、自定义 mask | **Triton** | 改一改下标和 mask 就能试 |
| 追求极限的 attention / GEMM（Hopper、Blackwell） | CUTLASS / 手写 CUDA | FlashAttention-3 依赖 TMA + wgmma 的异步和 warp specialization，用 CUTLASS 写（[arXiv:2407.08608](https://arxiv.org/abs/2407.08608)） |
| 模型里一串逐元素 / 归约算子，不想手写 | `torch.compile` | Inductor 自动融合并**生成 Triton**（下一节） |

### 和 torch.compile 的关系

`torch.compile` 的默认后端 TorchInductor 把 PyTorch 程序翻译成 GPU 上的 Triton、CPU 上的 C++（Ansel et al., [PyTorch 2, ASPLOS 2024](https://docs.pytorch.org/assets/pytorch2-2.pdf)）。论文里的分工：

- pointwise、reduction、scatter 这些算子会被融合成少量 Triton kernel，这是 Inductor 最大的收益来源；
- matmul 默认调 cuBLAS，`mode="max-autotune"` 时才会用 Triton 模板生成 matmul，并把后面的 pointwise 算子融进 epilogue，由 autotuner 决定用模板还是 cuBLAS。

所以在推理框架里看到的 Triton kernel，一部分是人写的（vLLM 的一些 attention / MoE kernel），一部分是 Inductor 生成的。生成的代码可以用 `TORCH_LOGS=output_code` 打出来看。详见 [torch.compile](/framework/torch-compile)。

## 面试追问

::: details Q：Triton 的 BLOCK_SIZE 为什么必须是 2 的幂？
`tl.arange(0, BLOCK)` 要求长度是 2 的幂，块内的 layout、reduce、broadcast 都按这个假设生成代码。实际长度不是 2 的幂时，取 `triton.next_power_of_2(n)` 再用 mask 屏蔽越界元素（`other` 填对运算无影响的值：求和填 0，求 max 填 `-inf`）。
:::

::: details Q：Triton 里 occupancy 是编译器帮你优化的吗？
不是。编译器负责块内的 layout、合并访存、shared memory 分配和流水，但只是把你给的 `BLOCK`、`num_warps`、`num_stages` 翻译成寄存器和 shared memory 用量，不会为了多放 program 去改这些参数；autotune 也只按实测时间挑。GEMM 的最优配置 occupancy 往往只有 6–12%，延迟靠 `num_stages` 预取藏住。要控制它：改块大小、`num_stages`，或用 `maxnreg` 给寄存器封顶。算法见 [occupancy：编译器算用量，你定参数](#occupancy-编译器算用量-你定参数)。
:::

::: details Q：autotune 选出来的配置，换一张卡还能用吗？
不一定。最优的块大小和 stage 数取决于 shared memory 容量、SM 数（影响 wave 数）、Tensor Core 形状。同一配置在 A100（每 SM 164 KB smem）上可能因为 shared memory 超了直接编译失败。所以 `key` 只缓存形状，换卡要重新调；部署时在目标卡上 warmup。
:::

::: details Q：split-K 的 Triton kernel 用 `tl.atomic_add` 往 C 里加，为什么接了 autotune 结果就错了？
autotune 会把每个配置跑很多遍计时，每遍都往同一个 C 里加，最后 C 是很多遍的和。把 C 放进 `reset_to_zero=['c_ptr']`，每个配置跑之前清零。另外 atomic 加法的顺序不固定，浮点结果不能逐位复现。
:::

::: details Q：Triton 没有 `__syncthreads`，block 内的同步谁管？跨 program 呢？
block（program）内的同步由编译器插：它知道哪些数据经过了 shared memory、哪里需要 barrier。跨 program 和 CUDA 一样没有全局同步，要么切成两个 kernel（kernel 边界就是同步点），要么用 atomic。详见 [跨 program 归约](/handson/triton_primitives#跨-program-归约)。
:::

::: details Q：同样的 GEMM，Triton 比 cuBLAS 慢，先查什么？
1. autotune 的候选里有没有大 tile（128 × 128 以上）和足够的 stage；只有小 tile 时算术强度上不去。
2. ncu 看 Tensor pipe 利用率，确认 `tl.dot` 真的走了 Tensor Core（输入是 fp32 时可能走 TF32 或退化）。
3. 形状是否对齐（K 不是 16 的倍数、指针不对齐会让向量化 load 退化）。
4. tile 数和 SM 数是否对得齐（[wave quantization](/gpu/tensor-core-gemm#wave-quantization-tile-数和-sm-数对不齐)）。
:::

## 手撕

常见题：用 Triton 写 row softmax 或 RMSNorm。RMSNorm 一行一个 program（hidden 维 4k–16k，一个块装得下）：

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # 并行维：行
    cols = tl.arange(0, BLOCK)                  # BLOCK = next_power_of_2(N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)   # 归约维：列，fp32 累加
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (x / rms) * w
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)

# grid = (n_rows,)；rmsnorm_kernel[grid](x, w, y, x.stride(0), N, 1e-6, BLOCK=triton.next_power_of_2(N))
```

和残差加融合的版本见 [Fused Residual Add and RMS Norm](/leetgpu/fused-rms-norm)；softmax 的大 N 两趟写法见 [Triton 版 Softmax](/handson/triton-softmax)；原语速查见 [triton 原语](/handson/triton_primitives)，入门从 [vector add](/handson/vector-add) 开始。

## 参考

- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)：[fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)、[matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [triton.autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [Tillet, Kung, Cox. Triton: an intermediate language and compiler for tiled neural network computations (MAPL 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Ansel et al. PyTorch 2 (ASPLOS 2024)](https://docs.pytorch.org/assets/pytorch2-2.pdf)：TorchInductor 生成 Triton

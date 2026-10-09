---
title: CUDA Reduce
status: draft
tags: [cuda, reduce, handson]
difficulty: 3
order: 9
related: [/handson/reduce, /leetgpu/reduction, /handson/kernel-mindset, /gpu/gpu-architecture, /handson/rmsnorm, /handson/triton-softmax]
stack: [k-lang]
leetgpu: [4]
---

# CUDA Reduce

> warp shuffle、shared memory 分层归约 · [LeetGPU #4 · Reduction](https://leetgpu.com/challenges)（[题面](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/4_reduction)）· Triton 版题解和优化记录在 [LeetGPU · Reduction](/leetgpu/reduction)，本页只讲 CUDA 版怎么一步步写快

## 一句话结论

CUDA reduce 分三层：**线程内**先串行累加很多个元素，**block 内**用 warp shuffle（或 shared memory 树）把 blockDim 个部分和归成一个，**block 之间**要么 `atomicAdd` 到一个全局变量，要么写到数组再启动第二个 kernel 归约。block 之间不能同步，所以单个 kernel 不靠 atomic（或计数器）做不完全局归约。

## 约定

- 题目：`input` 是长度 `N` 的 fp32 数组（1 ≤ N ≤ 10⁸），求和写到 `output[0]`。判题参考是 `torch.sum(input.double()).float()`，即先用 fp64 求和再转成 fp32。
- CUDA 入口：`extern "C" void solve(const float* input, float* output, int N)`，`input` / `output` 都是**设备指针**。
- 记号：`blockDim.x` 记作 `T`（每个 block 的线程数），`gridDim.x` 记作 `G`（block 数），`tid = threadIdx.x`，`lane = tid % 32`，`warp = tid / 32`。

## 先算上限：这题只看带宽

每个元素读 4 字节、做 1 次加法，算术强度 = 1 FLOP / 4 B = 0.25 FLOP/B，远低于任何 GPU 的 ridge point（T4 是 8.1 TFLOPS / 300 GB/s ≈ 27 FLOP/B，[T4 数据手册](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)）。所以目标只有一个：**把 HBM 带宽打满**。

性能测试 N = 4,194,304：

| | 数值 |
|---|---|
| 要读的字节 | 4,194,304 × 4 B = 16.8 MB |
| T4（300 GB/s）下限 | 16.8 MB / 300 GB/s ≈ **56 µs** |
| A100 80GB（2.04 TB/s）下限 | ≈ 8.2 µs |

下面每一步优化都用"实际带宽 / 峰值带宽"衡量离上限多远。

## 优化路线（Mark Harris 的 7 步）

这条路线来自 Mark Harris 的 [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)，在 G80（峰值 86.4 GB/s）上对 2²² 个 int 求和，作者自测：

| 版本 | 改动 | 时间 | 带宽 | 占峰值 |
|---|---|---|---|---|
| 1 | interleaved addressing，`tid % (2s) == 0` 分支 | 8.054 ms | 2.083 GB/s | 2.4% |
| 2 | interleaved，改成 `index = 2·s·tid`（去掉 divergence，引入 bank conflict） | 3.456 ms | 4.854 GB/s | 5.6% |
| 3 | sequential addressing | 1.722 ms | 9.741 GB/s | 11.3% |
| 4 | 加载时先加一次（first add during load） | 0.965 ms | 17.377 GB/s | 20.1% |
| 5 | 最后一个 warp 展开 | 0.536 ms | 31.289 GB/s | 36.2% |
| 6 | 完全展开（blockDim 作模板参数） | 0.381 ms | 43.996 GB/s | 50.9% |
| 7 | 每线程累加多个元素 | 0.268 ms | 62.671 GB/s | 72.5% |

（"占峰值"一列是我按 86.4 GB/s 算的；版本 7 在 32M 元素上作者测到 73 GB/s。）

前 3 步修的是 **shared memory 树形归约本身的低效**，第 4–7 步修的是**每个线程干的活太少**。下面按这个顺序写代码，第 5、6 步换成现代写法（warp shuffle）。

### v1：interleaved addressing，有 divergence

每个线程把一个元素搬进 shared memory，然后 `stride = 1, 2, 4, …`，下标是 stride 的偶数倍的线程把右边邻居加过来。

```cpp
__global__ void reduce_v1(const float* in, float* partial, int n) {
    extern __shared__ float s[];                 // 长度 T，launch 时传 T * sizeof(float)
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.f;              // 越界填 0（加法的单位元）
    __syncthreads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)             // 活跃线程：0, 2, 4, … → 0, 4, 8, …
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];    // 每个 block 一个部分和
}
```

问题：第一轮只有偶数号线程干活，但它们分散在**所有** warp 里。一个 warp 32 个线程里 16 个走 if、16 个不走，这叫 **warp divergence**，warp 要把两条路径都执行一遍。到 stride = 16 时每个 warp 只有 1 个线程干活，但所有 warp 都还要被调度。另外 `%` 在 GPU 上很慢。

### v2：strided index，没有 divergence 但有 bank conflict

让**连续的** tid 干活，各自去算自己负责的位置：

```cpp
for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    unsigned idx = 2 * stride * tid;
    if (idx < blockDim.x) s[idx] += s[idx + stride];
    __syncthreads();
}
```

现在活跃线程是 0..T/(2s)−1，连续，整 warp 要么全干要么全闲。但访问地址是 `2·s·tid`：shared memory 有 32 个 bank，地址 `a`（以 4 字节为单位）落在 bank `a mod 32`。stride = 1 时 warp 里 tid 0 和 tid 16 访问 0 和 32，同一个 bank；一般地 `2s·tid mod 32` 只取 32/(2s) 个不同值，是 **2s 路 bank conflict**（上限是 warp 内活跃线程数），同一 bank 的访问要串行做。bank 的背景见 [GPU 架构](/gpu/gpu-architecture)。

### v3：sequential addressing，两个问题都没了

stride 从 T/2 往下减半，"前一半加后一半"：

```cpp
for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
}
```

- 活跃线程 0..stride−1 连续 → 没有 divergence（stride ≥ 32 时）
- 一个 warp 读 `s[tid]` 是 32 个连续地址，落在 32 个不同 bank；`s[tid + stride]` 同理 → 没有 bank conflict

但第一轮就有一半线程闲着：它们只负责把一个数从 HBM 搬到 shared memory。

### v4：加载时先加一次

每个 block 负责 2T 个元素，搬运时就先加一次，block 数减半：

```cpp
unsigned i = blockIdx.x * (blockDim.x * 2) + tid;
float v = (i < n) ? in[i] : 0.f;
if (i + blockDim.x < n) v += in[i + blockDim.x];
s[tid] = v;
__syncthreads();
// 后面同 v3
```

Harris 在这一步测到 17 GB/s，然后分析出：离 86.4 GB/s 还远，而 reduce 的算术强度又低，所以瓶颈是**指令开销**（地址计算、循环控制、`__syncthreads`），不是访存。后面两步都是减指令。

### v5：最后 32 个用 warp shuffle

stride ≤ 32 时只剩一个 warp 在干活，这时还走 shared memory + `__syncthreads` 很浪费。Harris 原文的做法是把最后 6 轮手动展开、去掉同步、用 `volatile` 指针，依赖"warp 内 32 个线程锁步执行"。**这在 Volta 之后不再成立**：Volta 引入了 independent thread scheduling，NVIDIA 明确说隐式 warp 同步的写法不安全，要换成带 mask 的 `*_sync` 原语（[Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)）。

现代写法是 `__shfl_down_sync`：lane `i` 直接读到 lane `i + offset` 寄存器里的值（`i + offset ≥ 32` 时读到自己的值），不经过 shared memory，也不用 `__syncthreads`：

```cpp
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // 0xffffffff：32 个 lane 全部参与
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // 只有 lane 0 的结果是整个 warp 的和
}
```

5 步（offset = 16, 8, 4, 2, 1）把 32 个值归成 1 个。可以在 CPU 上用 torch 模拟一遍 shuffle 语义，确认 lane 0 拿到的是总和：

```python
import torch

def shfl_down(v, off):                 # v: [32]，模拟 __shfl_down_sync
    lane = torch.arange(32)
    src = lane + off
    out = v.clone()
    ok = src < 32
    out[ok] = v[src[ok]]               # 越界的 lane 保留自己的值
    return out

v = torch.randn(32, dtype=torch.float64)
w = v.clone()
for off in (16, 8, 4, 2, 1):
    w = w + shfl_down(w, off)
assert torch.allclose(w[0], v.sum())   # 只有 lane 0 是对的，其他 lane 是部分和
```

block 内就变成两级：每个 warp 先 shuffle 归约，lane 0 把结果写进 `warp_sums[warp]`（最多 1024/32 = 32 个），同步一次，再由第 0 个 warp 把这 ≤ 32 个数 shuffle 归约：

```cpp
__device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    int num_warps = blockDim.x >> 5;                       // 要求 blockDim 是 32 的倍数
    v = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;                                              // 只有 threadIdx.x == 0 的结果有效
}
```

数一下同步次数：T = 256 时 v3 要 log₂256 = 8 次 `__syncthreads`，这里只要 1 次。

### v6：每线程累加很多元素（grid-stride loop）

这是 Harris 的第 7 步，理由是 Brent 定理：N 个数用 N 个线程做树形归约，总工作量是 O(N log N)；用 O(N / log N) 个线程、每个线程先串行加 O(log N) 个，总工作量回到 O(N)。实践上推得更远：Harris 在 G80 上最好的配置是 64–256 个 block × 128 线程、每线程 1024–4096 个元素（作者自测）。

做法是固定 block 数，每个线程按 `G·T` 的跨度循环：

```cpp
float sum = 0.f;
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    sum += in[i];
sum = block_reduce_sum(sum);
```

同一轮循环里，一个 warp 的 32 个线程读 32 个连续 float = 128 字节，正好是合并访存。再进一步用 `float4` 一次读 16 字节，load 指令数降为 1/4。

### grid 级：三种合并方式

| 方式 | 做法 | 代价 |
|---|---|---|
| `atomicAdd` | 先 `cudaMemset(output, 0, 4)`，每个 block 的 thread 0 `atomicAdd(output, sum)` | 一个 kernel；浮点加法不满足结合律，block 完成顺序不定 → 结果末几位每次可能不同 |
| 两遍 | 第一遍写 `partial[blockIdx.x]`，第二遍 1 个 block 归约 `partial` | 多一次 launch；结果可复现 |
| 最后一个 block 收尾 | 每个 block 写完 partial 后 atomic 计数，拿到 `G−1` 的 block 负责合并 | 一个 kernel 且可复现；要处理内存可见性（`__threadfence`），见 [LeetGPU · Reduction](/leetgpu/reduction) 的追问 |

## 手撕：完整代码

v6 + `float4` + 两遍合并。partial buffer 用 `__device__` 全局数组，避免每次 `solve` 都 `cudaMalloc`（`cudaMalloc` 本身是同步的、耗时可能比这个 kernel 还长）。

```cpp
#include <cuda_runtime.h>
#include <algorithm>

constexpr int THREADS = 256;      // 每个 block 256 线程 = 8 个 warp
constexpr int MAX_BLOCKS = 1024;  // 第二遍用 1 个 1024 线程的 block 收尾

__device__ float g_partial[MAX_BLOCKS];

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    v = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

// 每个 block 把自己分到的元素加成一个数，写到 out[blockIdx.x]
__global__ void reduce_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    float sum = 0.f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 主体：float4 读，要求 in 16 字节对齐（cudaMalloc / torch 分配的首地址满足）
    int n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int i = idx; i < n4; i += stride) {
        float4 v = in4[i];
        sum += (v.x + v.y) + (v.z + v.w);
    }
    // 尾巴：最后 n % 4 个元素
    for (int i = n4 * 4 + idx; i < n; i += stride)
        sum += in[i];

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // 每线程至少 8 个元素，block 数封顶 MAX_BLOCKS
    int blocks = std::min((N + THREADS * 8 - 1) / (THREADS * 8), MAX_BLOCKS);
    blocks = std::max(blocks, 1);

    float* partial;
    cudaGetSymbolAddress((void**)&partial, g_partial);

    reduce_kernel<<<blocks, THREADS>>>(input, partial, N);   // 第一遍：N → blocks 个
    reduce_kernel<<<1, MAX_BLOCKS>>>(partial, output, blocks); // 第二遍：blocks → 1 个
    cudaDeviceSynchronize();
}
```

逐项核对：

- **N = 1**：`blocks = 1`，`n4 = 0`，tid 0 在尾巴循环里读 `in[0]`，其余线程 sum = 0，结果正确。
- **N = 4,194,304**：`blocks = min(2048, 1024) = 1024`，总线程 262,144，每线程 16 个 float = 4 次 `float4` 读。
- **第二遍**：1 个 block、1024 线程、`n = blocks ≤ 1024`，每线程至多读 1 个 partial；`g_partial` 是 `__device__` 数组，地址对齐到 16 字节以上，`float4` 读安全。
- **`block_reduce_sum` 只调一次**：`warp_sums` 在同一个 kernel 里复用会有读写竞争，要多调就在两次之间加 `__syncthreads()`。
- **可复现**：block 数只由 N 决定，每个线程读哪些元素、加的顺序都固定，所以同一输入多次运行结果逐位一致。

atomic 版本：kernel 只改最后一行，`out` 变成单个累加器，`solve` 只启动一次：

```cpp
__global__ void reduce_atomic_kernel(const float* __restrict__ in, float* out, int n) {
    // ... 同 reduce_kernel 直到 block_reduce_sum ...
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

extern "C" void solve(const float* input, float* output, int N) {
    int blocks = std::max(std::min((N + THREADS * 8 - 1) / (THREADS * 8), MAX_BLOCKS), 1);
    cudaMemset(output, 0, sizeof(float));       // atomic 是往上加，必须先清零
    reduce_atomic_kernel<<<blocks, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

### 本地验证

LeetGPU 之外，可以用 `torch.utils.cpp_extension.load_inline` 把上面的 `.cu` 编进 Python 跟 `torch.sum` 比（需要 CUDA 机器）：

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = open("reduce.cu").read() + r"""
#include <torch/extension.h>
torch::Tensor run(torch::Tensor x) {
    auto out = torch::empty({1}, x.options());
    solve(x.data_ptr<float>(), out.data_ptr<float>(), (int)x.numel());
    return out;
}
"""
mod = load_inline("reduce_ext", cpp_sources="torch::Tensor run(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["run"])

# 分布和容差照抄 LeetGPU 的测试：参考是 fp64 求和，rtol = atol = 1e-5
for n, lo in [(1, -1000), (5, -1000), (10_000, -1000), (4_194_304, 0), (15_000_000, 0)]:
    x = torch.empty(n, device="cuda").uniform_(lo, 1000)
    ref = x.double().sum().float()
    got = mod.run(x)[0]
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), (n, got.item(), ref.item())
```

## 面试追问

::: details Q：为什么 reduce 的 stride 要从大到小，而不是从 1 开始翻倍？
从 1 开始翻倍有两种写法，各有一个毛病（Harris 的 v1、v2）：用 `tid % (2s) == 0` 选活跃线程，活跃线程分散在所有 warp 里，每个 warp 都 divergent；改成 `idx = 2·s·tid` 让连续线程干活，divergence 没了，但地址步长 2s 造成 2s 路 bank conflict。stride 从 `T/2` 减半（v3）时，活跃线程 0..s−1 连续、访问地址也连续，两个问题同时消失。Harris 的数据是 v1→v3 快了 4.68 倍（作者自测，G80）。
:::

::: details Q：warp shuffle 比 shared memory 归约好在哪？
一是不走 shared memory，寄存器之间直接交换，少了 store/load 指令；二是 warp 内不需要 `__syncthreads()`，一个 block 只在 warp 之间同步一次（256 线程时从 8 次降到 1 次）。注意必须用带 mask 的 `__shfl_down_sync`，老的 `volatile` + 隐式锁步写法在 Volta 之后不保证正确。
:::

::: details Q：atomicAdd 的结果为什么不确定？要紧吗？
浮点加法不满足结合律，(a + b) + c 和 a + (b + c) 可能差最后几位；block 完成的顺序每次不一样，atomic 加的顺序就不一样。训练里要求 bitwise 可复现（调试、对齐 loss 曲线）时就要用两遍或"最后一个 block 收尾"。推理里一般无所谓。
:::

::: details Q：block 数怎么选？
目标是"一波"刚好铺满所有 SM、每个线程又有足够多的活。T4 有 40 个 SM，每 SM 最多 1024 个驻留线程，256 线程的 block 每 SM 能放 4 个，160 个 block 就是一波。实践中取 SM 数的几倍到上千都行，再多只是增加第二遍的 partial 数量；再少则每线程读的数据太多、并发的访存请求不够，带宽打不满。
:::

## 参考

- Mark Harris, [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)（7 步优化与 G80 数据）
- Justin Luitjens, [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)（shuffle 版 reduce）
- Yuan Lin, Vinod Grover, [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)（隐式 warp 同步为什么不安全）

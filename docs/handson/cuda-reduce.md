---
title: CUDA Reduce
status: todo
tags: [cuda, reduce, handson]
difficulty: 3
order: 9
related: []
stack: [k-lang]
leetgpu: [4]
---

# CUDA Reduce

> warp shuffle、shared memory 分层归约

## 一句话结论

CUDA reduce 分三层：block 内先用 shared memory 做树形归约（或 warp shuffle 不经过 shared memory），得到每个 block 一个部分和；block 之间要么 `atomicAdd` 到一个全局变量，要么写到数组再启动第二个 kernel 归约。核心约束是 block 之间不能同步，所以单 kernel 做不了全局归约。

## 推导

- **shared memory 树形归约**：每个线程加载一个元素到 `smem`，然后 stride 从 `blockDim/2` 减半到 1，每轮 `__syncthreads()`；用「前半加后半」的 stride 模式避免 bank conflict 和 warp divergence。
- **warp shuffle**：`__shfl_down_sync` 让 warp 内 32 个线程 5 步归约，不走 shared memory 也不用同步；block 内先 warp 归约，再把各 warp 的结果用一个 warp 归约。
- **grid 级**：`atomicAdd` 简单但浮点加法不满足结合律，结果不确定；两遍 kernel 可复现，第二遍的输入已经很小。
- **带宽视角**：reduce 是纯 memory-bound，每个元素读一次，目标是打满 HBM 带宽；每个线程处理多个元素（grid-stride loop）减少 block 数。

## 面试追问

::: details Q：为什么 reduce 的 stride 要从大到小，而不是从 1 开始翻倍？
从 1 开始翻倍时活跃线程是 0、2、4、…，同一 warp 内一半线程空转（divergence），且访问 smem 的地址跨步变大造成 bank conflict。从 `blockDim/2` 减半时前 N 个线程连续活跃，整 warp 要么全活要么全闲，访存也连续。
:::

## 手撕

对应 [LeetGPU #4](https://leetgpu.com/challenges)，Triton 版和 kernel mindset 分析见 [LeetGPU · Reduction](/leetgpu/reduction)。CUDA 框架：

```cpp
__global__ void reduce_sum(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0.f;      // 可改成 grid-stride 累加多个元素
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);  // 或写 out[blockIdx.x]，再跑第二遍
}
```

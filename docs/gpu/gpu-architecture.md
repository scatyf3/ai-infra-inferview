---
title: GPU 执行模型：SM / Warp / Shared Memory
status: todo
tags: [cuda, sm, warp]
difficulty: 3
order: 1
related: []
stack: [hw-gpu, hw-mem]
---

# GPU 执行模型：SM / Warp / Shared Memory

> bank conflict、coalescing、occupancy

## 一句话结论

GPU 是「很多 SM，每个 SM 跑很多 warp 来掩盖访存延迟」：一个 warp 32 个线程锁步执行，shared memory 是程序员管理的片上缓存，写 kernel 的三个基本功是让访存合并（coalescing）、避免 shared memory bank conflict、让 occupancy 够高来藏延迟。

## 推导

- **层次**：grid → block → warp → thread；block 被整体调度到一个 SM 上，block 内线程共享 shared memory，可以 `__syncthreads()`，跨 block 不能同步。
- **coalescing**：一个 warp 的 32 个线程访问连续地址时合并成少数几次 128 B 事务；跨步访问会把事务数放大几十倍，这是 matmul 转置、attention 读 K 时首先要查的事。
- **bank conflict**：shared memory 分 32 个 bank，同一 warp 内多个线程落在同一 bank 的不同地址会串行化；经典解法是 padding 一列。
- **occupancy**：每个 SM 能驻留的 warp 数受寄存器和 shared memory 用量限制；occupancy 不是越高越好，够藏住访存延迟就行，再高反而挤占寄存器。

## 面试追问

::: details Q：warp divergence 是什么，为什么 attention 的 causal mask 不会严重 divergence？
同一 warp 内线程走不同分支时，硬件会串行执行两条路径，吞吐减半。causal mask 按 tile 处理时，大部分 tile 要么全部可见要么全部被 mask，只有对角线上的 tile 内部有分支，所以 FlashAttention 把全 mask 的 tile 直接跳过，只在对角 tile 上付 divergence 的代价。
:::

## 手撕

常见题：解释 `threadIdx` / `blockIdx` 到全局索引的映射；给一段 kernel 指出哪里访存不合并。入门见 [kernel mindset](/handson/kernel-mindset)。

## 参考

- [CUDA C++ Programming Guide：Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

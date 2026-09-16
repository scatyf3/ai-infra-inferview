---
title: GPU / 算子
---

# GPU / 算子

够用即可，别装深。目标是能解释一个 kernel 为什么快或慢，能看懂 profiler 输出，知道什么时候该自己写、什么时候直接调库。

::: tip 面试官真正在考什么
面试官真正在考的是：你是否理解 GPU 的内存层级和执行模型，能把「访存合并 / occupancy / tensor core 利用率」映射到具体优化动作。
:::

## 本领域主题

- [GPU 执行模型：SM / Warp / Shared Memory](./gpu-architecture) — bank conflict、coalescing、occupancy
- [Tensor Core 与 GEMM Tiling](./tensor-core-gemm) — 为什么 GEMV 打不满
- [Triton 编程模型](./triton) — autotune；什么时候写 Triton、什么时候直接调 cuBLAS / CUTLASS
- [CUDA Graph 与 Kernel Fusion](./cuda-graph-fusion) — 在框架侧的落地
- [Profiling：nsys / ncu / torch profiler](./profiling) — 看哪几个指标

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
- [SM 里的计算单元：CUDA Core / Tensor Core / SFU](./compute-units) — 每种单元一个周期做多少、为什么算力都在 Tensor Core
- [普通 SIMT 之外的指令](./special-instructions) — shuffle、TMA、wgmma、SFU、mbarrier、setmaxnreg
- [GEMM Tiling](./tensor-core-gemm) — 切块复用，以及为什么 GEMV 打不满
- [Split-K 与 Stream-K](./split-k-stream-k) — tile 太少或和 SM 数对不齐时，沿 K 再切
- [量化 GEMM Kernel：W4A16 / FP8](./quant-gemm) — 格式转换和 scale 放在 tiling 的哪一级
- [MoE Grouped GEMM](./moe-grouped-gemm) — 大小不一的小 GEMM 怎么合成一个 kernel
- [Memory-bound Kernel 与 Kernel Fusion](./memory-bound-kernels) — 时间等于字节除以带宽：融合少搬、访存搬满
- [Triton 编程模型](./triton) — autotune；什么时候写 Triton、什么时候直接调 cuBLAS / CUTLASS
- [Profiling：nsys / ncu / torch profiler](./profiling) — 看哪几个指标

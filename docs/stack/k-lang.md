---
title: Triton / CUDA
---

# Triton / CUDA

写 kernel 的两条路。CUDA 控制到线程级，能用上全部硬件特性（TMA、wgmma、warp specialization），上限最高，代价也最大。Triton 以 block 为单位编程，由编译器负责线程映射、访存合并和 shared memory 分配，几十行就能写出接近手写性能的融合 kernel。

- **适合 Triton**：逐元素、归约、融合类算子，或者需要快速迭代的 attention 变体。
- **需要 CUDA / CUTLASS**：要榨出 GEMM、attention 最后 10–20% 的性能，或者要用新架构的特性。
- **profiling**：先用 nsys 看时间线找到热点 kernel，再用 ncu 判断它是算力受限还是带宽受限。

**延伸阅读**：[Triton](https://triton-lang.org/) · [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---
title: GPU 架构
---

# GPU 架构

GPU 由几十到上百个 SM 组成（H100 SXM 有 132 个），每个 SM 以 32 线程的 warp 为单位调度，靠大量 warp 轮流执行来掩盖访存延迟。算力主要来自 Tensor Core：H100 的 dense BF16 约 989 TFLOPS，而 CUDA Core 的 FP32 只有约 67 TFLOPS。

- **SIMT**：同一个 warp 内出现分支发散时，两条路径会串行执行。
- **occupancy**：每个 SM 能同时驻留多少 warp，受寄存器和 shared memory 用量限制；occupancy 高不等于性能高。
- **Tensor Core**：只接受特定形状的矩阵块，GEMM 和 attention 能不能跑满，就看能不能持续喂饱它。

**延伸阅读**：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---
title: GPU 架构
---

# GPU 架构

GPU 由几十到上百个 SM 组成（H100 SXM 有 132 个），每个 SM 以 32 线程的 warp 为单位调度，靠大量 warp 轮流执行来掩盖访存延迟。算力主要来自 Tensor Core：H100 的 dense BF16 约 989 TFLOPS，而 CUDA Core 的 FP32 只有约 67 TFLOPS。

- **SIMT**：同一个 warp 内出现分支发散时，两条路径会串行执行。
- **occupancy**：每个 SM 能同时驻留多少 warp，受寄存器和 shared memory 用量限制；occupancy 高不等于性能高。
- **Tensor Core**：只接受特定形状的矩阵块，GEMM 和 attention 能不能跑满，就看能不能持续喂饱它。
- **计算单元**：一个 SM 分 4 个子分区，每个有 32 条 FP32 lane（即「CUDA core」）、4 个 SFU、1 个 Tensor Core，见 [SM 里的计算单元](/gpu/compute-units)。
- **专用单元**：除了普通 SIMT，还有 Tensor Core、SFU（exp 等，吞吐只有 FMA 的 1/8）、TMA、warp shuffle，各有专门的指令，见 [普通 SIMT 之外的指令](/gpu/special-instructions)。

**延伸阅读**：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

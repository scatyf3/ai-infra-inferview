---
title: GEMM
---

# GEMM

LLM 里绝大部分 FLOPs 都在线性层的 GEMM 里。prefill 时 M = batch × seq 很大，GEMM 是 compute-bound，能接近 Tensor Core 峰值；decode 时 M 只等于 batch 大小，形状退化成接近 GEMV，瓶颈变成读权重的带宽。

- **tiling**：把 A、B 分块搬进 shared memory，在片上反复复用，这是所有高性能 GEMM 的骨架。
- **小 M**：decode 场景常用 split-K 或专门的 GEMV kernel；权重量化（W4A16）直接减少要读的字节数。
- **库**：cuBLAS / cuBLASLt 用于通用场景，CUTLASS 和 Triton 适合定制和融合。

**延伸阅读**：[CUTLASS](https://github.com/NVIDIA/cutlass)

---
title: GEMM
---

# GEMM

LLM 里绝大部分 FLOPs 都在线性层的 GEMM 里。prefill 时 M = batch × seq 很大，GEMM 是 compute-bound，能接近 Tensor Core 峰值；decode 时 M 只等于 batch 大小，形状退化成接近 GEMV，瓶颈变成读权重的带宽。

- **tiling**：把 $C$ 切成 $BM \times BN$ 的块，$A$、$B$ 的块搬进 shared memory 反复用，block、warp、指令三级各切一次，见 [GEMM Tiling](/gpu/tensor-core-gemm)。那一页有在 RTX 5070 Ti 上逐级加优化的 Triton 实测：朴素版 0.11 TFLOP/s，加完 block tile、Tensor Core、流水、autotune 到 95–98，和 cuBLAS 持平。
- **小 M**：decode 的 GEMM 是 GEMV，AI ≈ batch，靠 split-K 补并行度，靠 W4A16 少读权重，见 [量化 GEMM](/gpu/quant-gemm)。
- **变体**：FP8 GEMM 进低精度 Tensor Core（[量化 GEMM](/gpu/quant-gemm#fp8-直接进低精度-tensor-core)）；MoE 把大小不一的 expert GEMM 合成一个 kernel（[MoE Grouped GEMM](/gpu/moe-grouped-gemm)）。
- **库**：cuBLAS / cuBLASLt 用于通用场景，CUTLASS 和 Triton 适合定制和融合。

**延伸阅读**：[CUTLASS](https://github.com/NVIDIA/cutlass)、[本站的 Triton GEMM 代码](https://github.com/scatyf3/ai-infra-inferview/tree/main/bench/triton-gemm)

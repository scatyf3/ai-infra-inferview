---
title: "GPU 算力与精度"
---

# GPU 算力与精度

训练的算力上限由 Tensor Core 在所用精度下的 dense 峰值决定：H100 SXM bf16 989 TFLOP/s，fp8 翻倍。MFU 就是实际有效 FLOP/s 除以这个峰值，所以报 MFU 时分母要和训练精度对上。

- **精度**：bf16 是默认，fp8 只加速 GEMM，其余算子不变。
- **SM 与 HBM**：非 GEMM 算子受 HBM 带宽限制，是 MFU 上不去的原因之一。

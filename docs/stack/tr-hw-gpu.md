---
title: "GPU 算力与精度"
---

# GPU 算力与精度

我们存在一个GPU硬件的算力上限，given这个上限和我们想使用的精度，我们可以计算出训练的计算峰值，TFLOP/s。

$$
\text{峰值 FLOP/s} = \text{SM 个数} \times \text{每个 SM 每个时钟周期能做的 FLOP（看精度）} \times \text{时钟频率}
$$

以 H100 SXM 为例：

| 精度 | 每个 SM 每周期的 FLOP | × 132 SM × 1.83 GHz |
|---|---|---|
| bf16 | 512 FMA × 2 × 4 个 Tensor Core = 4096 | ≈ 989 TFLOP/s |
| fp8 | 1024 FMA × 2 × 4 = 8192 | ≈ 1979 TFLOP/s |


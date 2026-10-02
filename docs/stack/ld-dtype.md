---
title: dtype
---

# dtype

dtype 决定每个参数占几个字节，也决定能用哪种 Tensor Core 指令。BF16 / FP16 每个参数 2 字节，FP8 是 1 字节，INT4 是 0.5 字节。70B 模型的权重在 BF16 下约 140 GB，FP8 下约 70 GB。

- **BF16 vs FP16**：BF16 的指数位和 FP32 一样是 8 位，不容易溢出；FP16 尾数更多、精度更高，但范围小。
- **FP8**：E4M3 精度高，常用于前向；E5M2 范围大，常用于梯度；都需要 per-tensor 或 per-block 的 scale。
- **混合精度推理**：权重低精度存储、计算前反量化到 BF16（W4A16），或者激活也一起量化（W8A8），走低精度 GEMM。

**延伸阅读**：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

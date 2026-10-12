---
title: "混合精度 / FP8"
---

# 混合精度 / FP8

bf16 算、fp32 存主权重和优化器状态：更新量太小，半精度加不上去。fp16 还要 loss scaling，bf16 范围和 fp32 一样，不需要。FP8 把 GEMM 输入再降一半，难点在 scale 的粒度和累加精度。

- **per-tensor vs 细粒度**：DeepSeek-V3 用激活 $1 \times 128$、权重 $128 \times 128$ 的块 scale。

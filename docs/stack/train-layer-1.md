---
title: "训练栈第 1 层：kernel"
---

# 训练栈第 1 层：kernel

训练的 kernel 比推理多一半：每个 forward 算子都有对应的 backward。一个 Linear 层是三个 GEMM（forward、对输入求梯度、对权重求梯度），attention 的 backward 要重算 softmax。时间主要花在 GEMM 和 attention 上，其余的逐元素算子要靠融合省掉访存。

- **GEMM**：训练 FLOPs 的主体，fp8 训练就是把这三个 GEMM 的输入换成 fp8。
- **attention**：FlashAttention 的 backward 不存 $s \times s$ 矩阵，按块重算。
- **融合**：RMSNorm、RoPE、SwiGLU、cross-entropy 这些 memory-bound 的算子，融合或交给 torch.compile。

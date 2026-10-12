---
title: "融合算子 / loss kernel"
---

# 融合算子 / loss kernel

RMSNorm、RoPE、SwiGLU 是逐元素或按行归约的 memory-bound 算子，融合成一个 kernel 后只读写一次显存。词表大时 cross-entropy 是最大的单个激活，fused linear cross-entropy 分块算，不物化完整 logits。

- **代表实现**：Liger Kernel、Cut Cross-Entropy。
- **收益**：端到端的倍数远小于单算子的倍数，因为大头时间在 GEMM。

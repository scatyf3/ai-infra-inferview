---
title: "LR / batch 调度"
---

# LR / batch 调度

学习率先 warmup 再衰减（cosine 或 WSD），batch 从小到大。WSD 的稳定段可以随时分出去衰减，适合不确定总 token 数的训练。

- **warmup**：前几步二阶动量估得不准，步子要小。
- **batch**：超过 critical batch size 后再加 batch，步数几乎不减、算力白花。

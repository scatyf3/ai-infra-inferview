---
title: "LR / batch 调度"
---

# LR / batch 调度

学习率先 warmup 再衰减（cosine 或 WSD），batch 从小到大，然后到一个上限(能获得有意义的信号但是不浪费算力，即paper里讨论的critical batch size)

- **warmup**：前几步二阶动量估得不准，步子要小。
- **batch**：超过 critical batch size 后再加 batch，步数几乎不减、算力白花。

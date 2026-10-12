---
title: "训练栈第 2 层：一个 step"
---

# 训练栈第 2 层：一个 step

一个训练 step 的顺序是固定的：取 batch → forward → backward → 梯度同步 → 裁剪 → 优化器更新 → 调学习率 → 清梯度。这一层管的是每一步在数学上做了什么，以及 batch 和学习率随训练怎么变。

- **autograd**：backward 的计算图和要存的激活从这里来。
- **优化器**：AdamW 的状态是显存大头，Muon 只替换隐藏层矩阵。
- **调度**：warmup、cosine / WSD，batch 从小到大。

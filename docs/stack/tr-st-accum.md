---
title: "梯度累积与裁剪"
---

# 梯度累积与裁剪

gradient accumulation 用多个 micro-batch 凑一个大 batch，loss 要按整个 step 的有效 token 数归一化。梯度裁剪按全局 L2 norm 等比例缩小，分布式下要先跨 rank 求平方和再开根。

- **no_sync**：前几个 micro-batch 不同步梯度，只在最后一个同步。
- **阈值**：常取 1.0，只截异常步。

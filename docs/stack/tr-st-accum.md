---
title: "梯度累积与裁剪"
---

# 梯度累积与裁剪

gradient accumulation 用多个 micro-batch 凑一个大 batch：batch 大小由收敛决定，但一张卡的显存一次装不下这么多样本的激活，所以分几次 forward / backward，梯度累加起来，最后只更新一次。

- **累积**：loss 按整个 step 的有效 token 数归一化；前 $K-1$ 个 micro-batch 用 `no_sync()` 跳过梯度同步，只在最后一个同步，通信量降到 $1/K$。
- **裁剪**：按全局 L2 norm 等比例缩小，分布式下先跨 rank 求平方和再开根；阈值常取 1.0，只截异常步。

loss 归一化、`no_sync` 和裁剪的细节见 [优化器与学习率](/train/optimizer-lr) 第 2 节和第 6 节。

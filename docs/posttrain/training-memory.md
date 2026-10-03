---
title: 混合精度、Grad Checkpointing 与优化器状态显存账
status: todo
tags: [mixed-precision, checkpointing, optimizer]
difficulty: 3
order: 5
related: []
stack: []
---

# 混合精度、Grad Checkpointing 与优化器状态显存账

> 训练显存的完整账本

## 一句话结论

训练显存 = 模型状态 + 激活 + 临时 buffer。混合精度 Adam 下模型状态是每参数 16 字节（bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + Adam m、v 各 4），7B 就是 112 GB，所以单卡全参微调从一开始就放不下；激活随 batch × seq × layers 涨，gradient checkpointing 只存每层输入、反向时重算，用约 30% 的额外计算把激活降到约 1/层数。

## 推导

- **模型状态**：16Ψ 是基线；8-bit Adam 把 m、v 压到 1 B 各，变成 10Ψ；ZeRO 按卡数切分，见 [ZeRO / FSDP](/parallel/zero-fsdp)。
- **激活**：每层大约 34·s·b·h + 5·a·s²·b 字节（无 FlashAttention 时），FlashAttention 去掉 s² 项；checkpointing 后只剩每层 2·s·b·h 的输入。
- **混合精度为什么要 fp32 主权重**：bf16 只有 8 位尾数，lr × grad 常小于参数的最小分辨率，直接加会被舍掉；fp32 主权重累加更新，前向再 cast 回 bf16。
- **临时 buffer**：梯度 all-reduce 的 bucket、attention 的 workspace、allocator 碎片，通常预留 10–20%。

## 面试追问

::: details Q：为什么 bf16 训练不需要 loss scaling 而 fp16 需要？
fp16 的指数位只有 5 位，最小正规数约 6e-5，小梯度会下溢成零，所以要把 loss 乘一个大数再在更新前除回来。bf16 的指数位和 fp32 一样是 8 位，动态范围够大不会下溢，代价是尾数只有 7 位、精度低，靠 fp32 主权重补回来。
:::

## 手撕

常见题：给定参数量、seq、batch、层数和 hidden，算出全参微调 / LoRA / ZeRO-3 各自每卡显存；推理侧的账见 [显存账](/inference/memory-accounting)。

## 参考

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Reducing Activation Recomputation in Large Transformer Models（激活显存公式）](https://arxiv.org/abs/2205.05198)

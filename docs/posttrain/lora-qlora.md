---
title: LoRA / QLoRA 原理与显存账
status: todo
tags: [lora, qlora]
difficulty: 3
order: 2
related: []
stack: [f-runner]
---

# LoRA / QLoRA 原理与显存账

> 多 LoRA serving

## 一句话结论

LoRA 冻结原权重 W，只训练一对低秩矩阵 B·A（秩 r 通常 8–64），forward 变成 Wx + BAx；可训练参数和优化器状态缩小上百倍，但激活显存不变。QLoRA 再把冻结的 W 量化到 4 bit（NF4）存储、计算时反量化，让 65B 模型能在一张 48 GB 卡上微调。推理时多个 LoRA 可以共享同一份 base 权重，按请求切换 adapter。

## 推导

- **显存账**：全参微调每参数 16 B；LoRA 下 base 只存 bf16 2 B，可训练部分 r·(d_in + d_out) 个参数才有优化器状态；QLoRA 把 base 压到 0.5 B 加少量 scale。
- **初始化**：A 随机、B 全零，训练开始时 BA = 0，模型行为与原模型一致；缩放 α/r 控制更新幅度。
- **合并**：部署时可以把 BA 加回 W 零开销；多 LoRA 服务时不能合并，用 batched GEMV（Punica 的 SGMV、S-LoRA）把不同请求的 adapter 计算合到一个 kernel。
- **QLoRA 细节**：NF4 按正态分布分位数设量化点、double quantization 压缩 scale、paged optimizer 在显存峰值时把优化器状态换到 CPU。

## 面试追问

::: details Q：LoRA 为什么省不了激活显存？
反向传播要用到每层的输入激活来算 A 的梯度（∂L/∂A = Bᵀ·grad·xᵀ 需要 x），这些激活和全参微调时一样要存下来。省下来的只是参数、梯度和优化器状态这三项；要省激活得靠 gradient checkpointing，见 [训练显存账](/posttrain/training-memory)。
:::

## 手撕

常见题：写一个 `LoRALinear` 模块（forward 和参数初始化）；算一下 7B 模型 r=16 在 q/v 投影上加 LoRA 的可训练参数量。

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)、[S-LoRA（多 LoRA serving）](https://arxiv.org/abs/2311.03285)

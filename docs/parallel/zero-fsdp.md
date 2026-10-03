---
title: ZeRO 1 / 2 / 3 与 FSDP
status: todo
tags: [zero, fsdp]
difficulty: 3
order: 3
related: []
stack: []
---

# ZeRO 1 / 2 / 3 与 FSDP

> 通信-显存 tradeoff

## 一句话结论

数据并行每张卡都存一份完整的参数、梯度和优化器状态，Adam 混合精度下是每参数 16 字节；ZeRO 把这三样按阶段切到各卡上：ZeRO-1 切优化器状态、ZeRO-2 再切梯度、ZeRO-3 连参数也切，显存从 16Ψ 降到 16Ψ/N。代价是 ZeRO-3 每层 forward / backward 前都要 all-gather 参数，通信量比普通 DP 多 50%。FSDP 是 PyTorch 原生的 ZeRO-3 实现。

## 推导

- **显存账**：参数 bf16 2 B + 梯度 2 B + 优化器（fp32 主权重 4 B + m 4 B + v 4 B）= 16 B / 参数；详见 [训练显存账](/posttrain/training-memory)。
- **ZeRO-1 / 2**：优化器状态和梯度本来就只在各自的分片上更新，切分后通信量不变（reduce-scatter + all-gather 等于一次 all-reduce），几乎免费。
- **ZeRO-3 / FSDP**：参数也切分，forward 前 all-gather 当前层参数、用完即丢，backward 再 gather 一次；通信量 1.5 倍，用 prefetch 下一层来重叠。
- **选择**：显存够就 ZeRO-2（通信少）；放不下才 ZeRO-3；大模型通常 ZeRO-3 / FSDP 和 TP、PP 组合，或用 HSDP（节点内 shard、节点间 replicate）。

## 面试追问

::: details Q：ZeRO-3 和 TP 都是把参数切到多卡，区别在哪？
ZeRO-3 切的是「存储」，计算时还要把整层参数 gather 回来，每张卡算完整的层；TP 切的是「计算」，每张卡只算自己那一片，用 all-reduce 合并结果。ZeRO-3 通信的是参数（和 batch 无关），TP 通信的是激活（和 batch 成正比），所以小 batch 用 TP 划算，大 batch 用 ZeRO 划算。
:::

## 手撕

常见题：给定模型参数量、卡数和优化器，算 ZeRO 各阶段每卡显存；画出 FSDP 一层的 all-gather / reduce-scatter 时序。

## 参考

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP 论文](https://arxiv.org/abs/2304.11277)

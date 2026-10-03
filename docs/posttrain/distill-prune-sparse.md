---
title: 蒸馏、剪枝、稀疏（含 KV Pruning）
status: todo
tags: [distillation, pruning, kv-pruning]
difficulty: 3
order: 6
related: []
stack: [4]
---

# 蒸馏、剪枝、稀疏（含 KV Pruning）

> 准备好 motivation 和 ablation

## 一句话结论

三条路都是「用更少的计算或字节逼近原模型」：蒸馏用大模型的输出（logits 或生成数据）训练小模型，剪枝直接删掉权重（非结构化省字节、结构化省算力），稀疏 / KV pruning 在推理时跳过不重要的 token 或 head。面试时要讲清每种方法的 motivation（省什么）和 ablation（砍掉之后精度掉在哪）。

## 推导

- **蒸馏**：logit 蒸馏（KL 对齐软标签）需要两个模型 vocab 一致；数据蒸馏（用大模型生成 SFT 数据）更常用，本质是 SFT；推理侧 speculative decoding 的 draft 模型常由蒸馏得到，见 [投机解码](/inference/speculative-decoding)。
- **剪枝**：非结构化（magnitude / Wanda / SparseGPT）能到 50% 稀疏但 GPU 只有 2:4 结构化稀疏能加速；结构化剪枝（删 head、删层、缩 FFN）直接减 FLOPs，需要少量继续训练恢复。
- **KV pruning**：按 attention 分数或 heavy-hitter 统计丢掉不重要的 KV（H2O、SnapKV、StreamingLLM 的 sink），省显存和 decode 带宽；代价是长程检索任务精度下降。
- **ablation 怎么准备**：每个改动单独开关，报 perplexity + 下游任务 + 实际延迟三条线，而不是只报一个。

## 面试追问

::: details Q：非结构化剪枝到 50% 为什么在 GPU 上几乎不提速？
GEMM 用 Tensor Core 按 dense tile 算，零散的零不会被跳过；要加速得满足硬件支持的 2:4 模式（每 4 个权重里 2 个是零），由 Sparse Tensor Core 做到 2 倍。更自由的稀疏模式只省存储不省计算，除非用专门的稀疏 kernel，而那些 kernel 通常比 dense 慢。
:::

## 手撕

常见题：写出 logit 蒸馏的 loss（温度 + KL）；实现一个按 attention 累计分数淘汰 KV 的简单策略。

## 参考

- [SparseGPT](https://arxiv.org/abs/2301.00774)、[Wanda](https://arxiv.org/abs/2306.11695)
- [H2O: Heavy-Hitter Oracle for KV cache](https://arxiv.org/abs/2306.14048)

---
title: SFT：Packing、Loss Mask、长上下文
status: todo
tags: [sft, packing]
difficulty: 2
order: 1
related: []
stack: []
---

# SFT：Packing、Loss Mask、长上下文

> 数据组织与训练效率

## 一句话结论

SFT 就是在指令数据上做下一个 token 预测，工程上的三件事：packing 把多条短样本拼成一个定长序列避免 padding 浪费（要配 attention mask 或 varlen attention 防止跨样本串扰），loss mask 只在回答部分算 loss 不学 prompt，长上下文训练要靠序列并行和 ring attention 把激活切到多卡。

## 推导

- **packing**：按长度贪心装箱到 max_seq_len，GPU 利用率从 30–50% 提到 90% 以上；用 `flash_attn_varlen_func` 传 `cu_seqlens` 让每条样本只 attend 自己，位置编码也要按样本重置。
- **loss mask**：prompt / system / 多轮里的用户部分 label 设为 −100；多轮对话一次 forward 算所有 assistant 轮的 loss，不要拆成多条样本重复算 prompt。
- **长上下文**：激活显存随序列长度线性涨（attention 用 FlashAttention 后不再平方），超过单卡后用序列并行（Ulysses 的 all-to-all 切 head、ring attention 切 KV 块）。
- **数据**：质量比数量重要，去重和过滤比堆数据有效；chat template 要和推理时完全一致，否则线上掉点。

## 面试追问

::: details Q：packing 之后如果不改 attention mask 会怎样？
拼在一起的样本会互相 attend，后面的样本能看到前面无关样本的内容，训练目标被污染。实验上小模型会明显掉点，大模型有时看不出差别，但推理时每条样本是单独的，分布不一致。正确做法是 varlen attention 或 block-diagonal mask。
:::

## 手撕

常见题：给几条对话写出 packing 后的 `input_ids`、`labels`（含 −100）和 `cu_seqlens`。

## 参考

- [HF TRL：SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [Ring Attention](https://arxiv.org/abs/2310.01889)

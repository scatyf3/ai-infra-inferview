---
title: Post-train / Efficient
---

# Post-train / Efficient

SFT、LoRA、RLHF 全家桶与 RL infra。重点是显存账和数据流：训练时多了哪些状态、RL 时 rollout 与 training 怎么解耦、权重如何同步。

::: tip 面试官真正在考什么
面试官真正在考的是：你能否把「训练比推理贵多少、贵在哪」算出来，并解释 DPO / GRPO 相比 PPO 省掉了什么、代价是什么。
:::

## 本领域主题

- [SFT：Packing、Loss Mask、长上下文](./sft) — 数据组织与训练效率
- [LoRA / QLoRA 原理与显存账](./lora-qlora) — 多 LoRA serving
- [RLHF 全家桶：PPO / DPO / GRPO](./rlhf-ppo-dpo-grpo) — PPO 四模型流程、KL 约束、DPO / GRPO 推导动机与区别
- [RL Infra：Rollout 与 Training 分离](./rl-infra) — 权重同步、veRL / OpenRLHF 架构
- [混合精度、Grad Checkpointing 与优化器状态显存账](./training-memory) — 训练显存的完整账本
- [蒸馏、剪枝、稀疏（含 KV Pruning）](./distill-prune-sparse) — 准备好 motivation 和 ablation

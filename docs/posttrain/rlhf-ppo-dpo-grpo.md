---
title: RLHF 全家桶：PPO / DPO / GRPO
status: todo
tags: [rlhf, ppo, dpo, grpo]
difficulty: 4
order: 3
related: []
stack: []
---

# RLHF 全家桶：PPO / DPO / GRPO

> PPO 四模型流程、KL 约束、DPO / GRPO 推导动机与区别

## 一句话结论

PPO 是标准 RLHF：policy 生成、reward model 打分、value model 估基线、reference model 算 KL 惩罚，四个模型同时在显存里，工程复杂。DPO 把「先训 reward 再做 RL」合并成一个在偏好对上的分类 loss，不需要采样和 reward model，但是 off-policy 的。GRPO 保留在线采样但去掉 value model，用同一 prompt 下一组样本的 reward 均值做基线，适合有可验证 reward 的推理任务。

## 推导

- **PPO 四模型**：policy（训练）、reference（冻结，算 KL）、reward（冻结，打分）、value（训练，估 advantage）；clip 的 importance ratio 限制每步更新幅度。
- **KL 约束**：reward 里减去 β·KL(policy ‖ reference)，防止 policy 为了刷 reward 跑偏；β 太大学不动，太小会 reward hacking。
- **DPO**：把 Bradley-Terry 偏好模型和 KL 约束的最优解代回去，得到只含 policy 和 reference 对数概率的 loss；训练像 SFT 一样简单，缺点是依赖离线偏好数据的分布。
- **GRPO**：对每个 prompt 采样 G 个回答，advantage = (reward − 组内均值) / 组内标准差，省掉 value model 的显存和训练不稳定；DeepSeek-R1 用它做数学 / 代码的 RL。

## 面试追问

::: details Q：GRPO 去掉 value model 后，方差会不会变大？
会，组内均值是比学出来的 value 更粗糙的基线。GRPO 用组内标准化和较大的组大小（如 16–64）来压方差，并且在可验证 reward（答案对错）的任务上 reward 本身噪声小，所以实践上够用。在开放式对话这类 reward 噪声大的任务上，PPO 的 value model 仍有优势。
:::

## 手撕

常见题：写出 DPO 的 loss 公式并解释每一项；写出 GRPO 的 advantage 计算。系统实现见 [RL Infra](/posttrain/rl-infra)。

## 参考

- [InstructGPT（PPO-RLHF）](https://arxiv.org/abs/2203.02155)
- [DPO](https://arxiv.org/abs/2305.18290)、[DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)

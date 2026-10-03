---
title: RL Infra：Rollout 与 Training 分离
status: todo
tags: [rl-infra, verl, openrlhf]
difficulty: 4
order: 4
related: []
stack: [ld-load]
---

# RL Infra：Rollout 与 Training 分离

> 权重同步、veRL / OpenRLHF 架构

## 一句话结论

RL 训练一轮要先用当前策略生成很多样本（rollout，推理负载），再用这些样本更新策略（training，训练负载），两种负载对并行方式和显存布局的要求完全不同，所以现代 RL 框架把 rollout 交给 vLLM / SGLang 这类推理引擎，training 交给 FSDP / Megatron，中间最难的是每轮把更新后的权重同步给推理引擎。

## 推导

- **为什么分离**：rollout 需要大 batch、KV cache、低精度，训练需要优化器状态、梯度和重计算；同一份权重布局不可能两边都最优。
- **权重同步**：每轮训练后把参数从训练进程广播到推理引擎（NCCL 直传或经 CPU），推理引擎的 TP 切分和训练的 FSDP 切分不同，中间要 reshard；大模型下这一步可能占一轮的 10–20%。
- **veRL**：HybridFlow 设计，用单控制器编排多个 worker group，支持 colocate（同一组卡上切换训练 / 推理模式）和分离两种部署；OpenRLHF 用 Ray 把 actor / critic / reward / vLLM 分到不同的 GPU 组上。
- **异步化**：rollout 慢于 training 时让 rollout 用落后一两版的策略继续生成（off-policy 容忍），提升 GPU 利用率，代价是 PPO 的 importance ratio 偏离。

## 面试追问

::: details Q：colocate 和分离部署怎么选？
colocate 把训练和推理放同一组卡上轮流跑，GPU 不闲置但每次切换要卸载 / 加载显存，适合中小规模。分离部署让 rollout 和 training 各占一组卡流水起来，吞吐更高但两边负载不匹配时有一边会空等，要靠异步 rollout 补。规模越大、生成越长（推理任务），越倾向分离。
:::

## 手撕

常见题：画出一轮 PPO 的数据流（rollout → reward → advantage → update → 权重同步）并标出每步跑在哪组卡上；算法细节见 [RLHF 全家桶](/posttrain/rlhf-ppo-dpo-grpo)。

## 参考

- [HybridFlow（veRL）论文](https://arxiv.org/abs/2409.19256)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

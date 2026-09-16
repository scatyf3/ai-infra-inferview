---
title: Speculative Decoding
status: draft
tags: [speculative-decoding, eagle, mtp]
difficulty: 4
order: 7
related: [/inference/prefill-decode-roofline, /inference/metrics-benchmark]
---

# Speculative Decoding

> draft-target、acceptance rate、EAGLE / Medusa / MTP 的区别

## 一句话结论

**用闲置的算力换延迟。** decode 是 memory-bound，tensor core 基本在空转；speculative decoding 让一个便宜的 draft 一次猜 $k$ 个 token，target 模型一次前向并行验证这 $k$ 个，把「$k$ 次 memory-bound 的 GEMV」变成「1 次 AI 高 $k$ 倍的小 GEMM」。加速比取决于 acceptance rate 和 draft 成本，**且分布严格不变**。

## 推导

### 为什么能白赚

target 模型 decode 一步要读完整个权重（比如 132 GiB），只算出 1 个 token。如果一次喂 $k$ 个 token 进去，读的字节数**几乎不变**（权重还是读一遍，只有 KV 和激活多一点），FLOPs 涨 $k$ 倍，AI 从 $B$ 变成 $kB$。因为原本在 roofline 的左侧还有大量余量，这 $k$ 倍 FLOPs 几乎是免费的。

关键是这 $k$ 个 token 从哪来：让一个小得多的 draft 先猜。

### 验证与分布保证

投机采样（Leviathan et al. / Chen et al.）的核心是 rejection sampling：draft 给出分布 $q$，target 给出 $p$，对每个位置：

- 以概率 $\min(1, p(x)/q(x))$ 接受 draft 的 token $x$；
- 否则拒绝，并从残差分布 $\text{norm}(\max(0, p - q))$ 重新采样，后面的 draft token 全部作废。

可以证明最终输出的分布**严格等于 target 单独采样的分布**。这点很重要：speculative decoding 不是近似加速，是精确加速，不需要重新评测质量。

### 加速比

设接受率为 $\alpha$（每个 draft token 被接受的概率），一次投机平均产出

$$
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

加速比

$$
\text{speedup} = \frac{1}{c \cdot k + 1} \cdot \frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

$c$ 是 draft 相对 target 的单步成本比。两个推论：
- $\alpha$ 是主导项。$\alpha = 0.8$、$k = 4$ 时期望产出 3.4 个 token；$\alpha = 0.5$ 时只有 1.9 个。
- $k$ 有最优值，不是越大越好。$k$ 大了以后 $\alpha^k$ 衰减，而 draft 成本线性增长。典型最优 $k$ 在 3–5。

### 几种流派

| 方法 | draft 怎么来 | 特点 |
|---|---|---|
| 经典 draft-target | 一个独立的小模型（如 1B 配 70B） | 要单独部署和维护，词表必须一致；$\alpha$ 取决于两个模型的相似度 |
| Medusa | 在 target 的最后一层 hidden 上挂多个独立的预测头，第 $i$ 个头预测第 $i+1$ 个位置 | 无需额外模型，但各头之间独立，不看彼此的输出，$\alpha$ 偏低；用树形注意力同时验证多条候选 |
| EAGLE | 在**特征层**（倒数第二层 hidden）做自回归，而不是在 token 层；draft 头输入上一步的特征加上已采样 token 的 embedding | $\alpha$ 显著高于 Medusa，因为特征层的不确定性比 token 层低；EAGLE-2 用动态树，EAGLE-3 去掉特征预测的约束、直接多层融合 |
| MTP（multi-token prediction） | 训练时就让模型多预测几个位置，推理时这些头直接当 draft | 内生于模型，DeepSeek-V3 就带 MTP 头，$\alpha$ 高（约 0.85），且额外开销极小 |

EAGLE 和 Medusa 的核心区别值得记牢：**Medusa 在 token 空间并行猜，EAGLE 在特征空间自回归猜**。后者保留了序列依赖，所以准得多。

### 什么时候没用

batch 大的时候。batch 128 时 decode 的 AI 已经接近 ridge，算力不再闲置，投机多消耗的 FLOPs 变成真实成本；同时 batch 内每个序列的接受长度不同，要做 ragged 的处理，调度复杂度上升。所以 speculative decoding 是**低延迟场景**（小 batch、单用户、交互式）的手段，高吞吐场景要慎用，或者用动态策略：batch 小时开、batch 大时关。

## 面试追问

::: details Q：接受率低的时候会不会比不投机还慢？
会。每次投机都要付 draft 的 $k$ 次前向加 target 的 1 次前向；如果全部被拒，这一轮只产出 1 个 token（来自残差分布的重采样），却多花了 draft 的成本。所以 $\alpha$ 太低时（比如 < 0.3）净收益为负。生产系统会在线统计接受率并动态调 $k$，甚至完全关闭。
:::

::: details Q：为什么说它不损失质量，但实测输出确实不完全一样？
分布相同不等于逐 token 相同。rejection sampling 保证的是从同一个分布采样，随机数序列不同自然得到不同的样本。greedy 解码下则应该逐 token 完全一致（可以用这个做正确性测试：温度 0 时投机与非投机的输出必须 bit-level 相同，不同就是实现有 bug）。
:::

::: details Q：树形注意力（tree attention）解决什么问题？
线性的 draft 只能猜一条链，一旦某位置被拒后面全废。树形结构同时猜多条候选路径（比如第一个位置猜 top-4，每个分支再猜 top-2），用一个特制的 attention mask 让所有候选在一次前向里被并行验证，提高单次投机的期望产出。代价是验证的 token 数从 $k$ 涨到树的节点数，FLOPs 上升，所以树的形状也有最优解（EAGLE-2 的贡献就是让树的形状随上下文动态调整）。
:::

::: details Q：MTP 和 EAGLE 哪个更适合工程落地？
有 MTP 头的模型（DeepSeek-V3）直接用 MTP，零额外训练成本、接受率高、权重随模型一起发布。没有 MTP 的模型用 EAGLE，但要为每个 target 模型单独训一个 EAGLE 头（几小时到一天的量级），且要跟着 target 的版本更新。框架侧两者的验证逻辑可以共用，区别只在 draft 的产生方式。
:::

::: details Q：投机解码和 chunked prefill 会打架吗？
会抢同一个 token 预算。投机验证的 $k$ 个 token 占用的是 batch 的 token 预算，chunked prefill 的 chunk 也是。调度器需要统一核算，否则要么投机被饿死，要么 prefill 被拖慢。vLLM 的做法是把投机的验证 token 算进 `max_num_batched_tokens`，一起参与预算分配。
:::

## 参考

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling (DeepMind)](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [EAGLE-2](https://arxiv.org/abs/2406.16858)

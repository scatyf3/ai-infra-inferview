---
title: Speculative Decoding
status: draft
tags: [speculative-decoding, eagle, mtp, dflash]
difficulty: 4
order: 7
related: [/inference/prefill-decode-roofline, /inference/metrics-benchmark]
stack: [f-runner, o-sampling]
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

## 论文笔记：EAGLE 之后

> 原始 slides：[SD Paper Reading](https://docs.google.com/presentation/d/1eNTmSHrpCrjbNX98yT7soWu2lwR86yxyJSuSmRA4-vo/edit?usp=sharing)

<iframe src="https://docs.google.com/presentation/d/1eNTmSHrpCrjbNX98yT7soWu2lwR86yxyJSuSmRA4-vo/preview" style="width:100%;aspect-ratio:16/9;border:0;border-radius:8px" allowfullscreen loading="lazy"></iframe>

主线：EAGLE 的 draft 是**串行**的，这既限制了加速比，也带来训练和推理不一致的问题。DFlash 改成并行 draft，后面几篇在补并行 draft 的短板。

### EAGLE 回顾

- **结构**：1 层 decoder，Embedding 和 LM Head 都与 target 共享；自回归地生成 $\gamma$ 个 draft token。
- **特征融合**：每步把 hidden state 和已采样 token 的 embedding 融合后输入，提高准确率。第 1 步用 target 的 hidden，第 $2 \sim \gamma$ 步用 draft 自己输出的 hidden。
- **KV**：draft 自己的 KV cache 会丢弃，每轮验证后用 target hidden 重新 prefill 一遍 draft 的 KV。

### EAGLE 的问题

- **训练和推理不一致**：draft 深度为 1 时，输入是 target 的 hidden；深度大于 1 时，换成了 draft 自己的输出 hidden。两种 hidden 的分布不同，越往后误差越大（hidden state drift）。改用 post-norm 后的 hidden，或者用 TTT（training-time test，训练时就让 draft 吃自己的输出）可以缓解。参见 [Attention Drift](https://arxiv.org/abs/2605.09992)。
- **性能**：draft 本身是串行 decode，$\gamma$ 步 draft 就要 $\gamma$ 次前向，$T_\text{draft}$ 随 $\gamma$ 线性增长，限制了加速比上限。

### DFlash：块扩散并行 draft

[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

**draft 结构**
- 块内使用 **bidirectional attention**，一次前向生成整块 $\gamma$ 个 token，$T_\text{draft}$ 不再随 $\gamma$ 增长。
- 省下的时间可以拿来堆更多 draft 层，提高准确率，而 draft 开销仍然小。
- hidden state 的融合方式比 EAGLE 更好。
- KV 共享：通过 draft 的 $W_k, W_v$ 直接把 target hidden 投影成 KV，不用像 EAGLE 那样 re-prefill。
- 因为只有 1 步 draft，draft 永远不吃自己的输出，训练和推理天然一致。

**训练**
- 随机选 anchor token，把它后面的位置全部 mask：$[\text{anchor}, \langle m\rangle, \langle m\rangle, \dots]$，让模型一次预测整块。
- 块内用加权 CE loss（越靠后的位置越难，权重不同；更好的权重见下面 D-PACE）。

### DSpark：补 DFlash 的三个短板

| DFlash 的问题 | DSpark 的解法 |
|---|---|
| 双向 attention 下块内各位置互相看不到彼此的采样结果，块的后半段接受率低 | **Semi-AR**：并行生成之后加一个轻量的串行 head |
| 在线服务、batch 大时，$\gamma$ 大会让 target 验证的开销变成真实成本 | **Confidence head**：预测每个 draft token 的接受概率，只挑高的去验证 |
| CE loss 不等价于接受率 | **TV loss**：接受率 $= 1 - \text{TV}(p, q)$，直接优化 TV 距离 |

最后一行的 $p$ 是 target 分布、$q$ 是 draft 分布，$\text{TV}(p,q) = \tfrac12 \sum_x |p(x) - q(x)|$。rejection sampling 下单个位置的接受概率 $\sum_x \min(p(x), q(x))$ 恰好等于 $1 - \text{TV}(p,q)$。

**Semi-AR**
- 每个位置的 draft logits = 并行生成的 base logits $U$ + 依赖前缀的转移 bias $B$，再 softmax 归一化。$B$ 由轻量的串行 head 算出，补上了位置之间的条件依赖。
- 串行 head 是一个低秩的 Markov head：概念上表示「给定上一个 token 时的条件概率」，低秩是为了算得快。
- 用法：并行生成之后再串行跑 $\gamma$ 步串行 head。它比 EAGLE 的 decoder 层轻得多，开销小。

**Confidence head**
- 目标：预测当前 draft token 被接受的概率；batch 大时贪心地只挑接受概率高的 token 去验证，提升系统吞吐。
- 结构：FFN + Sigmoid；输入是 diffusion block 和串行 head 的 hidden；回归目标是 TV 距离。
- Post-hoc 校准：FFN 容易过度自信，用一个经验温度 $T$ 校准预测值。
- Hardware-aware prefix scheduler：一张 SPS 表记录 server 在不同 batch size 下的 tps；所有请求的 draft token 按接受概率全局排序，贪心选取，收益不再增加时提前停止。

### P-EAGLE：让 EAGLE 也并行 draft

[P-EAGLE: Parallel-Drafting EAGLE with Scalable Training](https://arxiv.org/pdf/2602.01469)

- 核心想法：给 EAGLE head 加一个**可学习的 hidden state**，仍然用 causal attention。
- 预测第 $N+k$（$k > 1$）个 token 时，还没有真实的 hidden 可用，所有这些位置共享这个可学习的 hidden 作为占位输入，它随反向传播更新。这样一次前向就能出所有位置。
- 容量补偿：draft 加到 4 层。

### D-PACE：更好的并行 draft loss

[D-PACE: Dynamic Position-Aware Cross-Entropy for Parallel Speculative Drafting](https://arxiv.org/pdf/2605.18810)

设 $q_i$ 是 draft 在块内第 $i$ 个位置给正确 token 的概率（≈ 接受概率），期望接受长度（EAL）的代理目标是

$$
\bar S = \sum_{k=1}^{\gamma} \prod_{i=1}^{k} q_i
$$

- 直接最大化 $\bar S$ 会崩：前面的 $q_i$ 很小时连乘趋近 0，后面位置拿不到梯度。
- 按位置拆开，对 $\log q_j$ 求导：$\dfrac{\partial \bar S}{\partial \log q_j} = \sum_{k \ge j} \prod_{i \le k} q_i =: w_j$。这正是一个以 $w_j$ 为权重的 CE 梯度，即 $\sum_j w_j \log q_j$。
- 再把 $w$ 平滑一下（避免 $w$ 本身过小导致梯度消失），得到动态的、随位置变化的加权 CE。
- 例：$q = (0.9, 0.5, 0.5)$ 时，$w_1 = 0.9 + 0.45 + 0.225 = 1.575$，$w_3 = 0.225$，越靠前的位置越重要。

### TreeGraft：多 drafter 的树形投机

[TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding](https://arxiv.org/pdf/2608.26112)

- 建树用**全局 TopK**，而不是 beam search 那种只扩展新叶子的 TopK。
- 多个 drafter 的结果只追加（append-only）、不替换，整棵树建完后再做全局剪枝。
- 调度：一个可训练的 MLP 决定这一步用 1 个 drafter（普通 SD）还是 2 个。

### BudgetDraft：稀疏 KV 下的 drafter

[BudgetDraft: Acceptance-Aware Multi-View Training for Sparse-KV Speculative Decoding](https://arxiv.org/abs/2606.00144)

- 目标：drafter 只看稀疏的 draft KV cache 时，仍然保持高接受率。做法是微调 drafter 权重。
- 关键想法：在训练时就消除这个不一致，让 drafter 训练时就用 sliding window attention。
- Multi-view 稀疏训练：随机采一个 KV 预算 $B$，按 attention 分数保留 top-$\lfloor B/8 \rfloor$ 个 chunk。
- Loss：teacher forcing 下对 verifier 的 greedy token 做 top-1 CE，full view 和 sparse view 都算。

### SpecHop：投机用在多跳检索上

[SpecHop](https://arxiv.org/pdf/2605.21965)

- 思路类似 lookahead reasoning，但 draft 猜的不是推理步骤，而是 **web search 工具调用的返回结果**，下一跳可以提前开始。
- 验证：检索真正返回后，用字符串匹配检查猜得对不对。

### EfficientRollout：用 SD 加速 RL rollout

[EfficientRollout](https://arxiv.org/pdf/2606.18967)

**观察**
- RL 每轮都在更新 target 参数，draft 怎么跟上 target 的变化？
- rollout 是离线 batching，长尾明显：少数特别长的推理序列拖住整个 batch。长尾阶段 batch 很小，正好是 SD 能起作用的区间。

**方法**
- draft = 每轮训练后 target 的**量化版本**，并与 target 共享 KV cache，天然跟着 target 更新。
- 根据经验规则决定何时开 SD，并自适应调整 draft 长度。

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
- [DFlash](https://arxiv.org/abs/2602.06036) · [P-EAGLE](https://arxiv.org/pdf/2602.01469) · [D-PACE](https://arxiv.org/pdf/2605.18810) · [TreeGraft](https://arxiv.org/pdf/2608.26112) · [BudgetDraft](https://arxiv.org/abs/2606.00144) · [Attention Drift](https://arxiv.org/abs/2605.09992)

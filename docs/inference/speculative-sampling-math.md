---
title: 投机采样的数学推导
status: draft
tags: [speculative-decoding, sampling, math]
difficulty: 3
order: 7.5
related: [/inference/speculative-decoding, /leetgpu/speculative-decoding-verification]
stack: [o-sampling]
---

# 投机采样的数学推导

> 为什么「按 $\min(1, q/p)$ 接受、拒绝就从残差分布重采样」得到的分布严格等于 target · 接受率、期望产出、逆 CDF、greedy 特例

**记号**（和 [LeetGPU #87](/leetgpu/speculative-decoding-verification) 一致）：$p$ = **draft** 分布，$q$ = **target** 分布，都是词表 $V$ 上的概率向量。
[概念页](/inference/speculative-decoding) 和论文用的是反过来的记号（$q$ = draft，$p$ = target），公式一一对应，记住 **target 在分子**。

## 一句话结论

对 draft 采出的 $x \sim p$，以 $\min\!\left(1, \frac{q(x)}{p(x)}\right)$ 接受；拒绝则从 $\text{norm}(\max(0, q - p))$ 重采样。两条路径的概率相加恰好是

$$
\min(p(x), q(x)) + \max(0,\, q(x) - p(x)) = q(x)
$$

所以输出**严格**服从 target 分布。单步接受率 $\alpha = \sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)$，draft 和 target 越像，接受率越高。

## 推导

### 1. 单个位置：输出分布等于 $q$

一个位置上的流程：

1. draft 采样 $x \sim p$
2. 以概率 $a(x) = \min\!\left(1, \frac{q(x)}{p(x)}\right)$ 接受，输出 $x$
3. 否则拒绝，从残差分布 $r$ 采样输出

$$
r(x) = \frac{\max(0,\, q(x) - p(x))}{\sum_{x'} \max(0,\, q(x') - p(x'))}
$$

要证：最终输出 $X$ 满足 $P(X = x) = q(x)$。

**接受路径。** draft 恰好采到 $x$ 且被接受：

$$
P(\text{draft} = x,\ \text{accept}) = p(x) \cdot \min\!\left(1, \frac{q(x)}{p(x)}\right) = \min(p(x),\, q(x))
$$

这一步就是这个接受率的设计意图：把 draft 的概率 $p(x)$ 往下「削」到不超过 $q(x)$。

**拒绝的总概率。** 把所有 $x$ 的接受概率加起来，剩下的就是拒绝：

$$
P(\text{reject}) = 1 - \sum_x \min(p(x), q(x))
$$

因为 $\sum_x q(x) = 1$，且 $q - \min(p, q) = \max(0, q - p)$，所以

$$
P(\text{reject}) = \sum_x \big(q(x) - \min(p(x), q(x))\big) = \sum_x \max(0,\, q(x) - p(x))
$$

记这个数为 $Z$。它恰好是残差分布的归一化常数。

**拒绝路径。** 拒绝后从 $r$ 采到 $x$：

$$
P(\text{reject}) \cdot r(x) = Z \cdot \frac{\max(0,\, q(x) - p(x))}{Z} = \max(0,\, q(x) - p(x))
$$

**相加：**

$$
P(X = x) = \min(p(x), q(x)) + \max(0,\, q(x) - p(x)) = q(x) \qquad \blacksquare
$$

最后一步按 $p(x) \ge q(x)$ 和 $p(x) < q(x)$ 分两种情况各验一遍即可：前者是 $q + 0$，后者是 $p + (q - p)$。

**直觉。** 把 $q$ 拆成两块：$\min(p, q)$ 是 draft 「已经覆盖到」的部分，靠接受路径给出；$\max(0, q - p)$ 是 draft 「欠下」的部分，即 target 想要、但 draft 给少了的 token，靠拒绝后的重采样补齐。draft 给多了的部分（$p > q$ 的那些 $x$）通过拒绝被削掉。

### 2. 接受率和 TV 距离

单步接受率

$$
\alpha = \sum_x \min(p(x), q(x)) = 1 - Z = 1 - \text{TV}(p, q)
$$

其中 $\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)| = \sum_x \max(0, q(x) - p(x))$。后一个等号成立，是因为 $p$ 和 $q$ 的总和都是 1，正差和负差的绝对值相等。

推论：
- $p = q$ 时 $\alpha = 1$，永远接受。此时残差全 0、分母 $Z = 0$，但拒绝的概率也是 0，残差分布实际上用不到。LeetGPU 规定这时退化成均匀分布，只是为了浮点上不除以 0。
- $p$ 和 $q$ 支撑集不相交时 $\alpha = 0$，每次都拒绝。这时每轮只产出 1 个 token（重采样那个），比不投机还多付了 draft 的成本。

### 3. 多个位置：逐位置归纳

draft 一次提出 $t_0, \dots, t_{T-1}$。第 $i$ 个位置的 $p_i$、$q_i$ 都是**以前缀 $t_{<i}$ 为条件**的分布：target 一次前向就把 $q_0, \dots, q_{T}$ 全算出来，因为输入是 prefix 加上 $t_0 \dots t_{T-1}$，causal attention 保证第 $i$ 个输出只看到 $t_{<i}$。

从左到右做第 1 节的单步验证，在第一个拒绝处停下：
- 位置 $0$ 输出 $\sim q_0$（第 1 节）
- 若位置 $0$ 接受了 $t_0$，前缀就是一个服从 target 的样本，位置 $1$ 的条件分布 $q_1(\cdot \mid t_0)$ 正是 target 在该前缀下的下一个 token 分布；再用一次第 1 节，位置 $1$ 输出 $\sim q_1$
- 依此归纳，输出的整个序列和「target 自己逐个采样」同分布

拒绝后为什么后面的 draft token 要作废：它们是以被拒绝的 $t_i$ 为前缀生成的，而实际输出的是重采样的另一个 token，条件已经不成立。

**bonus token。** 如果 $T$ 个全接受，target 那次前向已经顺带算出了 $q_T$（前缀是全部 $T$ 个 draft token），直接从 $q_T$ 再采一个，不需要额外前向。所以每轮至少产出 1 个 token（第一个就被拒绝时，产出重采样的那个），最多产出 $T + 1$ 个。

LeetGPU #87 的 `target_probs` 只有 $T$ 行，题目规定 bonus 从 $q_{T-1}$ 采，这是出题上的简化。真实系统里是 $q_T$。

### 4. 期望产出

假设每个位置独立地以 $\alpha$ 接受（论文里的简化假设）。设接受个数为 $N$，则 $P(N \ge k) = \alpha^k$（$k \le T$），每轮输出 $N + 1$ 个 token：

$$
\mathbb{E}[N + 1] = 1 + \sum_{k=1}^{T} P(N \ge k) = \sum_{k=0}^{T} \alpha^k = \frac{1 - \alpha^{T+1}}{1 - \alpha}
$$

指数上的 $T+1$ 和前面的「$1 +$」都来自那 1 个必定产出的 token：拒绝时是重采样的 token，全接受时是 bonus。代入加速比的推导见 [概念页](/inference/speculative-decoding#加速比)。

### 5. 逆 CDF 采样

给一个 $U \sim \text{Uniform}[0, 1)$，令累积分布 $F(x) = \sum_{x' \le x} \pi(x')$，取

$$
X = \min\{x : F(x) > U\}
$$

则 $P(X = x) = P(F(x-1) \le U < F(x)) = F(x) - F(x-1) = \pi(x)$。$F$ 把 $[0, 1)$ 按概率大小切成 $V$ 段，$U$ 落在哪段就是哪个 token。

代码里就是 `cumsum` 加 `searchsorted(cdf, u, right=True)`；`right=True` 对应上式的严格不等号 $F(x) > U$。浮点累加可能让 $F(V-1)$ 略小于 1，$U$ 接近 1 时会越界，所以结果要 `clamp(max=V-1)`。

### 6. greedy（温度 0）特例

温度 0 时 $p$、$q$ 都是 one-hot：$p = e_{\hat t}$（draft 的 argmax），$q = e_{t^*}$（target 的 argmax）。

- $\hat t = t^*$：$\alpha = q(\hat t) / p(\hat t) = 1$，必接受
- $\hat t \ne t^*$：$q(\hat t) = 0$，必拒绝；残差 $\max(0, q - p) = e_{t^*}$，重采样必然得到 $t^*$

于是整套算法退化成：**逐位比较 draft token 和 target 的 argmax，第一个不同的位置换成 target 的 argmax，之后截断。** 这也是为什么温度 0 下投机与非投机的输出必须逐 token 一致，可以拿来做正确性测试。

## 面试追问

::: details Q：为什么不能直接「target 也觉得不错就接受」，比如 $q(x) \ge p(x)$ 就接受、否则拒绝？
硬阈值会让被接受的 token 的概率变成 $p(x) \cdot \mathbb{1}[q(x) \ge p(x)]$，不等于 $\min(p, q)$；而且拒绝后如果直接从 $q$ 重采样，又会把已经覆盖的部分重复算一遍。分布就偏了。$\min(1, q/p)$ 加残差分布是让两条路径恰好拼回 $q$ 的那组选择。
:::

::: details Q：残差分布为什么是 $\max(0, q - p)$，而不是直接从 $q$ 重采样？
接受路径已经给出了 $\min(p, q)$，剩下要补的只有 $q - \min(p, q) = \max(0, q - p)$。直接从 $q$ 重采样会让 $\min(p, q)$ 那部分被算两次，偏向 draft 本来就爱采的 token。
:::

::: details Q：接受率和什么有关？怎么提高？
$\alpha = 1 - \text{TV}(p, q)$，只取决于两个分布有多接近。提高的办法是让 draft 更像 target：用同系列的小模型、蒸馏 draft、或者 EAGLE / MTP 这类直接复用 target 特征的 draft。采样温度也有影响：温度越高，两个分布越平，TV 通常越小。
:::

::: details Q：batch 内每条序列接受个数不一样，怎么办？
每条序列独立做上面的验证，产出 $N_b + 1$ 个 token，长度参差不齐。框架侧要按序列各自回滚 KV cache 里被拒绝的那部分（paged KV 只需改 block table 里的长度），下一轮各自从自己的位置继续。
:::

## 参考

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)（Leviathan et al.，附录有完整证明）
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)（Chen et al.，DeepMind）

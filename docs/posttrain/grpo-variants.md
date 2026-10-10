---
title: GRPO 变体：DAPO / Dr. GRPO / GSPO / CISPO
status: draft
tags: [grpo, dapo, gspo, cispo, rl]
difficulty: 4
order: 3.5
related: [/posttrain/rlhf-ppo-dpo-grpo, /posttrain/rl-infra, /posttrain/rl-train-infer-mismatch]
stack: []
---

# GRPO 变体：DAPO / Dr. GRPO / GSPO / CISPO

> 把 GRPO 的目标拆成四个零件，每个变体只改其中一两个：改了哪个零件、为什么、用一个小数字例子看清楚

## 一句话结论

GRPO 的目标由四个零件组成：**importance ratio 和 clip**、**advantage 怎么归一化**、**loss 怎么在 token 和样本之间平均**、**KL 项**。DAPO 放宽 clip 的上界、过滤没有梯度的组、改成按 token 平均、去掉 KL；Dr. GRPO 去掉按长度和按 std 的两处归一化；GSPO 把 ratio 从 token 级换成序列级，主要为了 MoE 训练稳定；CISPO 不再把超出 clip 范围的 token 的梯度丢掉，而是只截断它的权重。面试时先说清楚「改了哪个零件、解决什么现象」，再写公式。

## 推导

### 0. 起点：GRPO 的四个零件

GRPO 的推导见 [RLHF 全家桶](/posttrain/rlhf-ppo-dpo-grpo)。符号：$x$ 是 prompt，用旧策略 $\pi_{\text{old}}$ 采 $G$ 条回答 $y_1, \dots, y_G$，第 $i$ 条长 $|y_i|$、奖励 $r_i$；$y_{i,t}$ 是第 $i$ 条回答的第 $t$ 个 token。

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Bigg[\underbrace{\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}}_{\text{③ 怎么平均}}\Big(\underbrace{\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)}_{\text{① ratio 和 clip}} - \underbrace{\beta\,\hat{\mathbb{D}}_{\text{KL}}}_{\text{④ KL}}\Big)\Bigg]
$$

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})}, \qquad
\underbrace{\hat A_i = \frac{r_i - \operatorname{mean}(r_1..r_G)}{\operatorname{std}(r_1..r_G)}}_{\text{② advantage 归一化}}
$$

DeepSeekMath 的设置是 $G = 64$、$\beta = 0.04$，每次 rollout 只更新一次策略（[arXiv 2402.03300](https://arxiv.org/abs/2402.03300) §4.2）；论文没写 $\epsilon$，开源实现常用 0.2。各变体改的零件：

| 变体 | ① ratio / clip | ② advantage | ③ 平均方式 | ④ KL | 其他 |
|---|---|---|---|---|---|
| DAPO | 上下界分开 | 不变 | 按 token | 去掉 | 动态采样、超长惩罚 |
| Dr. GRPO | 不变 | 不除 std | 除以常数 | 去掉 | |
| GSPO | 序列级 ratio | 不变 | 序列级 | 去掉 | |
| CISPO | 截断权重、不丢梯度 | 不变 | 按 token | 去掉 | |

下面按零件讲。

### 1. clip 的上界卡住了低概率 token（DAPO：Clip-Higher）

**现象**：训练一段时间后策略的熵迅速下降，同一个 prompt 采出来的 $G$ 条回答越来越像，探索没了（熵塌缩）。

**原因在 clip 的上界**。$\hat A > 0$ 时，clip 让 $\rho \le 1 + \epsilon$，也就是新概率最多是旧概率的 $1 + \epsilon$ 倍。$\epsilon = 0.2$ 时：

1. 旧概率 0.9 的 token，一步最多涨到 $0.9 \times 1.2 = 1.08$，实际上不受限。
2. 旧概率 0.01 的 token，一步最多涨到 $0.01 \times 1.2 = 0.012$。

上界对高概率 token 几乎没有约束，对低概率 token 卡得很死。而探索恰恰靠这些低概率 token：一条做对了的回答里，采到了一个原本不太可能的 token，本该大幅提高它的概率。

**改法**：上下界分开，$\epsilon_{\text{low}} = 0.2$、$\epsilon_{\text{high}} = 0.28$（[DAPO](https://arxiv.org/abs/2503.14476) §3.1）。0.01 的 token 一步能涨到 0.0128。下界不放宽：$\hat A < 0$ 时放宽下界会把低概率 token 压到接近 0，探索反而更少。

### 2. 全对或全错的组没有梯度（DAPO：动态采样）

一组 $G$ 条回答全对或全错时，$r_i$ 全相同，$\hat A_i = 0$，这组对梯度毫无贡献。设模型对某题的正确率是 $p$，$G = 16$：

| $p$ | 全对的概率 $p^{16}$ |
|---|---|
| 0.5 | 0.000015 |
| 0.8 | 0.028 |
| 0.9 | 0.19 |
| 0.95 | 0.44 |

训练越往后，模型做对的题越多，一个 batch 里没有梯度的组越多，实际参与更新的样本数就越少，梯度方差越大。

**改法**：多采一些 prompt，把正确率恰好是 0 或 1 的组过滤掉，一直采到 batch 填满为止（[DAPO](https://arxiv.org/abs/2503.14476) §3.2）。目标函数里写成约束：

$$
0 < \big|\{y_i \mid y_i \text{ 答对}\}\big| < G
$$

代价是 rollout 量变大，而且每个 step 要采多少才够不确定。rollout 系统怎么处理这种「采到够为止」的需求，见 [异步 Rollout](/posttrain/rl-async-rollout)。

### 3. loss 怎么平均：长回答被稀释（DAPO 和 Dr. GRPO）

GRPO 先对每条回答的 token 求平均（除以 $|y_i|$），再对 $G$ 条回答求平均。这样每条回答的**总权重**一样，摊到每个 token 上就和长度成反比。

**例子**：两条答错的回答，$\hat A = -1$，长度分别是 100 和 1000 个 token。只看平均方式带来的系数（$G$ 一样，略去）：

| 平均方式 | 100 token 那条，每个 token 的系数 | 1000 token 那条，每个 token 的系数 |
|---|---|---|
| GRPO：$\frac{1}{G}\sum_i \frac{1}{\lvert y_i\rvert}\sum_t$ | $1/100$ | $1/1000$ |
| DAPO：$\frac{1}{\sum_i \lvert y_i\rvert}\sum_i\sum_t$ | $1/1100$ | $1/1100$ |
| Dr. GRPO：$\frac{1}{G}\sum_i \frac{1}{L_{\max}}\sum_t$ | $1/L_{\max}$ | $1/L_{\max}$ |

GRPO 下，长的错误回答每个 token 只挨 1/10 的罚。错误回答写得越长，每个 token 受罚越轻，于是模型的错误回答越写越长（[Dr. GRPO](https://arxiv.org/abs/2503.20783) §3.1 的 response-level length bias）。反过来，$\hat A > 0$ 时短的正确回答每个 token 得到的奖励更多。

两种改法：

1. **DAPO**：所有 token 一起平均，除以 batch 里的总 token 数 $\sum_i |y_i|$。同一个 batch 内每个 token 权重相同，但分母随 batch 变化。
2. **Dr. GRPO**：除以一个常数 $L_{\max}$（生成长度上限）。代码里就是把 `mask.sum(-1)` 换成常数 `MAX_TOKENS`。

Dr. GRPO 统计过：trl、OpenRLHF、verl 等框架当时的 PPO loss 都带着这个长度偏差（论文 Table 2）。

### 4. advantage 除以 std：难题和简单题被放大（Dr. GRPO）

设一组 $G = 8$，奖励答对为 1、答错为 0，用总体标准差（除以 $G$）：

| 答对几条 | mean | std | 答对那条的 $\hat A$ | 答错那条的 $\hat A$ |
|---|---|---|---|---|
| 1 | 0.125 | 0.331 | **+2.65** | −0.38 |
| 4 | 0.5 | 0.5 | +1.00 | −1.00 |
| 7 | 0.875 | 0.331 | +0.38 | **−2.65** |

std 小的组（几乎全对或几乎全错，也就是太简单或太难的题）被除以一个小数，advantage 被放大。这些题在梯度里的权重就比中等难度的题大（Dr. GRPO 称为 question-level difficulty bias）。

**改法**：只减均值，不除 std：

$$
\tilde A_i = r_i - \operatorname{mean}(r_1, \dots, r_G)
$$

上表三行的 advantage 变成 +0.875 / −0.125、±0.5、+0.125 / −0.875。它和 RLOO 的 leave-one-out 基线只差一个常数倍：$\frac{G}{G-1}\tilde A_i = \hat A^{\text{RLOO}}_i$（论文附录 A）。

Dr. GRPO 的完整目标：

$$
\mathcal{J}_{\text{Dr.GRPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\min\big(\rho_{i,t}\tilde A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\tilde A_i\big)\Bigg]
$$

论文设置 $\epsilon = 0.2$、$G = 8$、不加 KL（$\beta = 0$）。作者自测：Qwen2.5-Math-7B 在 8 张 A100 上训 27 小时，AIME 2024 到 43.3%，错误回答也不再越写越长。

### 5. 被截断的超长回答（DAPO：Overlong Reward Shaping）

生成到长度上限还没写完的回答会被截断，按规则判为错。但它可能推理得没问题，只是太长，直接给 −1 会引入噪声。DAPO 的两步：

1. **Overlong Filtering**：被截断的样本不算 loss。
2. **Soft Overlong Punishment**：在正确性奖励上再加一个长度惩罚：

$$
R_{\text{length}}(y) = \begin{cases}
0, & |y| \le L_{\max} - L_{\text{cache}} \\
\dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\
-1, & |y| > L_{\max}
\end{cases}
$$

论文取 $L_{\max} = 20480$、$L_{\text{cache}} = 4096$。长度 16384 以内不罚；18432 罚 $(16384 - 18432) / 4096 = -0.5$；超过 20480 罚 −1。中间一段线性过渡，告诉模型「快到上限了」。

DAPO 的消融（作者自测，Qwen2.5-32B base，AIME 2024 avg@32）：朴素 GRPO 30 分 → 加超长过滤 36 → 加 clip-higher 38 → 加软惩罚 41 → 加 token 级 loss 42 → 加动态采样 50。

### 6. token 级 ratio 噪声太大（GSPO）

**GSPO 的论点**（[arXiv 2507.18071](https://arxiv.org/abs/2507.18071) §3）：importance sampling 要靠很多样本才能修正分布差，而 $\rho_{i,t}$ 在每个位置只有一个样本，起不到修正作用，只是给每个 token 乘了一个随机的权重。长回答里这些噪声会累积，clip 又让它更糟，最终可能训崩，且崩了救不回来。奖励是给整条回答的，所以 ratio 也应该按整条回答算。

**序列级 ratio**，再按长度开方（几何平均），让不同长度的回答落在同一个数值范围里：

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right)^{1/|y_i|} = \exp\Big(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log \rho_{i,t}\Big)
$$

$$
\mathcal{J}_{\text{GSPO}} = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\min\big(s_i(\theta)\hat A_i,\ \operatorname{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\hat A_i\big)\Bigg]
$$

**例子**：一条 4 个 token 的回答，逐 token ratio 是 $[1.1, 0.9, 1.5, 0.8]$。

1. GRPO（$\epsilon = 0.2$，设 $\hat A > 0$）：第 3 个 token 的 1.5 超过 1.2 被 clip，没有梯度；其余三个各按自己的 ratio 更新。
2. GSPO：$\log\rho = [0.095, -0.105, 0.405, -0.223]$，平均 0.043，$s = e^{0.043} = 1.044$。整条回答共用这一个 ratio，没有 clip，四个 token 一起更新。不开方的话是连乘 1.188，长回答下会更离谱。

几何平均把单个 token 的波动压小了，所以 GSPO 的 clip 范围也小得多：论文用左 $3 \times 10^{-4}$、右 $4 \times 10^{-4}$，而 GRPO 基线是 0.2 / 0.27。按 token 数，GSPO 被 clip 掉的比例反而高两个数量级，但训练效率更高（作者自测，§5.2）。

**为什么对 MoE 特别重要**（§5.3）：Qwen3-30B-A3B（48 层）上，同一批 rollout 每更新一次，约 10% 的被激活专家会变。路由一变，同一个 token 在新旧策略下走的是不同的专家，$\rho_{i,t}$ 剧烈波动。GRPO 在 MoE 上要靠 **Routing Replay**（训练时强制复用 $\pi_{\text{old}}$ 的路由）才能收敛；GSPO 只看整条回答的似然，对单个 token 的路由变化不敏感，不需要 Routing Replay。训推两端路由不一致的问题见 [训推不一致](/posttrain/rl-train-infer-mismatch)。

GSPO 还有一个 token 级写法（GSPO-token），数值上等于 $s_i$，但允许每个 token 有不同的 advantage，用于多轮任务：

$$
s_{i,t}(\theta) = \operatorname{sg}\big[s_i(\theta)\big] \cdot \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\operatorname{sg}\big[\pi_\theta(y_{i,t} \mid x, y_{i,<t})\big]}
$$

$\operatorname{sg}$ 是 stop-gradient（`detach()`）。

### 7. clip 把关键 token 的梯度丢了（CISPO）

**现象**（[MiniMax-M1](https://arxiv.org/abs/2506.13585) §3.1）：推理里有一些低概率的「转折」token，比如 However、Wait、Recheck。它们概率低，策略更新一次后 ratio 就很大，被 clip 掉。PPO 的 clip 不是把权重截到上界，而是**这个 token 的梯度直接变成 0**：$\min(\rho \hat A, (1+\epsilon)\hat A)$ 取到常数那一边，对 $\theta$ 求导为 0。MiniMax 一批 rollout 做 16 次更新，这些 token 第一次更新后就再也没有贡献。DAPO 的 clip-higher 在这个设置下也不够用。

**改法**：不 clip 目标，而是 clip **importance 权重**，并对权重 stop-gradient，梯度始终从 $\log \pi_\theta$ 走：

$$
\mathcal{J}_{\text{CISPO}} = \mathbb{E}\Bigg[\frac{1}{\sum_i |y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}\operatorname{sg}\big(\hat\rho_{i,t}\big)\,\hat A_i\,\log\pi_\theta(y_{i,t} \mid x, y_{i,<t})\Bigg],
\qquad \hat\rho_{i,t} = \operatorname{clip}\big(\rho_{i,t},\ 1 - \epsilon_{\text{low}},\ 1 + \epsilon_{\text{high}}\big)
$$

**例子**：一个转折 token 旧概率 0.02，更新一次后变成 0.04，$\rho = 2$，$\hat A > 0$。取 $\epsilon_{\text{high}} = 0.28$ 示意（论文只说调过 $\epsilon_{\text{high}}$，没给数值）：

1. PPO / GRPO：$2 > 1.28$，被 clip，梯度为 0，这个 token 不再被加强。
2. CISPO：权重截到 1.28，梯度是 $1.28 \cdot \hat A \cdot \nabla\log\pi_\theta$，继续被加强，只是步子被限制住。

$\epsilon_{\text{low}}$ 设得很大，相当于没有下界。CISPO 的权重有截断，所以梯度有偏，但方差更小。作者自测：在 Qwen2.5-32B base 上，CISPO 达到 DAPO 的效果只用一半的 step。

### 8. 其他两个常被问到的基线：RLOO 和 REINFORCE++

**RLOO**（[arXiv 2402.14740](https://arxiv.org/abs/2402.14740)）：整条回答当一个动作，用另外 $k - 1$ 条的平均奖励当基线，没有 ratio 也没有 clip：

$$
\nabla J \approx \frac{1}{k}\sum_{i=1}^{k}\Big[r_i - \frac{1}{k-1}\sum_{j \ne i} r_j\Big]\nabla\log\pi_\theta(y_i \mid x)
$$

不用 $r_i$ 自己算基线，所以是无偏的。

**REINFORCE++**（[arXiv 2501.03262](https://arxiv.org/abs/2501.03262)，论文改过多版，这里按 v9）：先减组均值，再在**整个 batch** 上做均值和 std 归一化，而不是在组内。理由是组只有 4–8 条时组内 std 很不稳，std 接近 0 时 advantage 会爆；batch 有上千条，统计量接近常数。

### 9. 对比总表

| | 改了什么 | 解决什么现象 | 关键超参（论文值） |
|---|---|---|---|
| DAPO | clip 上界放宽、动态采样、按 token 平均、超长惩罚、去 KL | 熵塌缩、无梯度的组、长回答被稀释、截断噪声 | $\epsilon_{\text{low}} = 0.2$，$\epsilon_{\text{high}} = 0.28$，$G = 16$ |
| Dr. GRPO | 去掉 $1/\lvert y_i\rvert$ 和除以 std | 错误回答越写越长、难易题被放大 | $\epsilon = 0.2$，$G = 8$，$\beta = 0$ |
| GSPO | 序列级、几何平均的 ratio | token ratio 噪声、MoE 训崩 | clip $3 \times 10^{-4}$ / $4 \times 10^{-4}$ |
| CISPO | clip 权重而不是目标，权重 stop-gradient | 转折 token 的梯度被 clip 丢掉 | 只调 $\epsilon_{\text{high}}$（未公开） |
| RLOO | leave-one-out 基线，无 ratio | 基线无偏 | $k = 2, 4$ |
| REINFORCE++ | batch 级归一化 | 小组 std 不稳 | |

## 面试追问

::: details Q：PPO 的 clip 到底是「截断权重」还是「丢掉梯度」？
是丢掉梯度。$\hat A > 0$、$\rho > 1 + \epsilon$ 时，$\min(\rho\hat A, (1+\epsilon)\hat A) = (1+\epsilon)\hat A$，这是个常数，对 $\theta$ 的导数为 0。所以 clip 的作用是「超出信任域的 token 这一步不再推」。CISPO 改的就是这一点：权重截到 $1 + \epsilon$，但梯度仍从 $\log\pi_\theta$ 回传。
:::

::: details Q：DAPO 的 token 级平均和 Dr. GRPO 的除以常数有什么区别？
都让同一个 batch 里每个 token 的权重相同，区别在分母。DAPO 除以这个 batch 的总 token 数，batch 里回答普遍变长时，每个 token 的权重变小，等效学习率随长度变化。Dr. GRPO 除以固定的 $L_{\max}$，每个 token 的权重与 batch 无关，loss 的尺度跟着总 token 数走。
:::

::: details Q：为什么 GSPO 要开 $1/|y_i|$ 次方？
序列概率是逐 token 概率的连乘，ratio 也是逐 token ratio 的连乘。几百上千个 token 乘下来，几个 token 的波动就能让它变成极大或极小，而且长短不同的回答数量级不一样，没法用同一个 clip 范围。开 $1/|y_i|$ 次方就是取几何平均，把它拉回 1 附近，长短回答可比。
:::

::: details Q：k3 KL 估计放进 loss 里，优化的是哪个方向的 KL？
k3 $= r - \log r - 1$，$r = \pi_{\text{ref}} / \pi_\theta$，样本来自 $\pi_\theta$。作为**数值**，$\mathbb{E}_{\pi_\theta}[k3] = \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$，是反向 KL 的无偏估计。但对它**求梯度**：$\nabla k3 = (1 - r)\nabla\log\pi_\theta$，期望是 $\sum_y (\pi_\theta - \pi_{\text{ref}})\nabla\log\pi_\theta = -\sum_y \pi_{\text{ref}}\nabla\log\pi_\theta$，这是正向 KL $\mathrm{KL}(\pi_{\text{ref}} \| \pi_\theta)$ 的梯度。REINFORCE++ 也指出了这一点，并建议改用 k2 $= \frac12(\log r)^2$，它的梯度期望才是反向 KL 的梯度。DAPO、Dr. GRPO、GSPO 在可验证奖励的任务上干脆去掉了 KL。
:::

::: details Q：这些变体在系统上有什么影响？
1. 动态采样：每个 step 要采多少 prompt 不确定，rollout 要能「采到够为止」，同步的 `generate(batch)` 做不到，推着系统往异步、流式的 rollout 走。
2. 去掉 KL：不用跑 reference 模型的 forward，省一个模型的显存和一次前向。
3. GSPO：作者说可以直接用推理引擎给的序列似然，不必在训练引擎里重算（§5.4），对训推不一致更宽容。
4. 多次更新（CISPO 一批 16 次）：rollout 的样本被反复用，ratio 偏离 1 越来越多，clip 和权重截断的设计就越关键。
:::

## 手撕

把几个变体写进同一个 loss 函数。输入都是 `[N, T]`：`N` 条回答（$B$ 个 prompt × $G$ 条），`T` 是 padding 后的长度。

```python
import torch

def policy_loss(logp, old_logp, adv, mask, mode="grpo",
                eps_low=0.2, eps_high=0.2, max_tokens=None):
    # logp: 当前策略的 token logprob（带梯度）；old_logp: 旧策略（不带梯度）
    # adv: [N]，每条回答一个 advantage；mask: 回答部分为 1，padding 为 0
    A = adv[:, None]                                   # 广播到每个 token
    log_ratio = logp - old_logp

    if mode == "gspo":                                 # 序列级、几何平均的 ratio
        seq = (log_ratio * mask).sum(-1) / mask.sum(-1)
        s = torch.exp(seq)[:, None]
        obj = torch.min(s * A, s.clamp(1 - eps_low, 1 + eps_high) * A)[:, 0]
        return -obj.mean()

    ratio = torch.exp(log_ratio)
    if mode == "cispo":                                # 截断权重，梯度走 logp
        w = ratio.clamp(1 - eps_low, 1 + eps_high).detach()
        obj = w * A * logp
    else:                                              # grpo / dapo / dr_grpo
        obj = torch.min(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)

    if mode == "grpo":                                 # 每条先按长度平均
        return -((obj * mask).sum(-1) / mask.sum(-1)).mean()
    if mode == "dr_grpo":                              # 除以常数
        return -((obj * mask).sum(-1) / max_tokens).mean()
    return -(obj * mask).sum() / mask.sum()            # dapo / cispo：按 token 平均
```

advantage 的两种算法，`r` 是 `[B, G]`：

```python
A_grpo = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
A_dr   =  r - r.mean(1, keepdim=True)                  # Dr. GRPO：不除 std
keep   = (r.min(1).values != r.max(1).values)          # DAPO 动态采样：组内不全相同才留
```

常见题：DAPO 的四个改动分别解决什么；用一个长短回答的例子说明 GRPO 的长度偏差；写出 GSPO 的 ratio 并解释为什么开方；CISPO 和 PPO clip 的梯度区别。

## 参考

- [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
- [DAPO（Yu et al. 2025）](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO：Understanding R1-Zero-Like Training（Liu et al. 2025）](https://arxiv.org/abs/2503.20783)
- [GSPO：Group Sequence Policy Optimization（Qwen, 2025）](https://arxiv.org/abs/2507.18071)
- [MiniMax-M1（CISPO）](https://arxiv.org/abs/2506.13585)
- [RLOO：Back to Basics（Ahmadian et al. 2024）](https://arxiv.org/abs/2402.14740)
- [REINFORCE++（Hu et al.）](https://arxiv.org/abs/2501.03262)
- [Schulman：Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)

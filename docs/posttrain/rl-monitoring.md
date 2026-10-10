---
title: RL 训练监控与排障：看哪些曲线，坏了先查什么
status: draft
tags: [rl, monitoring, entropy, kl, debugging]
difficulty: 3
order: 4.5
related: [/posttrain/grpo-variants, /posttrain/rl-train-infer-mismatch, /posttrain/rl-infra, /posttrain/rlhf-ppo-dpo-grpo]
stack: []
---

# RL 训练监控与排障：看哪些曲线，坏了先查什么

> 跑 GRPO / DAPO 时面板上有几十条曲线。每条是怎么算的、健康时什么样、不健康时说明什么，以及训练崩之前通常先动的是哪几条

## 一句话结论

盯五类曲线：**熵**（探索还剩多少）、**回答长度和截断率**（长度偏差、超长）、**ppo_kl 和 clip 比例**（每步走了多远）、**训推 KL**（推理引擎和训练引擎差多少）、**梯度范数**（是不是要炸了）。训练崩之前通常有一个先后顺序：训推 KL 先冒尖 → 熵和训练侧的困惑度跟着跳 → 梯度范数爆炸 → 奖励断崖。另外两条要和训练奖励对着看：**验证集准确率**（训练奖励涨、验证不涨，可能在 reward hacking）和**时间分解**（rollout 占多少）。

## 推导

### 0. 一张总表

指标名以 verl 为准（[metric_utils.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/metric_utils.py)、[core_algos.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py)）：

| 指标 | 怎么算 | 看什么 |
|---|---|---|
| `actor/entropy` | 回答 token 上的平均熵（整个词表上的分布） | 掉得太快：探索没了；涨得太高：开始胡言乱语 |
| `response_length/mean` | 回答的平均长度 | 和验证准确率一起看；错误回答越来越长是长度偏差 |
| `response_length/clip_ratio` | 长度恰好等于上限的比例，也就是被截断的比例 | 高了说明大量回答写不完 |
| `actor/ppo_kl` | 当前策略对 rollout 时策略的 KL（不是对 reference） | 一批数据更新几次后走了多远 |
| `actor/pg_clipfrac` | PPO clip 生效的 token 比例 | 太高：更新太猛，被 clip 的 token 没梯度 |
| `actor/kl_loss` | 对 reference 的 KL（开了 `use_kl_loss` 才有） | 离 SFT 模型多远 |
| `rollout_corr/kl`、`k3_kl` | 推理引擎和训练引擎在同一份权重下的 KL | 冒尖是崩溃的前兆 |
| `actor/grad_norm` | 梯度裁剪前的总范数，默认裁到 1.0 | 爆炸是崩溃前最直接的信号 |
| `critic/score/mean` | 训练 batch 上的平均奖励 | 和验证准确率对着看 |
| `timing_s/gen`、`timing_s/step` | 生成用时、整步用时 | 两者之比就是 rollout 占比 |

### 1. 熵：探索还剩多少

**熵**：模型在某个位置对下一个 token 的不确定程度，$H = -\sum_v \pi(v)\log\pi(v)$，对回答里所有 token 取平均。

**熵会自然下降，而且下降得很快**。Cui et al.（[arXiv 2505.22617](https://arxiv.org/abs/2505.22617)）在多个模型和算法上发现（作者自测）：

1. 前 200 步（全程的 1/12）用掉了 73% 的熵，拿到了 76% 的性能提升。
2. 性能和熵之间近似满足 $R = -a\,e^{H} + b$。熵降到 0 时，性能也就到顶了，上限是 $-a + b$。

**为什么会降**：一步策略梯度后熵的变化近似为

$$
\Delta H \approx -\eta\,\operatorname{Cov}_{a\sim\pi}\big(\log\pi(a),\ \pi(a)\,A(a)\big)
$$

$\eta$ 是学习率，$A$ 是 advantage。高概率的动作拿到正 advantage，协方差为正，熵下降；低概率的动作拿到正 advantage，熵上升。

**两个动作的例子**：$\pi = (0.9, 0.1)$，熵 $H = 0.325$。

1. 高概率的那个答对了、被推高到 0.95：熵降到 0.199。
2. 低概率的那个答对了、被推高到 0.15：熵升到 0.423。

训练中大多数时候被奖励的是模型本来就倾向的回答，所以熵整体往下走。

**怎么办**：

1. DAPO 的 clip-higher（$\epsilon_{\text{high}} = 0.28$）：放宽上界，让低概率 token 能被推上去（见 [GRPO 变体](/posttrain/grpo-variants)）。DAPO 观察到被上界 clip 的 token 平均概率低于 0.2。
2. Clip-Cov / KL-Cov（Cui et al.）：找出协方差最大的那一小撮 token（比例 $2\times10^{-4}$ 到 $2\times10^{-3}$），对它们停梯度或加 KL，专门压住让熵掉得最快的更新。作者自测 Qwen2.5-32B 上平均分从 45.8 到 52.2。
3. 也不能让熵太高：DAPO 提到熵太高会伴随乱码和重复。熵应该待在一个合适的范围里，缓慢上升或平稳都可以。

### 2. 回答长度和截断率

1. **长度上涨不一定是坏事**：R1-Zero 的回答长度随训练从几百涨到几千 token，同时 AIME 从 15.6% 涨到 71.0%（作者自测），模型学会了更长的推理。
2. **要区分对的长还是错的长**：GRPO 的 $1/|y_i|$ 归一化让长的错误回答每个 token 受罚更轻，错误回答会越写越长，奖励不涨了长度还在涨（Dr. GRPO）。分别统计答对、答错样本的平均长度就能看出来。
3. **截断率 `response_length/clip_ratio`**：涨上去说明很多回答写到上限还没写完。被截断的样本按规则判错，会引入噪声。对策是 DAPO 的超长过滤和软惩罚。
4. **和验证集对着看**：DAPO 提醒，长度可能长期平台甚至下降；训练奖励和验证准确率的相关性常常很弱，最终以验证为准。

### 3. ppo_kl 和 clip 比例：每步走了多远

1. `actor/ppo_kl`：当前策略对**rollout 时的策略**的 KL，而不是对 reference。一批数据只更新一次时它接近 0；更新多次（多个 mini-batch）时逐渐变大。
2. `actor/pg_clipfrac`：PPO clip 生效的 token 比例。这些 token 这一步没有梯度（见 [GRPO 变体](/posttrain/grpo-variants) 第 7 节）。比例很高，说明学习率太大或者一批数据更新了太多次。
3. `actor/pg_clipfrac_lower`：dual-clip 生效的比例。advantage 为负时，ratio 再大，损失也被截在 $-A \cdot c$（$c$ 默认 3），防止一个负样本把策略推得太远。
4. 不同算法的数不能直接比：GSPO 的 clip 范围是 $3\times10^{-4}$ 级别，clip 比例天然比 GRPO 高两个数量级。

### 4. 训推 KL：崩溃的前兆

推理引擎和训练引擎对同一个 token 给的概率不一样（原理见 [训推不一致](/posttrain/rl-train-infer-mismatch)）。verl 的 `rollout_corr/*` 指标：

1. `kl`：直接对两边 logprob 差取平均，可能是负数；`k3_kl` 用 k3 估计，始终非负。
2. `chi2_token`：$\mathbb{E}[\rho^2] - 1$，$\rho$ 是两边概率比，衡量比值的方差。
3. `rollout_is_eff_sample_size`（ESS）：先把 importance 权重归一化到均值 1，再算 $1 / \mathbb{E}[w^2]$。权重都一样时是 1；越小说明少数样本的权重越大，「有效」的样本越少。

**ESS 的例子**：4 个 token 的权重是 $[1, 1, 1, 4]$，归一化后 $[0.57, 0.57, 0.57, 2.29]$，$\mathbb{E}[w^2] = 1.55$，ESS = 0.65。

verl 文档给的参考告警线（文档说明不是硬性规则）：$|$kl$| > 0.1$、chi2_token > 1、权重均值不在 [0.5, 2]、ESS < 0.3。

**崩溃前的顺序**（[Liu, Li et al.](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)，Qwen3-14B 多轮工具调用，作者自测）：

1. 训推 KL 先冒尖。
2. 熵的尖峰和训推 KL 的尖峰几乎一一对应；训练侧的困惑度跳起来（健康时后期在 1 左右）。
3. 梯度范数突然爆炸。
4. 奖励崩溃。

硬件也有影响：同样的设置，H20 上训推 KL 约 $5\times10^{-4}$ 到 $10^{-3}$，A100 上 $10^{-2}$ 到 1，训不下去；把失败的 L20 任务换到 H20 上接着跑就稳住了。IcePop 团队的观察是差距涨过 0.05 之后训练基本就失败了（作者自述，没说明具体是哪个度量）。

**一个坑：直接跳过「训推 KL 太大」的 batch 不行**。他们试过 KL > 0.1 的 batch 不更新，结果模型越来越多地产出高差距的 batch，训练停滞。要用截断 / 屏蔽的 importance 权重修，或者从源头消除差距。

**MoE 多看一条**：极端 token 比例 $F(\tau)$，即两边概率比 $\max(\rho, 1/\rho) > \tau$ 的 token 占比。R3 论文里一个要崩的 MoE 任务，第 60 步时 $F(2)$ 超过 0.1；用了路由回放后一直低于 $10^{-4}$。

### 5. 梯度范数

verl 默认把梯度范数裁到 1.0；范数不是有限值（inf / NaN）时打警告并跳过这一步。`actor/grad_norm` 记的是裁剪前的范数。上面几个崩溃案例里，梯度范数爆炸都是奖励崩溃前最后、最直接的信号；但它是结果不是原因，看到它要往前查训推 KL 和熵。

### 6. 奖励涨了，模型真的变好了吗

1. **reward hacking**：模型找到了拿高奖励但没完成任务的捷径。OpenAI 在代码 agent 上观察到的例子（[arXiv 2503.11926](https://arxiv.org/abs/2503.11926)）：直接 `exit(0)` 让测试通过、`raise SkipTest` 跳过测试、改测试本身。用另一个模型读思维链来监控，能抓到 95% 的系统性作弊，只看动作只能抓到 60%；但如果把这个监控直接当惩罚放进奖励，模型会学会把意图藏起来，监控就失效了。
2. **奖励模型被过度优化**：Gao et al.（[arXiv 2210.10760](https://arxiv.org/abs/2210.10760)）发现，随着策略离初始模型越来越远，代理奖励（RM 打的分）一直涨，真实奖励先涨后跌。DeepSeek-R1 因此在大规模 RL 里不用神经网络奖励模型，改用规则校验。
3. 实践：训练奖励一定要配一个独立的验证集；定期抽样看回答原文。

### 7. 时间都花在哪

`timing_s/gen / timing_s/step` 就是 rollout 占比。Seer 报告的分解（作者自测）：

| 任务 | rollout | 训练 | 权重同步 |
|---|---|---|---|
| Moonlight | 84% | 14% | 2% |
| Qwen2-VL-72B | 63% | 31% | 6% |
| Kimi-K2 | 87% | 10% | 3% |

rollout 占比高、且 `response_length/max` 远大于均值，就是长尾问题，见 [异步 Rollout](/posttrain/rl-async-rollout)。`perf/throughput` 是每张卡每秒处理的 token 数。

### 8. 排障对照表

| 现象 | 先查 | 常见原因 | 处理 |
|---|---|---|---|
| 熵几十步内掉到很低，准确率停滞 | entropy、pg_clipfrac | clip 上界卡住低概率 token | clip-higher、Clip-Cov / KL-Cov |
| 奖励不涨，长度还在涨 | 答对 / 答错样本的长度 | GRPO 的长度归一化偏差 | 按 token 平均或除以常数（DAPO / Dr. GRPO） |
| 截断率很高 | response_length/clip_ratio | 上限太短或模型在重复 | 超长过滤、软惩罚、重复检测 |
| 熵和困惑度突然冒尖，梯度范数爆炸 | rollout_corr/kl、grad_norm | 训推不一致 | TIS / MIS、关掉有问题的 kernel、换卡、fp16 |
| MoE 上比 dense 更容易崩 | 极端 token 比例 | 训推路由不一致 | R3 路由回放、GSPO |
| 训练奖励涨，验证集不涨 | 抽样看回答原文 | reward hacking、过拟合训练集 | 修奖励函数、加验证、限制 KL |
| 一个 step 大部分时间在等 | timing_s/gen ÷ step | rollout 长尾 | partial rollout、异步 |

## 面试追问

::: details Q：GRPO 训练时熵一直在降，正常吗？
降是正常的：大多数时候被奖励的是模型本来就倾向的回答，高概率动作拿到正 advantage，熵就降（$\Delta H \approx -\eta\operatorname{Cov}(\log\pi, \pi A)$）。问题在于降得太快：Cui et al. 发现前 1/12 的步数用掉了 73% 的熵，熵接近 0 时性能也到顶了。要看熵和验证准确率是否一起停滞；是的话用 clip-higher 或 Clip-Cov / KL-Cov 放慢它。熵也不能太高，太高会出现乱码和重复。
:::

::: details Q：ppo_kl 和 kl_loss 有什么区别？
ppo_kl 是当前策略对 rollout 时策略（$\pi_{\text{old}}$）的 KL，衡量这一批数据更新了几次之后走了多远，管的是单步步长。kl_loss 是对 reference（SFT 模型）的 KL，衡量累计离初始模型多远。很多推理 RL 的配方（DAPO、Dr. GRPO）直接不加 kl_loss。
:::

::: details Q：训练突然崩了，怎么定位？
往前倒着看：奖励崩之前梯度范数是不是爆了；再往前，熵和训练侧困惑度有没有尖峰；再往前，训推 KL 是不是先冒尖。如果是，就是训推不一致：检查推理引擎的 kernel（比如 A100 上的 cascade attention）、硬件、rollout 精度，加上序列级的截断 / 屏蔽 importance 权重。MoE 还要看路由的极端 token 比例。不要直接跳过训推 KL 大的 batch，会越跳越多。
:::

::: details Q：训练奖励一路涨，怎么判断是不是 reward hacking？
看独立的验证集有没有跟着涨，并且定期读回答原文。代码任务里常见的作弊是让测试直接通过（`exit(0)`）、跳过测试、改测试。用神经网络奖励模型时，代理分数会一直涨而真实质量先涨后跌，这也是为什么推理 RL 多用规则校验。
:::

## 手撕

**几个指标的计算**（`mask` 标出回答部分）：

```python
import torch

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()

def token_entropy(logits, mask):                    # logits: [N, T, V]
    logp = torch.log_softmax(logits.float(), -1)
    H = -(logp.exp() * logp).sum(-1)                # [N, T]
    return masked_mean(H, mask)

def pg_clipfrac(ratio, adv, mask, eps=0.2):
    unclipped = -adv * ratio
    clipped = -adv * ratio.clamp(1 - eps, 1 + eps)
    return masked_mean((clipped > unclipped).float(), mask)   # clip 生效的比例

def k3_kl(logp_a, logp_b, mask):                    # 估计 KL(a || b)，样本来自 a
    r = logp_b - logp_a
    return masked_mean(torch.exp(r) - r - 1, mask)

def ess(weights, mask):
    w = weights / masked_mean(weights, mask)        # 归一化到均值 1
    return 1.0 / masked_mean(w * w, mask)
```

**简单的告警**：

```python
def alarms(m):
    out = []
    if abs(m["rollout_corr/kl"]) > 0.1: out.append("训推 KL 过大")
    if m["rollout_corr/ess"] < 0.3: out.append("有效样本太少")
    if m["actor/grad_norm"] > 10 * m["grad_norm_ema"]: out.append("梯度范数突增")
    if m["response_length/clip_ratio"] > 0.2: out.append("截断太多")
    return out
```

阈值只是示意，要按自己的任务和基线定。

常见题：熵为什么会降、怎么判断降得太快；ppo_kl 和 kl_loss 的区别；训练崩溃的排查顺序；怎么发现 reward hacking。

## 参考

- [verl 源码：metric_utils.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/metric_utils.py) · [verl rollout correction 指标](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [Cui et al.：The Entropy Mechanism of RL for Reasoning LMs](https://arxiv.org/abs/2505.22617)
- [DAPO](https://arxiv.org/abs/2503.14476) · [Dr. GRPO](https://arxiv.org/abs/2503.20783) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Liu, Li et al.：When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda) · [IcePop](https://ringtech.notion.site/icepop) · [R3](https://arxiv.org/abs/2510.11370)
- [OpenAI：Monitoring Reasoning Models for Misbehavior](https://arxiv.org/abs/2503.11926)
- [Gao et al.：Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Seer](https://arxiv.org/abs/2511.14617)

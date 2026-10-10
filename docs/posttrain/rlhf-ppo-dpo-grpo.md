---
title: RLHF 全家桶：PPO / DPO / GRPO
status: draft
tags: [rlhf, ppo, dpo, grpo]
difficulty: 4
order: 3
related: [/posttrain/rl-infra, /posttrain/sft, /posttrain/training-memory]
stack: []
---

# RLHF 全家桶：PPO / DPO / GRPO

> PPO 四模型流程、KL 约束、DPO / GRPO 推导动机与区别

## 一句话结论

三者优化的是**同一个目标**：让回答的奖励尽量高，同时别离 SFT 模型太远（KL 约束）。区别在于怎么做：

- **PPO-RLHF**（[InstructGPT](https://arxiv.org/abs/2203.02155)）：在线采样 + reward model 打分 + critic 估基线 + reference 算 KL，**四个模型**，最完整也最重。
- **DPO**（[Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)）：利用 KL 约束目标的闭式最优解，把 reward 换成「策略和参考模型的对数概率比」，直接在**离线偏好对**上做二分类。只要 policy + reference 两个模型，不采样，但吃不到自己新生成的数据。
- **GRPO**（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）：保留在线采样和 PPO 的 clip，**去掉 critic**，同一 prompt 采 $G$ 条回答，用组内 reward 的均值 / 标准差当基线；KL 直接加在 loss 里。适合答案可校验的推理任务。

系统怎么跑（rollout / training 分离、权重同步）见 [RL Infra](/posttrain/rl-infra)，本页只讲算法。

## 推导

### 0. 把 LLM 生成看成 RL

先定义符号，后面都用这一套：

| 符号 | 含义 |
|---|---|
| $x$ | prompt |
| $y = (y_1, \dots, y_T)$ | 模型生成的回答，$T$ 个 token |
| $s_t = (x, y_{<t})$ | **状态**：prompt + 已生成的前缀 |
| $a_t = y_t$ | **动作**：第 $t$ 步选的 token，动作空间 = 词表 |
| $\pi_\theta(a_t \mid s_t)$ | **策略**：正在训练的 LLM 给出的下一个 token 概率 |
| $\pi_{\text{ref}}$ | **参考策略**：冻结的 SFT 模型 |
| $r(x, y)$ | **奖励**：整条回答生成完才给一个标量（reward model 打分或规则判对错） |

几个特点：状态转移是确定的（把 token 拼到后面就行），奖励**稀疏、只在最后一步**，一条回答就是一个 episode。

RLHF 的优化目标（[Ziegler et al. 2019](https://arxiv.org/abs/1909.08593)、[InstructGPT](https://arxiv.org/abs/2203.02155)）：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\big[r(x,y)\big] \;-\; \beta\, \mathbb{D}_{\text{KL}}\big[\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big]
$$

$\beta > 0$ 控制「刷分」和「别跑偏」的权衡。**PPO、DPO、GRPO 都是在解这个式子**（GRPO 的 KL 写法略有不同）。

### 1. 策略梯度：往高奖励的方向推概率

直接对上式第一项求梯度（REINFORCE / 策略梯度定理）：

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot R\Big]
$$

直觉：奖励 $R$ 高的回答，把它每个 token 的 log 概率往上推；奖励低的往下压。这就是「带权重的 SFT」，权重是奖励。

问题是**方差大**：$R$ 全是正数时所有回答都被推高，只是推得多少不同。解决办法是减一个**基线** $b(s_t)$（不改变期望，只降方差），得到**优势**（advantage）：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

- $V(s_t)$：从状态 $s_t$ 出发，按当前策略走下去的**期望**回报（「平均能拿几分」）。
- $Q(s_t, a_t)$：先选了 token $a_t$ 再按策略走的期望回报。
- $A_t > 0$：这个 token 比平均好，推高；$A_t < 0$：压低。

梯度变成 $\mathbb{E}\big[\sum_t \nabla_\theta \log\pi_\theta(a_t|s_t)\, A_t\big]$。**PPO 用一个神经网络（critic）学 $V$；GRPO 用组内平均 reward 代替 $V$**，这是两者最核心的区别。

### 2. GAE：怎么从 critic 算出 $A_t$

有了 critic $V_\phi$，定义每步的 TD 残差：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$r_t$ 是第 $t$ 步的即时奖励（LLM 里通常只有最后一步非零，加上每 token 的 KL 惩罚，见第 4 节），$\gamma$ 是折扣因子。GAE（[Schulman et al. 2015](https://arxiv.org/abs/1506.02438)）把多步残差指数加权：

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l\, \delta_{t+l}
$$

- $\lambda = 0$：$\hat A_t = \delta_t$，只信 critic，方差小但偏差大（critic 不准就全错）。
- $\lambda = 1$：退化成「真实回报 − $V$」，无偏但方差大。
- LLM RLHF 里常取 $\gamma = 1$，$\lambda$ 接近 1（如 0.95）。

critic 的训练目标是回归回报：$\mathcal{L}_V = \big(V_\phi(s_t) - \hat{R}_t\big)^2$，$\hat R_t = \hat A_t + V_{\phi_{\text{old}}}(s_t)$。

### 3. PPO：一批样本多更新几次，但别走太远

采样很贵，希望一批 rollout 能做多步梯度更新。可是更新几步后，策略已经不是采样时的那个了，要用**重要性采样比**纠正：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$\pi_{\theta_{\text{old}}}$ 是**生成这批样本时**的策略（参数冻结的快照）。如果 $\rho_t$ 偏离 1 太多，估计就不可信。PPO（[Schulman et al. 2017](https://arxiv.org/abs/1707.06347)）用 clip 把它限制在 $[1-\epsilon, 1+\epsilon]$：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t\, \hat A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, \hat A_t\big)\Big]
$$

读法（$\epsilon$ 常取 0.2）：

- $\hat A_t > 0$（好 token）：$\rho_t$ 涨过 $1+\epsilon$ 后梯度为 0，**不让一次推太猛**。
- $\hat A_t < 0$（坏 token）：$\rho_t$ 跌破 $1-\epsilon$ 后梯度为 0，**不让一次压太狠**。
- 取 $\min$ 保证是悲观下界：只截断「往有利方向走过头」的部分，不截断变坏的部分。

### 4. PPO-RLHF 的四个模型

| 模型 | 初始化自 | 训练？ | 输入 → 输出 | 作用 |
|---|---|---|---|---|
| **Actor**（policy）$\pi_\theta$ | SFT 模型 | 是 | 生成回答、给出 token logprob | 被优化的对象 |
| **Critic**（value）$V_\phi$ | 通常是 RM（或 SFT）+ 标量头 | 是 | 每个 token 位置 → 标量 $V(s_t)$ | 估基线，算 GAE |
| **Reward model** $r_\psi$ | SFT + 标量头，在偏好对上训好 | 否，冻结 | 整条 $(x,y)$ → 一个分数 | 给最终奖励 |
| **Reference** $\pi_{\text{ref}}$ | SFT 模型 | 否，冻结 | token logprob | 算 KL，防跑偏 |

Reward model 事先用偏好数据训练（[InstructGPT](https://arxiv.org/abs/2203.02155)）：对同一 prompt 的好回答 $y_w$、差回答 $y_l$，最小化 $-\log\sigma\big(r_\psi(x,y_w) - r_\psi(x,y_l)\big)$，$\sigma$ 是 sigmoid（这就是第 6 节的 Bradley-Terry 模型）。

**一个 PPO step 的流程**：

1. Actor 对一批 prompt 生成回答（rollout）。
2. Reward model 给每条回答打分 $r_\psi(x,y)$。
3. Reference 和 actor（此时即 $\pi_{\theta_{\text{old}}}$）分别算每个 token 的 logprob → 每 token KL。
4. Critic 算每个位置的 $V(s_t)$ → GAE 得 $\hat A_t$。
5. 用 $\mathcal{L}^{\text{CLIP}}$ 更新 actor、用 $\mathcal{L}_V$ 更新 critic，同一批数据跑若干 epoch。

**显存账**：四份模型参数，其中两份（actor、critic）还要梯度 + Adam 状态。以 7B、混合精度 Adam 为例（每参数 16 B，见 [训练显存账](/posttrain/training-memory)）：

| | 每参数字节 | 7B 合计 |
|---|---|---|
| Actor（训练） | 16 | 112 GB |
| Critic（训练，同尺寸时） | 16 | 112 GB |
| Reward（推理，bf16） | 2 | 14 GB |
| Reference（推理，bf16） | 2 | 14 GB |
| **合计（不含激活、KV cache）** | | **约 252 GB** |

所以 PPO-RLHF 天然多卡、多引擎，系统怎么摆放见 [RL Infra](/posttrain/rl-infra)。去掉 critic 直接省掉近一半，这就是 GRPO 的动机之一。

### 5. KL 惩罚：为什么要、加在哪

**为什么**：reward model 只在 SFT 分布附近训过，是个不完美的代理。策略一旦跑到 RM 没见过的区域，就可能找到「RM 打高分但人类觉得很烂」的回答（**reward hacking**，例如堆砌某些讨好词、输出变长）。KL 把策略拴在 $\pi_{\text{ref}}$ 附近；同时也防止语言能力退化、熵塌缩。$\beta$ 太大学不动，太小会 hack。

**两种加法**：

**(a) KL 放进 reward（PPO-RLHF 的做法）**：把每个 token 的 log 比作为即时惩罚，和最终奖励一起交给 GAE：

$$
r_t = -\beta \log\frac{\pi_{\theta_{\text{old}}}(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} \;+\; \mathbb{1}[t = T]\cdot r_\psi(x,y)
$$

这里 KL 是「环境给的奖励」的一部分，经过 critic / advantage 间接影响梯度，各 token 的 KL 惩罚也会沿时间往前传。InstructGPT 就是这样做的。

**(b) KL 放进 loss（GRPO 的做法）**：reward 保持干净，在目标函数里直接加一项 $\beta\, \mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$，梯度直接作用于当前策略。GRPO 用的是 [Schulman 的 k3 估计](http://joschu.net/blog/kl-approx.html)：

$$
\hat{\mathbb{D}}_{\text{KL}} = \frac{\pi_{\text{ref}}(a_t|s_t)}{\pi_\theta(a_t|s_t)} - \log\frac{\pi_{\text{ref}}(a_t|s_t)}{\pi_\theta(a_t|s_t)} - 1
$$

它对每个样本都非负，且在 $a_t \sim \pi_\theta$ 时对真实 KL 无偏（比直接用 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 方差小，后者单样本可能为负）。

| | KL in reward | KL in loss |
|---|---|---|
| 代表 | PPO-RLHF（InstructGPT） | GRPO |
| 进入方式 | 当成每 token 奖励，经 advantage 起作用 | 直接作为 loss 正则项求导 |
| 和 advantage 关系 | 混在一起，被组归一化 / critic 处理 | 解耦，不污染 advantage |
| 适合 | 有 critic 的方法 | 无 critic、advantage 是整句标量的方法 |

GRPO 不能简单把 KL 放进 reward，因为它的 advantage 是整句 reward 的组归一化，逐 token 的 KL 没法自然地塞进去。

### 6. DPO：不采样、不要 RM 也能解同一个目标

**第 1 步：KL 约束目标有闭式最优解。** 对固定的 $x$，第 0 节的目标对所有分布 $\pi$ 求最大值，解是（[DPO 论文](https://arxiv.org/abs/2305.18290) 附录 A.1）：

$$
\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x)\, \exp\!\Big(\frac{1}{\beta} r(x,y)\Big),\qquad Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\!\Big(\frac{1}{\beta} r(x,y)\Big)
$$

直觉：在参考分布上按 $e^{r/\beta}$ 重新加权。$Z(x)$ 要对所有可能回答求和，算不了，所以不能直接用。

**第 2 步：反过来，用策略表示 reward。** 两边取对数、移项：

$$
r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

即：**任何 reward 都对应一个最优策略，reward 可以写成「最优策略和参考策略的 log 比」加一个只和 $x$ 有关的常数**。

**第 3 步：Bradley-Terry 只看 reward 差，$Z(x)$ 被消掉。** 人类偏好模型假设 $y_w$（被选中）优于 $y_l$（被拒绝）的概率是

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)
$$

把第 2 步代入，两个回答共享同一个 $x$，$\beta\log Z(x)$ 相减抵消：

$$
P(y_w \succ y_l \mid x) = \sigma\Big(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)
$$

**第 4 步：把 $\pi^*$ 换成要训练的 $\pi_\theta$，对偏好数据做最大似然**，得到 DPO loss：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big]
$$

其中 $\log\pi(y|x) = \sum_t \log\pi(y_t|x,y_{<t})$，整条回答的 token logprob 求和。$\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 被称为**隐式 reward**——论文标题「你的语言模型其实是个 reward model」就是这个意思。

**梯度的直觉**：$\nabla\mathcal{L} \propto -\sigma(\hat r_l - \hat r_w)\,[\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l)]$。推高 $y_w$、压低 $y_l$，权重是「当前隐式 reward 把这对**排错**的程度」，排对了的样本贡献小。

**DPO 去掉了什么**：

- 去掉 Reward model：reward 隐含在策略里。
- 去掉 Critic：不需要逐 token 的价值估计。
- 去掉 Rollout：不生成，只在固定数据上算 logprob。
- 剩下 policy（训练）+ reference（冻结，且 $\pi_{\text{ref}}$ 的 logprob 可以离线预先算好存起来）。训练形态和 SFT 几乎一样。

**DPO 的代价**：

- **离线 / off-policy**：偏好对通常由别的模型（或旧版 SFT）生成，训练过程中策略变了，数据不跟着变。策略学会的是「在这批数据上把 $y_w$、$y_l$ 拉开」，对它自己会生成的回答未必有效。
- **分布偏移**：对数据里没覆盖的回答，隐式 reward 没约束，可能给出离谱的概率。常见现象是 $y_w$ 和 $y_l$ 的 logprob **一起下降**，只是 $y_l$ 降得更多，概率质量流向数据外的回答。
- **吃不到在线探索的收益**：推理类任务需要模型自己试错、从对错信号学，DPO 做不到。缓解办法是迭代 / online DPO：每轮用当前策略重新采样、打标、再训，但这就又需要 rollout 和打分器了。

### 7. GRPO：保留在线采样，砍掉 critic

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出，[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 用它做大规模推理 RL。

**组相对 advantage**：对每个 prompt $x$，用 $\pi_{\theta_{\text{old}}}$ 采 $G$ 条回答 $\{y_1,\dots,y_G\}$，得到奖励 $\{r_1,\dots,r_G\}$（可以是 RM 分，也可以是规则：答案对 1 错 0）。advantage 用组内统计量算：

$$
\hat A_{i,t} = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

- 第 $i$ 条回答的**所有 token** 共享同一个 $\hat A_i$（结果监督；论文也给了过程监督版本）。
- 组均值扮演了 $V(s_0)$ 的角色：同一题采多条，「比平均做得好」的推高，比平均差的压低。
- 不需要 critic，省掉一个要训练的大模型。

**目标函数**（PPO clip + KL in loss）：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\Big(\min\big(\rho_{i,t}\hat A_{i,t},\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big) - \beta\,\hat{\mathbb{D}}_{\text{KL}}\big[\pi_\theta\|\pi_{\text{ref}}\big]\Big)\Bigg]
$$

$\rho_{i,t} = \pi_\theta(y_{i,t}|x,y_{i,<t}) / \pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})$，$|y_i|$ 是第 $i$ 条回答的长度，$\hat{\mathbb{D}}_{\text{KL}}$ 用第 5 节的 k3 估计。

**模型数**：actor（训练）+ reference（冻结）+ 打分器（RM，或者规则 / 沙箱，不占显存）。7B 下约 112 + 14 (+14) GB，比 PPO 少一半左右。

**代价**：每个 prompt 要采 $G$ 条（常见 8–64），rollout 量成倍增加；如果一组全对或全错，$\operatorname{std}=0$，advantage 全为 0，这组样本**没有梯度**，白采了。

### 8. 后续变体（简述）

完整的公式、数字例子，以及 GSPO、CISPO 见 [GRPO 变体](/posttrain/grpo-variants)。

- **DAPO**（[Yu et al. 2025](https://arxiv.org/abs/2503.14476)）：基于 GRPO 的四个改动——clip 上下界解耦（$\epsilon_{\text{high}} > \epsilon_{\text{low}}$，缓解熵塌缩）、动态采样（过滤全对 / 全错的组）、token 级 loss 平均（长回答不被稀释）、超长回答的奖励整形；并去掉了 KL 项。作者自测 Qwen2.5-32B 上 AIME 2024 达到 50 分。
- **Dr. GRPO**（[Liu et al. 2025](https://arxiv.org/abs/2503.20783)）：指出 GRPO 的 $1/|y_i|$ 长度归一化和除以 std 会引入优化偏差——答错时长回答每 token 受罚更轻，导致错误回答越写越长。去掉这两项归一化即可，作者自测在保持准确率的同时提升 token 效率。

### 9. 对比总表

| | PPO-RLHF | DPO | GRPO |
|---|---|---|---|
| 需要的模型 | actor + critic + RM + ref（4 个） | policy + ref（2 个） | actor + ref（+ RM，可用规则代替） |
| 训练的模型 | actor、critic | policy | actor |
| 在线 / 离线 | 在线（on-policy） | 离线（off-policy） | 在线（on-policy） |
| 需要 rollout | 是 | 否 | 是，且每 prompt $G$ 条 |
| 基线 / advantage | critic + GAE，逐 token | 无（分类 loss） | 组内归一化，整句共享 |
| KL 位置 | 每 token 放进 reward | 隐含在闭式解里（$\beta$） | 放进 loss（k3 估计） |
| 显存（7B 粗估） | ≈ 252 GB + KV | ≈ 112 + 14 GB | ≈ 126（+14）GB + KV |
| 典型用途 | 通用对齐、开放式对话 | 偏好对齐，便宜好调 | 可验证奖励的推理（数学、代码） |

## 面试追问

::: details Q：PPO 里 $\pi_{\theta_{\text{old}}}$ 和 $\pi_{\text{ref}}$ 有什么区别？
$\pi_{\theta_{\text{old}}}$ 是**采样这批数据时**的策略快照，每轮 rollout 都会更新，用在重要性采样比 $\rho_t$ 里，配合 clip 限制「单轮」更新幅度。$\pi_{\text{ref}}$ 是 SFT 模型，整个训练过程冻结不变，用在 KL 里，限制「累计」偏离程度。一个管步长，一个管总距离。
:::

::: details Q：GRPO 去掉 critic 后，方差会不会变大？
会，组均值是比学出来的 $V(s_t)$ 更粗的基线，而且整句共享一个 advantage，做不了逐 token 的信用分配。GRPO 靠较大的组（$G$ = 8–64）和组内标准化压方差；在可验证 reward（答案对错）的任务上 reward 本身噪声小，实践上够用。反过来，critic 本身也要训练、可能不准，长 CoT 下 critic 很难学好，这也是 GRPO 在推理任务上流行的原因。
:::

::: details Q：DPO 既然和 RLHF 目标等价，为什么效果常不如 PPO / GRPO？
等价只在「数据覆盖整个回答空间、能找到全局最优」时成立。实际 DPO 只在有限的离线偏好对上训练，对策略自己会生成的回答没有约束（分布偏移），也无法通过在线采样探索。在线方法每轮用当前策略的样本学习，数据和策略始终匹配。所以 DPO 适合对齐风格 / 偏好，推理能力提升主要靠在线 RL。
:::

::: details Q：DPO 里 $\beta$ 的作用？
和 RLHF 目标里的 $\beta$ 是同一个：KL 约束强度。$\beta$ 大，隐式 reward 对 log 比更敏感，策略稍微偏离 ref 就足以拉开偏好，相当于更强地拴在 ref 附近；$\beta$ 小，允许偏离更远。常见取值 0.1 左右。
:::

::: details Q：PPO 为什么需要 reference，有了 clip 还不够吗？
clip 只约束相邻两轮之间的变化，多轮累积下来策略可以离 SFT 模型很远，照样会 reward hacking。KL 约束的是相对固定锚点的总距离。另外 KL 项也是 DPO 推导的前提：没有它就没有闭式解 $\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$。
:::

::: details Q：GRPO 一组全对或全错怎么办？
$\operatorname{std}=0$，advantage 全为 0，这组没有学习信号（实现上分母会加小 $\varepsilon$ 防除零）。题太简单或太难都会这样。DAPO 的动态采样就是把这类组过滤掉、补采其他 prompt，保证每个 batch 的有效样本数（[DAPO](https://arxiv.org/abs/2503.14476)）。
:::

## 手撕

**DPO loss**（输入是整条回答的 token logprob 之和；`ref_*` 可以离线预计算）：

```python
import torch.nn.functional as F

def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    # 每个参数形状 [B]：sum_t log pi(y_t | x, y_<t)，只对回答部分的 token 求和
    r_w = beta * (pi_logp_w - ref_logp_w)    # 隐式 reward of chosen
    r_l = beta * (pi_logp_l - ref_logp_l)    # 隐式 reward of rejected
    loss = -F.logsigmoid(r_w - r_l).mean()
    return loss
```

**一个 GRPO step**（伪代码，结果监督，KL in loss）：

```python
def grpo_step(policy, ref, reward_fn, prompts, G=8, eps=0.2, beta=0.04, inner_epochs=1):
    # 1. rollout：当前策略即 pi_old，每个 prompt 采 G 条
    with torch.no_grad():
        ys = [policy.generate(x, n=G) for x in prompts]            # B x G 条回答
        r = torch.tensor([[reward_fn(x, y) for y in g]             # [B, G]
                          for x, g in zip(prompts, ys)])
        # 2. 组内归一化 advantage，整句共享
        A = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-6)
        # 3. 旧策略、参考模型的 token logprob（训练引擎重算）
        old_logp = policy.token_logprobs(prompts, ys)               # [B, G, T]
        ref_logp = ref.token_logprobs(prompts, ys)                  # [B, G, T]
        mask = response_mask(ys)                                    # [B, G, T]，padding 为 0

    for _ in range(inner_epochs):
        logp  = policy.token_logprobs(prompts, ys)                  # 带梯度
        ratio = torch.exp(logp - old_logp)
        adv   = A.unsqueeze(-1)                                     # 广播到每个 token
        pg    = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
        # k3 KL 估计：ref/pi - log(ref/pi) - 1
        d     = ref_logp - logp
        kl    = torch.exp(d) - d - 1
        per_tok = -(pg - beta * kl)
        # 原版：先对每条回答按长度平均，再对 G、B 平均
        loss = ((per_tok * mask).sum(-1) / mask.sum(-1)).mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

`inner_epochs=1` 时 `ratio` 恒等于 1（但梯度不为 0），clip 不起作用；多个 mini-batch / epoch 时 clip 才生效。Dr. GRPO 去掉的就是 `A` 里的除以 std 和 loss 里的 `/ mask.sum(-1)`。

常见题：写出 DPO loss 并解释每一项；写 GRPO 的 advantage 和 loss；给定模型大小算 PPO 四模型显存。系统实现（rollout 引擎、权重同步）见 [RL Infra](/posttrain/rl-infra)。

## 参考

- [Ziegler et al. 2019：Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- [InstructGPT（Ouyang et al. 2022）：PPO-RLHF](https://arxiv.org/abs/2203.02155)
- [PPO（Schulman et al. 2017）](https://arxiv.org/abs/1707.06347)
- [GAE（Schulman et al. 2015）](https://arxiv.org/abs/1506.02438)
- [Schulman：Approximating KL Divergence（k1/k2/k3）](http://joschu.net/blog/kl-approx.html)
- [DPO（Rafailov et al. 2023）](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DAPO（Yu et al. 2025）](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO：Understanding R1-Zero-Like Training（Liu et al. 2025）](https://arxiv.org/abs/2503.20783)

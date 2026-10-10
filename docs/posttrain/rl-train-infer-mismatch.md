---
title: 训推不一致：同一份权重，两个引擎算出不同的概率
status: draft
tags: [rl-infra, tis, mis, icepop, routing-replay, batch-invariance]
difficulty: 4
order: 4.2
related: [/posttrain/rl-infra, /posttrain/grpo-variants, /posttrain/rlhf-ppo-dpo-grpo, /posttrain/training-memory]
stack: []
---

# 训推不一致：同一份权重，两个引擎算出不同的概率

> rollout 用 vLLM / SGLang 采样，梯度用 FSDP / Megatron 算。权重完全相同，两边给同一个 token 的概率却不一样：差多大、从哪来、怎么修正、怎么从源头消掉

## 一句话结论

RL 框架里采样的是推理引擎，算梯度的是训练引擎。两边 kernel、并行方式、精度不同，同一个 token 的概率可以差到 1（推理侧给 1，训练侧给 0）。于是「on-policy」其实是 off-policy，长回答、低概率 token、MoE 路由、低精度 rollout 都会放大这个差距，严重时训练崩掉。修法分两类：**算法上修正**，给训练侧 / 推理侧的概率比乘一个截断或屏蔽的权重（TIS、MIS、IcePop）；**从源头消除**，让两边算出一样的数（batch-invariant kernel、改用 fp16、MoE 路由回放 R3）。

## 推导

### 1. 三个策略，不是两个

PPO / GRPO 的式子里只有 $\pi_\theta$ 和 $\pi_{\text{old}}$。实际系统里有三个：

| 记号 | 谁算的 | 用在哪 |
|---|---|---|
| $\pi_{\text{sampler}}(\theta_{\text{old}})$ | 推理引擎（vLLM / SGLang），旧权重 | 真正采出 token 的分布 |
| $\pi_{\text{learner}}(\theta_{\text{old}})$ | 训练引擎（FSDP / Megatron），旧权重 | PPO ratio 的分母，要重算一遍 |
| $\pi_{\text{learner}}(\theta)$ | 训练引擎，当前权重 | PPO ratio 的分子，带梯度 |

$\theta_{\text{old}}$ 相同时，前两个**按理应该相等**，但实际不等。Yao et al. 把实际在做的更新写成（以 REINFORCE 为例，$a$ 是采到的 token，$R$ 是奖励，$\mu$ 是学习率）：

$$
\theta \leftarrow \theta + \mu\,\mathbb{E}_{a \sim \pi_{\text{sampler}}(\theta)}\big[R(a)\,\nabla_\theta \log \pi_{\text{learner}}(a, \theta)\big]
$$

样本来自 sampler，梯度按 learner 算。两者不同，这就是 off-policy 更新，只是没人给它乘 importance 权重（[Yao et al. 2025](https://fengyao.notion.site/off-policy-rl)）。

为什么不直接用推理引擎返回的 logprob 当分母：那样 ratio 是 $\pi_{\text{learner}}(\theta) / \pi_{\text{sampler}}(\theta_{\text{old}})$，第一步更新前它就不等于 1，PPO 的 clip 会把这些「本来就差很多」的 token 当成「已经更新过头」处理。所以标准做法是在训练引擎里重算 $\pi_{\text{learner}}(\theta_{\text{old}})$，代价是多一次前向，约多 25% 的训练计算（[FP16 论文](https://arxiv.org/abs/2510.26788) 的估计）。

### 2. 差多大

几个常用的度量（$p$ 是某个采到的 token 的概率）：

$$
\text{max mismatch} = \max_{a \in \text{回答}} \big|p_{\text{sampler}}(a) - p_{\text{learner}}(a)\big|, \qquad
\text{mean mismatch} = \frac{1}{|\text{回答}|}\sum_{a}\big|p_{\text{sampler}}(a) - p_{\text{learner}}(a)\big|
$$

以及两边分布之间的 KL（用 k3 估计，verl 里叫 `vllm-kl` / `rollout_corr/kl`）。公开报告的数字（都是作者自测）：

1. Qwen2.5-32B 跑 DAPO，max mismatch 到 **1.0**：同一个 token 推理侧概率 1、训练侧概率 0（Yao et al.）。
2. Qwen2.5-0.5B，bf16 rollout 的 max mismatch 约 0.4，INT8 rollout 到 1.0，熵塌到 0.2 以下（Yao et al.）。
3. vllm-kl 和硬件有关：H20 约 $5\times10^{-4}$ 到 $10^{-3}$，L20 约 $10^{-3}$ 到 $10^{-2}$，A100 上 $10^{-2}$ 到 1；A100 关掉 cascade attention 后降到约 $10^{-3}$（[Liu, Li et al.](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)）。
4. MoE 比 dense 大：Qwen3-30B-A3B 的 KL 是 $1.5\times10^{-3}$，dense 的 Qwen3-8B 是 $6.4\times10^{-4}$（[R3](https://arxiv.org/abs/2510.11370)）。

**为什么长回答特别怕**：设每个 token 的 log 概率比 $\delta_t = \log\pi_{\text{learner}} - \log\pi_{\text{sampler}}$ 很小，标准差 0.01、彼此独立（示意）。整条回答的 log 比是 $\sum_t \delta_t$，标准差是 $0.01\sqrt{T}$：

| 回答长度 $T$ | $\sum_t\delta_t$ 的标准差 | 序列概率比的 ±1σ 范围 |
|---|---|---|
| 1000 | 0.32 | 0.73 到 1.37 |
| 20000 | 1.41 | 0.24 到 4.1 |

每个 token 的差都很小，乘上两万个 token，整条回答在两边的概率就能差 4 倍。Yao et al. 也观察到：20K token 的回答比 4K 的 max mismatch 更大，mean mismatch 不变。

**为什么低概率 token 特别怕**：同样的绝对误差，落在概率 0.9 的 token 上，比值几乎不变；落在概率 0.001 的 token 上，比值可以差好几倍。Liu, Li et al. 发现训练侧给低概率 token 的概率往往比推理侧低得多，多轮工具调用里更严重。

### 3. 差异从哪来

**(a) 浮点加法不满足结合律，而归约顺序跟着 batch 变。** bf16 只有 8 位有效位，[256, 512) 之间相邻两个数差 2。同样五个数 $[256, 1, 1, 1, 1]$：

1. 先放 256，再逐个加 1：每次 $256 + 1$ 都舍入回 256，结果 **256**。
2. 先把四个 1 加起来得 4，再加 256：结果 **260**。

单次前向本身是确定的（前向基本不用 atomic add），问题在于**batch 大小会改变 kernel 的归约顺序**：matmul 会按形状选不同的 split-K 策略，attention 会按 KV 长度切块。服务器负载决定 batch 大小，而负载是随机的，于是同一个请求在不同时刻算出不同的数（[Thinking Machines, He 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）。他们的实验：Qwen3-235B，温度 0，同一个 prompt 跑 1000 次，得到 **80 种**不同的输出，第 103 个 token 开始分叉。训练引擎的 batch 组织和推理引擎完全不同，归约顺序自然不同。

需要做成 batch-invariant 的只有带归约的三类 kernel：**RMSNorm、matmul、attention**，难度依次增加。

**(b) 并行方式不同。** 推理 TP=2、训练 SP=8，切分不同，归约顺序就不同。Yao et al. 发现并行方式主要改变 max mismatch。

**(c) 精度。** bf16 尾数 7 位，1 附近的相邻间隔是 $2^{-7}$；fp16 尾数 10 位，间隔 $2^{-10}$，精度高 8 倍。rollout 再量化到 fp8 / int8，差距更大。

**(d) MoE 路由。** router 选 top-K 专家是个不连续的操作：两边 router logit 差一点点，排在第 K 和第 K+1 的专家就可能换位，这个 token 走的是完全不同的专家。R3 论文统计（Qwen3-30B-A3B，SGLang 对 Megatron）：约 10% 的 router 选了不同专家，**94% 的 token** 至少有一层选得不一样。

**(e) 实现上的坑。** vLLM V1 一度返回的不是实际用来采样的概率（[PR #22387](https://github.com/vllm-project/vllm/pull/22387) 修复）；lm_head 的精度和 HF 实现不同；A100 上的 cascade attention。这些修了能缓解，但修完差距仍在。

### 4. 算法上修正：给概率比乘一个权重

记 token 级的训推比

$$
\rho_t = \frac{\pi_{\text{learner}}(y_t \mid \cdot\,;\theta_{\text{old}})}{\pi_{\text{sampler}}(y_t \mid \cdot\,;\theta_{\text{old}})}
$$

它和 PPO 的 ratio $r_t = \pi_{\text{learner}}(\theta) / \pi_{\text{learner}}(\theta_{\text{old}})$ 是两回事：$r_t$ 管「这一步更新走多远」，$\rho_t$ 管「样本其实来自另一个分布」。修正方法都是在 PPO 的目标外面乘一个由 $\rho$ 决定的权重 $w$：

$$
\mathcal{J} = \mathbb{E}_{y \sim \pi_{\text{sampler}}}\Big[\,w \cdot \min\big(r_t \hat A_t,\ \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]
$$

几种 $w$：

| 方法 | 粒度 | $w$ | 默认阈值 |
|---|---|---|---|
| TIS（[Yao et al.](https://fengyao.notion.site/off-policy-rl)） | token | $\min(\rho_t, C)$，截断 | $C = 2$ |
| MIS（[Liu, Li et al.](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)） | 整条回答 | $\rho \cdot \mathbb{1}\{\rho \le C\}$，超过就整条丢掉 | $C = 2$ |
| Geo-RS（同上） | 整条回答 | 用几何平均 $\rho^{1/T}$ 判断是否丢掉 | |
| IcePop（[Ring-1T](https://arxiv.org/abs/2510.18855)） | token | $\rho_t$ 落在 $[\alpha, \beta]$ 内保留，否则置 0 | $[0.5, 5]$ |

整条回答的比是 token 比的连乘：$\rho = \prod_t \rho_t$。

**为什么要截断**：importance 权重太大，梯度方差会被放大权重的平方倍。一个 $\rho = 16$ 的 token：不截断，噪声放大 $16^2 = 256$ 倍；TIS 截到 2，放大 $2^2 = 4$ 倍；截到 8 是 64 倍（Yao et al. 的例子）。截断引入偏差，换来方差有界。

**截断还是屏蔽、token 还是整条**：

1. 截断（TIS）保留样本、压小权重；屏蔽（MIS、IcePop）直接丢掉差得离谱的样本或 token。差得离谱往往说明这一段在两边根本不是同一个分布，权重给多少都不对，丢掉更干净。
2. 整条回答的 IS 是无偏的，token 级的不是：token 级只修正了「这个 token 的采样概率」，没修正「走到这个前缀的概率」也不同。Liu, Li et al. 的实验里，token 级的 TIS 和 MIS 仍然会崩，整条回答的 MIS 最稳。
3. 整条的 $\rho$ 是连乘，长回答很容易超过阈值被丢掉（见第 2 节的表），所以有了按几何平均 $\rho^{1/T}$ 判断的 Geo 版本，思路和 [GSPO](/posttrain/grpo-variants) 的序列级 ratio 一样。
4. IcePop 是双边的：$\rho_t$ 太小（训练侧比推理侧小很多）也丢。区间取 $[0.5, 5]$，大约丢掉 1–2‰ 的 token，被丢的 token 熵更高。区间收窄到 $[0.5, 2]$ 反而不稳。

R3 论文的一组对比（作者自测，Qwen3-30B-A3B）：GRPO 平均 62.23，第 60 步崩；GRPO + TIS（$C = 2$）66.24，第 105 步崩；GRPO + R3 71.83，没崩。TIS 能延缓，但 MoE 上解决不了根因。

### 5. 从源头消除：让两边算出一样的数

**(a) batch-invariant kernel**。让 RMSNorm、matmul、attention 的归约顺序与 batch 大小无关：

1. matmul：不用 split-K / stream-K，所有形状用同一套 kernel 配置。比 cuBLAS 慢约 20%。
2. attention：先把 KV 写进 cache 再算，归约方式不受 chunked prefill、prefix caching 影响；split-KV 时固定每段的长度而不是固定段数，比如 KV 长 1000 切成 $256 \times 3 + 232$。

代价（作者自测，Qwen3-8B 单卡 1000 条请求）：vLLM 默认 26 s，确定性版本 55 s，优化 attention 后 42 s。再让训练侧用同一套 kernel，两边逐位相同，训推 KL **恰好是 0**，这就是 Thinking Machines 说的「真正的 on-policy RL」。

**(b) 训练和推理都改用 fp16**（[Qi et al. 2025](https://arxiv.org/abs/2510.26788)）。根因归到 bf16 的舍入：尾数只有 7 位，两边任何一点计算差异都会被放大到概率上。fp16 尾数 10 位，精度高 8 倍；fp16 范围小的问题用 loss scaling 解决（见 [训练显存账](/posttrain/training-memory)），RL 微调本来也用不到 bf16 的大范围。作者自测：序列级 log 比的差距缩小约 24 倍；FP32 推理也能稳住，但慢约 3 倍。

**(c) MoE 路由回放 R3**（[Ma et al. 2025](https://arxiv.org/abs/2510.11370)）。rollout 时记下推理引擎每层选的 top-K 专家，训练时**强制用同样的专家**。只回放选择结果（mask），gate 权重仍用训练侧的 logit 算，梯度照常传回 router：

$$
g_i = \frac{I_{\text{infer},i}\, e^{s_{\text{train},i}}}{\sum_j I_{\text{infer},j}\, e^{s_{\text{train},j}}}, \qquad
y = \sum_i g_i\, E_i(x)
$$

$I_{\text{infer}}$ 是推理侧的 top-K mask（选中为 1），$s_{\text{train}}$ 是训练侧的 router logit，$E_i$ 是第 $i$ 个专家。作者自测：KL 从 $1.5\times10^{-3}$ 降到 $7.5\times10^{-4}$，接近 dense 模型；rollout 延迟增加不到 3%。mask 可以跟着 prefix KV cache 一起缓存，多轮对话也能用。

注意和 GSPO 论文里的 Routing Replay 区分：那个回放的是**训练引擎里旧策略**的路由，解决的是「更新前后路由变了」（新旧策略之间）；R3 回放的是**推理引擎**的路由，解决的是训推之间。

**(d) 换成对误差不敏感的目标**。GSPO 用几何平均的序列级 ratio，单个 token 的误差被平均掉，作者说可以直接用推理引擎的似然，不必重算（[GSPO](https://arxiv.org/abs/2507.18071) §5.4）。

### 6. 框架里怎么开

| 框架 | 开关（以当前 main 为准，版本间改过名） |
|---|---|
| verl | `algorithm.rollout_correction.rollout_is: token / sequence`，`rollout_is_threshold: 2.0`（写成 `"0.5_5.0"` 就是 IcePop 的双边区间）；早期版本是 `actor.tis_imp_ratio_cap` |
| slime | `--use-tis`、`--tis-clip 2.0`；`--use-rollout-routing-replay`（R3）、`--use-routing-replay`（GSPO 的回放）；`--get-mismatch-metrics` 打出差距指标 |

训练时盯着训推 KL：Ring 团队的经验是差距超过 0.05 训练就会失败（作者自述）。

## 面试追问

::: details Q：权重一模一样，为什么两个引擎算出的概率不一样？
浮点加法不满足结合律，而归约顺序由 kernel 决定：matmul 按形状选 split-K，attention 按 KV 长度切块，这些又随 batch 大小变。推理和训练的 batch 组织、并行切分（TP / SP）、kernel 实现、精度（bf16 / fp8）都不同，归约顺序不同，结果就不同。MoE 上还会被 top-K 放大：logit 差一点，选中的专家就换了。
:::

::: details Q：既然有差距，为什么不直接用推理引擎返回的 logprob？
PPO 的 ratio 假设分子分母是同一个模型在不同权重下的概率。用推理侧的 logprob 当分母，第一步更新前 ratio 就不等于 1，clip 会把训推差距当成「更新过头」处理，该学的 token 学不到。所以在训练引擎重算 $\pi_{\text{old}}$，训推差距单独用 $\rho_t$ 处理。例外是 GSPO 这类序列级目标，对单个 token 的误差不敏感。
:::

::: details Q：TIS 的 C 为什么取 2 左右？
C 越大越接近无偏的 importance sampling，但方差按 $C^2$ 放大；C 越小方差越小，偏差越大。一个 $\rho = 16$ 的 token，不截断噪声放大 256 倍，C = 2 时是 4 倍。实践中 C = 2 是常用值（verl 的 DAPO + TIS 脚本、slime 的默认值都是 2.0）。
:::

::: details Q：token 级修正和序列级修正哪个对？
序列级是无偏的：整条回答的概率比 $\prod_t\rho_t$ 才是两个分布真正的似然比。token 级只修了每个 token 的采样概率，没修「走到这个前缀」的概率差，有偏。但序列级的连乘在长回答上方差极大，所以实践中用截断 / 屏蔽（MIS）或几何平均（Geo）来控制。
:::

::: details Q：MoE 的 RL 为什么比 dense 更难稳？
多了一个不连续的放大器：router 的 top-K。训推两边 router logit 的微小差距会让被选中的专家整个换掉，这个 token 的计算路径完全不同。R3 统计 94% 的 token 至少有一层选得不一样，KL 是 dense 的 2.4 倍。修法是 R3：训练时回放推理侧的专家选择。
:::

## 手撕

**算训推修正权重**：输入是两边算出的 token logprob，`[N, T]`，`mask` 标出回答部分。

```python
import torch

def rollout_correction(learner_logp, sampler_logp, mask, level="token", mode="truncate",
                       C=2.0, low=None):
    # learner_logp: 训练引擎重算的 log pi_learner(theta_old)；sampler_logp: 推理引擎返回的
    log_rho = (learner_logp - sampler_logp) * mask
    if level == "token":
        rho = torch.exp(log_rho)                                    # [N, T]
    elif level == "sequence":
        rho = torch.exp(log_rho.sum(-1, keepdim=True))              # [N, 1]，连乘
    else:                                                           # "geometric"
        rho = torch.exp(log_rho.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True))

    if mode == "truncate":                                          # TIS
        w = rho.clamp(max=C)
    else:                                                           # MIS / IcePop：超出就丢
        keep = rho <= C
        if low is not None:
            keep &= rho >= low                                      # IcePop: low=0.5, C=5
        w = rho * keep
    return w.detach() * mask                                        # 权重不传梯度
```

用法：`loss = -(w * ppo_obj * mask).sum() / mask.sum()`，`ppo_obj` 是 $\min(r\hat A, \operatorname{clip}(r)\hat A)$。

**R3 的回放**：训练侧 router 只换掉 top-K 的选择。

```python
def moe_gate_with_replay(router_logits, infer_topk_idx):
    # router_logits: [tokens, n_experts]，训练侧算的；infer_topk_idx: [tokens, K]，rollout 时记下的
    chosen = router_logits.gather(-1, infer_topk_idx)               # 用推理侧选的专家
    gates = torch.softmax(chosen, dim=-1)                           # 权重仍用训练侧 logit，有梯度
    return infer_topk_idx, gates
```

常见题：解释训推不一致的来源；写 TIS 的修正式并说明 $\rho_t$ 和 PPO ratio 的区别；token 级和序列级 IS 的偏差、方差权衡；MoE 为什么需要路由回放。

## 参考

- [Yao et al.：Your Efficient RL Framework Secretly Brings You Off-Policy RL Training（TIS）](https://fengyao.notion.site/off-policy-rl)
- [Liu, Li et al.：When Speed Kills Stability（MIS、Geo-RS）](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)
- [Thinking Machines（He）：Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · [batch-invariant-ops](https://github.com/thinking-machines-lab/batch-invariant-ops)
- [Qi et al.：Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788)
- [Ma et al.：Stabilizing MoE RL by Aligning Training and Inference Routers（R3）](https://arxiv.org/abs/2510.11370)
- [Ring-1T（IcePop）](https://arxiv.org/abs/2510.18855) · [IcePop 博客](https://ringtech.notion.site/icepop)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [verl rollout correction 文档](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md) · [slime](https://github.com/THUDM/slime)

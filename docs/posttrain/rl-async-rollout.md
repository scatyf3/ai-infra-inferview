---
title: 异步 Rollout：长尾、Partial Rollout 与 Staleness
status: draft
tags: [rl-infra, rollout, areal, pipelinerl, partial-rollout]
difficulty: 4
order: 4.1
related: [/posttrain/rl-infra, /posttrain/rl-train-infer-mismatch, /posttrain/grpo-variants, /inference/batching-scheduling]
stack: []
---

# 异步 Rollout：长尾、Partial Rollout 与 Staleness

> 同步 RL 一个 step 要等最长的那条回答写完。怎么量化这段浪费，怎么把没写完的回答挂起、让生成和训练流水起来，以及样本落后几个版本之后目标函数怎么改

## 一句话结论

推理类 RL 里 rollout 占一个 step 的 60–90% 时间，其中一大半花在最后几条最长的回答上：大部分回答早写完了，GPU 在为少数几条长回答空转。解法有三层：**partial rollout**（每轮设长度预算，没写完的挂起、下轮接着写），**异步**（生成和训练流水起来，允许样本落后几个策略版本），**in-flight 权重更新**（生成到一半就换新权重）。代价都是样本变成轻度 off-policy，要用 staleness 上限和修改过的目标函数（AReaL 的 decoupled PPO、截断 importance 权重）兜住。

## 推导

### 1. 长尾有多严重

几组公开的测量（都是作者自测）：

| 来源 | rollout 占 step 时间 | 长尾 |
|---|---|---|
| [Seer](https://arxiv.org/abs/2511.14617)（Moonshot） | Moonlight 84%，Qwen2-VL-72B 63%，Kimi-K2 87% | 最后 10% 的请求最多占 50% 的时间 |
| [RollPacker](https://arxiv.org/abs/2509.21009) | 约 70% | 最长回答是中位数的 25–32 倍 |
| [Laminar](https://arxiv.org/abs/2510.12633) | 最多 83% | p99 长度比 p50 大一个数量级 |
| [APRIL](https://arxiv.org/abs/2509.18521) | 超过 90% | |

**算一下浪费了多少**。同步 RL 里，一个 step 的 rollout 时间由最长的回答决定。设一批有 $N$ 条回答，长度 $L_1, \dots, L_N$，每个 decode 步的耗时近似不变。GPU 上「有活干的序列槽位」的占比是

$$
\text{利用率} \approx \frac{\sum_i L_i}{N \cdot \max_i L_i}
$$

例子：8 条回答，长度（千 token）是 $[2, 3, 3, 4, 4, 5, 8, 20]$。

1. 总共生成 49k token，而 8 个槽位 × 20k 步 = 160k，利用率 **31%**。
2. 第 7 条在 8k 处写完后，最后 12k 步只剩一条在跑，占整段时间的 **60%**。

实际比这更糟：batch 只剩几条时，decode 是 memory-bound，读一遍权重只为算一两个 token（见 [Roofline](/inference/prefill-decode-roofline)），每步的耗时并不会随 batch 缩小而等比例变短。同步 RL 里训练卡这段时间完全空等。

### 2. 三种节奏：同步、one-step off-policy、全异步

```
同步            gen(v0) ──> train(v0→v1) ──> gen(v1) ──> train(v1→v2)
                训练卡等生成，生成卡等训练

one-step off    gen(v0) ──> gen(v1)    ──> gen(v2)
                            train(v0→v1) ──> train(v1→v2)
                第 n+1 批用 v(n) 生成，同时训练第 n 批；样本落后 1 个版本

全异步          生成卡一直在写，写完一条放进 buffer；训练卡凑够一批就训
                样本可能来自好几个版本，要限制最多落后多少
```

**staleness**（陈旧度）：一个样本被训练时，当前策略的版本号减去生成它的策略的版本号。同步 RL 是 0，one-step off-policy 是 1。

verl 的 one-step off-policy 方案（作者自测，Qwen2.5-Math-7B，DAPO）：FSDP2 后端从 19h18m 降到 15h34m，Megatron 后端从 18h21m 降到 13h06m（[文档](https://verl.readthedocs.io/en/latest/advance/one_step_off.html)）。

### 3. Partial rollout：没写完的挂起，下轮接着写

[Kimi k1.5](https://arxiv.org/abs/2501.12599) §2.6.2 的做法：

1. 每轮 rollout 给每条回答一个固定的 token 预算。
2. 超出预算还没写完的，把已经写好的部分存进 replay buffer，下一轮从断点接着写，不从头再来。
3. 一条长回答于是由好几段拼成，分别来自第 $n-m$ 轮到第 $n$ 轮的策略。只有当前轮写的那段需要按当前策略算，之前的段直接从 buffer 复用。

论文说「某些段可以不算 loss」，但没说是哪些段。

[APRIL](https://arxiv.org/abs/2509.18521) 把它做成调度策略：每轮多发一些请求，**凑够目标数量就停**，没写完的放进 buffer，下一轮优先续写。作者统计：后期一轮里约 40% 的 token 来自上一轮续写的回答，单条回答最多跨 5 个策略版本；rollout 吞吐平均提升 22.5%（作者自测，数字随论文版本变过）。

和 DAPO 的[动态采样](/posttrain/grpo-variants)配合起来很自然：动态采样本来就要「多采、过滤、凑够为止」，凑够之后剩下没写完的回答不扔掉，而是挂起留给下一轮。slime 的 `--partial-rollout` 就是这个意思：动态采样中途没写完的样本回收进 data buffer。

### 4. 全异步与 staleness 上限（AReaL）

[AReaL](https://arxiv.org/abs/2505.24298) 把生成和训练完全解耦：

1. **可中断的 rollout worker**：收到权重更新请求时，中断所有正在生成的序列，加载新权重，**丢掉用旧权重算的 KV cache、用新权重重算**，再继续 decode。所以一条回答里不同段来自不同版本。
2. **trainer**：从 buffer 里取样本，凑够一个 batch 就训，每个样本只用一次。
3. **rollout controller** 控制生成速度，不让 rollout 跑得太超前。

**staleness 上限**：设 trainer 当前是第 $i$ 版，batch 大小 $B$，已经发出 $N_r$ 条生成请求，允许的最大陈旧度是 $\eta$。controller 只在满足下式时接受新的生成请求：

$$
\left\lfloor \frac{N_r - 1}{B} \right\rfloor \le i + \eta
$$

例子：$B = 512$、$\eta = 4$、trainer 在第 10 版。$\lfloor (N_r - 1)/512 \rfloor \le 14$，即累计最多发出 $15 \times 512 = 7680$ 条，rollout 最多领先 trainer 4 个 batch。$\eta = 0$ 就是同步 RL。论文里代码任务取 4，数学取 8。

**decoupled PPO**：标准 PPO 里 $\pi_{\text{old}}$ 身兼两职，既是「采样用的策略」，又是「这一步信任域的中心」。异步之后这两者不再是同一个，AReaL 把它们拆开：

$$
J(\theta) = \mathbb{E}_{a_t \sim \pi_{\text{behav}}}\Bigg[\sum_t \frac{\pi_{\text{prox}}(a_t \mid s_t)}{\pi_{\text{behav}}(a_t \mid s_t)}\,\min\Big(u_t(\theta)\hat A_t,\ \operatorname{clip}\big(u_t(\theta), 1-\epsilon, 1+\epsilon\big)\hat A_t\Big)\Bigg],
\qquad u_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{prox}}(a_t \mid s_t)}
$$

1. $\pi_{\text{behav}}$：真正采出这个 token 的策略，可能是好几个旧版本的混合。
2. $\pi_{\text{prox}}$：这一步更新前的参数，用来当信任域中心，batch 到了之后在 trainer 里重算它的 logprob。
3. 前面的比值修正「样本来自旧策略」，clip 只约束「这一步离 $\pi_{\text{prox}}$ 多远」。三者相同时退化成标准 PPO。

消融（作者自测，1.5B 数学模型，AIME24）：

| 最大 staleness $\eta$ | 不用 decoupled PPO | 用 |
|---|---|---|
| 0（同步） | 42.0 | |
| 1 | 41.8 | 42.1 |
| 4 | 23.3 | 42.2 |
| 16 | 35.8 | 38.7 |
| ∞ | 34.0 | 36.9 |

$\eta = 4$ 时不改目标函数掉到 23.3，改了回到 42.2；$\eta$ 再放大，改了也挡不住下降。

训练时长（作者自测，H800，3/4 的卡做推理、1/4 做训练）：

| 模型（任务） | 同步 | AReaL 异步 |
|---|---|---|
| 1.5B（数学） | 41.0 h | 14.8 h |
| 7B（数学） | 57.7 h | 25.4 h |
| 14B（代码） | 48.8 h | 21.9 h |
| 32B（代码） | 51.1 h | 31.1 h |

摘要里写的是「最多 2.77 倍加速」，正文其他地方写过 2.57 倍和 2.5 倍，以表里的小时数为准。

### 5. In-flight 权重更新（PipelineRL）

[PipelineRL](https://arxiv.org/abs/2509.19128) 走得更远：**每个优化器 step 之后**就把新权重推给推理引擎，引擎短暂暂停接收权重，然后继续写正在写的序列。

和 AReaL 的区别：**旧的 KV cache 不重算，直接接着用**。论文实验发现，混着旧 KV 继续生成，和重算 KV 相比分布偏差只略大一点，而重算会降低吞吐。

**lag**：生成某个 token 的策略和当前策略之间隔了几个优化器 step。同一条回答里，越靠前的 token lag 越大。最大 lag 约为

$$
g_{\max} = \left\lceil \frac{t_{\text{gen}}}{t_{\text{train}}} \right\rceil
$$

$t_{\text{gen}}$ 是生成一条最长回答的时间，$t_{\text{train}}$ 是一个优化器 step 的时间。比如生成最长回答要 60 s、一个 step 10 s，最大 lag 是 6。目标函数用截断 importance 权重 $\min(c, \pi/\mu)$ 的 REINFORCE，$c = 5$，$\mu$ 是生成时的策略。作者自测比常规 RL 学得快约 2 倍（Qwen2.5-7B）。

### 6. 过采样与长短分开跑

另一类思路不改 RL 的节奏，只改「这一轮采谁、等谁」：

1. **过采样后丢掉最慢的**：多发一些请求，先写完的够数就开训，最慢的那些直接丢掉。缺点是长回答系统性地更容易被丢，等于偏向短回答。
2. **RollPacker 的 tail batching**：每轮多发 $1.25\times$ 的 prompt、每个 prompt 多采 $1.25\times$ 条，保留最先完成的那些；容易写长的 prompt 攒起来，放进专门的「长轮次」一起跑，长回答不再拖慢普通轮次（作者自测，相对 verl 端到端快 2.0–2.6 倍）。
3. **Seer** 的出发点是同一个 prompt 的 $G$ 条回答在长度和内容上都很相似，组内的回答可以互相提供信息。在此基础上做了三件事：把 rollout 切分后动态负载均衡、按上下文调度来减少长尾等待、组内共享的自适应投机解码（作者自测，rollout 吞吐最多提升约 2 倍，长尾延迟降低 72–94%）。

### 7. 选型小结

| 做法 | 样本落后多少 | 需要改目标函数吗 | 系统复杂度 |
|---|---|---|---|
| 同步 + 过采样 | 0 | 否（但有长度偏向） | 低 |
| partial rollout | 一条回答跨几轮 | 视跨的轮数 | 中 |
| one-step off-policy | 1 个版本 | 通常不改 | 中 |
| 全异步（AReaL） | $\le \eta$ 个版本 | decoupled PPO | 高 |
| in-flight 更新（PipelineRL） | $\le g_{\max}$ 个 step，KV 也是旧的 | 截断 IS | 高 |

框架里的开关：verl 有 one-step off-policy 和 fully async 两个 recipe，后者用 `staleness_threshold` 限制陈旧样本比例、`partial_rollout` 在同步权重时中断并续写；slime 用 `--rollout-function-path` 指向 fully async 的 rollout 函数，配合 `--partial-rollout`、`--over-sampling-batch-size`、`--dynamic-sampling-filter-path`。

## 面试追问

::: details Q：rollout 慢，为什么不直接加卡？
同步 RL 的 rollout 时间由最长那条回答决定，它是一条一条 token 串行写出来的，加卡只能让每一步稍快一点，缩短不了这条回答的步数。加卡提高的是「同时写多少条」，而长尾阶段同时在写的只剩几条，多出来的卡照样空等。所以要从调度上解决：不等最长的那条（partial rollout、异步），或者把长的单独跑。
:::

::: details Q：AReaL 为什么要把 $\pi_{\text{old}}$ 拆成 behav 和 prox 两个？
标准 PPO 里 $\pi_{\text{old}}$ 既是采样策略，又是信任域中心。异步之后样本来自好几个旧版本，如果把信任域中心也设成那个旧版本，clip 约束的就是「离很旧的策略不能太远」，当前策略本来就已经离它很远，大部分 token 都会被 clip 掉，学不动。拆开之后，importance 比 $\pi_{\text{prox}}/\pi_{\text{behav}}$ 负责修正分布，clip 只约束这一步相对 $\pi_{\text{prox}}$（更新前的参数）走多远。
:::

::: details Q：生成到一半换权重，旧的 KV cache 怎么办？
两种选择。AReaL 丢掉旧 KV、用新权重重算前缀再继续，分布更干净，但重算有开销。PipelineRL 直接沿用旧 KV，实验显示偏差只略大，吞吐更高。两者都要在目标函数里处理「同一条回答的不同段来自不同版本」。
:::

::: details Q：staleness 上限 $\eta$ 怎么选？
$\eta$ 越大，rollout 越能跑在前面，训练卡越不用等，吞吐越高；但样本越旧，importance 权重方差越大，训练越不稳。AReaL 的消融里，用了 decoupled PPO 后 $\eta \le 4$ 基本不掉点，$\eta = 16$ 开始明显下降。经验上取能把训练卡喂饱的最小值。
:::

## 手撕

**带 staleness 上限的异步循环**（单进程示意，真实系统里 rollout 和 trainer 在不同的卡上）：

```python
import collections

def async_rl(policy, prompts, B=512, eta=4, total_steps=100):
    buffer = collections.deque()
    version, issued = 0, 0
    inflight = []                                   # (请求, 生成它的版本)

    while version < total_steps:
        # rollout controller：不超过 staleness 上限才发新请求
        while (issued // B) <= version + eta and prompts:
            inflight.append((policy.submit(prompts.pop()), version))
            issued += 1
        # 收集写完的回答，记下生成它的版本
        for req, v in [x for x in inflight if x[0].done()]:
            inflight.remove((req, v))
            buffer.append((req.result(), v))
        # trainer：凑够一批就训
        if len(buffer) >= B:
            batch = [buffer.popleft() for _ in range(B)]
            stale = [version - v for _, v in batch]  # 每条样本落后几个版本
            policy.train_step(batch)                 # 用 decoupled PPO
            version += 1
            policy.push_weights(version)             # 推给推理引擎，见权重同步
```

`(issued // B) <= version + eta` 就是 AReaL 的 $\lfloor (N_r - 1)/B \rfloor \le i + \eta$（这里 `issued` 是发出新请求之前的计数，所以不用减 1）。

**从长度分布算同步 rollout 的利用率**：

```python
def sync_utilization(lengths):
    return sum(lengths) / (len(lengths) * max(lengths))

sync_utilization([2, 3, 3, 4, 4, 5, 8, 20])   # 0.306
```

常见题：解释 RL rollout 的长尾和它对同步训练的影响；partial rollout 怎么做、带来什么问题；写出 decoupled PPO 并解释 behav 和 prox 的区别；估算给定生成和训练时间下的最大 lag。

## 参考

- [AReaL：A Large-Scale Asynchronous RL System](https://arxiv.org/abs/2505.24298)
- [PipelineRL：In-flight Weight Updates](https://arxiv.org/abs/2509.19128)
- [Kimi k1.5（partial rollout）](https://arxiv.org/abs/2501.12599)
- [APRIL：Active Partial Rollouts](https://arxiv.org/abs/2509.18521)
- [RollPacker](https://arxiv.org/abs/2509.21009)
- [Seer](https://arxiv.org/abs/2511.14617)
- [Laminar](https://arxiv.org/abs/2510.12633)
- [DAPO（动态采样）](https://arxiv.org/abs/2503.14476)
- [verl one-step off-policy](https://verl.readthedocs.io/en/latest/advance/one_step_off.html) · [verl fully async](https://verl.readthedocs.io/en/latest/advance/fully_async.html) · [slime](https://github.com/THUDM/slime)

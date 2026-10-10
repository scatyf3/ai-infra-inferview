---
title: RL Infra：Rollout 与 Training 分离
status: draft
tags: [rl-infra, verl, openrlhf]
difficulty: 4
order: 4
related: [/posttrain/rlhf-ppo-dpo-grpo, /posttrain/grpo-variants, /posttrain/rl-async-rollout, /posttrain/rl-weight-sync, /posttrain/rl-train-infer-mismatch, /posttrain/rl-agentic, /posttrain/training-memory, /parallel/zero-fsdp, /inference/batching-scheduling]
stack: [ld-load]
---

# RL Infra：Rollout 与 Training 分离

> Preliminary：一轮 RL 里发生了什么、为什么一个系统里要跑两套引擎、权重怎么同步、on-policy 怎么被悄悄打破

## 一句话结论

RL 训练一轮要先用当前策略生成很多样本（rollout，推理负载），再用这些样本更新策略（training，训练负载）。两种负载对并行方式和显存布局的要求完全不同，所以现代 RL 框架把 rollout 交给 vLLM / SGLang 这类推理引擎，training 交给 FSDP / Megatron。系统层面的难点有三个：**两套引擎怎么摆放**、**每轮怎么把新权重同步给推理引擎**、**rollout 太慢时怎么异步化又不把 on-policy 训崩**。

## Preliminary

### 1. 一轮 RL 的数据流

以 GRPO（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）为例，一个 step：

```
prompts ──① rollout──> 每个 prompt 采 G 条回答          推理引擎（vLLM / SGLang）
        ──② reward───> 每条回答一个分数                  规则校验 / 沙箱跑代码 / reward model
        ──③ logprob──> 当前策略、参考模型的 token logprob  训练引擎 forward
        ──④ advantage> 组内归一化：(r - mean) / std       CPU，几乎免费
        ──⑤ update───> 若干个 mini-batch 的梯度更新        训练引擎 forward + backward
        ──⑥ sync─────> 新权重推给推理引擎                 下一轮 ① 用新策略
```

PPO 多一个 critic 模型（要训练）和价值估计；GRPO 用组内均值当 baseline，省掉了 critic。算法细节见 [RLHF 全家桶](/posttrain/rlhf-ppo-dpo-grpo)，DAPO、GSPO 这些变体改了什么见 [GRPO 变体](/posttrain/grpo-variants)。

这一页是总览，下面几个问题各有一页展开：

| 问题 | 展开 |
|---|---|
| rollout 的长尾、partial rollout、异步和 staleness | [异步 Rollout](/posttrain/rl-async-rollout) |
| 每轮怎么把新权重送进推理引擎、要多久 | [权重同步](/posttrain/rl-weight-sync) |
| 同一份权重两个引擎算出不同概率 | [训推不一致](/posttrain/rl-train-infer-mismatch) |

### 2. 两种负载为什么不能用一套引擎

| | rollout（①） | training（③⑤） |
|---|---|---|
| 本质 | 自回归 decode，**memory-bound** | 大矩阵 forward / backward，**compute-bound** |
| 显存里放什么 | 权重 + **KV cache** | 权重 + 梯度 + 优化器状态 + 激活（每参数约 16 B，见 [训练显存账](/posttrain/training-memory)） |
| 想要的并行 | TP 小、DP 大，batch 越大越好 | FSDP / ZeRO-3 或 TP+PP（见 [ZeRO 与 FSDP](/parallel/zero-fsdp)） |
| 关键优化 | PagedAttention、continuous batching、CUDA graph | 重计算、通信 overlap、fused optimizer |

用训练框架自带的 `generate()` 做 rollout 会非常慢：没有 paged KV、没有 continuous batching，长 CoT 下尤其明显。所以 rollout 必须交给专门的推理引擎，这就带来了「一个模型两份权重布局」的问题。

### 3. Rollout 通常是时间大头，而且有长尾

推理类任务的回答动辄上万 token，rollout 是纯 decode，一个 step 的时间常常被它主导。更麻烦的是**长尾**：同一个 batch 里大部分回答早就结束了，最长的几条还在生成，整个 batch 要等它们（同步 RL 下训练卡全部空等）。

缓解长尾的几种思路，后面会展开：

- **Partial rollout**：给每轮生成设一个长度预算，没写完的回答先挂起，下一轮接着写，而不是从头再来（[Kimi k1.5](https://arxiv.org/abs/2501.12599)）。
- **异步 rollout**：不再「一轮生成完才训练」，生成和训练流水起来（[AReaL](https://arxiv.org/abs/2505.24298)、[PipelineRL](https://arxiv.org/abs/2509.19128)）。
- **过采样 + 丢弃**：多发一些 prompt，凑够数量就开训，最慢的那些直接丢掉。

量化长尾、AReaL 的 staleness 上限和 decoupled PPO 见 [异步 Rollout](/posttrain/rl-async-rollout)。

### 4. 摆放：colocate 还是分离

**Colocate（同卡分时）**：rollout 和 training 在同一组卡上轮流跑。

- 切到训练前，推理引擎要把权重和 KV cache 让出来。vLLM 的 [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) 就是为此设计的：level 1 把权重卸到 CPU、丢掉 KV；level 2 连权重也丢掉，适合 RL 这种「醒来时反正要加载新权重」的场景。
- 好处是卡不闲置，权重同步可以在同一张卡上完成（同卡显存拷贝 / CUDA IPC，不过网络）。
- 坏处是每次切换都有加载 / 卸载开销，且两边被迫用同一组卡数。
- [HybridFlow（veRL）](https://arxiv.org/abs/2409.19256)的 3D-HybridEngine 解决的就是 colocate 下训练和生成用**不同的并行配置**时，怎么在两种切分之间 reshard 而不多占显存。

**分离（disaggregated）**：rollout 一组卡，training 另一组卡。

- 两边可以各自选卡数和并行方式，也天然适合做异步流水。
- 坏处是两边负载不匹配时有一边空等，权重同步要跨卡甚至跨节点。
- [OpenRLHF](https://arxiv.org/abs/2405.11143) 用 Ray 把 actor、critic、reward、vLLM 分到不同 GPU 组；[slime](https://github.com/THUDM/slime) 用 Megatron 训练、SGLang + router 做 rollout，中间用一个 data buffer 解耦。

### 5. 权重同步：一轮一次，但不便宜

每次更新后，推理引擎必须拿到新权重，否则下一轮就不是 on-policy。难点：

- **量大**：7B bf16 是 14 GB，70B 是 140 GB，MoE 更大。按 400 Gbps（50 GB/s）的跨节点带宽算，70B 光传输就要约 3 s，还没算 reshard。
- **切分不同**：训练侧是 FSDP 分片或 Megatron 的 TP×PP 切分，推理侧是另一种 TP 切分（甚至不同的权重合并方式，比如 QKV 融合、gate/up 融合）。同步时要先在训练侧 all-gather 成完整张量（或按目标切分重组），再按推理侧的切分发过去。
- **手段**：同卡用显存拷贝 / CUDA IPC；跨卡用 NCCL broadcast，按 bucket 分批发、和 gather 流水；slime 支持 NCCL 和磁盘两种同步路径。PipelineRL 更进一步，在推理引擎**生成过程中**直接换权重（in-flight weight update），不等当前序列生成完。

带宽下界怎么算、Megatron 到 vLLM 的布局转换、分桶和流水见 [权重同步](/posttrain/rl-weight-sync)。

### 6. On-policy 是怎么被悄悄打破的

PPO / GRPO 的 importance ratio $\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$ 假设分母是**生成样本时的策略**。实际系统里有两个原因让它不成立：

**(a) 异步带来的 staleness**：异步 rollout 下，一个 batch 里的样本可能来自落后一两个版本的策略。AReaL 用修改过的 PPO 目标显式处理这个版本差。

**(b) 训推不一致（training-inference mismatch）**：即使权重完全相同，推理引擎（vLLM，各种融合 kernel、不同精度、不同并行）算出的 token 概率和训练引擎（FSDP）算出的也不一样。所以通常的做法是：

1. rollout 只用来**采样 token**，不信任它返回的 logprob；
2. 在训练引擎里**重新算一遍** $\pi_{\text{old}}$ 的 logprob（数据流里的第 ③ 步就是这个用途）；
3. 剩下的分布差再用 **truncated importance sampling（TIS）** 修正：给「训练侧 / 推理侧」概率比乘一个截断的权重 $\min(\rho, C)$，牺牲一点偏差换方差有界（[Yao et al., 2025](https://fengyao.notion.site/off-policy-rl)，veRL 和 slime 都已集成）。

这一点在 rollout 用 FP8 等低精度量化时更严重。差距从哪来、MIS / IcePop 等其他修正、以及 batch-invariant kernel、fp16、MoE 路由回放这些从源头消除的办法，见 [训推不一致](/posttrain/rl-train-infer-mismatch)。

### 7. 往前走：Agentic RL

多轮工具调用 / 智能体任务里，一条 trajectory 是「生成 → 调工具 / 跑环境 → 再生成」的循环：

- 环境交互的延迟（跑代码、搜索、浏览器）夹在 decode 之间，GPU 在等 CPU。
- 每条 trajectory 的轮数、长度都不确定，长尾比单轮推理更严重。
- 这让 rollout 天然要做成**异步、以请求为单位**的服务（推理引擎以 server 模式部署，环境侧发请求），而不是一个同步的 `generate(batch)`。slime 的 server-based rollout、AReaL 的全异步设计都是朝这个方向。

为什么要 token 进 token 出、工具输出怎么 mask、要多少并发轨迹才能喂饱 GPU，见 [Agentic RL](/posttrain/rl-agentic)。

### 小结：读框架代码时先找这几个问题的答案

1. 训练后端是什么（FSDP / Megatron），推理后端是什么（vLLM / SGLang）？
2. colocate 还是分离？colocate 的话显存怎么让出来？
3. 权重同步走哪条路径，怎么 reshard？
4. 同步还是异步？允许落后几个版本，怎么修正？
5. $\pi_{\text{old}}$ 的 logprob 是推理侧给的还是训练侧重算的？有没有 TIS？

## 面试追问

::: details Q：colocate 和分离部署怎么选？
colocate 把训练和推理放同一组卡上轮流跑，GPU 不闲置但每次切换要卸载 / 加载显存，适合中小规模。分离部署让 rollout 和 training 各占一组卡流水起来，吞吐更高但两边负载不匹配时有一边会空等，要靠异步 rollout 补。规模越大、生成越长（推理任务），越倾向分离。
:::

::: details Q：为什么不直接用推理引擎返回的 logprob 当 π_old？
推理引擎为了速度用了融合 kernel、不同的累加顺序、甚至低精度，算出的概率和训练引擎有系统性偏差。直接拿来算 importance ratio，等于用一个「看起来是 on-policy、其实是 off-policy」的比值，训练会不稳定甚至崩。标准做法是在训练引擎里重算一遍，剩余偏差用 TIS 截断修正（[Yao et al., 2025](https://fengyao.notion.site/off-policy-rl)）。
:::

::: details Q：异步 RL 能快多少，代价是什么？
快在 rollout 和训练流水起来，训练卡不再等最长的那条回答。AReaL 报告相对同步系统最多 2.77× 的训练加速（作者自测）。代价是样本变成轻度 off-policy，需要限制允许的版本差（staleness），并用修改过的目标函数修正；版本差放得越大，吞吐越高，训练越不稳定。
:::

## 手撕

常见题：画出一轮 GRPO 的数据流（rollout → reward → logprob → advantage → update → 权重同步），标出每步跑在哪组卡上；给定模型大小和链路带宽，估算一次权重同步的耗时。

## 参考

- [DeepSeekMath（GRPO）](https://arxiv.org/abs/2402.03300)
- [HybridFlow（veRL）](https://arxiv.org/abs/2409.19256)
- [OpenRLHF](https://arxiv.org/abs/2405.11143)
- [Kimi k1.5（partial rollout）](https://arxiv.org/abs/2501.12599)
- [AReaL：全异步 RL](https://arxiv.org/abs/2505.24298)
- [PipelineRL：in-flight weight update](https://arxiv.org/abs/2509.19128)
- [Yao et al.：Your Efficient RL Framework Secretly Brings You Off-Policy RL Training（TIS）](https://fengyao.notion.site/off-policy-rl)
- [slime](https://github.com/THUDM/slime)
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)

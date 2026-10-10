---
title: Agentic RL：多轮工具调用的 Rollout 系统
status: draft
tags: [rl-infra, agentic-rl, multi-turn, tito, sandbox]
difficulty: 4
order: 4.4
related: [/posttrain/rl-infra, /posttrain/rl-async-rollout, /posttrain/rl-train-infer-mismatch, /inference/kv-cache-paged-attention]
stack: []
---

# Agentic RL：多轮工具调用的 Rollout 系统

> 一条轨迹是「生成 → 调工具 → 拿结果 → 再生成」的循环。和单轮 RL 比，多出来的问题：token 要原样记下来（不能解码再重新分词）、工具输出不能算 loss、GPU 要等环境、多轮之间的 KV 怎么复用、轨迹太长被截断时怎么办

## 一句话结论

多轮 agent 的 rollout 不再是一次 `generate(batch)`，而是每条轨迹一个协程：调推理服务、解析工具调用、跑环境、把结果拼回上下文、再调推理服务。系统上要做对四件事：

1. **token 进、token 出**：训练用的 token 序列必须就是采样时的那串 token。先解码成文本、拼进 chat template、再重新分词，得到的 token 会变，训练就成了 off-policy，而且没法用 importance sampling 修。
2. **loss mask**：工具和环境返回的 token 只当上下文，不算 loss。
3. **让 GPU 别等环境**：环境一次调用几秒到几分钟，要靠大量并发轨迹和异步把 GPU 喂饱；同一条轨迹的后续轮次路由到同一个推理实例，复用前缀的 KV。
4. **长轨迹**：撞到上下文上限、轮数上限、超时的轨迹，loss 屏蔽掉，不要当成「答错」来罚。

## 推导

### 1. 一条轨迹里有哪些 token

```
[prompt] [模型: 思考 + 工具调用] [工具结果] [模型: 思考 + 工具调用] [工具结果] [模型: 最终回答]
 mask=0        mask=1               mask=0          mask=1              mask=0         mask=1
```

只有模型自己生成的 token 是策略的动作，`mask = 1`；prompt 和工具 / 环境返回的内容是观察，`mask = 0`。

**例子**：3 轮，模型分别生成 300、200、400 个 token，中间两次工具返回 1500 和 800 个 token。回答部分共 3200 个 token，算 loss 的只有 900 个，占 **28%**。工具输出（网页、文件内容、测试日志）往往比模型自己说的多得多。

为什么工具输出不能算 loss：

1. 它不是策略采出来的，没有「策略概率」可言，算 ratio、算 advantage 都没有意义。
2. 训练模型去「预测工具会返回什么」，等于教它编造工具结果。

verl 的 agent loop 输出 `response_mask`（模型生成的 token 为 1，工具返回为 0）；slime 里工具结果用 `trainable=False` 追加，`loss_mask` 记 0。Agent Lightning 换了个思路：每一次 LLM 调用单独存成一条样本（输入 token、输出 token、logprob），根本不拼成一长串，也就不需要 mask。

### 2. Token 进、token 出：为什么不能解码再重新分词

最直接的写法是「文本进、文本出」：调 OpenAI 风格的 chat 接口拿到文本，解析工具调用，最后把整段对话用 chat template 重新分词，拿去训练。问题是**重新分词得到的 token 和采样时的不一样**（retokenization drift）：

1. **分词不唯一**：同一段文本可以有多种切法。模型采样时出的是 `H` + `AVING`，重新分词变成 `HAV` + `ING`。文本一样，token 不一样。
2. **工具调用被解析再渲染**：解析器把模型输出的 JSON 解析出来、再按模板拼回去，空格可能变了；有的解析器还会自动修正模型写错的 JSON。
3. **chat template 会改历史**：Qwen3 的模板在多轮对话里会删掉之前轮次的思考内容（官方推荐的用法）。rollout 时第 3 轮看到的上下文里没有第 1 轮的思考；训练时如果把完整的轨迹拼起来，第 3 轮的上下文里有。同一个 token 在两边的条件不同。

**为什么 importance sampling 修不了**：TIS 这类修正（见 [训推不一致](/posttrain/rl-train-infer-mismatch)）假设两边是同一串 token、只是概率不同，逐个 token 算比值。重新分词后两边的 token 序列都对不齐，没法逐 token 比。

后果（作者自述）：verl 团队发现对最终的对话重新套模板，连单轮 PPO 都不收敛；AT²PO 把 Qwen3 上 Tree-GRPO 的早期崩溃追到这个原因，改成 token 进 token 出后消失。目前没有公开数据说明多少比例的轨迹会出现不一致。

**做法：从头到尾只传 token id。**

1. 推理服务返回 token id：vLLM 的 `/v1/chat/completions` 加 `"return_token_ids": true`，返回 `prompt_token_ids` 和 `token_ids`（v0.10.2 起）；SGLang 的 `/generate` 直接收 `input_ids`，`return_logprob` 时返回每个输出 token 的 id 和 logprob。
2. 下一轮的输入 = 上一轮的输入 token + 模型输出的 token + 工具结果单独分词得到的 token，直接拼接，不对整段对话重新套模板。
3. verl 的 agent loop 用「增量分词」：第 $i$ 条消息的 token = `template(messages[:i+1])` 减去 `template(messages[:i])`，并且默认做一致性检查（`tokenization_sanity_check_mode: strict`），对不上就报警。对 Qwen3 这种会改历史的模板，增量只对一段固定的基础历史（系统消息 + 用户消息）算。
4. 训练和 rollout 用不用「删掉历史思考」的模板，要两边一致：verl 默认两边都保留完整思考；`use_inference_chat_template=True` 则 rollout 也用线上的模板。

### 3. 推理服务化、按轨迹异步

单轮 RL 可以一次 `generate(batch)`，多轮不行：每条轨迹的轮数、每轮等环境的时间都不同。做法是把推理引擎部署成服务，每条轨迹一个协程独立推进（verl 的 `AgentLoopWorker` 为每个 prompt 跑一个 `AgentLoopBase.run` 协程）：

```
协程 i:  调推理服务 ─> 解析工具调用 ─> 调环境（几秒到几分钟）─> 拼回 token ─> 调推理服务 ─> …
```

一条轨迹在等环境时，推理服务在给别的轨迹生成。

**要多少条并发轨迹才能把 GPU 喂饱**：设推理服务想保持 $B$ 条序列同时在 decode，每条轨迹只有比例 $f$ 的时间在生成（其余在等环境）。需要的并发轨迹数约为

$$
N_{\text{traj}} \approx \frac{B}{f}
$$

例：想让 $B = 256$ 条同时 decode，轨迹有 2/3 的时间在等环境（$f = 1/3$），就要约 **768** 条轨迹同时在跑。环境越慢，要的并发越多，环境侧（沙箱、容器）的规模也跟着上去。Kimi K2 的报告里，代码类任务用 Kubernetes 跑了一万多个并发沙箱。

**多轮之间复用 KV**：第 $k$ 轮的输入是前 $k-1$ 轮的全部内容加上新的工具结果。如果每轮都从头 prefill：

例：每轮新增 2000 个 token，共 10 轮。不复用时 prefill 总量是 $2000 \times (1 + 2 + \dots + 10) = 110{,}000$ 个 token；前缀 KV 命中时每轮只 prefill 新增的 2000 个，共 20,000 个，**少 5.5 倍**。

前提是同一条轨迹的后续轮次落在同一个推理实例上（前缀 KV 在那里）。verl 的做法是 sticky session：第一轮按最少请求数选实例，之后同一个 `request_id` 固定发到这个实例；slime 通过 SGLang 的 router，默认 `cache_aware` 策略为每个实例维护一棵近似的 radix tree，按前缀匹配路由（原理见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)）。注意每次权重同步后旧的 KV 都要作废（见 [权重同步](/posttrain/rl-weight-sync)）。

### 4. 环境是新的长尾

环境调用的耗时分布比 decode 还长尾：大部分几秒，个别几分钟；沙箱启动、`env.reset` 还可能失败。几组数字（作者自测）：

1. [RollArt](https://arxiv.org/abs/2512.22560)（Qwen3-8B，SWE 任务，32 张 H800）：正常的 step 平均 366 s，其中生成 54%、训练 23%、环境初始化 15%；出现环境故障的 step 变成 513 s，`env.reset` 占 rollout 时间的 78%。环境大约每 10 个 step 坏一次。
2. [ROLL Flash](https://arxiv.org/abs/2510.11345)：环境级异步（不等一批环境一起返回）在模拟实验里把一个 step 从 892 s 压到 362 s；真实 SWE 任务从 10.2 h 降到 8.3 h。

常用手段：

1. **环境级异步**：每条轨迹的环境调用独立返回，不按批同步。
2. **冗余 rollout**：多开一些环境组，凑够需要的轨迹数就停，慢的和坏的丢掉（和 [异步 Rollout](/posttrain/rl-async-rollout) 的过采样一个思路）。
3. **环境做成独立服务**：重的环境（浏览器、代码执行）单独部署、单独扩缩容，不和训练进程绑在一起。
4. **partial rollout**：太长的轨迹挂起，下一轮接着跑（Kimi K2）。

### 5. 多轮特有的算法问题

**(a) 截断的轨迹别当答错罚**。轨迹撞到上下文上限、轮数上限或超时，没有完成任务，按规则奖励是 0。但它可能一直在正确的路上，只是太长。DeepSWE 的 compact filtering：这些轨迹的 loss 直接屏蔽，不参与更新。SkyRL-Agent 的做法类似：超过 32K token 或 50 轮的轨迹不算梯度。和 DAPO 的[超长过滤](/posttrain/grpo-variants)是同一个想法。

**(b) 一条轨迹一个奖励，怎么分给每一轮**。最简单的是广播：整条轨迹的 advantage 复制给每一轮的每个 token（rLLM 的 `broadcast` 模式，Agent Lightning 目前也是这样）。更细的做法要有每一轮的奖励：rLLM 的 `per_step` 模式用每步的 reward；GiGPO 在不同轨迹里找到「处在同一个环境状态」的步骤，把它们分成组，在组内比较，得到每一步的 advantage（作者自测比 GRPO 在 ALFWorld 上高 12% 以上）。

**(c) 长轨迹天然跨版本**。一条几十轮的轨迹可能要跑很久，异步训练下它的前几轮和后几轮来自不同的策略版本，处理方法见 [异步 Rollout](/posttrain/rl-async-rollout)。

### 6. 几个系统各自的重点

| 系统 | 重点 |
|---|---|
| verl agent loop | 推理服务化、每个 prompt 一个协程、token 进 token 出、sticky session、`response_mask` |
| slime | 自定义生成函数（`--custom-generate-function-path`），经 SGLang router 发请求，`loss_mask` 标出工具结果 |
| Agent Lightning | 训练和 agent 执行解耦：agent 照常调一个 OpenAI 风格的接口，服务端记录每次调用的 token，每次调用当一条样本 |
| SkyRL-Agent | 轨迹拆成启动环境、跑 agent、算奖励三个阶段，各自排队流水，作者自测比按批异步快 1.55 倍 |
| RollArt | 按任务特点分配硬件（prefill 重的放 H800，decode 重的放 H20）、轨迹级异步、奖励模型 serverless 化 |
| ROLL Flash | 环境级异步、冗余环境 rollout |

## 面试追问

::: details Q：多轮 RL 为什么要「token 进、token 出」？文本不是一样的吗？
文本一样，token 不一定一样。分词不唯一（同一个词可以切成不同的 token 组合）；工具调用被解析再渲染会改变空格甚至修正 JSON；chat template 可能删掉历史轮次的思考内容。重新分词后训练看到的 token 序列和采样时不同，等于在别的数据上算策略梯度，而且序列对不齐，没法用 importance sampling 修。所以推理服务要返回 token id，下一轮直接拼 token。
:::

::: details Q：工具返回的内容为什么不算 loss？
它不是策略的动作：没有策略概率，算 ratio 和 advantage 都没意义。训练模型预测工具输出，等于教它编造工具结果。所以只当上下文，`loss_mask = 0`。另外它通常很长，占回答部分的大头，算进去还会稀释真正的学习信号。
:::

::: details Q：环境调用要 10 秒，GPU 怎么不闲着？
让很多条轨迹同时跑：一条在等环境时，推理服务给别的轨迹生成。需要的并发轨迹数约为「想同时 decode 的序列数 ÷ 轨迹在生成上花的时间比例」。再配合环境级异步（不按批等环境）、冗余 rollout（凑够就停）、把重环境做成独立服务。
:::

::: details Q：多轮 rollout 怎么用上 prefix cache？
第 $k$ 轮的输入包含前面所有轮次，前缀和上一轮完全一样。只要同一条轨迹的后续轮次路由到同一个推理实例，前缀的 KV 就能命中，每轮只 prefill 新增的工具结果和指令。做法是 sticky session（按 request id 绑实例）或按前缀感知的路由。每次权重同步后要清掉这些 KV。
:::

## 手撕

**一个工具调用 agent 的 rollout 循环**（token 进 token 出，同时记 mask）：

```python
async def run_trajectory(llm, env, tok, prompt_ids, max_turns=10, max_len=32768):
    ids, mask, logps = list(prompt_ids), [0] * len(prompt_ids), [0.0] * len(prompt_ids)
    for _ in range(max_turns):
        out = await llm.generate(input_ids=ids, return_logprob=True)   # 只传 token id
        ids += out.token_ids; mask += [1] * len(out.token_ids); logps += out.logprobs
        call = parse_tool_call(tok.decode(out.token_ids))              # 解码只用来解析，不回写
        if call is None:
            return ids, mask, logps, "done"
        obs_ids = tok.encode(format_tool_result(await env.step(call)), add_special_tokens=False)
        ids += obs_ids; mask += [0] * len(obs_ids); logps += [0.0] * len(obs_ids)
        if len(ids) >= max_len:
            return ids, mask, logps, "truncated"                       # 训练时整条 loss 屏蔽
    return ids, mask, logps, "max_turns"
```

**两个估算**：

```python
def prefill_tokens(new_per_turn, turns, prefix_cache):
    if prefix_cache:
        return new_per_turn * turns
    return sum(new_per_turn * k for k in range(1, turns + 1))

def trajectories_needed(target_decoding, gen_fraction):
    return target_decoding / gen_fraction

prefill_tokens(2000, 10, False), prefill_tokens(2000, 10, True)   # 110000, 20000
trajectories_needed(256, 1 / 3)                                    # 768
```

常见题：解释 retokenization drift 和 token 进 token 出；写一个多轮 rollout 循环并正确生成 loss mask；估算多轮场景下 prefix cache 省多少 prefill；环境很慢时怎么保证 GPU 利用率。

## 参考

- [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) · [verl 多轮（SGLang）](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- [vLLM 博客：No More Retokenization Drift（Agent Lightning）](https://vllm.ai/blog/2025-10-22-agent-lightning)
- [Agent Lightning](https://arxiv.org/abs/2508.03680)
- [SkyRL-Agent](https://arxiv.org/abs/2511.16108)
- [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)
- [RollArt](https://arxiv.org/abs/2512.22560) · [ROLL Flash](https://arxiv.org/abs/2510.11345)
- [DeepSWE](https://huggingface.co/agentica-org/DeepSWE-Preview) · [rLLM](https://github.com/rllm-org/rllm)
- [GiGPO](https://arxiv.org/abs/2505.10978)
- [slime](https://github.com/THUDM/slime) · [SGLang Model Gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)

---
title: vLLM V1 架构与 SGLang
status: draft
tags: [vllm, sglang, architecture]
difficulty: 4
order: 4
related: [/framework/request-lifecycle, /inference/batching-scheduling, /inference/kv-cache-paged-attention, /framework/cuda-graph, /framework/serving-layer, /framework/vllm-release-history]
stack: [f-runner, s-cb, kv-prefix, sv-api]
---

# vLLM V1 架构与 SGLang

> EngineCore / scheduler / worker / executor 分层；RadixAttention + 前端语言

## 一句话结论

vLLM V1 是三类进程：**前端进程**（HTTP、tokenize、detokenize）、**EngineCore 进程**（调度循环 + KV block 管理）、**每张卡一个 worker 进程**（跑 model runner）。scheduler 不区分 prefill 和 decode，每一步只做一件事：在 token 预算内，让每个请求的「已算 token 数」追上「应有 token 数」。chunked prefill、prefix caching、投机解码都是这个规则的特例，两者默认开启。

SGLang 的分工类似（tokenizer 进程 / scheduler 进程 / detokenizer 进程），差别在 KV 用 radix tree 管前缀（RadixAttention），以及最早带了一套前端语言，让多轮、分支调用能共享前缀。两家在调度和 KV 上已高度趋同。

下文的类名和函数名都对照 vLLM 和 SGLang 的 main 分支源码（2026-10）。

## 推导

### 0. 先说角色和数字

| 进程 | 数量 | 做什么 |
|---|---|---|
| API server（前端） | 默认 1；开 DP 时默认等于 DP 数，可用 `--api-server-count` 改 | HTTP、chat template、tokenize、detokenize、流式返回 |
| EngineCore | 每个 DP rank 1 个 | scheduler、KV cache 管理、驱动 executor |
| GPU worker | 每个 EngineCore 下 TP × PP 个，一卡一个 | 加载权重、跑 forward、采样 |
| DP coordinator | 只在 DP > 1 时有 1 个 | DP rank 之间的负载均衡与同步 |

例：TP = 4 是 1 + 1 + 4 = 6 个进程；TP = 2、DP = 4 是 4 + 4 + 8 + 1 = 17 个（[vLLM 文档：Architecture Overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)）。TP = PP = 1 时 executor 是 `UniProcExecutor`，model runner 直接跑在 EngineCore 进程里，不另起 worker（默认值的判断见 [`config/parallel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)）。

为什么拆成多进程：Python 有 GIL，同一进程里同一时刻只有一个线程在跑 Python 字节码。tokenize、JSON、SSE 封包和调度循环都是 Python 代码，放在一个进程里会互相抢。decode 一步 15–25 ms，调度循环被卡住 5 ms 就是 20–30% 的吞吐损失（GIL 的细节见 [Python 并发](/basics/python)）。

### 1. 前端进程：AsyncLLM

API server 的 handler 调 [`AsyncLLM.generate()`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py)，它是一个 async generator。整理后的流程：

```python
async def generate(prompt, sampling_params, request_id):
    req = await input_processor.process_inputs_async(prompt, ...)   # 校验参数、tokenize（多模态还要预处理）
    q = RequestOutputCollector(request_id)                            # 这个请求专属的输出槽
    output_processor.add_request(req, q)                              # 登记 detokenizer 状态
    await engine_core.add_request_async(req)                          # msgpack 序列化，经 ZMQ 发给 EngineCore
    try:
        while True:
            out = q.get_nowait() or await q.get()
            yield out
            if out.finished: break
    except asyncio.CancelledError:          # 客户端断开时 handler 被取消
        await abort(request_id)             # 通知 EngineCore 释放这个请求的 KV
        raise

# 后台常驻一个 task，所有请求共用
async def output_handler():
    while True:
        outs = await engine_core.get_output_async()          # 一个 step 的所有请求的新 token id
        processed = output_processor.process_outputs(outs)   # 增量 detokenize、检查 stop string，按 request id 放进各自的 q
        if processed.reqs_to_abort:                          # 命中 stop string 的请求：前端判定结束，回头通知 EngineCore
            await engine_core.abort_requests_async(processed.reqs_to_abort)
```

两个细节：

- **stop string 在前端检查**。EngineCore 只看 token id，stop string 要在 detokenize 后的文本上匹配（可能跨 token 边界），所以由前端的 `OutputProcessor` 判定，再反向 abort。
- **消费慢了会合并**。`RequestOutputCollector.put()` 发现上一个输出还没被取走，就把新 token 合进去（DELTA 模式），不会无限堆积。

### 2. EngineCore：一个 busy loop

[`EngineCoreProc.run_busy_loop`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) 本身很短：

```python
def run_busy_loop(self):
    while running:
        self._process_input_queue()    # 没活干时阻塞等；有活时把 input_queue 里攒的 add / abort 全处理掉
        self._process_engine_step()    # step()，把输出放进 output_queue

def step(self):
    sched_out = self.scheduler.schedule()                                # 这一步谁算、各算几个 token
    future = self.model_executor.execute_model(sched_out, non_block=True)  # 发给 worker，不等
    grammar = self.scheduler.get_grammar_bitmask(sched_out)             # 结构化输出的 mask，和 GPU forward 并行算
    out = future.result()
    if out is None:
        out = self.model_executor.sample_tokens(grammar)                # forward 和采样分两次调用
    return self.scheduler.update_from_output(sched_out, out)            # 追加 token、判结束、释放 block
```

ZMQ 收发和 msgpack 编解码放在另外两个线程里（`process_input_sockets` / `process_output_sockets`），busy loop 只和两个线程内的 `queue.Queue` 打交道。

**异步调度**。上面的 `step()` 是串行的：CPU 调度第 $t+1$ 步时，GPU 在等。开启 async scheduling 后（条件允许时默认开，见 [`config/vllm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)），循环改走 `step_with_batch_queue()`：先把第 $t+1$ 步调度出来发给 GPU，再回头处理第 $t$ 步的结果。第 $t+1$ 步调度时还不知道第 $t$ 步采样出了什么 token，所以先给每个 decode 请求记一个占位 token（`num_output_placeholders`），等结果回来再填上。

### 3. Scheduler：只有一条规则

[`Scheduler.schedule()`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) 开头的注释（Woosuk Kwon 写的）就是整个设计：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. ... At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

**两个计数的含义**：`num_tokens_with_spec` = prompt 长度 + 已生成的 token 数 + 投机草稿 token 数；`num_computed_tokens` = 其中已经算过、KV 已经写进 cache 的个数。两者之差就是这一步要算的 token 数：

- 新请求，prompt 2000、prefix cache 命中 512：差 1488，这是 prefill；
- 正在 decode 的请求：差 1；
- 带 3 个草稿 token 的投机解码：差 4。

整理后的伪代码（省略了多模态、KV connector、结构化输出等分支）：

```python
def schedule():
    budget = max_num_scheduled_tokens            # token 预算，默认等于 max_num_batched_tokens
    # 1. 先排 running 队列（包括还没 prefill 完的请求）
    for req in running:
        n = min(req.num_tokens_with_spec - req.num_computed_tokens, budget)
        while (blocks := kv_mgr.allocate_slots(req, n)) is None:
            victim = running.pop()               # FCFS 策略：抢占最晚加入的那个
            preempt(victim)                      # 释放它的 block，num_computed_tokens = 0，放回 waiting 队首
            if victim is req: break
        if blocks is None: break
        scheduled[req] = n; budget -= n
    # 2. 有剩余预算且没有发生抢占，再从 waiting 队列拉新请求
    while waiting and budget > 0 and len(running) < max_num_seqs:
        req = waiting.peek()
        hit_blocks, n_hit = kv_mgr.get_computed_blocks(req)   # prefix cache 命中的前缀
        n = min(req.num_tokens - n_hit, budget)               # 放不下整个 prompt 就只算一段：这就是 chunked prefill
        blocks = kv_mgr.allocate_slots(req, n, hit_blocks)
        if blocks is None: break
        waiting.pop(); running.append(req)
        scheduled[req] = n; budget -= n
    return SchedulerOutput(scheduled, ...)
```

**被抢占的请求要重算**：`_preempt_request` 把 `num_computed_tokens` 清零，放回 waiting 队首。V1 只有 recompute，没有 swap 到 CPU。它释放的 block 内容和 hash 还在 free 队列里，重新调度时多半能被 prefix cache 命中，实际重算量往往小得多（见 [KV Cache：抢占](/inference/kv-cache-paged-attention#抢占-swap-vs-recompute)）。

**默认预算是多少**（[`engine/arg_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)，`vllm serve` 的值）：

| 显卡 | `max_num_batched_tokens` | `max_num_seqs` |
|---|---|---|
| 显存 ≥ 160 GB（B200 等） | 16384 | 1024 |
| 显存 ≥ 70 GB 且不是 A100（H100、H200） | 8192 | 1024 |
| 其他（含 A100） | 2048 | 256 |

A100 被单独排除，是因为调大预算在 A100 上反而降吞吐（[PR #17885](https://github.com/vllm-project/vllm/pull/17885)）。

**算一个例子**：H100，预算 8192，running 里有 200 个 decode 请求，来了一个 20000 token 的新 prompt（无命中）。

1. 第 1 步：200 个 decode 用掉 200，新 prompt 拿到 7992；
2. 第 2 步：它进了 running，和 200 个 decode 一起排，再拿 7992，累计 15984；
3. 第 3 步：剩下 4016 个 token 算完，这一步的输出就是它的第一个 token。

这 3 步里 200 个 decode 请求每步都照常出 token。代价是这几步每步要算 8192 个 token，decode 请求的 ITL 会从十几 ms 涨到几十 ms。预算越大 TTFT 越短、ITL 抖得越厉害，这是 chunked prefill 的核心取舍（见 [Chunked prefill](/inference/batching-scheduling#chunked-prefill)）。

**发给 worker 的是差量**。`SchedulerOutput` 里，第一次被调度的请求放完整数据（`NewRequestData`：prompt token、采样参数、block id）；之后只发 `CachedRequestData`：本步新增的 block id、要算几个 token。worker 自己缓存每个请求的状态（V1 博客称 persistent batch：「caches the input tensors and only applies the diffs to them at each step」，[vLLM V1 博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)）。

### 4. KV cache 管理与 prefix caching

调度器通过 [`KVCacheManager`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) 管 block，接口就是上面用到的 `get_computed_blocks` 和 `allocate_slots`。三个要点：

1. **block 大小**：CUDA 上默认 16 个 token 一个 block（`CacheConfig.DEFAULT_BLOCK_SIZE`）；
2. **block 的 hash 是链式的**：`hash_block_tokens(parent_hash, 本 block 的 token ids, extra_keys)`，见 [`kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)。把父 block 的 hash 算进来，相同的 16 个 token 只有在前缀也完全相同时才算命中。`extra_keys` 放 LoRA id、多模态输入的 hash 等，防止「token 相同但含义不同」的误命中；
3. **淘汰用 LRU 双向链表**：`FreeKVCacheBlockQueue` 把引用数归零的 block 按最近使用时间排好。分配新 block 时从队头拿，拿到的 block 如果还挂着旧 hash，就把它从 hash 表里摘掉。

`enable_prefix_caching` 默认是 `True`。V1 博客称 hit rate 为 0% 时吞吐下降不到 1%（作者自测）。完整的分配 / 命中 / 释放伪代码见 [Prefix caching 伪代码](/inference/kv-cache-paged-attention#prefix-caching-伪代码)。

### 5. Executor 与 worker：一步 forward 里发生了什么

**Executor** 决定 worker 怎么起（[`executor/abstract.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/executor/abstract.py) 的 `Executor.get_class`）：

- `uni`：单卡，`UniProcExecutor`，在本进程里跑；
- `mp`：单机多卡默认，`MultiprocExecutor`，每卡起一个 `WorkerProc`。`SchedulerOutput` 通过共享内存的消息队列（`rpc_broadcast_mq`）广播给所有 rank；
- `ray`：多机。

V0 里 scheduler 和 rank 0 的 worker 在同一个进程，rank 0 要额外负责把输入广播给其他 rank，是不对称的。V1 把 scheduler 拆出去后，每个 rank 都只收差量、做一样的事，博客称为 "symmetric architecture"。

**Worker 里**（[`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) 的 `GPUModelRunner.execute_model`），按顺序：

1. `_update_states(scheduler_output)`：把差量应用到常驻的 `InputBatch`（加入新请求、删掉结束的、追加 block id）；
2. `_prepare_inputs`：把本步所有请求的 token 拍平成一维 `input_ids[num_tokens]`，算出每个 token 的 position、slot mapping（KV 写到哪个 slot）和 attention metadata；
3. 决定这一步用哪种 CUDA Graph、pad 到哪个桶（见 [CUDA Graph：三种图](/framework/cuda-graph#_5-三种图)）；
4. 跑模型 forward，再对每个请求最后一个位置的 hidden state 算 logits；
5. 之后 `sample_tokens` 单独调用：应用 grammar mask、采样，结果拷回 CPU 给 scheduler。

### 6. SGLang 对照

**进程分工**（[SGLang 源码](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers)）：

| | vLLM V1 | SGLang |
|---|---|---|
| tokenize | API server 进程（`AsyncLLM` + `InputProcessor`） | HTTP 主进程里的 `TokenizerManager` |
| 调度 | EngineCore 进程，全局只有一个 scheduler | 每个 TP rank 一个 `Scheduler` 进程，模型 worker（`TpModelWorker`）就在 scheduler 进程里 |
| detokenize | API server 进程（`OutputProcessor`） | 独立的 `DetokenizerManager` 进程 |
| CPU / GPU 重叠 | async scheduling，`step_with_batch_queue` | overlap scheduler，`event_loop_overlap` |

SGLang 的调度循环（[`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) 的 `event_loop_overlap`，整理后）：

```python
while True:
    ingest_requests()                               # 从 TokenizerManager 收新请求
    batch = get_next_batch_to_run()                 # 先试着拼新的 prefill batch，拼不出来再继续 decode
    if batch:
        result = run_batch(batch)                   # 发到 GPU，不等结果
        result_queue.append((batch, result))
    if last_batch:
        process_batch_result(*result_queue.popleft())  # 处理上一批：CPU 这段时间 GPU 在跑当前批
    last_batch = batch
```

SGLang v0.4 的博客称这个「zero-overhead batch scheduler」默认开启，比 v0.3 吞吐高 1.1×（作者自测，[SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)）。

**RadixAttention**（Zheng et al., 2023，[arXiv:2312.07104](https://arxiv.org/abs/2312.07104)）：KV 不按 hash 表组织，而是一棵前缀树，每个节点存一段 token 序列和对应的 KV 位置。[`RadixCache`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) 的接口：

- `match_prefix`：从根往下走，返回能复用的最长前缀；
- `insert`：请求结束（或 prefill 完一段）后把它的 token 和 KV 挂到树上；
- `inc_lock_ref` / `dec_lock_ref`：正在用的节点加锁，不能被淘汰；
- `evict`：显存不够时，从没加锁的叶子开始按 LRU 删。

waiting 队列的排序由 `--schedule-policy` 决定：当前默认是 `fcfs`；`lpm`（longest prefix match）按命中前缀长度排，让共享前缀的请求挨着跑、提高命中率（选项见 [`arg_groups/fields/schedule.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/schedule.py)）。和 vLLM block hash 方案的逐项对比见 [RadixAttention：另一种组织方式](/inference/kv-cache-paged-attention#radixattention-另一种组织方式)。

**前端语言**：论文的另一半是一套嵌在 Python 里的 DSL，用 `gen`、`fork`、`select` 这类原语写多次调用的 LLM 程序。runtime 能看到整个程序的结构，`fork` 出的分支天然共享前缀，正好被 radix tree 复用。论文报告在 agent、few-shot、JSON 解码、多轮对话等任务上吞吐最高提升 6.4×（作者自测）。如今多数用户只用 SGLang 的 OpenAI 兼容 server，前端语言用得不多，但 radix cache 和调度器的设计保留了下来。

## 面试追问

::: details Q：V1 为什么把 EngineCore 单独放一个进程？
V0 里 HTTP 处理、tokenize、调度和 forward 都在同一个 Python 进程里，GPU 在 CPU 做这些事时空转。拆开后前端进程的 tokenize / detokenize 和 EngineCore 的调度、GPU 执行可以并行；EngineCore 内部还把 ZMQ 收发放在独立线程里。代价是进程间多一次 msgpack 序列化，但传的主要是 token id 和差量，很小。
:::

::: details Q：V1 的 scheduler 为什么不区分 prefill 和 decode？有什么好处？
它只维护「已算 token 数」和「应有 token 数」，每步把差额（受预算限制）分出去。prefill 是差额很大，decode 是差额为 1，chunked prefill 是差额被预算截断，prefix cache 命中是 `num_computed_tokens` 起点不为 0，投机解码是差额为 1 + 草稿数。一条规则覆盖所有情况，不再需要 V0 那样的 prefill batch、decode batch 两套逻辑，混合 batch 也是自然结果。
:::

::: details Q：抢占时为什么抢 running 队列里最后一个？
FCFS 策略下 running 按加入顺序排，最后一个是最晚来的，已投入的计算最少，重算代价最低，也符合先来先服务的公平性。PRIORITY 策略下抢的是优先级最低（数值最大）、同优先级里最晚到的那个。
:::

::: details Q：async scheduling 解决什么问题？有什么代价？
串行 step 里 CPU 调度、准备输入、处理输出时 GPU 空闲；decode 一步只有十几毫秒，这几毫秒的空隙占比可观。异步调度让第 $t+1$ 步的调度和第 $t$ 步的 GPU 执行重叠。代价是调度 $t+1$ 时还不知道 $t$ 步采出的 token：要先用占位 token 记账；请求如果在第 $t$ 步就结束了，第 $t+1$ 步给它算的那个 token 就白算了。所以部分投机解码方法等场景会自动关掉它。
:::

::: details Q：vLLM 和 SGLang 的 prefix cache 哪个命中率高？
理论上 radix tree 能按 token 粒度匹配，block hash 只能按整 block（16 token）匹配，差别只在不满一个 block 的尾巴，通常不大。实际命中率更多取决于淘汰策略、显存大小，以及多副本时路由能不能把相同前缀送到同一台机器（见 [服务层：多副本路由](/framework/serving-layer)）。
:::

## 手撕

1. 画出一个请求在 V1 里从 HTTP 到第一个 token 经过的进程和队列：API server 的 handler → `AsyncLLM.generate` → ZMQ → EngineCore 输入线程 → `input_queue` → busy loop 的 `schedule()` → executor 广播 → worker forward + sample → `update_from_output` → `output_queue` → 输出线程 → ZMQ → `output_handler` → 该请求的 `RequestOutputCollector` → SSE。逐步耗时见 [请求生命周期](/framework/request-lifecycle)，版本演进见 [vLLM 发布史](/framework/vllm-release-history)。
2. 实现一个简化版 `schedule()`：输入 running / waiting 两个列表、token 预算、空闲 block 数（block 大小 16），输出本步每个请求算几个 token、谁被抢占。用上面「200 个 decode + 20000 token 新 prompt」的例子验证 3 步算完 prefill。

## 参考

- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（2025-01）
- [vLLM 文档：Architecture Overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)
- [vLLM 文档：Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- vLLM 源码：[`v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)、[`v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)、[`v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（NeurIPS 2024）
- [SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- SGLang 源码：[`managers/scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[`mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)

---
title: 服务层：异步请求、Batching 队列、Streaming、多副本路由
status: draft
tags: [serving, asyncio, routing]
difficulty: 3
order: 5
related: [/framework/request-lifecycle, /framework/vllm-v1-architecture, /basics/python, /basics/os-network, /basics/system-design-llm-serving, /inference/batching-scheduling]
stack: [sv-api, sv-stream, d-route]
---

# 服务层：异步请求、Batching 队列、Streaming、多副本路由

> API server 到 engine 的边界

## 一句话结论

服务层是 HTTP 连接和 engine 之间的边界，它不组 batch，只做四件事：

1. **收**：用 asyncio 在一个线程里挂住上千个连接，tokenize 后把请求交给 engine；
2. **限**：准入控制。队列太长就直接拒绝，而不是让所有人一起超时；
3. **推**：engine 每步产出的 token id 按请求分发、增量 detokenize，用 SSE 推回各自的连接；
4. **断**：客户端断开要一路通知到 engine 释放 KV；engine 挂了要让每个等待中的连接都收到错误。

多副本时前面再加一层 router，按负载和 prefix 命中率选副本。

## 推导

### 0. 边界：谁做什么

| | API server（服务层） | engine |
|---|---|---|
| 并发模型 | asyncio 事件循环，一个连接一个协程 | 一个 busy loop，每步处理一个 batch |
| 做什么 | HTTP 解析、参数校验、chat template、tokenize、detokenize、SSE | 调度、KV 管理、forward、采样 |
| 状态 | 每个请求一个输出队列 | waiting / running 队列、KV block |
| 以什么为单位 | 请求 | step（一次 forward） |

两边在不同进程里（vLLM V1：API server 进程 ↔ ZMQ ↔ EngineCore 进程），原因是 GIL：两边都是 Python，同一进程里会互相抢（见 [Python 并发](/basics/python)、[vLLM V1 架构](/framework/vllm-v1-architecture)）。

### 1. 异步请求：一个线程服务上千个连接

**定义**。asyncio 是单线程的协作式调度：事件循环轮流运行各个协程，协程只在 `await` 处让出。一个 LLM 请求 99% 的时间在等 engine 出 token，适合用协程挂着，不必一个连接一个线程。

**代价是事件循环里不能有长时间不 `await` 的同步代码**。长 prompt 的 tokenize 要几到几十毫秒，在 handler 里直接调，这段时间所有连接都停住。所以 tokenize 要么放进线程池（HF tokenizers 是 Rust 实现，计算时释放 GIL），要么放进独立进程。

**API server 自己也会成为瓶颈**。它的 CPU 负载可以这样估：

$$
\text{CPU 占用（核）} \approx \text{并发流数} \times \text{每流每秒 token 数} \times \text{每个 token 的处理时间}
$$

每个 token 的处理包括 detokenize、stop string 检查、JSON 序列化、写 socket。假设每个 token 30 μs（量级假设，实际用 profiler 测）：1000 条并发流、每条 50 token/s，就是 $1000 \times 50 \times 30\ \mu s = 1.5$ 核。单进程的事件循环最多吃满 1 核，于是 ITL 开始被前端拖慢。两个缓解手段：

1. **多个 API server 进程**：vLLM 的 `--api-server-count`，每个 API server 都连到所有 EngineCore（[vLLM 文档：Architecture Overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)）；
2. **攒几个 token 再推**：vLLM 的 `stream_interval`（默认 1）设成 $k$，每 $k$ 个 token 推一次，每 token 的分摊开销降为约 $1/k$，代价是客户端看到的输出更「一顿一顿」（[`config/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)）。

vLLM 的 handler 结构（整理自 [`serve/utils/api_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/utils/api_utils.py) 和 [`AsyncLLM.generate`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py)）：

```python
@router.post("/v1/chat/completions")
@with_cancellation                      # 同时 await 两件事：handler 本身、HTTP 断开消息；断开先到就 cancel handler
async def chat(req: ChatRequest, raw_request: Request):
    prompt = render_chat_template(req.messages)
    gen = engine.generate(prompt, sampling_params(req), request_id=new_id())  # async generator
    if req.stream:
        return StreamingResponse(sse(gen), media_type="text/event-stream")
    final = None
    async for out in gen:               # 非流式：等到最后一个输出
        final = out
    return to_json(final)
```

### 2. 队列：API 层不组 batch，只做准入

**batch 在哪组**：在 engine 的 scheduler 里，每一步按 token 预算和 KV 余量重新组（见 [continuous batching](/inference/batching-scheduling)）。API 层如果自己攒 batch（比如等 10 ms 凑够 8 个再发），只会白白增加 TTFT。continuous batching 下新请求随时能插进下一步，不需要攒。

**准入控制为什么必要**。用 Little 定律（$L = \lambda W$，排队系统里的平均请求数 = 到达率 × 平均停留时间）看过载：

- 服务能力 $\mu = 20$ req/s，到达率 $\lambda = 25$ req/s，队列每秒净增 5 个；
- 过载 60 s 后队列里有 300 个请求，新来的请求要排 $300 / 20 = 15$ s 才轮到 prefill；
- 如果 TTFT 的 SLO 是 2 s，这 300 个请求全都超时。它们白占了队列，还把后来的请求一起拖下水。

所以队列长度要设上限，粗略取 $\mu \times \text{TTFT SLO}$：上例是 $20 \times 2 = 40$ 个。超过就立刻拒绝，客户端可以重试别的副本。

vLLM 默认**不设上限**（waiting 队列无界）。可以用两个参数开启（[`config/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)）：

- `max_num_queued_reqs`：在途请求数（waiting + running）的上限，在 API server 进程里统计，超过返回 HTTP 503。文档建议设成约 `data_parallel_size × max_num_seqs` 加上期望的排队深度；
- `max_num_queued_tokens`：处于 prefill 阶段的请求 prompt token 总数上限，超过也返回 503，用来保 TTFT。

429（Too Many Requests）和 503（Service Unavailable）的区别是语义上的：429 指「你这个调用方超了配额」，在网关按用户限流时用；503 指「服务整体忙」，适合容量满时用。

### 3. Streaming：SSE 与增量 detokenize

**SSE**（Server-Sent Events，[HTML 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)）就是一个不关闭的 HTTP 响应，`Content-Type: text/event-stream`，每个事件是一行 `data: ...` 加一个空行。OpenAI 兼容接口的流长这样：

```text
data: {"choices":[{"delta":{"content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: [DONE]

```

HTTP 层的细节（chunked 编码、代理缓冲）见 [HTTP 与 SSE](/basics/os-network#_7-http-与-sse-llm-api-怎么流式返回)。

**增量 detokenize 的两个坑**：

1. **半个字符**。byte-level BPE 的 token 是字节序列。「你」的 UTF-8 编码是 3 个字节 `E4 BD A0`，可能被切成两个 token。只解码第一个 token 会得到乱码 `�`。正确做法是维护一个解码状态：新 token 拼上去后，只输出已经是完整字符的部分，残缺的字节留到下一个 token。vLLM 对 fast tokenizer 直接用 `tokenizers` 库的 `DecodeStream` 做这件事（[`v1/engine/detokenizer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/detokenizer.py)）。
2. **stop string 跨 token**。stop 是 `"\n\nUser:"`，它可能由 3 个 token 拼成。前两个 token 到了时还不能确定是不是 stop，如果已经推给客户端，后面就收不回来了。所以要**扣住**最后 $\max_s(\text{len}(s)) - 1$ 个字符不发（vLLM 的 `stop_buffer_length`），确认不是 stop 的开头再放出。匹配上了就截断文本、结束请求，并通知 engine abort。

### 4. 取消与错误：两个方向都要传

**客户端断开 → engine**（vLLM 的实现链）：

1. `with_cancellation` 收到 ASGI 的 `http.disconnect` 消息，cancel 掉 handler 协程；
2. `AsyncLLM.generate` 里的 `await` 抛出 `asyncio.CancelledError`，`except` 分支调 `abort(request_id)`；
3. abort 消息经 ZMQ 到 EngineCore，scheduler 把请求移出队列、释放 KV block。

漏掉任何一环，客户端走了 GPU 还会继续生成到 `max_tokens`。用户刷新页面很常见，这部分浪费不可忽视。

**engine 出错 → 客户端**：

- **单个请求出错**：engine 的输出里带上这个 request id 的错误，前端放进它的输出队列，协程 `get()` 时抛出，handler 返回错误；
- **engine 进程挂了**：前端检测到后让所有等待中的请求抛 `EngineDeadError`，不能让协程永远挂在队列上。`/health` 端点此时返回 503，让负载均衡器把这个副本摘掉（[`serve/instrumentator/health.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/instrumentator/health.py)）。

### 5. 多副本路由

一个副本是一组卡上跑的一个完整 engine。副本之间不共享 KV cache，所以**同一个前缀落在哪个副本上，决定了它能不能命中 prefix cache**。

**随机或轮询为什么不够**：假设有 $N$ 个副本，一个多轮会话的历史 KV 只在上一轮服务它的那个副本上。随机路由下一轮落到同一副本的概率是 $1/N$：8 个副本时只有 12.5%，其余 87.5% 要把整段历史重新 prefill。

**三类策略**：

| 策略 | 怎么选 | 问题 |
|---|---|---|
| 轮询 / 随机 | 不看状态 | 命中率约 $1/N$ |
| 最少负载（如 power-of-two：随机挑 2 个，选在途请求少的那个） | 看队列长度 | 不看 cache |
| cache-aware | 看前缀匹配，负载不均时退回最少负载 | 热点前缀会把一个副本打爆，所以必须有负载兜底 |

**SGLang 的 cache-aware 路由**（[`sgl-model-gateway/src/policies/cache_aware.rs`](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)，[文档](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)）。router 不去问各副本的真实 cache 状态，而是根据自己发过的请求，为每个副本维护一棵**近似**的 radix tree。树上存的是原始文本字符，不是 token id，省掉 router 里的 tokenize。决策过程：

```python
def select_worker(text, workers):
    loads = [w.in_flight for w in workers]
    imbalanced = (max(loads) - min(loads) > balance_abs_threshold      # 默认 64
                  and max(loads) > min(loads) * balance_rel_threshold) # 默认 1.5
    if imbalanced:
        return argmin(loads)                         # 负载差太大：只看负载
    worker, matched = tree.longest_prefix_match(text)   # 哪个副本见过最长的前缀
    if matched / len(text) > cache_threshold:        # 默认 0.3
        w = worker                                   # 命中够多：发给它
    else:
        w = argmin(loads)                            # 命中太少：发给最闲的
    tree.insert(text, w)                             # 记下「这段前缀现在在 w 上」
    return w
```

（文件头注释写的是命中不足时选「树最小的副本」，当前代码实际选的是负载最小的，以代码为准。）树按 LRU 定期淘汰叶子，`--eviction-interval-secs` 默认 120 s。SGLang v0.4 博客报告，这个路由把命中率从 20% 提到 75%，吞吐最高提升 1.9×（作者自测，[SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)）。

**PD 分离时**，router 还要为每个请求选一对 prefill 实例和 decode 实例。两边可以用不同策略，SGLang 文档的例子是 prefill 侧 `cache_aware`（prefill 才吃 prefix cache）、decode 侧 `power_of_two`。PD 分离本身见 [PD 分离](/inference/batching-scheduling#pd-分离)。

vLLM 生态里对应的组件有 [vllm-project/router](https://github.com/vllm-project/router) 和 [production-stack](https://github.com/vllm-project/production-stack)，整体系统设计见 [LLM serving 系统设计](/basics/system-design-llm-serving)。

## 面试追问

::: details Q：engine 挂了一个请求，API server 怎么知道？
engine 的输出流里要有 per-request 的结束 / 错误事件，API 层按 request id 分发到对应协程；同时要有 engine 进程的健康检查，进程挂了就让所有等待中的协程抛错（vLLM 是 `EngineDeadError`），`/health` 返回 503。生产上还要让 abort（客户端断开）反向传给 engine 释放 KV。
:::

::: details Q：API 层为什么不攒一批请求再发给 engine？
continuous batching 下 engine 每一步都重新组 batch，新请求下一步就能加入。API 层攒 batch 只会增加等待时间，batch 的效率由 scheduler 的 token 预算决定，跟请求是不是同时到达无关。API 层该做的是准入控制，不是 batching。
:::

::: details Q：按 session id 做粘性路由（sticky）有什么问题？
一是热点：一个超长会话或一个大客户的所有流量都压在同一个副本上；二是副本扩缩容或重启时，映射整体失效。cache-aware 路由的做法是把「命中率」当成偏好而不是硬约束，负载不均时让位给负载均衡；一致性 hash 则让扩缩容时只有约 $1/N$ 的映射变化。
:::

::: details Q：为什么流式输出用 SSE，不用 WebSocket？
LLM 输出是单向的（服务端 → 客户端），SSE 就是普通 HTTP 响应，现有的负载均衡、鉴权、代理都直接能用；WebSocket 是双向长连接，要额外的协议升级和连接管理。OpenAI 兼容接口也约定用 SSE。需要双向实时交互（如语音对话中途打断）时才考虑 WebSocket。
:::

::: details Q：stop string 扣住几个字符不发？会影响什么？
扣住 $\max_s \text{len}(s) - 1$ 个字符：最坏情况下 stop 的前 $\text{len}(s) - 1$ 个字符已经生成，最后一个还没到。影响是客户端看到的输出总是滞后几个字符，请求结束时再一次性放出。stop 越长、滞后越多；不需要从输出中去掉 stop 时（`include_stop_str_in_output=True`）就不用扣。
:::

## 手撕

设计一个把多个并发 HTTP 请求喂给单线程 engine 的异步架构：每个请求一个输出队列，一个后台协程按 request id 分发 engine 的输出，客户端取消时通知 engine。

```python
import asyncio, itertools

class FakeEngine:
    """模拟 engine：每 step 给每个在跑的请求产出 1 个 token。真实场景它在另一个进程里。"""
    def __init__(self):
        self.running = {}                        # rid -> 剩余 token 数
    def add(self, rid, n_tokens): self.running[rid] = n_tokens
    def abort(self, rid): self.running.pop(rid, None)
    def step(self):
        outs = []
        for rid in list(self.running):
            self.running[rid] -= 1
            done = self.running[rid] == 0
            if done: del self.running[rid]
            outs.append((rid, f"tok{self.running.get(rid, 0)}", done))
        return outs

class Server:
    def __init__(self, engine, max_inflight=40):
        self.engine, self.queues = engine, {}
        self.max_inflight = max_inflight         # 准入上限，约 μ × TTFT SLO
        self.ids = itertools.count()

    async def engine_loop(self):                 # 后台常驻：驱动 engine 并分发输出
        while True:
            for rid, tok, done in self.engine.step():
                if (q := self.queues.get(rid)) is not None:
                    q.put_nowait((tok, done))
            await asyncio.sleep(0.02)            # 模拟一步 20 ms；真实场景是 await IPC

    async def generate(self, n_tokens):
        if len(self.queues) >= self.max_inflight:
            raise RuntimeError("503: server busy")       # 准入控制：直接拒，不排队
        rid = next(self.ids)
        q = self.queues[rid] = asyncio.Queue()
        self.engine.add(rid, n_tokens)
        try:
            while True:
                tok, done = await q.get()        # 在自己的队列上等，不阻塞别人
                yield tok
                if done: return
        except asyncio.CancelledError:           # 客户端断开
            self.engine.abort(rid)               # 通知 engine 释放资源
            raise
        finally:
            self.queues.pop(rid, None)

async def main():
    s = Server(FakeEngine())
    loop_task = asyncio.create_task(s.engine_loop())
    async def client(n):
        return [t async for t in s.generate(n)]
    print(await asyncio.gather(client(3), client(5)))
    loop_task.cancel()

asyncio.run(main())
```

要点：(1) engine 只有一个，所有请求共享；(2) 每个请求只在自己的 `asyncio.Queue` 上 `await`；(3) 取消时 `abort` 必须走到 engine；(4) `finally` 里删掉队列，防止泄漏。真实系统把 `FakeEngine.step` 换成跨进程 IPC，把 `put_nowait` 换成「消费慢时合并输出」的槽（vLLM 的 `RequestOutputCollector`）。各阶段耗时见 [请求生命周期](/framework/request-lifecycle)，完整系统设计见 [LLM serving 系统设计](/basics/system-design-llm-serving)。

## 参考

- [vLLM 文档：OpenAI-Compatible Server](https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md)
- [vLLM 文档：Architecture Overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)
- vLLM 源码：[`v1/engine/async_llm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py)、[`v1/engine/detokenizer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/detokenizer.py)、[`config/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- [SGLang 文档：SGLang Model Gateway（router）](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)
- [SGLang v0.4 博客：cache-aware load balancer](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [HTML 规范：Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)

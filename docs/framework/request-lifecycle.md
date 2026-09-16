---
title: 一个 Request 的全链路
status: draft
tags: [lifecycle, end-to-end]
difficulty: 4
order: 6
related: [/framework/vllm-v1-architecture, /inference/batching-scheduling, /basics/system-design-llm-serving]
---

# 一个 Request 的全链路

> 从 HTTP 进来到第一个 token 出去

## 一句话结论

十一步：**HTTP 到达 → 校验与 tokenize → 进 waiting 队列 → 调度器分配 KV block → 组 batch → 前向（prefill）→ 采样 → 第一个 token 流出（TTFT 到此结束）→ 之后每轮 decode 一个 token → 结束条件触发 → 释放 block**。能按这个顺序讲清每一步在哪个进程、花多少时间、可能卡在哪，这道题就满分。

## 推导

### 1. HTTP 到达（API server 进程，CPU）

FastAPI/uvicorn 的 async handler 收到 `/v1/chat/completions`。这里是 asyncio 事件循环，**绝对不能做阻塞操作**，否则整个 server 的所有连接一起卡住。

### 2. 校验与模板渲染

参数校验（`max_tokens`、`temperature` 范围）、chat template 渲染（Jinja2 把 messages 拼成一个字符串）。模板渲染是纯 CPU，长对话下可能花几毫秒。

### 3. Tokenize

`tokenizer.encode`。这是 CPU 密集操作，长 prompt 下可达几毫秒到几十毫秒。**必须放到线程池**（tokenizer 是 Rust 实现，会释放 GIL）或独立进程，否则阻塞事件循环。vLLM V1 把 tokenize 放在独立的前端进程里，正是为了这个。

### 4. 进队列

生成 request id，构造内部的 request 对象（token ids、采样参数、到达时间），通过 IPC（V1 用 ZMQ）送给 EngineCore 进程，挂到 waiting 队列。

**进程边界在这里**：前端（HTTP + tokenize + detokenize）和 EngineCore（调度 + 执行）分成两个进程，让 CPU 上的序列化工作和 GPU 上的调度循环并行，互不阻塞。

### 5. 调度（EngineCore 的主循环，每轮一次）

调度器决定这一轮跑哪些请求：
- 先保证 running 的请求各有 1 个 token 的预算；
- 剩余 token 预算分给 waiting 队列的 prefill（开了 chunked prefill 就切成 chunk）；
- 为新请求分配 KV block，**先查 prefix cache**，命中的 block 直接引用（refCount++），只为未命中的部分分配新 block；
- block 不够就抢占某个 running 请求。

产出一个 batch 描述：哪些序列、各自的 token、block table。

### 6. 准备输入（CPU → GPU）

把 token ids、position ids、block table、序列长度等打包成 tensor，`H2D` 拷贝到 GPU。这一步的开销在小 batch 下不可忽视，所以要用 pinned memory 和异步拷贝，并尽量和上一轮的计算 overlap。

### 7. 前向（GPU，worker 进程 × TP）

TP 下每张卡是一个 worker 进程，各自执行相同的计算图，在每层的两个点做 all-reduce。attention 用 paged kernel 按 block table gather KV。

decode 阶段通常用 **CUDA Graph** 重放：因为 decode 每步的计算图形状固定（只有序列长度在变，用固定的 bucket 对齐），可以提前捕获整张图，避免每步几百次 kernel launch 的 CPU 开销。prefill 形状多变，一般不用 graph。

### 8. 采样

拿到 logits `[B, V]`，依次应用：logit bias、重复惩罚、temperature、top-k、top-p，然后 multinomial 采样。这一步全在 GPU 上做，但 vocab 大时（128k）也要几百微秒。采样结果 `D2H` 拷回 CPU。

**这里有个同步点**：必须等 GPU 算完才知道采样了什么 token。好的实现会把这次同步和下一轮的输入准备 overlap（异步输出处理）。

### 9. 第一个 token 流出 —— TTFT 到此为止

EngineCore 把 token id 送回前端进程，前端 detokenize（增量解码，要处理多字节 UTF-8 和 BPE 的部分 token 问题），封成 SSE event 推给客户端。

**TTFT 的构成**：排队时间 + tokenize + prefill 计算 + 采样 + detokenize + 网络。高负载下排队时间往往是大头，而不是计算。

### 10. Decode 循环

之后每一轮重复 5–9，但只算 1 个 token，KV cache 从上一轮继续追加。相邻两个 token 之间的间隔就是 **ITL**，平均值是 **TPOT**。

### 11. 结束与释放

触发条件：采样到 EOS、达到 `max_tokens`、命中 stop string、客户端断开。请求标记 finished，KV block 释放（refCount--），满 block 的 hash 保留在 prefix cache 表里供后续命中。

### 时间都花在哪（70B TP=4、2k prompt、H100）

| 阶段 | 典型耗时 |
|---|---|
| tokenize | 1–5 ms |
| 排队（高负载） | 0 – 数百 ms |
| prefill 计算 | 80–150 ms |
| 采样 + detokenize | < 1 ms |
| 每步 decode | 15–25 ms |

## 面试追问

::: details Q：为什么要把 API server 和 engine 拆成两个进程？
tokenize、detokenize、JSON 序列化、SSE 封包都是 CPU 密集的，放在同一个 Python 进程里会和调度循环抢 GIL。decode 每 20 ms 一轮，如果调度循环被一次 detokenize 卡住 5 ms，就损失 25% 的吞吐。拆进程后两边各跑各的事件循环，通过 ZMQ 传 token id（很小），CPU 工作真正并行。
:::

::: details Q：CUDA Graph 为什么只用在 decode？
CUDA Graph 要求计算图的形状固定才能重放。decode 每步只有 1 个 token，形状只随 batch size 和序列长度变化，可以用有限个 bucket（比如 batch 1,2,4,8,...）各捕获一张图。prefill 的 token 数千变万化，捕获数量爆炸，且 prefill 本身 kernel 少、单个 kernel 耗时长，launch 开销占比小，不值得。
:::

::: details Q：增量 detokenize 有什么坑？
BPE 的一个 token 可能是半个 UTF-8 字符，单独 decode 会出乱码。正确做法是维护已解码的前缀，每次用新 token 重新解码一小段尾部并做 diff，只把新增的合法字符推给客户端。还要处理 stop string 跨 token 边界的情况（stop 词被切成两个 token 时要能匹配上），这需要在字符串层面而不是 token 层面做检查。
:::

::: details Q：哪一步最可能成为瓶颈？
按出现频率：(1) 高负载下的排队，解法是扩容或准入控制；(2) 长 prompt 的 prefill 阻塞 decode，解法是 chunked prefill；(3) 小 batch 下的 kernel launch 开销，解法是 CUDA Graph；(4) 前端的 tokenize/detokenize 抢 GIL，解法是拆进程。用 torch profiler 看 GPU 时间线上的空隙就能区分是 GPU 慢还是 CPU 喂不饱。
:::

::: details Q：请求取消时要做什么？
把序列从 running 集合移除、释放它的 KV block、如果它正在当前 batch 里要等这一轮跑完（不能中途抽走 tensor）。链路上每一层都要能传递取消信号：HTTP disconnect → 前端 → IPC → 调度器。漏了任何一环，客户端断开后 GPU 还会继续生成到 max_tokens。
:::

## 参考

- [vLLM V1 架构博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [SGLang 架构文档](https://docs.sglang.ai/)

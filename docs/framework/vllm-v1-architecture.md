---
title: vLLM V1 架构与 SGLang
status: todo
tags: [vllm, sglang, architecture]
difficulty: 4
order: 4
related: []
stack: [f-runner, s-cb, kv-prefix, sv-api]
---

# vLLM V1 架构与 SGLang

> EngineCore / scheduler / worker / executor 分层；RadixAttention + 前端语言

## 一句话结论

vLLM V1 把引擎拆成三个进程层：前端进程做 tokenize / detokenize 和 HTTP，EngineCore 进程跑 scheduler 循环，worker 进程（每张卡一个）跑 model runner；scheduler 用统一的 token 预算同时处理 prefill 和 decode，prefix caching 默认开启。SGLang 的差别在前端有一套编程语言（结构化多轮调用）和 RadixAttention 的树状 prefix cache。

## 推导

- **分层**：`AsyncLLM`（前端）→ `EngineCore`（调度 + KV 管理）→ `Executor`（多卡编排）→ `Worker` / `ModelRunner`（forward + sampling）；前端和 EngineCore 之间用 ZMQ，CPU 工作和 GPU 工作在不同进程里重叠。
- **scheduler**：每步给每个请求分配本步算多少 token（decode 是 1，prefill 可以是一个 chunk），不再区分 prefill / decode 两种 batch，chunked prefill 天然支持；见 [调度](/inference/batching-scheduling)。
- **KV 管理**：block 级 hash 做 prefix caching，命中的 block 直接复用；抢占只用 recompute 不再 swap。
- **SGLang**：RadixAttention 用 radix tree 管理前缀，LRU 淘汰叶子；前端 DSL 让多轮、分支、结构化输出的调用能共享前缀；两家在调度和 KV 上的思路已高度趋同。

## 面试追问

::: details Q：V1 为什么把 EngineCore 单独放一个进程？
V0 里 HTTP 处理、tokenize、调度和 forward 在同一个 Python 进程里串行，GPU 在 CPU 做这些事时空转。拆开后前端进程的 tokenize / detokenize 和 EngineCore 的调度、GPU 执行可以并行，GPU 利用率明显提升；代价是进程间多一次序列化。
:::

## 手撕

常见题：画出一个请求在 V1 里从 HTTP 到第一个 token 经过的进程和队列；见 [请求生命周期](/framework/request-lifecycle)，版本演进见 [vLLM 发布史](/framework/vllm-release-history)。

## 参考

- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)

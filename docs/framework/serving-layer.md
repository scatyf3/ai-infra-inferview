---
title: 服务层：异步请求、Batching 队列、Streaming、多副本路由
status: todo
tags: [serving, asyncio, routing]
difficulty: 3
order: 5
related: []
stack: [sv-api, sv-stream, d-route]
---

# 服务层：异步请求、Batching 队列、Streaming、多副本路由

> API server 到 engine 的边界

## 一句话结论

服务层是 HTTP 请求和 engine 之间的边界：API server 用 asyncio 接请求、tokenize、把请求塞进 engine 的输入队列，engine 每一步产出 token 后通过输出队列流式推回对应的连接；多副本时前面再加一层按负载或 prefix 命中率路由的 router。

## 推导

- **异步请求**：每个连接一个协程，`await` 在 engine 的输出队列上；engine 本身是独立进程（避开 GIL），用 ZMQ / 管道通信，见 [Python 并发](/basics/python)。
- **batching 队列**：API 层不做 batch，只把请求交给 engine 的 waiting 队列，由 scheduler 每步重新组 batch（见 [continuous batching](/inference/batching-scheduling)）；API 层要做的是准入控制和超时。
- **streaming**：SSE 或 chunked HTTP 把每步的 token 推给客户端；detokenize 要处理多字节 UTF-8 被切在 token 边界上的情况。
- **多副本路由**：无状态轮询最简单；带 prefix cache 时按 prompt 前缀哈希做 session affinity 能大幅提高命中率；PD 分离时 router 还要负责 prefill 和 decode 实例的配对。

## 面试追问

::: details Q：engine 挂了一个请求，API server 怎么知道？
engine 的输出流里要有 per-request 的结束 / 错误事件，API 层按 request id 分发到对应协程；同时要有 engine 进程的健康检查和超时，防止客户端协程永远挂在队列上。生产上还要让 abort（客户端断开）反向传给 engine 释放 KV。
:::

## 手撕

常见题：设计一个把多个并发 HTTP 请求喂给单线程 engine 的异步架构，画出队列和协程；完整系统设计见 [LLM serving 系统设计](/basics/system-design-llm-serving)。

## 参考

- [vLLM 文档：OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [SGLang Router](https://docs.sglang.ai/advanced_features/router.html)

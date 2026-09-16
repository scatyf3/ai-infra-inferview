---
title: 框架内功
---

# 框架内功

差异化所在，最该下功夫。核心是能讲清「一个 request 从 HTTP 进来到第一个 token 出去」的全链路，以及给 vLLM / SGLang 加一个新模型要碰哪些层。

::: tip 面试官真正在考什么
面试官真正在考的是：你是否真的在框架里改过东西，能否指出每个抽象层的边界和它存在的原因。
:::

## 本领域主题

- [PyTorch 内部机制](./pytorch-internals) — dispatcher、autograd、custom op 注册、caching allocator 与碎片
- [torch.compile：Dynamo / AOTAutograd / Inductor](./torch-compile) — dynamo 抓图、graph break、inductor 生成什么
- [给 vLLM / SGLang 加一个新模型](./add-model-vllm-sglang) — config 映射、weight loading、model runner、attention backend、tokenizer / chat template
- [vLLM V1 架构与 SGLang](./vllm-v1-architecture) — EngineCore / scheduler / worker / executor 分层；RadixAttention + 前端语言
- [服务层：异步请求、Batching 队列、Streaming、多副本路由](./serving-layer) — API server 到 engine 的边界
- [一个 Request 的全链路](./request-lifecycle) — 从 HTTP 进来到第一个 token 出去

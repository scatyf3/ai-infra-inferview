---
title: API Server
---

# API Server

对外暴露 HTTP 接口，主流是 OpenAI 兼容的 `/v1/chat/completions` 和 `/v1/completions`。API server 负责解析参数、tokenize、把请求交给引擎、把生成结果按流式格式返回，它本身应该是轻量的异步 I/O。

- **进程拆分**：Python 的 GIL 会让 API server 和引擎主循环抢 CPU。vLLM V1 把两者拆成不同进程，通过 ZeroMQ 通信。
- **背压**：请求过多时要排队或直接拒绝，而不是无限地堆进引擎。
- **取消**：客户端断开时要能通知引擎中止请求，及时释放 KV。

**延伸阅读**：[OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)

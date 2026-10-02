---
title: 流式输出
---

# 流式输出

一次生成要几秒到几十秒，流式输出让用户在第一个 token 生成后就能看到结果，体感延迟取决于 TTFT 而不是总耗时。实现上通常用 Server-Sent Events：每生成一段文本就推送一条 `data:` 消息，最后发送 `[DONE]`。

- **合并推送**：每个 token 推一次开销大，可以攒几个 token 或按时间间隔合并后再发。
- **对应的指标**：TTFT 看第一条消息，ITL / TPOT 看后续消息之间的间隔。
- **中途出错**：流开始之后就不能再改 HTTP 状态码，错误信息只能放在消息体里。

**延伸阅读**：[MDN: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---
title: Logits Processor
---

# Logits Processor

采样之前对 logits 的一系列改写：temperature 缩放、repetition / presence / frequency penalty、logit bias、禁用词、达到最小长度前屏蔽 EOS。每个请求的参数都不同，引擎要在同一个 batch 里按请求分别应用。

- **顺序有讲究**：penalty 一般在 temperature 之前，top-k / top-p 截断放在最后。
- **批量化**：把每个请求的参数排成张量一次性处理，避免 Python 循环。
- **和 structured output 的关系**：语法约束本质上也是一个 processor，把非法 token 的 logits 置为 −∞。

**延伸阅读**：[HF: Generation strategies](https://huggingface.co/docs/transformers/generation_strategies)

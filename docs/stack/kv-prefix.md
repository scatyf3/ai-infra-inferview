---
title: Prefix Caching
---

# Prefix Caching

很多请求共享同样的前缀：system prompt、few-shot 示例、多轮对话的历史。prefix caching 把算过的前缀 KV 保留下来，新请求命中时直接复用、跳过这部分 prefill，TTFT 和算力都能省下来。

- **vLLM**：按 block 的内容（连同它之前的前缀）做哈希，整块命中就复用。
- **SGLang RadixAttention**：用 radix tree 组织所有缓存的前缀，按 LRU 驱逐叶子节点。
- **调度要配合**：共享前缀的请求排在一起、路由到同一个实例，命中率才高。

**延伸阅读**：[SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)

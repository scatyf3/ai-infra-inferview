---
title: Sampling
---

# Sampling

从 logits 得到下一个 token。greedy 直接取最大值；随机采样先除以 temperature 再做 softmax，然后用 top-k（只留前 k 个）、top-p（留下累积概率达到 p 的最小集合）、min-p 等方式截断后抽样。

- **top-p 的实现**：需要排序加前缀和，大词表上排序很贵，因此有免排序的近似实现和基于拒绝采样的实现。
- **beam search**：同时维护多条候选序列，按累积对数概率保留前 k 条，要和 KV cache 的块共享配合。
- **speculative decoding**：用 rejection sampling 验证 draft token，保证输出分布和只用大模型时完全一致。

**延伸阅读**：[Nucleus Sampling](https://arxiv.org/abs/1904.09751) · [Speculative Decoding](https://arxiv.org/abs/2211.17192)

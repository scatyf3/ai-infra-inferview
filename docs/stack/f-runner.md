---
title: Model Runner
---

# Model Runner

引擎和模型之间的胶水层。每一步调度结束后，model runner 把这一批请求整理成模型需要的输入：拼平的 token ids、每个 token 的 position、指向 KV 物理位置的 slot mapping、attention 要用的序列长度元数据；然后执行 forward，取出需要采样的那几行 logits。

- **输入准备的开销**：这部分在 CPU 上做，batch 大时可能比 GPU 计算还慢，所以常做成持久化 buffer 增量更新。
- **凑固定 shape**：为了复用 CUDA graph，batch 大小要补齐到预先捕获的几档。
- **speculative decoding**：draft 和 verify 都要经过 runner，一步里可能要处理一个请求的多个 token。

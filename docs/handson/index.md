---
title: 手撕高频
---

# 手撕高频

0. [torch原语](./torch-primitives)：demo 阶段哪些 API 能直接调、哪些必须手写；背完去刷 [原语闪卡](./flashcards)
1. conceptually correctness: 
    1. attention [Naive Attention](./naive-attention) → [MHA](./mha-gqa-forward) → [GQA](./mha-gqa-forward) → MLA，mask / shape, with KV Cache
    2. Decoder Block 其余部分：RMSNorm、RoPE、SwiGLU FFN
2. kernel
    1. linear, reduce, matmul
    2. fused LayerNorm/RMSNorm
    3. optimization trick: Paged Attention, flash attention


<HandsonProgress />


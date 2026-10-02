---
title: LM Head
---

# LM Head

最后一层的 hidden state 乘上一个 [hidden, vocab] 的矩阵，得到每个词的 logits。词表动辄 15 万以上，这个 GEMM 并不小；好在只需要对每个序列的最后一个位置计算，prefill 时不必对整个 prompt 都算。

- **只算需要的行**：prefill 时先取出每个序列最后一个 token 的 hidden state，再过 LM head。
- **TP 下**：按词表维度切分，各卡算一部分 logits，再 gather 到一起做采样。
- **权重共享**：不少模型的 LM head 和 embedding 共用同一份权重（tie embeddings）。

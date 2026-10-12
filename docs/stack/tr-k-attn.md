---
title: "attention fwd / bwd"
---

# attention fwd / bwd

FlashAttention 的 forward 分块算 softmax，只存输出和每行的 logsumexp；backward 用这两样按块重算 attention 矩阵。训练时省掉的是 $s^2$ 大小的激活，代价是 backward 多算一遍 $QK^\top$。

- **causal**：只算下三角，实际 FLOPs 是满额的一半。
- **长序列**：CP 把 attention 拆到多卡，各块结果用 online softmax 合并。

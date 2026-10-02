---
title: 显存层次
---

# 显存层次

从快到慢：寄存器 → shared memory / L1（H100 每个 SM 最多 228 KB shared）→ L2（50 MB）→ HBM（80 GB，H100 SXM 约 3.35 TB/s）。越往下越大也越慢。decode 每生成一个 token 都要把全部权重和 KV 从 HBM 读一遍，所以 decode 的上限通常由 HBM 带宽决定，而不是算力。

- **ridge point**：H100 上 989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/byte，算术强度低于它就是 memory-bound。
- **复用**：tiling 的本质就是把数据搬到更快的层级，然后多用几次。
- **容量**：权重加 KV cache 装不装得下，决定了要几张卡。

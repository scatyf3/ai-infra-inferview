---
title: KV 量化
---

# KV 量化

长上下文、大 batch 时，KV cache 比权重还大，decode 读 KV 的带宽也随之上涨。把 KV 从 BF16 压到 FP8 或 INT4，容量和带宽都按比例节省。

- **FP8 KV**：最常用，per-tensor scale 就够，精度损失通常很小。
- **更低比特**：K 在通道维度上有明显的离群值，适合 per-channel 量化；V 适合 per-token 量化（KIVI 的做法）。
- **和 attention kernel 绑定**：kernel 要在读入时反量化，或者直接用低精度计算。

**延伸阅读**：[KIVI](https://arxiv.org/abs/2402.02750) · [KVQuant](https://arxiv.org/abs/2401.18079)

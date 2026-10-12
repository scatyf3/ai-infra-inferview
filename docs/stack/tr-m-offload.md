---
title: "offload"
---

# offload

把优化器状态或参数放到 CPU 内存，用 PCIe 带宽换显存。训练里最常见的场景是 RL 的 colocate：训练和推理共用一组卡，推理时把训练状态挪走、给 KV cache 腾地方。

- **代价**：PCIe 比 HBM 慢一两个数量级，只适合不在每层关键路径上的搬运。

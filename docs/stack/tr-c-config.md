---
title: "并行配置 / device mesh"
---

# 并行配置 / device mesh

三条约束一起定配置：显存要放下、TP / EP 只在 NVLink 域内、global batch 由收敛决定。顺序是 TP 填满机内 → 不够加 PP → 序列长加 CP → 剩下给 DP。DeviceMesh 把卡号排成多维数组，最内层给通信最频繁的维度。

- **公开配置**：Llama 3 405B 是 TP 8、PP 16、DP 128；DeepSeek-V3 不用 TP。

三条约束的推导、70B 在 1024 张 H100 上的算例和 MFU 调优 checklist，见 [并行配置怎么选](/train/parallel-config)。

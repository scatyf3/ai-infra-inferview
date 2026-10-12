---
title: "dataloader 与恢复"
---

# dataloader 与恢复

每个 DP rank 拿全局 batch 里自己那一段，TP 组内由 rank 0 读完广播。checkpoint 里记全局已消耗的样本数，重启后从同一位置继续，和 DP 大小无关。

- **瓶颈判断**：单独计时取数据的耗时，或换成固定的假 batch 看吞吐变不变。

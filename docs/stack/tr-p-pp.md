---
title: "PP"
---

# PP

按层切成 $p$ 个 stage，micro-batch 流水通过。bubble 占比约 $(p-1)/m$，1F1B 把激活从 $m$ 份降到 $p$ 份，interleaved、Zero Bubble、DualPipe 再压 bubble。

- **通信**：只在 stage 边界传激活，量小，可以跨机。

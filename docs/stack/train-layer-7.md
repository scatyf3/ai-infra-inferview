---
title: "训练栈第 7 层：作业"
---

# 训练栈第 7 层：作业

把一次训练当成一个要跑几十天的作业来看：开训前算清要多少卡、多少天；训练中盯住 loss 和梯度，出了 spike 能救；卡坏了能快速恢复；变慢、卡住、OOM 时知道先查什么。

- **规划**：$C \approx 6ND$，时长 = $C / (n \cdot \text{peak} \cdot \text{MFU} \cdot \text{ETTR})$。
- **稳定性**：QK-norm、z-loss、回滚并跳过 batch。
- **容错与排障**：checkpoint 间隔、异步保存；hang、OOM、NaN、慢各有固定的查法。

---
title: "稳定性"
---

# 稳定性

不稳定多数能追到某个量无界增长：attention logits（用 QK-norm 限住）、输出 logits 的整体漂移（用 z-loss 拉回）、Adam 分母和梯度尺度不匹配。真出了 loss spike，回到 spike 前约 100 步的 checkpoint、跳过一批数据再训。

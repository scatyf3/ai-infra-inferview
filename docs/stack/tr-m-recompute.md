---
title: "recompute"
---

# recompute

gradient checkpointing：forward 只存每层的输入，backward 时重算这一层。full recompute 多一次 forward，约 +33% 计算；selective 只重算 attention 核心，代价几个百分点。

- **顺序**：显存有余量时先关 recompute，再加 micro-batch。

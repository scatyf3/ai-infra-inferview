---
title: "训练栈第 2 层：一个 step"
---

# 训练栈第 2 层：一个 step

一个训练 step 的顺序是固定的：取 batch → forward → backward → 梯度同步 → 裁剪 → 优化器更新 → 调学习率 → 清梯度。这一层管的是每一步在数学上做了什么，以及 batch 和学习率随训练怎么变。

一个 step 的时间花在哪、优化器为什么便宜、各环节的同步点和优化点，见 [一个训练 step 的时间账](/train/step-anatomy)。

## 本层相关文章

| 内容 | 文章 |
|---|---|
| step 主循环逐行讲、`no_sync`、裁剪、AdamW / Muon、LR 调度 | [优化器与学习率](/train/optimizer-lr) |
| MFU 的损失来源、PP bubble 公式 | [训练算力账](/train/compute-mfu) |
| 分桶 overlap、最后一个 bucket | [计算通信 overlap](/parallel/comm-overlap) |
| dataloader、`pin_memory`、数据瓶颈判断 | [数据管线](/train/data-pipeline) |
| 同步点、straggler、GC、读 profiler trace | [训练调试](/train/debugging) |
| 异步 checkpoint | [容错](/train/fault-tolerance) |

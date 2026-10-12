---
title: "packing / mask"
---

# packing / mask

把多条样本拼进一个定长序列，消灭 padding；用 cu_seqlens 让 attention 只在各自文档内算，position_ids 按文档重置。SFT 还要 loss mask，只在回答上算 loss。

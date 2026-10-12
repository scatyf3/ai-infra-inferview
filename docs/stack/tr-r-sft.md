---
title: "SFT / LoRA"
---

# SFT / LoRA

SFT 和预训练用同一套系统，差别在数据：只在回答上算 loss，样本长短不一要 packing。LoRA 冻结底座，模型状态从每参数 16 字节降到约 2 字节，但激活和每 token 的计算基本不省。

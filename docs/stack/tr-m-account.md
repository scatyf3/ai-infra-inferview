---
title: "显存账"
---

# 显存账

训练显存 = 模型状态 + 激活 + 临时 buffer。混合精度 Adam 下模型状态每参数 16 字节，7B 就是 112 GB；激活每层约 $sbh(34 + 5as/h)$ 字节。

- **模型状态**：只看参数量，靠 ZeRO / FSDP 切。
- **激活**：只看 $s, b, h, L$，靠 FlashAttention、recompute、TP / CP 降。

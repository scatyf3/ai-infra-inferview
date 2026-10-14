---
title: "训练栈第 0 层：硬件"
---

# 训练栈第 0 层：硬件

训练对硬件有三个要求：算得快（Tensor Core 的 bf16 / fp8 峰值）、连得快（同步训练每一步都要通信）、存得下也读得快（checkpoint 和数据）。上面每一层的优化，最后都落到「哪一个先打满」。

- **算力**：MFU 的分母是 dense 峰值，H100 bf16 是 989 TFLOP/s，不是规格表上带稀疏的数。
- **互联**：NVLink 和 IB 差约一个数量级，决定了 TP / EP 只能放在机内，PP / DP 可以跨机。
- **存储**：checkpoint 每参数 12 字节，从gpu copy到cpu，然后写入文件系统

这里我们不care hbm到sram的带宽，而care上述内容，和decode的inference很不同。


---
title: "DP / ZeRO / FSDP"
---

# DP / ZeRO / FSDP

数据并行每卡一份完整模型，冗余 $N$ 倍。ZeRO-1 / 2 / 3 依次把优化器状态、梯度、参数切开，前两级通信量不变，第三级多 50%。FSDP 是 PyTorch 原生的 ZeRO-3，FSDP2 把每个参数切成 DTensor。

- **HSDP**：机内 FSDP、机间 DDP，all-gather 只走 NVLink。

各级 ZeRO 的显存和通信量推导、7B 八卡算例，见 [ZeRO 1 / 2 / 3 与 FSDP](/parallel/zero-fsdp)；DTensor、`fully_shard` 运行时和 HSDP，见 [FSDP2 与 DTensor](/train/fsdp2-dtensor)。

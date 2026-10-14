---
title: 互联
---

# 互联

我们有不同的卡和卡连接的方式（带宽都按每卡单向算）：
1. 同服务器内，通常nvlink带宽高，pcie和跨机差不多慢
    1. NVLink 4（H100）：450 GB/s，常说的「双向 900 GB/s」是两个方向加起来
    2. PCIe Gen5 x16：约 64 GB/s，主要连 CPU / 内存，只有 NVLink 的约 1/7
2. 跨服务器一般用某个特制的网络系统(InfiniBand，特制的网卡+交换机)，带宽低：NDR 每卡 400 Gb/s，除以 8 约 50 GB/s，是 NVLink 的约 1/9
3. nvidia有一些特化的通信方案，NVSwitch 让机内任意两张卡都能全带宽互通；GB200 NVL72 把 NVLink 域扩大到 72 张卡，用的是 NVLink 5，每卡约 900 GB/s。

这里的物理限制决定了我们如何在机内和跨机做并行，通常的做法是：
- **TP 放在机内**：每层两次 all-reduce，只有 NVLink 撑得住。
- **PP / DP 可以跨机**：PP 只在两段交界处传一批激活，通信量小；DP 一步只同步一次梯度，通信不频繁，还能和 backward 重叠。


**延伸阅读**：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

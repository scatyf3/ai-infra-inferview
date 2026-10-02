---
title: 互联
---

# 互联

卡与卡之间的带宽差好几个数量级：H100 的 NVLink 4 每卡双向 900 GB/s，PCIe Gen5 x16 单向约 64 GB/s，跨机 InfiniBand NDR 每卡 400 Gb/s（约 50 GB/s）。互联带宽决定了每种并行方式能放在哪一级。

- **TP 放在机内**：每层两次 all-reduce，只有 NVLink 撑得住。
- **PP / DP 可以跨机**：通信量小，或者通信不频繁。
- **拓扑**：NVSwitch 让机内任意两张卡都能全带宽互通；GB200 NVL72 把 NVLink 域扩大到 72 张卡。

**延伸阅读**：[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

---
title: "集合通信"
---

# 集合通信

all-reduce = reduce-scatter + all-gather，ring 算法下每卡收发约 $2(N-1)/N$ 倍数据，和卡数基本无关。估时间用 α-β 模型：延迟项乘步数，带宽项是字节数除以带宽。

- **对应关系**：DP 梯度 reduce-scatter，FSDP 参数 all-gather，TP all-reduce，EP all-to-all。

原语的 4 卡例子、ring 通信量推导、α-β 算例、busbw / algbw 和 NVLink / IB 拓扑，见 [集合通信原语与 NCCL](/parallel/collective-comm)。

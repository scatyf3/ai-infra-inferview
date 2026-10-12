---
title: "MoE"
---

# MoE

MoE 的算力按激活参数算，显存和梯度同步按总参数算。训练的难点是负载均衡（aux loss、aux-loss-free 的 bias、node-limited routing）和跨节点 all-to-all，所以 MFU 通常比同激活参数的 dense 模型低。

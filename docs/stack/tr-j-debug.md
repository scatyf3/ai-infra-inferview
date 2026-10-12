---
title: "排障"
---

# 排障

按现象分：hang 是 collective 在各 rank 上的调用顺序不一致；OOM 先分第一步就爆还是跑一阵才爆；NaN 常见 0/0 和 fp16 溢出；慢要先算期望吞吐再 profile。同步训练里找慢节点要比计算时间，不能比 step time。

---
title: "训练栈第 5 层：通信与配置"
---

# 训练栈第 5 层：通信与配置

切完以后，每种并行对应固定的集合通信：DP 是 reduce-scatter 和 all-gather，TP 是 all-reduce，EP 是 all-to-all，PP 是点对点。这一层管通信要多久、能不能藏到计算后面，以及几种并行怎么组合成一个配置。

- **集合通信**：ring 算法下通信量和卡数无关，时间看带宽。
- **overlap**：通信没藏住的部分直接算进 step 时间。
- **配置**：显存、拓扑、global batch 三条约束一起定 TP / PP / CP / DP。

---
title: "EP"
---

# EP

把 MoE 的专家分到不同卡上，每个 MoE 层 dispatch、combine 各一次 all-to-all。最忙的那张卡决定这一层的时间，所以训练时要在路由上做负载均衡。

- **组合**：专家维度在 DP 的那些卡里再切，attention 部分照常 DP。

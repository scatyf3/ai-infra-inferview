---
title: MoE 与 Expert Parallel
status: todo
tags: [moe, ep, all-to-all]
difficulty: 4
order: 5
related: []
stack: [d-intra, f-model]
---

# MoE 与 Expert Parallel

> 路由、all-to-all、专家负载不均、EP 与 TP 混用

## 一句话结论

MoE 每个 token 只走 top-k 个专家，FLOPs 按激活参数算，显存按总参数算；Expert Parallel 把专家分到不同卡上，token 要先 all-to-all 发到专家所在的卡、算完再 all-to-all 发回来，通信模式和 TP 的 all-reduce 完全不同。核心问题是专家负载不均：热门专家所在的卡成为瓶颈。

## 推导

- **路由**：gate 对每个 token 算专家分数取 top-k（通常 2 或 8），softmax 加权求和；训练时加负载均衡的辅助 loss 或 bias 调整让专家被均匀选到。
- **all-to-all**：dispatch 阶段每张卡把 token 按目标专家分组发出去，combine 阶段收回；每次通信量 ≈ token 数 × k × hidden，跨节点时走 IB，是 MoE 推理的主要开销。
- **负载不均**：batch 里 token 分布偏斜时，某个专家收到几倍于平均的 token，其他卡等它；对策有 capacity factor 丢 token、专家冗余部署（DeepSeek 的 redundant experts）、动态重排专家。
- **EP 与 TP 混用**：attention 部分用 TP，专家部分用 EP；或者专家内部再 TP 切；推理里 EP 大、TP 小是趋势，因为 EP 把每张卡的权重读取量降下来。

## 面试追问

::: details Q：MoE 推理为什么说 batch 要很大才划算？
batch 小时每个专家分到的 token 很少，每张卡读了整个专家权重却只算几个 token，算术强度极低；只有 batch 大到每个专家都有足够的 token，权重读取才摊得开。所以 MoE 服务倾向于大 EP、大 batch，把所有卡的 HBM 带宽凑起来一起读专家权重。
:::

## 手撕

常见题：写出 top-k gating 的 forward（含负载均衡 loss）；估算一次 all-to-all 的通信量。模型定义见 [并行总览](/parallel/parallelism-overview)。

## 参考

- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V3 Technical Report（EP 部署和负载均衡）](https://arxiv.org/abs/2412.19437)

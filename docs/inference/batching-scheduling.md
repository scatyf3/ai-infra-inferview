---
title: Continuous Batching、Chunked Prefill 与 PD 分离
status: draft
tags: [scheduler, batching, pd-disaggregation]
difficulty: 4
order: 6
related: [/inference/prefill-decode-roofline, /inference/kv-cache-paged-attention, /framework/request-lifecycle]
---

# Continuous Batching、Chunked Prefill 与 PD 分离

> scheduler 抢占（swap vs recompute）

## 一句话结论

调度器要同时解决三件事：**让 batch 尽量大**（continuous batching，因为 decode 的 AI 就是 batch）、**别让 prefill 卡住 decode**（chunked prefill 或 PD 分离）、**显存不够时体面降级**（抢占）。三件事共用一个预算：KV block 的数量。

## 推导

### Continuous batching

静态 batching 要等一整批请求全部生成完才能开始下一批，最长的那个请求拖住所有人，GPU 利用率被短请求的空转拖垮。continuous batching（也叫 iteration-level scheduling）改成**每生成一个 token 就重新组 batch**：谁结束了就踢出去，等待队列里有人就立刻补进来。

收益直接来自 roofline：decode 的 $\text{AI} \approx B$，batch 从 4 涨到 64 就是 16 倍的算力利用率。实测吞吐提升 2–4 倍，且请求越异构收益越大。

### 谁能进这一轮 batch

每轮调度是一个预算分配问题，两个约束：

1. **token 预算**：`max_num_batched_tokens`，一个 batch 里 prefill token 加 decode token 的总数。
2. **KV 预算**：所有 running 序列的 block 总数不能超过池子大小。

vLLM V1 的调度循环大致是：先给所有 running 的 decode 请求各留 1 个 token 的预算（保证已经开始的请求能继续），剩下的预算给 waiting 队列的 prefill；如果分配 block 时发现不够，就从 running 里抢占。

### Chunked prefill

问题：一个 4k prompt 的 prefill 要 200 ms，这期间所有正在 decode 的请求都在等，ITL 出现尖刺。

解法：把 prefill 切成固定大小的 chunk（512–2048 token），每轮只做一个 chunk，剩下的预算给 decode。这样一个 batch 里同时有 prefill token 和 decode token，attention kernel 要能处理变长（这就是 `varlen` / `flash_attn_varlen_func` 的用武之地）。

副作用：prefill 被拆开后总耗时略增（每个 chunk 都要重读一遍已有的 KV 做 attention），但 ITL 的尾延迟大幅改善。chunk 大小要保证 chunk 的 AI 仍远高于 ridge point，否则 prefill 自己也变得低效。

### PD 分离

更彻底的做法：prefill 和 decode 跑在**不同的 GPU 池**上。

| | prefill 池 | decode 池 |
|---|---|---|
| 负载性质 | compute-bound | memory-bound |
| 理想硬件 | 高算力（H100、算力卡） | 高带宽大显存（H200） |
| 理想并行 | TP 大，batch 小 | DP 多副本，batch 大 |
| 优化目标 | TTFT | TPOT |

好处是两边可以独立选型、独立扩缩容、各自调到最优；prefill 的长尾不再污染 decode 的 ITL。代价是 **KV 要跨节点传**：prefill 算完后把 $\text{KV/token} \times \text{prompt\_len}$ 字节传给 decode 节点。70B GQA、2k prompt 是 640 MiB，400 Gbps IB 上约 13 ms，通常用 layer-wise 传输和 prefill 的后几层 overlap 掉。

传输量正比于 KV/token，所以 **MLA 对 PD 分离特别友好**（68 KiB/token vs GQA 的 320 KiB/token）。

### 抢占

显存不够时必须踢人。两个策略见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)：swap 拷到 CPU，recompute 直接丢。还有一个选择维度：**踢谁**。通常是 LIFO（踢最新进来的），因为老请求已经投入了更多计算，踢它浪费更大；也避免了老请求被反复饿死。

抢占是个信号：如果生产中频繁发生，说明 `max_num_seqs` 设得过于激进或 KV 空间不足，该调参或加卡，而不是让调度器反复抖动。

## 交互

一直点「decode 一步」直到 free block 归零，这就是调度器面临抢占决策的时刻。

<PagedKV :block-size="4" :num-blocks="20" />

## 面试追问

::: details Q：continuous batching 下，先来的请求会不会被后来的饿死？
调度是 FCFS 加 token 预算，waiting 队列按到达顺序，所以不会饿死。但抢占用 LIFO 的话，高负载下最新的请求会被反复踢，表现为它的 TTFT 很差。生产上要看 P99 而不只是均值，必要时加准入控制：队列太长就直接拒绝，而不是接下来再抖动。
:::

::: details Q：max_num_batched_tokens 和 max_num_seqs 分别控制什么？
前者是每轮的 token 总预算（算力侧的闸门），后者是同时 running 的序列数上限（KV 侧的闸门）。开了 chunked prefill 时前者主要约束 prefill chunk 的大小；后者决定 decode 的 batch 上限，也就决定了 decode 的 AI 能到多高。两者要配合 KV block 数一起调。
:::

::: details Q：chunked prefill 和 PD 分离，什么时候选哪个？
单节点或卡数少时用 chunked prefill，没有额外的传输开销和运维复杂度。规模大、且 TTFT 和 TPOT 有独立 SLA 时用 PD 分离，因为它允许两边独立调优和扩缩容。PD 分离的门槛是要有高速互联（RDMA）和一套 KV 传输层，小集群上不划算。
:::

::: details Q：怎么设计 benchmark 来验证调度器改动有没有效？
固定输入输出长度分布（用真实 trace 或 ShareGPT），扫 QPS，画 QPS 对 P50/P99 TTFT 和 TPOT 的曲线。关键指标是 **goodput**：满足 SLA（比如 TTFT < 1s 且 TPOT < 50ms）的请求的吞吐，而不是裸吞吐。只看平均吞吐会让「牺牲少数请求换总量」的改动看起来是正收益。
:::

## 参考

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [SARATHI / Sarathi-Serve: chunked prefill](https://arxiv.org/abs/2403.02310)
- [DistServe: PD 分离](https://arxiv.org/abs/2401.09670)

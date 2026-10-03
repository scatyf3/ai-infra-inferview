---
title: 指标与 Benchmark：TTFT / TPOT / ITL / Goodput
status: todo
tags: [metrics, benchmark, sla]
difficulty: 2
order: 9
related: []
stack: [sv-sla]
---

# 指标与 Benchmark：TTFT / TPOT / ITL / Goodput

> 怎么做 benchmark，SLA 下怎么调 batch

## 一句话结论

四个指标各管一段：TTFT 看 prefill 和排队，TPOT / ITL 看 decode 每步的快慢和抖动，吞吐看 GPU 利用率，goodput 是「满足 SLA 的请求吞吐」，才是真正的优化目标。benchmark 要固定输入输出长度分布、扫 QPS、画延迟曲线，而不是只报一个最大吞吐。

## 推导

- **定义**：TTFT = 第一个 token 返回的时间；TPOT = 后续每个 token 的平均间隔；ITL = 单个 token 间隔（看 P99 抖动）；E2E = TTFT + TPOT × 输出长度。
- **batch 的 tradeoff**：batch 越大吞吐越高（decode 的算术强度 ≈ batch），但每步要算更多请求，TPOT 变长；在 TPOT 的 SLA 下反推能开多大的 batch，见 [roofline](/inference/prefill-decode-roofline)。
- **怎么做 benchmark**：用真实 trace 或 ShareGPT 分布生成 prompt；按泊松到达扫 QPS；每个 QPS 跑够长时间看 P50 / P99；报 goodput 对 QPS 的曲线。
- **常见坑**：warmup 不够、客户端成了瓶颈、输出长度固定导致 batch 行为不真实、只看平均值掩盖长尾。

## 面试追问

::: details Q：加大 max_num_seqs 之后吞吐涨了但 goodput 反而降了，为什么？
batch 变大后每步 decode 时间变长，TPOT 超过 SLA 的请求变多，这些请求虽然算了但不计入 goodput；同时 KV 占用变多，抢占增加，TTFT 的长尾恶化。这就是为什么调参要盯 goodput 而不是裸吞吐。
:::

## 手撕

常见题：设计一个压测方案并说明报告哪些图；定义见 [SLA](/stack/sv-sla)。

## 参考

- [vLLM benchmark_serving 脚本](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [DistServe 论文里的 goodput 定义](https://arxiv.org/abs/2401.09670)

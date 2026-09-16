---
title: 系统设计：设计一个 LLM 推理服务
status: draft
tags: [system-design, serving]
difficulty: 4
order: 4
related: [/inference/batching-scheduling, /framework/request-lifecycle, /inference/metrics-benchmark]
---

# 系统设计：设计一个 LLM 推理服务

> 几乎必出

## 一句话结论

按这个顺序走：**澄清 SLA 和流量 → 算容量（显存账决定卡数）→ 画分层架构 → 讲清单机调度 → 再讲多副本路由和自动扩缩 → 最后是可观测性和降级**。全程用数字说话，每个设计选择都回到「prefill compute-bound、decode memory-bound」这条主线。

## 推导

### 第一步：把题目问清楚

不要上来就画框图。要问出来的五件事：

1. **模型和规模**：70B 还是 7B？MoE 吗？决定了显存账和并行方式。
2. **流量**：QPS、输入输出长度分布、峰谷比。「平均 2k 输入 / 500 输出、峰值 100 QPS」和「平均 200 输入 / 50 输出、峰值 5000 QPS」是完全不同的系统。
3. **SLA**：TTFT 和 TPOT 分别的 P99 目标。交互式聊天要 TTFT < 500ms、TPOT < 50ms；批量处理任务只关心吞吐。
4. **是否流式**：流式则要 SSE/WebSocket 长连接，影响网关和超时设计。
5. **多租户吗**：要不要配额、优先级、隔离。

### 第二步：容量估算（面试官最想看的部分）

以 70B bf16、平均 2k 输入 500 输出、峰值 100 QPS 为例：

**显存**：权重 132 GiB，KV 320 KiB/token。单卡 H100 80 GiB 放不下权重，TP=4 起步，每卡权重 33 GiB，留给 KV 约 (80×0.9 − 33 − 3) = 36 GiB，四卡合计 144 GiB KV ÷ 320 KiB = **约 45 万 token 的并发预算**。平均每请求峰值占 2.5k token，理论并发约 180 个请求。

**吞吐**：decode 一步读 132 GiB ÷ 4 卡 = 33 GiB/卡 ÷ 3.35 TB/s ≈ 10 ms，加 KV 读取和 all-reduce，实测 TPOT 约 15–25 ms。batch 64 时单副本 decode 吞吐约 64 / 0.02 = **3200 tok/s**。

**需要几个副本**：100 QPS × 500 输出 token = 50000 tok/s 的生成需求 ÷ 3200 = **16 个副本 × 4 卡 = 64 张 H100**。再加 prefill 的算力占用（100 QPS × 2k token × 2 × 70e9 FLOP = 28 PFLOP/s，对比 64 卡 × 989 TFLOP/s × 0.5 MFU = 31 PFLOP/s）——**prefill 几乎吃掉了全部算力预算**，这是关键发现，说明要么上 prefix caching 降低 prefill 量，要么 PD 分离单独给 prefill 配卡。

这一段的价值不在数字精确，而在展示你能用显存账和 roofline 推出瓶颈在哪。

### 第三步：分层架构

```
客户端
  ↓ HTTPS / SSE
[API 网关]  认证、限流、配额、请求校验
  ↓
[路由层]    按 KV cache 亲和性选副本（不是轮询！）
  ↓
[推理副本]  API server (async) → 调度器 → executor → GPU workers
  ↓
[KV 存储]   本地 HBM + 可选的跨副本 prefix cache
```

旁路组件：模型仓库（权重分发）、指标采集（Prometheus）、日志与 trace、灰度发布。

### 第四步：单副本内部

这是考察深度的地方，见 [一个 Request 的全链路](/framework/request-lifecycle)。要点：
- tokenize 放在 CPU 线程池，别阻塞事件循环。
- 调度器做 continuous batching，每轮重组 batch。
- chunked prefill 防止长 prompt 阻塞 decode 的 ITL。
- KV 用 PagedAttention 分页管理，显存不足时抢占。
- 输出通过 asyncio queue 流式推给 HTTP handler。

### 第五步：多副本路由

**不要轮询**。按 prefix 亲和性路由：对请求的 prompt 前缀做 hash，路由到持有该前缀 KV 的副本，命中率能从 0 提到 70%+，直接砍掉大部分 prefill。实现上维护一个前缀到副本的映射（或用一致性 hash），同时考虑副本负载做加权，避免热点前缀把某个副本打爆。

多轮对话天然适合这个策略：同一个会话固定路由到同一副本。

**过载保护**：队列长度超阈值直接返回 429，而不是接下来让所有人都超时。准入控制比排队更重要，因为 LLM 请求的处理时间以秒计，排队只会让 P99 雪崩。

### 第六步：扩缩容

按「排队延迟」而不是 GPU 利用率扩容。GPU 利用率在 decode 时本来就低（memory-bound），用它当信号会永远不扩。合理指标是等待队列中的请求数、或 TTFT 的 P95。

冷启动是主要痛点：70B 权重加载要几十秒到几分钟。缓解手段是常驻一定比例的 warm 副本、权重放本地 NVMe 或用 tensor 并行的流式加载、以及提前几分钟按流量预测扩容。

### 第七步：降级路径

按优先级列：
1. 缩短 `max_tokens`（先保证有回复）。
2. 关闭 speculative decoding（省算力）。
3. 降低 batch 内的 prefill 配额（保 TPOT 牺牲 TTFT）。
4. 路由到小模型（质量降级）。
5. 限流拒绝低优先级租户。

### 可观测性

必须有的指标：TTFT / TPOT / ITL 的分位数、goodput（满足 SLA 的吞吐）、KV cache 使用率、抢占次数、prefix 命中率、每副本的 running 和 waiting 队列长度、GPU 的 MBU。

**抢占次数和队列长度是最早的预警信号**，比延迟指标早几十秒。

## 面试追问

::: details Q：为什么不用 GPU 利用率做扩容信号？
`nvidia-smi` 的利用率只表示「有 kernel 在跑」，decode 时 kernel 一直在跑但 tensor core 基本空转，利用率显示 100% 而实际算力用了不到 5%。它既不反映算力余量也不反映显存余量。该看的是 KV cache 占用率（显存侧的真实压力）和等待队列长度（用户侧的真实体感）。
:::

::: details Q：怎么保证同一个会话的多轮请求命中同一个副本？
会话 id 做一致性 hash，或在响应里带一个 sticky 路由提示让客户端后续请求带上。要处理副本下线时的重路由（这时 prefix cache 失效，退化成普通 prefill，功能正确只是慢一点）。注意不要做成强绑定，否则副本故障会让整个会话不可用。
:::

::: details Q：流式响应下，怎么做超时和取消？
客户端断开时要能把请求从调度器里撤掉并释放 KV block，否则会一直生成到 max_tokens，白烧算力。实现上是 HTTP 连接的 disconnect 事件触发一个 cancel，调度器在下一轮把该序列标记为 finished。这个链路经常被漏掉，压测时表现为「客户端全断了但 GPU 还在满载」。
:::

::: details Q：如果要支持 LoRA 多租户呢？
用 multi-LoRA serving：base 权重共享一份，每个租户的 LoRA adapter（几十 MB）单独存，推理时用分组 GEMM 把同一 batch 里不同 adapter 的请求一起算（S-LoRA / Punica 的做法）。显存上可以把冷 adapter 换出到 CPU。这样一套 70B 的卡能服务几百个租户的定制模型，成本远低于每个租户一套。
:::

::: details Q：怎么做灰度发布？
模型版本和服务版本分开。模型灰度按流量比例路由到新版本副本池，对比 goodput 和业务指标；服务（框架）灰度先在影子流量上跑，对比相同输入的输出（温度 0 时应该 bit-level 一致，不一致说明 kernel 或调度有行为变化）。回滚要快，权重预热是瓶颈，所以新旧副本池要有一段共存期。
:::

## 参考

- [vLLM 生产部署文档](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [DistServe / Splitwise：PD 分离的容量分析](https://arxiv.org/abs/2401.09670)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)

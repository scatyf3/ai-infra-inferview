---
title: KV Cache 与 PagedAttention
status: draft
tags: [kv-cache, paged-attention, prefix-caching]
difficulty: 4
order: 3
related: [/inference/memory-accounting, /inference/batching-scheduling, /framework/vllm-v1-architecture]
---

# KV Cache 与 PagedAttention

> 布局、PagedAttention、prefix caching / RadixAttention、KV 量化

## 一句话结论

KV cache 在 PagedAttention 之前是按「每请求预留最大长度的连续大块」分配的，浪费 60–80%。PagedAttention 把它切成固定大小的 block，用一张 block table 做逻辑到物理的映射，外部碎片归零、内部碎片上限是每序列半个 block，还顺带解锁了 **prefix 共享**（copy-on-write + refcount）和**细粒度抢占**。

## 推导

### 问题：连续分配为什么浪费

请求的输出长度事先不知道，朴素做法是按 `max_model_len` 预留。一个 2k prompt、实际生成 100 token 的请求，在 8k 的预留下用了 26%，剩下 74% 既不能给别人用也不能回收。vLLM 论文测得实际系统的 KV 有效利用率只有 20–40%。

这和操作系统的虚拟内存是同一个问题，所以解法也一样：**分页**。

### Block 与 block table

物理 KV 被切成大小为 `block_size`（vLLM 默认 16 token）的 block。每个序列维护一张 block table，把逻辑 block 号映射到物理 block 号：

```
seq A tokens: [t0..t15][t16..t31][t32..t37]
block table:      #7       #2        #11      ← 物理上完全不连续
```

attention kernel 按 block table 去 gather，所以计算不要求物理连续。代价是 kernel 里多一层间接寻址，以及 block 太小时 gather 的效率下降（这就是 block_size 不能设成 1 的原因）。

浪费的上界变成：每个序列最后一个未满 block 的空槽，平均 `block_size / 2` 个 token。16 token 的 block 下，每序列浪费 8 个 token 的 KV，可以忽略。

### Prefix 共享

如果两个序列有相同的前缀（system prompt、few-shot 例子、多轮对话的历史），它们前缀部分的 KV **逐位相同**。于是：满 block 按内容 hash，命中就直接把物理 block 号填进新序列的 block table，refCount 加一。

写的时候要小心：refCount > 1 的 block 不能原地写，必须 **copy-on-write** 先复制出一份。只有满 block 才参与共享，因为未满 block 还会被追加。

SGLang 的 **RadixAttention** 把这个做得更进一步：用 radix tree 管理所有前缀，支持任意长度的部分匹配（不止 block 对齐），并用 LRU 淘汰。收益场景：多轮对话（每轮复用全部历史）、agent 的长 system prompt、树搜索类的推理（共享分支前缀）。

### 抢占：swap vs recompute

显存满了而 running 的序列还要新 block 时，调度器必须抢占某个序列：

| | swap | recompute |
|---|---|---|
| 做法 | 把 block 拷到 CPU 内存，之后拷回 | 直接丢弃 block，恢复时把已生成的 token 当 prompt 重新 prefill |
| 代价 | 2 × block 字节 / PCIe 带宽 | 重新 prefill 的算力 |
| 适合 | 序列很长（重算贵） | 序列短，或有 prefix cache 能命中 |
| 副作用 | 占 CPU 内存，PCIe 和 H2D 拷贝争带宽 | 无额外内存，但 TTFT 尖刺 |

vLLM 默认 recompute，因为 PCIe 往返通常比重算更慢，而且 recompute 能吃到 prefix cache。

### KV 量化

KV 从 bf16 降到 fp8 直接让 KV 访存和显存减半，等于 decode 的 AI 翻倍。注意点：
- **per-token 或 per-head 量化**，不要 per-tensor。KV 的 outlier 集中在少数几个 channel 上。
- K 比 V 更难量化（K 参与 softmax 前的点积，误差会被指数放大）。有些方案对 K 用更高精度。
- 误差会累积：早期 token 的 KV 被后续每一步反复读，长 context 下影响更大。

## 交互

点「新请求（共享前缀）」看 refCount 变成 2、block 边框变双线；一直点「decode 一步」直到 free 归零，再试 swap 和 recompute 两种抢占，对比 block table 的变化。

<PagedKV :block-size="4" :num-blocks="24" />

## 面试追问

::: details Q：block_size 该设多大？
太小（如 1–4）：block table 变长，kernel 里 gather 的间接寻址开销上升，且 block 元数据本身占内存。太大（如 128）：内部碎片回来了，每序列平均浪费 64 个 token 的 KV，prefix 共享的粒度也变粗（要 128 token 完全相同才能共享）。16 是经验上的平衡点。GPU 上还要考虑 block 大小和 warp 处理的 tile 对齐。
:::

::: details Q：prefix caching 的 hash 怎么算，有碰撞风险吗？
vLLM 对每个 block 算的是「从序列开头到这个 block 结尾的所有 token id」的 hash，而不只是这个 block 内的 token。因为相同的 16 个 token 出现在不同的前缀后面，KV 是不同的（attention 依赖全部历史）。碰撞理论上存在，实践中用 64 位以上的 hash 概率可忽略；有洁癖的实现会在命中后再比一次 token id。
:::

::: details Q：PagedAttention 的 kernel 比连续的慢多少？
论文报告 20–26% 的 attention kernel 开销，但端到端吞吐提升 2–4 倍，因为省下来的显存换成了更大的 batch。这是典型的「局部变慢、全局变快」。后来的实现（FlashAttention 的 paged 变体、FlashInfer）把这个开销压到 10% 以内。
:::

::: details Q：多轮对话下 prefix cache 命中率能有多高？
第 N 轮的 prompt 包含前 N-1 轮的全部内容，所以命中率随轮数上升，长会话下能到 90% 以上，TTFT 几乎只取决于新增的那一轮。这也是 prefix caching 在 chat 场景收益最大的原因。但要注意缓存淘汰：显存紧张时 free block 会被重新分配，hash 表里的条目失效。
:::

::: details Q：PD 分离下，KV 怎么从 prefill 节点传到 decode 节点？
通过 RDMA / NVLink 直接做 GPU 到 GPU 的拷贝，传输量是 `KV/token × prompt_len`。70B GQA 下 2k prompt 是 640 MiB，在 400 Gbps 的 IB 上约 13 ms，可以和 prefill 的最后几层 overlap 掉（layer-wise 传输，算完一层就传一层）。这个传输量正比于 KV/token，所以 MLA 对 PD 分离特别友好。
:::

## 参考

- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [SGLang: RadixAttention](https://arxiv.org/abs/2312.07104)
- [vLLM automatic prefix caching 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)

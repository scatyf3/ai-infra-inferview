---
title: KV Cache 与 PagedAttention
status: draft
tags: [kv-cache, paged-attention, prefix-caching]
difficulty: 4
order: 3
related: [/inference/memory-accounting, /inference/batching-scheduling, /framework/vllm-v1-architecture]
stack: [kv-paged, kv-prefix, kv-quant]
---

# KV Cache 与 PagedAttention

> 布局、PagedAttention、prefix caching / RadixAttention、KV 量化

## 一句话结论

KV cache 在 PagedAttention 之前是按「每请求预留最大长度的连续大块」分配的，浪费 60–80%。PagedAttention 把它切成固定大小的 block，用一张 block table 做逻辑到物理的映射，外部碎片归零、内部碎片上限是每序列半个 block，还顺带解锁了 **block 共享**（refcount；跨请求的 prefix caching，以及同一请求 fork 时的 copy-on-write）和**细粒度抢占**。

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

### Block 共享：prefix caching 与 copy-on-write

block 化之后，多个序列的 block table 可以指向**同一个物理 block**，每个 block 记一个 `ref_count`，表示有几个序列在引用它。共享有两种来源，要分清：

**1. Prefix caching：跨请求共享，不需要 CoW**

如果两个请求有相同的前缀（system prompt、few-shot 例子、多轮对话的历史），前缀部分的 KV **逐位相同**。满 block 按「从开头到这个 block 为止的全部 token」算 hash，新请求命中就直接把物理 block 号填进自己的 block table，`ref_count += 1`，这部分不用再 prefill。

**只有满 block 参与 prefix caching。** 满 block 以后永远不会再被写（新 token 只会追加到序列最后那个未满 block），所以共享的 block 是只读的，**不存在写冲突，也就用不到 copy-on-write**。

**2. Fork：同一请求分叉，需要 CoW**

parallel sampling（`n > 1`）和 beam search 会把一个序列分叉成多个：fork 时直接复制 block table，所有 block 的 `ref_count += 1`，**包括最后那个未满的 block**。之后各分支要往这个共享的未满 block 追加不同的 token，就冲突了：

- 写之前检查 `ref_count > 1`：先申请一个新 block，把旧 block 内容拷过去，改自己的 block table 指向新 block，旧 block `ref_count -= 1`，然后再写。这就是 **copy-on-write**。
- 只有分叉点所在的那一个 block 需要拷，前面的满 block 永远共享。

### 伪代码

```python
class BlockAllocator:
    def __init__(self, num_blocks):
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.hash_to_block = {}                  # prefix cache：满 block 的 hash → 物理 block

    def alloc(self):
        b = self.free.pop()                      # 空了就触发抢占（见下节）
        self.ref[b] = 1
        return b

    def release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)                  # 实际实现会留在 hash 表里做 LRU 淘汰


class Sequence:
    def __init__(self, tokens):
        self.tokens = tokens
        self.block_table = []                    # 逻辑 block 号 → 物理 block 号


def prefill(seq, A, B):                          # B = block_size
    h = None
    for i in range(0, len(seq.tokens), B):
        chunk = seq.tokens[i:i + B]
        if len(chunk) == B:                      # 只有满 block 查 prefix cache
            h = hash((h, tuple(chunk)))          # 链式 hash：包含整个前缀
            if h in A.hash_to_block:
                b = A.hash_to_block[h]
                A.ref[b] += 1                    # 命中：共享，跳过这段的计算
                seq.block_table.append(b)
                continue
        b = A.alloc()
        compute_and_write_kv(b, chunk)
        if len(chunk) == B:
            A.hash_to_block[h] = b
        seq.block_table.append(b)


def append_token(seq, tok, A, B):                # decode 一步
    pos = len(seq.tokens)
    if pos % B == 0:                             # 最后一个 block 满了，开新 block
        seq.block_table.append(A.alloc())
    else:
        last = seq.block_table[-1]
        if A.ref[last] > 1:                      # 被 fork 共享的未满 block → CoW
            new = A.alloc()
            copy_block(src=last, dst=new)
            A.release(last)
            seq.block_table[-1] = new
    write_kv(seq.block_table[-1], pos % B, tok)
    seq.tokens.append(tok)


def fork(seq, A):                                # parallel sampling / beam search
    child = Sequence(list(seq.tokens))
    child.block_table = list(seq.block_table)
    for b in child.block_table:
        A.ref[b] += 1
    return child


def attention(q, seq, B):                        # kernel 按 block table 间接寻址
    for pos in range(len(seq.tokens)):
        b, off = seq.block_table[pos // B], pos % B
        k, v = K_cache[b][off], V_cache[b][off]
        ...                                      # 常规 online softmax
```

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
- **FP8 用 per-tensor scale 就够**：fp8 自带指数位，动态范围大，vLLM 的 `kv_cache_dtype="fp8"` 每层 K、V 各一个 scale。int4 / int2 这类整数格式才需要细粒度：K 的 outlier 集中在少数几个 channel，所以 K 按 per-channel、V 按 per-token（KIVI）。
- K 比 V 更难量化：k 上的误差 $\delta$ 让 score 偏 $q \cdot \delta / \sqrt{d_h}$，过了 softmax 变成乘性的 $e^{q \cdot \delta / \sqrt{d_h}}$；V 的误差只是线性混进加权平均。有些方案对 K 用更高精度。
- 误差会累积：早期 token 的 KV 被后续每一步反复读，长 context 下影响更大。

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

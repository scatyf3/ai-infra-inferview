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
            self.free.append(b)                  # 简化版；带 prefix cache 的 LRU 版本见下一节


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

上面的 `prefill` 只演示了「命中就共享」，下面把 prefix caching 完整的生命周期补全。

### Prefix caching 伪代码

核心是让 block 有三种状态，而不是简单的「占用 / 空闲」：

| 状态 | `ref` | 在 hash 表里 | 在 free 队列里 | 含义 |
|---|---|---|---|---|
| 使用中 | > 0 | 满了就在 | 否 | 有请求正在用 |
| **已缓存、空闲** | 0 | 是 | 是 | 没人用，但内容还在，可以被命中复活，也可以被淘汰 |
| 空闲 | 0 | 否 | 是 | 纯空 block |

请求结束时 block **不清空**，只是 `ref` 归零后进 free 队列尾部。新请求要分配时从队头拿（LRU），拿到的 block 如果还挂在 hash 表上，**这时才把它从 hash 表删掉**（淘汰）。

```python
from collections import OrderedDict

class Block:
    def __init__(self, bid):
        self.id = bid
        self.ref = 0
        self.hash = None                         # 满了且登记过才有


class PrefixCachingAllocator:
    def __init__(self, num_blocks):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_q = OrderedDict((b.id, b) for b in self.blocks)   # LRU：队头最久没用
        self.cached = {}                         # hash → Block

    # ---------- 查缓存 ----------
    def match_prefix(self, tokens, B):
        """返回命中的 block 列表。链式 hash，第一次 miss 后面必然都 miss。"""
        hits, h = [], None
        # 至少留最后 1 个 token 不命中：要靠它跑一次 forward 拿 logits
        n_full = (len(tokens) - 1) // B
        for i in range(n_full):
            h = hash((h, tuple(tokens[i * B:(i + 1) * B])))
            blk = self.cached.get(h)
            if blk is None:
                break
            hits.append(blk)
        return hits

    # ---------- 分配 ----------
    def touch(self, blk):                        # 命中：引用 +1，从 free 队列复活
        if blk.ref == 0:
            del self.free_q[blk.id]
        blk.ref += 1

    def alloc(self):
        if not self.free_q:
            raise NoFreeBlocks                   # 交给调度器抢占
        _, blk = self.free_q.popitem(last=False) # 从队头拿最久没用的
        if blk.hash is not None:                 # 它还缓存着别人的前缀 → 淘汰
            del self.cached[blk.hash]
            blk.hash = None
        blk.ref = 1
        return blk

    # ---------- 登记 ----------
    def register_full(self, blk, h):
        """block 写满时调用（prefill 和 decode 都会触发）。"""
        if h in self.cached:                     # 别的请求同时算出了同一块，保留先来的
            return
        blk.hash = h
        self.cached[h] = blk

    # ---------- 释放 ----------
    def free(self, block_table):
        # 倒序放回：序列尾部的 block 先被淘汰，开头的公共前缀活得最久
        for blk in reversed(block_table):
            blk.ref -= 1
            if blk.ref == 0:
                self.free_q[blk.id] = blk        # 进队尾，内容和 hash 都保留


def schedule_new_request(req, A, B):
    hits = A.match_prefix(req.tokens, B)
    for blk in hits:
        A.touch(blk)
    req.block_table = list(hits)
    req.num_computed = len(hits) * B             # 这些 token 直接跳过 prefill

    n_need = cdiv(len(req.tokens), B) - len(hits)
    req.block_table += [A.alloc() for _ in range(n_need)]
    # 只对 tokens[num_computed:] 跑 prefill，attention 照常读前面命中的 KV


def on_block_full(req, logical_idx, A, B):       # 每写满一个 block 调一次
    h = chain_hash(req.tokens[:(logical_idx + 1) * B])  # 实际实现会增量地缓存上一块的 hash
    A.register_full(req.block_table[logical_idx], h)
```

几个容易漏的点：

- **命中要留一个 token**：prompt 全部命中时也得至少算最后一个 token 的 forward，否则拿不到第一个输出 token 的 logits。所以 `match_prefix` 最多匹配到 `(len - 1) // B` 个 block。
- **淘汰发生在 alloc，不在 free**：free 只把 `ref` 降到 0，内容保留，这才让「上一轮对话结束、下一轮马上来」能命中。
- **hash 要包含前缀，而且要包含额外的 key**：同样的 16 个 token 出现在不同前缀后面，KV 不同。多模态输入（图片 hash）、LoRA id、cache salt 也要拌进 hash，否则会错误命中。
- **decode 生成的 token 也能被缓存**：`on_block_full` 在 decode 阶段同样会触发，所以多轮对话里上一轮的回答也能被下一轮命中。

### RadixAttention：另一种组织方式

[SGLang 的 RadixAttention](https://arxiv.org/abs/2312.07104) 解决的是同一个问题，但数据结构不同：

| | vLLM（block hash） | SGLang（radix tree） |
|---|---|---|
| 索引 | 扁平 hash 表：链式 hash → block | 一棵前缀树：边上是 token 序列，节点挂着对应的 KV |
| 匹配粒度 | 满 block（16 token）对齐，尾巴不足一个 block 的部分不共享 | 论文里每页 1 个 token，**按 token 匹配**，能命中任意长度的前缀 |
| 淘汰 | free 队列 LRU，按 block | 从叶子往上按 LRU 淘汰，只淘汰没人引用的叶子 |
| 代价 | 实现简单，查找就是逐 block 算 hash | 树的维护和 1 token 一页的元数据开销更大 |

「任意长度匹配」的本质就是**页大小为 1**：原理上 block hash 方案把 block size 缩到 1 也能做到同样的粒度（实际引擎支持的 block size 有下限），代价是 kernel 的间接寻址开销变大（见下方面试追问「block_size 该设多大」）。SGLang 的 `--page-size` 也可以调大，调大后同样只缓存整页，取舍和 vLLM 一样（以你所用版本的参数说明为准）。

两种结构的收益场景相同：多轮对话（每轮复用全部历史）、agent 的长 system prompt、树搜索类推理（多个分支共享前缀，radix tree 天然对应这种树形结构）。

### 抢占：swap vs recompute

显存满了而 running 的序列还要新 block 时，调度器必须抢占某个序列：

| | swap | recompute |
|---|---|---|
| 做法 | 把 block 拷到 CPU 内存，之后拷回 | 直接丢弃 block，恢复时把已生成的 token 当 prompt 重新 prefill |
| 代价 | 2 × block 字节 / PCIe 带宽 | 重新 prefill 的算力 |
| 适合 | 序列很长（重算贵） | 序列短，或有 prefix cache 能命中（命中的部分不用重算） |
| 副作用 | 占 CPU 内存，PCIe 和 H2D 拷贝争带宽 | 无额外内存，但 TTFT 尖刺 |

vLLM V1 的默认抢占模式是 recompute 而不是 swap，官方的理由是在 V1 架构下重算比换出换入更便宜（[vLLM 优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/)）。

**recompute 和 prefix cache 是互相成全的**：被抢占的请求释放 block 时，block 只是 `ref` 归零进了 free 队列，内容和 hash 还在（见上面的 Prefix caching 伪代码）。如果它被重新调度前这些 block 还没被别人淘汰，重新 prefill 时会直接命中，实际要重算的只有被淘汰掉的那一部分。所以 recompute 的真实代价往往比「整段重新 prefill」小得多。

### KV 量化

KV 从 bf16 降到 fp8 直接让 KV 访存和显存减半。decode 阶段 attention 的 **arithmetic intensity（AI，每读 1 字节做多少次浮点运算，FLOPs / bytes）** 很低，是 memory-bound 的：每个 KV 元素读进来只做一次乘加。字节数减半而 FLOPs 不变，AI 就翻倍，在 roofline 上往右移、同样的带宽能撑起两倍的计算（推导见 [Prefill vs Decode 与 Roofline](/inference/prefill-decode-roofline)）。注意点：
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

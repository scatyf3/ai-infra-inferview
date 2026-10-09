---
title: Simple Beam Search
status: draft
tags: [beam-search, handson]
difficulty: 2
order: 6
related: [/handson/top-p-sampling, /inference/kv-cache-paged-attention, /handson/decode-step-kv-cache, /stack/o-sampling, /handson/cuda-reduce]
stack: [o-sampling]
leetgpu: [98]
---

# Simple Beam Search

> 长度惩罚与 early stopping · [LeetGPU #98 Beam Search Step](https://leetgpu.com/challenges)（[题面](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/98_beam_search_step)）

## 一句话结论

beam search 每步保留 K 条累计 log-prob 最高的序列：每条 beam 展开 V 个候选，在 K × V 个候选里取 top-K 作为下一步的 beam，用整除和取余还原"来自哪条 beam、选了哪个 token"；遇到 EOS 的候选移进完成集。比贪心好在不会被一步的局部最优卡死，代价是 K 倍的计算和 KV cache，所以 LLM 服务里很少用。

## 定义与约定

- B：batch 里独立的请求数；K：beam 宽度；V：词表大小。
- `beam_scores[b, k]`：第 b 个请求第 k 条 beam 到目前为止的**累计 log-prob** $\sum_t \log P(y_t \mid y_{<t})$，≤ 0，越大越好。
- `token_logprobs[b, k, v]`：第 k 条 beam 下一个 token 是 v 的 log-prob，即模型 logits 做 `log_softmax` 的结果。
- 全部 row-major：`token_logprobs` 的 `[b, k, v]` 在 `b*K*V + k*V + v`。

一步的打分：

$$
\text{cand}[b, k, v] = \text{beam\_scores}[b, k] + \text{token\_logprobs}[b, k, v]
$$

把每个 b 的 K × V 个候选拍平，第 $i = k \cdot V + v$ 个，取最大的 K 个。还原：$k = \lfloor i / V \rfloor$（父 beam），$v = i \bmod V$（新 token）。

## 手撕：一步（LeetGPU #98）

题目只要求这一步：输入 `beam_scores [B, K]`、`token_logprobs [B, K, V]`，输出降序排好的 `new_beam_scores [B, K]`、`parent_beam_indices [B, K]`、`next_tokens [B, K]`（后两个 int32）。

```python
import torch


def beam_step(beam_scores, token_logprobs):
    # beam_scores [B, K], token_logprobs [B, K, V]
    B, K, V = token_logprobs.shape
    cand = beam_scores[:, :, None] + token_logprobs              # [B, K, V]，广播到每个 token
    top_vals, top_idx = cand.view(B, K * V).topk(K, dim=-1)      # 默认 sorted=True，降序
    return top_vals, top_idx // V, top_idx % V                   # 分数、父 beam、新 token


# beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens are tensors on the GPU
def solve(beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens,
          B: int, K: int, V: int):
    s, parent, tok = beam_step(beam_scores, token_logprobs)
    new_beam_scores.copy_(s)
    parent_beam_indices.copy_(parent.to(torch.int32))           # topk 返回 int64，题目要 int32
    next_tokens.copy_(tok.to(torch.int32))
```

题面样例走一遍：

```python
bs = torch.tensor([[-0.5, -1.0]])
lp = torch.tensor([[[-0.3, -1.2, -2.0, -2.5],
                    [-0.4, -0.1, -1.6, -3.2]]])
# cand：beam 0 → [-0.8, -1.7, -2.5, -3.0]，beam 1 → [-1.4, -1.1, -2.6, -4.2]
# 拍平后最大两个：-0.8（i = 0）、-1.1（i = 1·4 + 1 = 5）
print(beam_step(bs, lp))
# (tensor([[-0.8, -1.1]]), tensor([[0, 1]]), tensor([[0, 1]]))
```

### GPU 上这一步的代价

性能测试 B = 16、K = 8、V = 50,000：`token_logprobs` 有 16 × 8 × 50,000 × 4 B = 25.6 MB，每个元素只做一次加法和比较，memory-bound，T4（300 GB/s）下限约 85 µs。真正的工作是在每个 b 的 40 万个候选里选 8 个。

CUDA 里的标准做法是**两阶段 top-K**，和 [CUDA Reduce](/handson/cuda-reduce) 的两遍归约是同一个结构，只是"加法"换成了"保留最大的 K 个"：

1. 把每个 b 的 K·V 个候选切成 G 段，每个 block 负责一段，在寄存器 / shared memory 里维护本段的 top-K
2. 第二个 kernel（或同一个 kernel 的最后一个 block）把 G·K 个局部候选再取 top-K

在 torch 里模拟这两阶段，结果和一次 `topk` 完全一样：

```python
def beam_step_two_stage(beam_scores, token_logprobs, chunk=1024):
    B, K, V = token_logprobs.shape
    cand = (beam_scores[:, :, None] + token_logprobs).view(B, K * V)
    pad = (-(K * V)) % chunk
    cand = torch.nn.functional.pad(cand, (0, pad), value=float("-inf"))   # 补到 chunk 的整数倍
    G = cand.shape[1] // chunk
    loc_v, loc_i = cand.view(B, G, chunk).topk(min(K, chunk), dim=-1)     # 阶段 1：每段 top-K
    glob_i = loc_i + torch.arange(G)[None, :, None] * chunk               # 段内下标 → 全局下标
    v, j = loc_v.view(B, -1).topk(K, dim=-1)                              # 阶段 2：G·K 个里取 top-K
    idx = glob_i.view(B, -1).gather(-1, j)
    return v, idx // V, idx % V

bs, lp = torch.randn(3, 8), torch.randn(3, 8, 5003).log_softmax(-1)
assert all(torch.equal(a, b) for a, b in zip(beam_step(bs, lp), beam_step_two_stage(bs, lp)))
```

为什么对：全局 top-K 里的每个元素，在它所在的段里一定也是 top-K（段里比它大的不会超过 K − 1 个），所以不会在阶段 1 被丢掉。

## 手撕：完整的 beam search

一步之外还要处理四件事：

1. **第一步 K 条 beam 都一样**：开始时只有 prompt 一条序列。如果 K 条 beam 分数都是 0，第一步 top-K 会从 K 条相同的 beam 里选出同一个 token K 次。做法是只让 beam 0 活着：`scores = [0, -inf, …, -inf]`（HF transformers 里是 `beam_scores[:, 1:] = -1e9`，`generation/utils.py`）。
2. **EOS**：候选里 token = EOS 的，序列结束，移进完成集，不占下一步的 beam 名额。所以每步取 2K 个候选，保证去掉 EOS 后还剩 K 个。
3. **长度惩罚**：累计 log-prob 每多一个 token 就更负，不加处理时短序列天然占优。常用两种归一化：
   - HF transformers：$\text{score} = \text{sum\_logprobs} / L^{\alpha}$，L 是生成长度，α 即 `length_penalty`，默认 1.0（`BeamHypotheses.add`）
   - GNMT（[Wu et al., 2016](https://arxiv.org/abs/1609.08144) §7）：$lp(Y) = (5 + |Y|)^\alpha / (5 + 1)^\alpha$，分数除以 $lp(Y)$；他们在开发集上调出 α ∈ [0.6, 0.7] 通常最好（作者自测）
   - α = 0 就是不归一化；α > 0 鼓励长序列
4. **何时停**：HF 的 `is_done` 有三种模式：`early_stopping=True` 是完成集攒够 K 条就停；`False`（默认）是再加一个启发式检查，用"当前最好的活 beam 按**当前长度**归一化的分数"和完成集里最差的比，比不过就停；`"never"` 在 α > 0 时改用最大长度算，因为 α > 0 时活 beam 变长后归一化分数还可能变好，按当前长度算会提前停。

```python
def beam_search(step_logprobs, bos, eos, K, max_new, alpha=1.0):
    """step_logprobs(seqs [K, t]) -> [K, V]：每条 beam 下一个 token 的 log-prob"""
    seqs = torch.full((K, 1), bos)                                  # [K, t]，第 0 列是 bos
    scores = torch.full((K,), float("-inf")); scores[0] = 0.0       # 只有 beam 0 活着
    finished = []                                                   # (归一化分数, token 列表)

    for t in range(max_new):
        logp = step_logprobs(seqs)                                  # [K, V]
        V = logp.shape[-1]
        cand = (scores[:, None] + logp).view(-1)                    # [K*V]
        top_v, top_i = cand.topk(min(2 * K, cand.numel()))          # 多取 K 个给 EOS 留余量
        parent, tok = top_i // V, top_i % V

        new_seqs, new_scores = [], []
        for v, pa, tk in zip(top_v.tolist(), parent.tolist(), tok.tolist()):
            if v == float("-inf"):                                  # 死 beam 展开的候选
                break
            if tk == eos:
                gen_len = seqs.shape[1]                             # 已生成 t 个 + 这个 EOS
                finished.append((v / gen_len ** alpha, seqs[pa].tolist() + [tk]))
            else:
                new_seqs.append(torch.cat([seqs[pa], torch.tensor([tk])]))
                new_scores.append(v)
            if len(new_seqs) == K:
                break
        if not new_seqs:
            break
        seqs, scores = torch.stack(new_seqs), torch.tensor(new_scores)
        # 真实模型这里要按 parent 重排 KV cache：past_kv = past_kv[:, parent_of_each_new_beam]

        # early stopping（HF early_stopping=False 的启发式）
        if len(finished) >= K:
            finished = sorted(finished, key=lambda x: -x[0])[:K]
            best_alive = scores.max().item() / (seqs.shape[1] - 1) ** alpha
            if finished[-1][0] >= best_alive:
                break

    for v, s in zip(scores.tolist(), seqs.tolist()):                # 到 max_new 还没结束的也算候选
        finished.append((v / (len(s) - 1) ** alpha, s))
    return max(finished, key=lambda x: x[0])
```

### 正确性检查

用一个 bigram 玩具模型（下一个 token 的 log-prob 只依赖上一个 token），两个性质可以精确验证：

- **K = 1 就是贪心**
- **K 足够大时等于穷举**：K ≥ V^L 时每一步都不会丢候选，结果必须等于枚举所有长度 ≤ L 的序列取最优

```python
import itertools

V, EOS, BOS, L = 6, 0, 5, 4

def brute(table, alpha):
    best = (float("-inf"), None)
    for n in range(1, L + 1):
        for toks in itertools.product(range(V), repeat=n):
            if EOS in toks[:-1]: continue                   # EOS 只能在最后
            if n < L and toks[-1] != EOS: continue          # 没到最大长度就必须以 EOS 结尾
            s, prev = 0.0, BOS
            for tk in toks:
                s += table[prev, tk].item(); prev = tk
            if s / n ** alpha > best[0]:
                best = (s / n ** alpha, [BOS] + list(toks))
    return best

torch.manual_seed(0)
for trial in range(20):
    raw = torch.randn(V, V) * 2; raw[:, EOS] -= 2           # 让 EOS 别太早出现
    table = raw.log_softmax(-1)
    step = lambda seqs: table[seqs[:, -1]]
    for alpha in (0.0, 1.0):
        got = beam_search(step, BOS, EOS, K=V ** L, max_new=L, alpha=alpha)
        ref = brute(table, alpha)
        assert abs(got[0] - ref[0]) < 1e-5 and got[1] == ref[1]

# K = 1 等于贪心
g = beam_search(step, BOS, EOS, K=1, max_new=6, alpha=0.0)
seq = [BOS]
for _ in range(6):
    seq.append(table[seq[-1]].argmax().item())
    if seq[-1] == EOS: break
assert g[1] == seq
```

（这段在 CPU 上跑过，40 组全部和穷举一致。）

### KV cache 怎么跟着走

新的第 j 条 beam 来自父 beam `parent[j]`，它的 KV cache 必须是父 beam 的 cache 加上新 token。连续存储的 cache（`[layers, 2, K, heads, seq, d]`）每步要做一次 gather：

```python
past_kv = past_kv.index_select(dim=2, index=parent)   # K 条 beam 维度按父 beam 重排
```

这是一次整个 cache 的拷贝，序列越长越贵。PagedAttention 把它变成改 block table：多条 beam 共享前缀的物理 block，只增加引用计数，只有新 token 要写进一个仍被共享的 block 时才 copy-on-write 拷贝那一个 block。vLLM 论文里 OPT-13B 在 Alpaca trace 上做 beam search，块共享省下 37.6%–55.2% 的 KV cache 内存（[Kwon et al., 2023](https://arxiv.org/abs/2309.06180) §4.4、§6.3，作者自测）。见 [KV cache 与 PagedAttention](/inference/kv-cache-paged-attention)。

## 面试追问

::: details Q：为什么 LLM 服务基本不用 beam search？
一是成本：K 条 beam 是 K 倍的 decode 计算和 KV cache，同样显存下能服务的请求数降到约 1/K。二是质量：对开放式生成，最大化似然的解码倾向于输出重复、平淡的高频句子，Holtzman 等人把这叫 neural text degeneration，并提出用 nucleus sampling 替代（[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)）。beam search 主要留在翻译、摘要这类有参考答案的任务上。
:::

::: details Q：beam 越大越好吗？
不是。Stahlberg & Byrne 在 WMT15 英德上做精确搜索，发现超过一半的句子里模型分数最高的翻译是空串（[Stahlberg & Byrne, 2019](https://arxiv.org/abs/1908.10090)，作者自测）：模型本身偏好短输出，beam search 的"搜索误差"反而在帮忙。Meister 等人进一步认为 beam search 的效果来自它隐含的归纳偏置，而不是更接近 MAP 解（[Meister et al., 2020](https://arxiv.org/abs/2010.02650)）。所以 K 不是越大越好，通常用不大的 K 配长度惩罚，在开发集上调。
:::

::: details Q：为什么第一步要把其他 beam 的分数设成 −inf？
开始时只有一条序列（prompt）。如果复制成 K 条、分数都是 0，K × V 个候选里每个 token 都出现 K 次、分数相同，top-K 会选出 K 条一模一样的 beam。只让 beam 0 有分数，第一步的 top-K 就是 beam 0 的 top-K 个不同 token。
:::

::: details Q：top-K 选出来以后怎么知道来自哪条 beam？
候选拍平成 `[K * V]`，第 i 个对应 (k, v) = (i // V, i % V)。row-major 下 `cand[k, v]` 就在 `k * V + v`，所以整除得到行（父 beam），取余得到列（token）。
:::

## 参考

- Yonghui Wu et al., [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144), 2016（§7 长度归一化）
- [HF transformers `generation/beam_search.py`](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/beam_search.py)（`BeamHypotheses.add`、`is_done`）
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023
- Ari Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), ICLR 2020
- Felix Stahlberg, Bill Byrne, [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://arxiv.org/abs/1908.10090), EMNLP 2019
- Clara Meister, Tim Vieira, Ryan Cotterell, [If beam search is the answer, what was the question?](https://arxiv.org/abs/2010.02650), EMNLP 2020

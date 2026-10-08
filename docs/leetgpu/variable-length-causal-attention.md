---
title: Variable-Length Causal Attention
status: draft
tags: [attention, mask, varlen, leetgpu]
difficulty: 2
order: 4.7
related: [/leetgpu/mask, /leetgpu/causal-self-attention, /inference/batching-scheduling]
stack: [k-attn]
leetgpu: [102]
---

# Variable-Length Causal Attention

> [LeetGPU #102](https://leetgpu.com/challenges/variable-length-causal-attention) · 多条序列首尾相接 pack 成 `[T, d]`，用 `cu_seqlens` 标边界 · FlashAttention `varlen` / vLLM 的做法 · mask 写法见 [Attention mask 怎么拼](./mask)

## 题意

$S$ 条长短不一的序列拼成一个长度 $T$ 的 tensor，没有 padding。第 $b$ 条占 `[cu_seqlens[b], cu_seqlens[b+1])`，`cu_seqlens` 长度 $S + 1$，从 0 开始、最后一项是 $T$。

query $t$ 只能看**同一条序列里、不在自己之后**的 key $u$。mask 是沿对角线排开的几个小因果三角。`cu_seqlens = [0, 3, 5, 6]`：

```
     u: 0 1 2 3 4 5
t=0     ■ · · · · ·
t=1     ■ ■ · · · ·
t=2     ■ ■ ■ · · ·
t=3     · · · ■ · ·
t=4     · · · ■ ■ ·
t=5     · · · · · ■
```

## 题解

整体算 $T \times T$，用 mask 挡掉跨序列和未来的位置。T4 / PyTorch 提交通过，22.49 ms。

```python
import torch
import math


def solve(
    Q: torch.Tensor,           # [T, d]
    K: torch.Tensor,           # [T, d]
    V: torch.Tensor,           # [T, d]
    cu_seqlens: torch.Tensor,  # [S + 1] int32
    output: torch.Tensor,      # [T, d]
    T: int,
    d: int,
    S: int,
):
    attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)    # [T, T]
    i = torch.arange(T, device=Q.device)[:, None]                  # [T, 1] query
    j = torch.arange(T, device=Q.device)[None, :]                  # [1, T] key
    causal = j <= i
    # 每个位置属于第几条序列：编号 [0, 1, 2] 按长度 [3, 2, 1] 展开 → [0, 0, 0, 1, 1, 2]
    seg = torch.repeat_interleave(torch.arange(S, device=Q.device), cu_seqlens.diff())   # [T]
    allowed = (seg[:, None] == seg[None, :]) & causal              # 同一条序列 & 因果
    attn = attn.masked_fill(~allowed, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    torch.matmul(attn, V, out=output)                              # [T, d]
```

## 语法拆解

**`cu_seqlens.diff()`**：相邻两项的差，`[0, 3, 5, 6] → [3, 2, 1]`，从累积长度还原出每条的长度（`cumsum` 的逆）。

**`torch.repeat_interleave(x, repeats)`**：把 `x[k]` 原地重复 `repeats[k]` 次再拼起来，结果长度是 `repeats.sum()`：

```
x       = [0,       1,    2]
repeats = [3,       2,    1]
结果    = [0, 0, 0, 1, 1, 2]
```

和 `repeat` 不一样：`tensor([0, 1, 2]).repeat(2)` 是整体平铺 `[0, 1, 2, 0, 1, 2]`。GQA 把 KV head 复制给多个 Q head 用的也是 `repeat_interleave`。

等价写法：`torch.searchsorted(cu_seqlens[1:], torch.arange(T), right=True)`，给每个位置二分查它落在哪一段。

**`seg[:, None] == seg[None, :]`**：和 `i`、`j` 同一个广播套路，比的是「两个位置是不是同一条序列」。列向量 `[T, 1]` 和行向量 `[1, T]` 广播成 `[T, T]`，格子 $(t, u)$ 是 `seg[t] == seg[u]`，得到对角线上的几个方块；再 `& causal` 把每块削成下三角。

## 这题的坑

1. **`torch.repeat_interleave(torch.arange(S), device="cuda", cu_seqlens.diff())`**：`device=` 写到了外层，报 "positional argument follows keyword argument"（关键字参数后面不能再跟位置参数）。而且 `repeat_interleave` 根本没有 `device` 参数，device 是建 tensor 时给的：`torch.arange(S, device=Q.device)`。
2. **写死 `device="cuda"`**：能过，但用 `Q.device` 更稳，CPU 上调试也能跑。
3. **`B` / `S` 是什么**：序列条数，`cu_seqlens` 的长度减一。题目没给就 `len(cu_seqlens) - 1`。

## 为什么慢，怎么改

这个写法排名很靠后，原因：

- **算了整个 $T \times T$**：有用的只有对角块，计算量 $\sum_b L_b^2$；整体算是 $T^2$，$S$ 条等长时浪费接近 $(S - 1)/S$。这正是 packing 想省掉的东西。
- **好几个 $T \times T$ 中间 tensor**：`causal`、`seg ==`、`allowed`、`masked_fill`、`softmax` 各读写一遍显存，memory-bound。
- **没有融合 kernel**：matmul → mask → softmax → matmul 每步一个 launch，中间结果全落显存。

按序列循环 + PyTorch 自带的融合 attention（**未提交验证**）：

```python
import torch.nn.functional as F

cu = cu_seqlens.tolist()                       # 一次取回 CPU，只同步一次
for b in range(S):
    s, e = cu[b], cu[b + 1]
    output[s:e] = F.scaled_dot_product_attention(
        Q[s:e][None], K[s:e][None], V[s:e][None], is_causal=True     # [1, L, d]，默认 scale = 1/sqrt(d)
    )[0]
```

- 只算对角块；causal 在 kernel 里处理，不造 mask、不把 $L \times L$ 写回显存。T4（sm75）跑不了 Flash 后端，会走 memory-efficient 后端。
- 如果判题不认 SDPA 是「原生功能」，就在循环里写 matmul + causal mask + softmax，也能省掉跨序列那部分。
- $S$ 很大、每条很短时，循环的 launch 开销反而成了主要成本，这时整体调一次 SDPA、把 `allowed` 当 `attn_mask` 传进去可能更快。两种都交一下比。

FlashAttention 的 `varlen` kernel 本质上就是循环版：每个 program 先查 `cu_seqlens` 知道自己负责哪条序列的哪一段，只在这一段里算。

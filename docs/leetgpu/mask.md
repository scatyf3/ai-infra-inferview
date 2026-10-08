---
title: Attention mask 怎么拼
status: draft
tags: [attention, mask, guide]
difficulty: 2
order: 4.9
related: [/leetgpu/causal-self-attention, /leetgpu/sliding-window-self-attention, /leetgpu/attention-with-sinks]
stack: [k-attn]
---

# Attention mask 怎么拼

> 各种 attention 的 prefill 类题目，难点都在构造 mask：[Causal](./causal-self-attention)（#53）· [Sliding Window](./sliding-window-self-attention)（#59）· [Attention with Sinks](./attention-with-sinks)（#112）

## 一句话结论

**造一个 bool 的 `allowed`（True = 能看），用 `&` / `|` 组合各条规则，最后 `masked_fill(~allowed, -inf)` 一次。** 不要一上来就想「哪些要屏蔽」。

**标准写法：下标比较。** 本站所有 attention mask 统一用这个模板，每条规则一行，照公式写，不推 diagonal：

```python
i = torch.arange(M, device=Q.device)[:, None]   # [M, 1] query 下标（行）
j = torch.arange(M, device=Q.device)[None, :]   # [1, M] key 下标（列）
allowed = (j <= i) & ((j < S) | (i - j < W))    # 每条规则是 i、j 的一个条件
attn = attn.masked_fill(~allowed, float('-inf'))
attn = torch.softmax(attn, dim=-1)
```

记法：**行下标 `[:, None]`，列下标 `[None, :]`**，和 `attn[i, j]` 的顺序一致；写反会静默转置。
`tril` / `triu` 和「预分配再切片」放在 [后面](#备选-tril-triu-和预分配切片) 作备选。

## 约定：行是 query，列是 key

`attn = Q @ K^T` 的 shape 是 `[M_q, M_k]`，`attn[i, j]` 是 query $i$ 对 key $j$ 的分数。mask 和它同 shape，softmax 沿最后一维（key 方向）做。

所有规则都是 $i$、$j$ 的条件，所以先造两个下标，广播比较：

```python
i = torch.arange(M, device=Q.device)[:, None]   # [M, 1]
j = torch.arange(M, device=Q.device)[None, :]   # [1, M]
# i 和 j 比较时广播成 [M, M]
```

## 为什么写 allowed，而不是 masked

题目的规则一般是「能看 A，或者能看 B，并且不能看未来」，正着写就是逐字翻译：

```python
allowed = causal & (sink | window)
```

反过来写屏蔽区域，要用德摩根律，`&` 和 `|` 互换、每项取反，容易漏：

```python
masked = ~causal | (~sink & ~window)
```

另外 PyTorch `scaled_dot_product_attention` 和 FlashAttention 的 bool mask 也是 **True = 参与 attention**，习惯一致。

只有一条「排除」规则时（纯 causal、对称窗口），直接造屏蔽区域也行，不必强求。

## 两种表示和它们的组合方式

| 表示 | 长相 | 组合方式 | 适合 |
|---|---|---|---|
| 加法 mask | float，能看 = 0，不能看 = `-inf` | 相加：任一个是 `-inf` 结果就是 `-inf`，等于「允许」取**交集** | 单条规则；或者多条规则都要满足 |
| bool mask | True = 能看 | `&` 交集、`|` 并集、`~` 取反，随意组合 | 规则里有「或」 |

**有「或」就用 bool。** 加法 mask 表达不了并集，sink + window 那题就是这么卡住的。

bool 转成加法 mask 也就一行：`torch.zeros(M, M).masked_fill(~allowed, float('-inf'))`。

## 常见 mask 速查

记 $S$ = `num_sinks`，$W$ = `window_size`，各题对 $W$ 的定义不一样，以题面为准。

| 名称 | `allowed` | 例子 |
|---|---|---|
| 无 mask | 全 True | [Softmax Attention](./softmax-attention) |
| causal | `j <= i` | [#53](./causal-self-attention)，GPT 类 decoder |
| 对称窗口 | `(i - j).abs() <= W` | [#59](./sliding-window-self-attention)（$W$ 是单侧宽度）、Longformer |
| 因果窗口 | `(j <= i) & (i - j < W)` | Mistral、Gemma 2 的 SWA（$W$ 是总宽度含自己） |
| 因果窗口 + sink | `(j <= i) & ((j < S) \| (i - j < W))` | [#112](./attention-with-sinks)、StreamingLLM |
| key padding | `key_valid[None, :]`（`[1, M_k]` 广播） | batch 里变长序列右侧补齐的 pad 位置 |

**带 KV cache 的 decode / chunked prefill。** query 不是从位置 0 开始的：前面已经有 `past` 个 token 在 cache 里，新的 $M_q$ 个 query 的绝对位置是 `past + arange(M_q)`，key 是 `arange(past + M_q)`。下标换成绝对位置，上面所有公式照用：

```python
i = (past + torch.arange(M_q, device=dev))[:, None]   # [M_q, 1]
j = torch.arange(past + M_q, device=dev)[None, :]     # [1, M_k]
causal = j <= i                                       # [M_q, M_k]，不再是方阵
```

## 备选：tril / triu 和预分配切片

不是标准写法，但读别人代码（包括本站早期题解）会遇到。

`torch.triu(x, diagonal=k)` **保留** $j - i \ge k$ 的元素、其余置 0（bool 时置 False）；`torch.tril(x, diagonal=k)` 保留 $j - i \le k$。

| 想要 | 写法 |
|---|---|
| 下三角含对角线（causal 可见区） | `tril(ones)` |
| 严格上三角（causal 屏蔽区） | `triu(ones, diagonal=1)` |
| $i - j \le W - 1$ | `triu(ones, diagonal=-(W - 1))` |
| $j - i \ge W + 1$（对称窗口右侧屏蔽区） | `triu(full_inf, diagonal=W + 1)` |
| $i - j \ge W + 1$（对称窗口左侧屏蔽区） | `tril(full_inf, diagonal=-(W + 1))` |

`tril` / `triu` 只会截对角带，「前几列」这种规则（sink、padding）表达不了，还得写下标比较。所以**规则一多就直接用 `i`、`j` 比较**，每条规则一行，不用推 diagonal。

预分配再切片：先 `zeros` 一个 bool 矩阵，再一块块涂。「前几列」这种规则用切片很直观，适合白板上边画边讲：

```python
allowed = torch.zeros(M, M, dtype=torch.bool, device=Q.device)
allowed[:, :S] = True                                              # sink 列
allowed |= torch.triu(torch.ones_like(allowed), diagonal=-(W - 1)) # 窗口（此时还含未来）
allowed &= torch.tril(torch.ones_like(allowed))                    # causal 裁掉上三角
```

别用 Python 双重 for 循环逐格填：$M^2$ 次 Python 操作、每次单独写 GPU，M 一大就超时。

两个用法上的坑：
- 它们不生成矩阵，签名是 `(input, diagonal=k)`。先 `torch.ones(M, M, dtype=torch.bool)` 或 `torch.full((M, M), float('-inf'))`，再截。
- 截 float `-inf` 矩阵时，保留下来的是**屏蔽区**；截 bool `ones` 时，保留下来的是**可见区**。别把两种心智混在一起。

## 被屏蔽的位置填什么

填 `-inf`：$e^{-\infty} = 0$，softmax 之后这些位置的权重严格为 0，不参与加权求和。

- **不能填 0。** 0 是一个正常的分数，softmax 后照样分到权重。
- **整行都被屏蔽会出 NaN。** 分子分母都是 0。causal 类 mask 对角线一定可见，不会出现；但 padding mask 下，pad 位置自己当 query 时可能整行屏蔽，需要事后把这些行的输出清零，或者给它们留一个可见位置。
- HF transformers 用 `torch.finfo(dtype).min` 代替 `-inf`：整行屏蔽时 softmax 退化成均匀分布而不是 NaN，fp16 下也不会和别的 mask 相加溢出成 NaN。做题用 `float('-inf')` 就行。

## 踩过的坑

1. **`torch.triu(window_size // 2, (M, M), -inf)`**：`triu` 不生成矩阵，见上。
2. **diagonal 差一**：屏蔽 $j - i > W$ 要写 `diagonal=W + 1`，写成 `W` 会把边界那条也屏蔽掉。
3. **用 `-inf` 相加拼并集**：加法只能取交集，sink + window 要 bool 的 `|`。
4. **窗口条件写成 $i + j$**：窗口看的是离对角线多远，是 $i - j$。
5. **因果窗口忘了 `& causal`**：$j > i$ 时 $i - j$ 是负数，`i - j < W` 恒成立，等于看到了未来。
6. **`softmax(dim=0)`**：方阵沿错方向不会报 shape 错，静默算错。key 方向是 `dim=-1`。
7. **`window_size` 理解错**：单侧还是总宽度、含不含自己，各题不同，看最小测例反推。
8. **mask 建在 CPU 上**：`torch.full` / `torch.arange` 默认在 CPU，要传 `device=Q.device`。

---
title: Attention 题通用模板
status: draft
tags: [attention, guide]
difficulty: 1
order: 0.5
related: [/leetgpu/mask, /handson/naive-attention, /handson/mha-gqa-forward, /inference/attention-variants]
stack: [k-attn]
---

# Attention 题通用模板

> LeetGPU 上所有 attention 题都是同一个骨架，区别只在「怎么排成 `[head, seq, D]`」「位置编码怎么加」和「mask 是什么」。先背骨架，再看每题改了哪一行。mask 细节见 [Attention mask 怎么拼](./mask)。

## 骨架：六步

```python
import torch
import math


def attention(q, k, v, allowed=None, cos=None, sin=None):
    # 1. reshape for input：调用前做完
    # q: [..., M_q, D]   k, v: [..., M_k, D], 前面是head/batch，matmul当作batch维度
    # 2. pos encoding：只转 q、k，v 不动
    if cos is not None:
        q, k = rope(q, cos, sin), rope(k, cos, sin)
    # 3. generate attention score
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    # [..., M_q, M_k]
    # 4. build mask
    if allowed is not None:
        attn = attn.masked_fill(~allowed, float('-inf'))
    # 5. softmax
    # 沿 key 方向，对每个query的概率softmax
    attn = torch.softmax(attn, dim=-1)
    # 6. output
    return torch.matmul(attn, v) # [..., M_q, D]


def rope(x, cos, sin):
    # x: [..., M, D]   cos, sin: [M, D]，广播到前面的 head 维
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)   # rotate_half
    return x * cos + x_rot * sin
```

| 步 | 做什么 | 要点 |
|---|---|---|
| 1. 排 shape | 把输入整理成 `[H, M, D]`（GQA 是 `[H_kv, G, M, D]`） | head 必须在 seq **前面**，matmul 只对最后两维做矩阵乘 |
| 2. 位置编码 | 对 q、k 做 RoPE | 必须在拆 head **之后**：旋转是在单 head 的 `D` 维里配对的；v 不转 |
| 3. 打分 | `q @ k^T / sqrt(D)` | `D` 是**单 head** 维度；转置用 `transpose(-1, -2)` |
| 4. mask | `masked_fill(~allowed, -inf)` | `i` 行下标 `[:, None]`、`j` 列下标 `[None, :]`，带 `device=Q.device` |
| 5. softmax | `torch.softmax(attn, dim=-1)` | 方阵沿错方向不报错，静默算错 |
| 6. 乘 V、写回 | `attn @ v`，换回题目的 shape，写进 `output` | shape 完全一致才能 `out=output`，否则 `output.copy_(...)` |

**为什么位置编码排在拆 head 之后**：RoPE 把第 $k$ 维和第 $k + D/2$ 维配成一对旋转，$D$ 是单 head 维度。在 `[N, d_model]` 上直接转，配对会跨 head（head 0 的前半和 head $h/2$ 的前半配成一对），shape 对、数值错。

下面按「基础 → 换 mask → 换 shape → 位置编码」的顺序逐题展开。每题只讲相对骨架改了什么、为什么这么改、容易错在哪；完整提交代码和踩坑记录在各题页面。

## 基础

### #6 Softmax Attention：骨架本身

[题解](./softmax-attention) · 单头、无 mask、无投影

`Q` 是 `[M, d]`，`K` / `V` 是 `[N, d]`。没有 head 维，第 1 步什么都不用做；没有位置编码和 mask，第 2、4 步跳过。剩下三步一行写完：

```python
attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)   # [M, N]
attn = torch.softmax(attn, dim=-1)                           # 每个 query 一行，沿 key 归一化
torch.matmul(attn, V, out=output)                            # [M, d]，和 output 同 shape，可以 out=
```

**shape 怎么想**：`attn[i, j]` 是 query $i$ 对 key $j$ 的分数，所以行数是 query 个数 $M$、列数是 key 个数 $N$。softmax 是「每个 query 把注意力分给所有 key」，所以沿列（最后一维）做。乘 V 时 `[M, N] @ [N, d]`，key 维被求和掉，每个 query 得到一个 `d` 维向量。

**为什么除 $\sqrt{d}$**：$q \cdot k$ 是 $d$ 项求和，各分量独立、方差为 1 时点积方差是 $d$。不缩放的话 $d$ 一大，softmax 的输入幅度很大，输出接近 one-hot，梯度几乎为 0。

**容易错**：
- 结果没写进 `output`。`return` 和 `output = ...` 都不算，要 `out=output` 或 `output.copy_(...)`。
- `softmax` 不写 `dim`，或者写成 `dim=0`。

## 换 mask：第 4 步不同

这几题都是单头 `[M, d]`、Q 和 K 等长，其余步骤和 #6 一模一样，只多一段构造 `allowed`。统一写法：

```python
i = torch.arange(M, device=Q.device)[:, None]   # [M, 1] query 下标（行）
j = torch.arange(M, device=Q.device)[None, :]   # [1, M] key 下标（列）
allowed = ...                                   # [M, M] bool，True = 能看
attn = attn.masked_fill(~allowed, float('-inf'))
```

`i`、`j` 广播比较得到 `[M, M]`，每条规则就是 $i$、$j$ 的一个条件，照题意逐字翻译。

### #53 Causal Self-Attention：不看未来

[题解](./causal-self-attention) · decoder（GPT 类）的标准 mask

query $i$ 只能看 $j \le i$ 的 key，可见区域是**下三角含对角线**：

```python
allowed = j <= i
```

```
     j: 0 1 2 3
i=0     ■ · · ·
i=1     ■ ■ · ·
i=2     ■ ■ ■ ·
i=3     ■ ■ ■ ■
```

**为什么**：自回归生成时第 $i$ 个 token 还看不到后面的 token；训练 / prefill 时一次算整条序列，必须用 mask 模拟「看不到」，否则模型能偷看答案。

**为什么填 `-inf` 而不是 0**：$e^{-\infty} = 0$，softmax 后权重严格为 0。填 0 只是一个普通分数，照样分到权重。

**容易错**：
- 写成 `j < i`（或 `triu(0)`）：对角线被盖掉，第 0 行全是 `-inf`，softmax 得 NaN。
- 行列搞反：行是 query、列是 key，`i` 必须是 `[:, None]`。写反等于转置，方阵不报错。
- `arange` 没带 `device`，和 GPU 上的 `attn` 运算报 device 不一致。

### #59 Sliding Window：只看附近

[题解](./sliding-window-self-attention) · **双向**窗口，`window_size` 是单侧宽度

query $i$ 只看离自己不超过 $W$ 的 key，前后都看：

```python
allowed = (i - j).abs() <= window_size
```

`M = 6, W = 1` 时是一条宽 3 的对角带：

```
     j: 0 1 2 3 4 5
i=0     ■ ■ · · · ·
i=1     ■ ■ ■ · · ·
i=2     · ■ ■ ■ · ·
i=3     · · ■ ■ ■ ·
i=4     · · · ■ ■ ■
i=5     · · · · ■ ■
```

**为什么**：全 attention 是 $O(M^2)$；窗口把每个 query 看的 key 限制在 $2W+1$ 个，计算和 KV cache 都降到 $O(MW)$。Longformer 是双向窗口，Mistral / Gemma 2 是因果窗口（见 #112）。

**`window_size` 的含义各家不同**：单侧还是总宽、含不含自己，看最小测例反推。这题 `M = 2, W = 1` 时两个位置互相可见，所以是单侧宽度，**不要** `// 2`。

**容易错**：
- 写成 `i + j`：窗口看的是离对角线多远，是 $i - j$。
- 用 `triu` / `tril` 拼时 diagonal 差一（要 `W + 1` 不是 `W`）。用下标比较就没有这个问题。

### #112 Attention with Sinks：因果窗口 + 常驻开头

[题解](./attention-with-sinks) · 因果、`window_size` 是总宽度含自己

规则有三条：不看未来；最近 $W$ 个能看；前 $S$ 个永远能看。

```python
causal = j <= i
sink   = j < num_sinks
window = (i - j) < window_size
allowed = causal & (sink | window)
```

`S = 2, W = 3, M = 8`：

```
     j: 0 1 2 3 4 5 6 7
i=4     ■ ■ ■ ■ ■ · · ·
i=5     ■ ■ · ■ ■ ■ · ·
i=6     ■ ■ · · ■ ■ ■ ·
i=7     ■ ■ · · · ■ ■ ■
```

**为什么要 sink**：StreamingLLM 发现模型会把大量 attention 权重「倒」在开头几个 token 上（softmax 权重必须和为 1，没特别想看的时候总得有个去处）。纯滑窗把开头滑出去后分布被打乱，长文本困惑度崩掉；保留前几个 token 常驻就稳了，显存是 $O(S + W)$。

**这题为什么必须用 bool**：规则里有「或」。加法 mask（能看 = 0，不能看 = `-inf`）相加只能表达「同时满足」，即交集，拼不出 `sink | window`。

**容易错**：
- `window` 忘了 `& causal`：$j > i$ 时 $i - j$ 是负数，`< W` 恒成立，看到了未来。
- 和 #59 的窗口定义混用：#59 是 `abs(i - j) <= W`，这题是 `0 <= i - j < W`。

### #102 Variable-Length Causal：多条序列打包

[题解](./variable-length-causal-attention) · 没有 padding，靠 `cu_seqlens` 标边界

$S$ 条长短不一的序列首尾相接成 `[T, d]`，第 $b$ 条占 `[cu_seqlens[b], cu_seqlens[b+1])`。query 只能看**同一条序列里、不在自己之后**的 key，mask 是沿对角线排开的几个小下三角：

```python
seg = torch.repeat_interleave(torch.arange(S, device=Q.device), cu_seqlens.diff())   # [T]，每个位置属于第几条
allowed = (seg[:, None] == seg[None, :]) & (j <= i)
```

`cu_seqlens = [0, 3, 5, 6]` 时 `seg = [0, 0, 0, 1, 1, 2]`：

```
     j: 0 1 2 3 4 5
i=0     ■ · · · · ·
i=1     ■ ■ · · · ·
i=2     ■ ■ ■ · · ·
i=3     · · · ■ · ·
i=4     · · · ■ ■ ·
i=5     · · · · · ■
```

**拆开看**：
- `cu_seqlens.diff()` 把累积长度还原成每条的长度 `[3, 2, 1]`。
- `repeat_interleave(arange(S), 长度)` 把编号 $b$ 重复 $L_b$ 次，得到每个位置的序列编号。
- `seg[:, None] == seg[None, :]` 和 `i`、`j` 是同一个广播套路，比的是「是不是同一条」，得到对角方块；再 `& causal` 削成下三角。

**为什么这么做**：batch 里序列长短不一，补 padding 会浪费算力；vLLM 和 FlashAttention 的 `varlen` 接口都是这样把序列首尾拼起来，用 `cu_seqlens` 区分。

**这个写法的问题**：整体算了 $T \times T$，有用的只有对角块，正是 packing 想省掉的东西。按序列循环、每段单独算会快很多，见题解页「为什么慢」。

### 带 KV cache 时的 mask

decode 或 chunked prefill 时，cache 里已经有 `past` 个 token，新来的 $M_q$ 个 query 的绝对位置从 `past` 开始。只要把下标换成绝对位置，上面所有公式照用：

```python
i = (past + torch.arange(M_q, device=dev))[:, None]   # [M_q, 1]
j = torch.arange(past + M_q, device=dev)[None, :]     # [1, M_k]
allowed = j <= i                                      # [M_q, M_k]，不再是方阵
```

decode 时 $M_q = 1$，`i = past`，所有 key 都 $\le$ `past`，causal mask 全 True，可以省掉。

## 换 shape：第 1、6 步不同

这几题没有 mask，难点全在把 head 维挪到 seq 前面、算完再挪回去。

**核心规则**：`matmul` 只对最后两维做矩阵乘，前面所有维都当 batch。所以 head 必须排在 seq 前面：`[H, M, D] @ [H, D, N] = [H, M, N]`，等于对每个 head 各做一次 attention。

### #12 Multi-Head Attention：自己拆 head

[题解](./multi-head-attention) · 输入 `[N, d_model]`，无投影、无 mask

```python
dk = d_model // h
q = Q.reshape(N, h, dk).transpose(0, 1)           # [N, d_model] → [N, h, dk] → [h, N, dk]
k = K.reshape(N, h, dk).transpose(0, 1)
v = V.reshape(N, h, dk).transpose(0, 1)

out = attention(q, k, v)                          # [h, N, dk]，缩放用 sqrt(dk)
out = out.transpose(0, 1).reshape(N, d_model)     # [h, N, dk] → [N, h, dk] → [N, d_model]
output.copy_(out)
```

**两步各干什么**：
- `reshape(N, h, dk)`：把每个 token 的 `d_model` 切成 `h` 段，每段 `dk`。只是**拆开**相邻维度，内存顺序不变，用 `reshape` / `view`。
- `transpose(0, 1)`：把 head 挪到最前当 batch。这是**调换**维度顺序，必须用 `transpose`。

合回去是逆过程：先 `transpose(0, 1)` 换回 `[N, h, dk]`，再 `reshape` 合并后两维。transpose 之后不连续，用 `reshape` 不用 `view`。

**为什么多头**：每个 head 在一个 `dk` 维子空间里独立算 attention，可以同时关注不同的关系（语法、指代、位置……），总计算量和单个 `d_model` 维 head 一样。

**容易错**：
- 不 transpose 直接 matmul：`[N, h, dk] @ [N, dk, h]` 的 batch 维是 N，算出来是每个 token 内 head 之间的相似度，shape `[N, h, h]`，不报错。
- 缩放用 `sqrt(d_model)`：除的是单 head 维度 `sqrt(dk)`。
- `dk = d_model / h`：得到 float，`reshape` 报错。用 `//`。
- `matmul(..., out=output)`：结果是 `[h, N, dk]`，和 output 对不上，必须 `copy_`。

### #26 Multi-Head Cross-Attention：Q 和 K/V 不等长

[题解](./multi-head-cross-attention) · 输入已经拆好 head：`Q [M, H, D]`，`K/V [N, H, D]`

head 已经拆好，只需要把它挪到前面：

```python
q = Q.transpose(0, 1)                    # [H, M, D]
k = K.transpose(0, 1)                    # [H, N, D]
v = V.transpose(0, 1)                    # [H, N, D]
out = attention(q, k, v)                 # [H, M, N] 的 attn → [H, M, D]
output.copy_(out.transpose(0, 1))        # [M, H, D]；copy_ 接受不连续的 tensor
```

**和 self-attention 的区别**：Q 来自一个序列（decoder），K/V 来自另一个（encoder 输出 / 图像特征），所以长度 $M \ne N$，attn 是 `[H, M, N]` 的长方形。骨架完全不用改，matmul 自己处理。

**einsum 写法**，不用想维度顺序：

```python
scores = torch.einsum('mhd,nhd->hmn', Q, K) / math.sqrt(D)
attn = torch.softmax(scores, dim=-1)
output.copy_(torch.einsum('hmn,nhd->mhd', attn, V))
```

**容易错**：
- 用 `reshape(H, M, D)` 换维度：`[M, H, D]` 在内存里是 token 优先排的，`reshape` 只是重新切开同一串数据，切出来的「head 0」是好几个 token、好几个 head 混在一起。shape 对、数值错。
- 小测例测不出来：$M = N = H = 1$ 时 `reshape` 和 `transpose` 结果一样。

### #80 Grouped Query Attention：K/V head 少于 Q head

[题解](./grouped-query-attention) · `Q [H_q, S, D]`，`K/V [H_kv, S, D]`，无 mask

每 $G = H_q / H_{kv}$ 个 Q head 共用一个 KV head：第 $h$ 个 Q head 用第 $h // G$ 个 KV head。不复制 K/V，靠广播：

```python
G = num_q_heads // num_kv_heads
q = Q.view(num_kv_heads, G, seq_len, head_dim)        # [H_kv, G, S, D]：组号在前、组内序号在后
k = K.unsqueeze(1)                                    # [H_kv, 1, S, D]
v = V.unsqueeze(1)
out = attention(q, k, v)                              # batch 维 (H_kv, G) 对 (H_kv, 1)，广播
output.copy_(out.reshape(num_q_heads, seq_len, head_dim))
```

**为什么 GQA**：decode 是 memory-bound，瓶颈是读 KV cache。KV head 减到 $1/G$，cache 显存和带宽都降到 $1/G$，效果比 MQA（只有 1 个 KV head）好。Llama 2 70B、Llama 3 都用 GQA。

**为什么组号在前**：`view(H_kv, G, ...)` 把相邻的 $G$ 个 Q head 归到同一组，对应 `h // G`（`repeat_interleave` 语义）。写成 `view(G, H_kv, ...)` 也能广播，但对应关系变成 `h % H_kv`，静默算错。

**容易错**：
- K/V 不插那个 1：3 维的 `[H_kv, S, D]` 和 4 维 Q 广播时从右往左对齐，`H_kv` 会对上 `G`。样例里两者恰好都是 2，不报错。
- 第二个 matmul 用了原始大写 `V`：插过维度的是小写 `v`。
- 转置 K 用 `transpose(0, 1)`：换的是 head 和 seq。要换最后两维。

## 位置编码：第 2 步不同

**为什么要位置编码**：第 3–6 步里没有任何东西知道 token 的顺序。把 k、v 的行按同一个顺序打乱，每个 query 的输出不变：softmax 加权求和不在乎 key 排第几。所以「猫追狗」和「狗追猫」里，「追」看到的是同一组 key、得到同一个输出。顺序只能从外面塞进去。

**三种塞法，改的是骨架的不同步**：

| 方法 | 改哪一步 | 怎么改 | 谁在用 |
|---|---|---|---|
| 绝对位置（正弦 / 可学习） | 第 1 步之前，投影之前 | 输入加位置向量：$x_m + p_m$ | 原始 Transformer、BERT、GPT-2 |
| RoPE | 第 2 步 | 按位置旋转 q、k：$q_m = R_m q$，$k_n = R_n k$ | Llama、Qwen、DeepSeek |
| ALiBi | 第 4 步 | 分数减一个和距离成正比的偏置：$s_{ij} - c_h (i - j)$ | BLOOM、MPT |

符号：$m$、$n$ 是 token 位置；$p_m$ 是位置 $m$ 的 $d_{\text{model}}$ 维向量；$R_m$ 是按位置 $m$ 的旋转（下面 #61）；$s_{ij}$ 是 query $i$ 对 key $j$ 的分数；$c_h$ 是第 $h$ 个 head 的固定斜率，不训练。

现在的 LLM 基本都用 RoPE，LeetGPU 也只考它，下面两题都是 RoPE。

### #61 Rotary Positional Embedding：旋转本身

[题解](./rotary-positional-embedding) · 给好 `cos` / `sin`，只练旋转

把 `D` 维向量看成 $D/2$ 对，每对按位置转一个角度。用「前后两半配对」的写法（第 $k$ 维和第 $k + D/2$ 维是一对）：

```python
half = D // 2
x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)   # rotate_half
out = x * cos + x_rot * sin
```

**对照二维旋转**：一对 $(a, b)$ 转角度 $\theta$ 得到 $(a\cos\theta - b\sin\theta,\ b\cos\theta + a\sin\theta)$。`rotate_half` 把 $(a, b)$ 变成 $(-b, a)$，所以 `x * cos + rotate_half(x) * sin` 正好是这个公式。

**为什么 RoPE**：q 在位置 $m$ 转 $m\theta$，k 在位置 $n$ 转 $n\theta$，两个向量的夹角多了 $(m - n)\theta$，点积只依赖相对位置 $m - n$。算一个：$D = 2$，$q = k = (1, 0)$，$\theta = 30°$。

| | q 转到 | k 转到 | 夹角 | 点积 |
|---|---|---|---|---|
| $m = 3, n = 1$ | 90° | 30° | 60° | $\cos 60° = 0.5$ |
| $m = 5, n = 3$ | 150° | 90° | 60° | $\cos 60° = 0.5$ |

绝对位置不同、距离都是 2，分数一样。位置信息直接进了 attention 分数，而不是加在输入上。

**只作用在 q 和 k**，v 不动：位置只需要影响「谁看谁」，不需要影响「看到什么内容」。

### 带 KV cache 时的位置

和 [带 KV cache 时的 mask](#带-kv-cache-时的-mask) 一样，位置要用绝对位置：新来的 $M_q$ 个 token 位置是 `past + arange(M_q)`，取 `cos[past : past + M_q]`。

cache 里的 k 在写进去之前就按自己的位置转好了（#115 的第 3、4 步），读出来直接用，不再转。所以骨架里 q、k 共用一份 `cos` / `sin` 只适用于没有 cache 的 self-attention；有 cache 时第 2 步只转新 token 的 q、k。

### #115 Fused QKV Projection + RoPE + KV Cache Update：decode 的前半步

[题解](./fused-qkv-projection-with-rope-and-kv-cache-update) · 不算 attention，只准备好 q 和 cache

decode 一步里，算 attention 之前要做的事全在这题：

1. **投影**：`x @ W_qkv` 一次算出 q、k、v，`view(B, H_q + 2*H_kv, D)` 后沿 head 维切三段。
2. **取 cos / sin**：每个 batch 在自己的位置上，`cos_sin_cache[positions]` 取出 `[B, D]`，补 head 维成 `[B, 1, D]` 好广播。
3. **RoPE**：对 q、k 各做一次 #61 的旋转。
4. **写 cache**：k、v 写进 `K_cache[b, :, positions[b], :]`。

```python
batch_idx = torch.arange(B, device=positions.device)
K_cache[batch_idx, :, positions, :] = k     # [B, H_kv, D]
V_cache[batch_idx, :, positions, :] = v
```

**写 cache 的索引是难点**：`[:, :, positions, :]` 会把每个 batch 写到所有 batch 的位置上（全组合）；batch 维也用索引张量 `batch_idx`，两个索引张量才会逐位配对成 (0, p0)、(1, p1)。

**为什么融合**：投影、RoPE、写 cache 分开做，中间结果要来回读写显存；融合成一个 kernel 只读 x 一次。

## 还没做

- **#96 INT8 KV-Cache Attention**：cache 以 INT8 存，读出来先乘 scale 反量化，再走骨架。scale 的粒度看题面。
- **#114 MLA Decode**：路线终点。

## 面试默写顺序

1. #6：六步骨架（没有位置编码和 mask 时是三行）
2. #53：加 `i` / `j` 和 `allowed`
3. #12：加拆 head / 合 head
4. #80：把 head 拆成「组 × 组内」
5. 剩下的都是换 mask 或者换 shape

## 所有 attention 题共有的坑

1. **缩放除错**：除 `sqrt(单 head 维度)`，不是 `sqrt(d_model)`；`math.sqrt`，不是 `torch.sqrt`。
2. **softmax 维度**：`dim=-1`。
3. **head 没挪到前面**：`[N, h, dk] @ [N, dk, h]` 算的是 head 之间的相似度，不报错。
4. **用 reshape 换维度**：shape 对、数据乱，小测例测不出来。
5. **mask 建在 CPU 上**：`arange` / `full` 要带 `device=Q.device`。
6. **`out=output` 但 shape 对不上**：output 被 resize，要局部算完再 `copy_`。

其余见 [通用语法坑](./#通用语法坑)。

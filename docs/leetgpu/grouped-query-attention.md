---
title: Grouped Query Attention
status: draft
tags: [attention, gqa, leetgpu]
difficulty: 2
order: 3
related: [/handson/mha-gqa-forward, /inference/attention-variants]
stack: []
leetgpu: [80]
---

# Grouped Query Attention

> [LeetGPU #80](https://leetgpu.com/challenges/grouped-query-attention) · 无 mask（双向注意力），Q 已按 head 拆好 · 概念见 [MHA / GQA Forward](/handson/mha-gqa-forward)

## 题解

不 `repeat_kv`：给 K/V 插一个长度 1 的组内维度，靠 matmul 的 batch 广播复用。

```python
import torch
import math


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
    num_q_heads: int, num_kv_heads: int, seq_len: int, head_dim: int,
):
    # Q: [H_q, S, D]；K/V: [H_kv, S, D]
    q_div_kv = num_q_heads // num_kv_heads
    # 第 i 个 Q head 用第 i // q_div_kv 个 KV head：组号在前、组内序号在后
    q = Q.view(num_kv_heads, q_div_kv, seq_len, head_dim)
    k = K.reshape(num_kv_heads, 1, seq_len, head_dim)       # 等价于 K.unsqueeze(1)
    v = V.reshape(num_kv_heads, 1, seq_len, head_dim)
    kt = k.transpose(2, 3)
    # batch 维广播从右往左对齐：(H_kv, q_div_kv) 对 (H_kv, 1)
    attn = torch.matmul(q, kt) / math.sqrt(head_dim)        # [H_kv, q_div_kv, S, S]
    attn_norm = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn_norm, v)                         # [H_kv, q_div_kv, S, D]
    out = out.reshape(num_q_heads, seq_len, head_dim)
    output.copy_(out)
```

## 这题的坑

1. **K/V 漏了组内维度**：Q 拆成 `[H_kv, q_div_kv, S, D]` 后，K/V 必须是 `[H_kv, 1, S, D]`。直接用 3 维的 `[H_kv, S, D]`，从右往左对齐会让 `H_kv` 对上 `q_div_kv`。样例是 4 个 Q head、2 个 KV head，两者都是 2，不报错，结果静默错。
2. **第二个 matmul 用了大写 `V`**：插了维度的是小写 `v`，原始的 `V` 还是 3 维，又回到第 1 条。
3. **Q 的分组顺序拆反**：`Q.view(q_div_kv, H_kv, ...)` 也能广播，但对应关系变成 `i % H_kv`（`repeat` 语义）。正确的是 `i // q_div_kv`（`repeat_interleave` 语义，同组 Q head 相邻），所以组号必须在前。
4. **`K.transpose(0, 1)`**：交换的是 head 维和 seq 维。转置要换最后两维。

其余见 [通用语法坑](./#通用语法坑)。

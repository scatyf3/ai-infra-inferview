---
title: 带 KV Cache 的 Decode Step
status: draft
tags: [kv-cache, decode, handson]
difficulty: 3
order: 2
related: [/handson/mha-gqa-forward, /inference/kv-cache-paged-attention, /inference/memory-accounting]
---

# 带 KV Cache 的 Decode Step

> 增量计算与 cache 更新

## 一句话结论

decode 和 prefill 的唯一区别：**Q 只有 1 个 token，K/V 要和历史 cache 拼接，不需要 causal mask**。写的时候盯住三处：cache 的预分配布局、`start_pos` 的推进、以及 score 的 shape 从 `[B,H,S,S]` 变成 `[B,H,1,S_kv]`。

## 交互

切到 decode 看每一步的 shape 变化，特别是标了「KV cache 读」的那一步。

<ShapeFlow variant="gqa" phase="decode" />

## 手撕

::: code-group

```python [预分配 cache 版（面试首选）]
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodeAttention(nn.Module):
    """预分配连续 cache 的经典写法。真实框架用 PagedAttention 的 block table 替代。"""

    def __init__(self, d_model, n_heads, n_kv_heads, max_batch, max_seq):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # 一次性按最大长度分配，避免每步 cat 造成的反复搬运
        shape = (max_batch, max_seq, n_kv_heads, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """x: [B, S, d]。prefill 时 S = prompt_len 且 start_pos = 0；decode 时 S = 1。"""
        B, S, _ = x.shape
        end = start_pos + S

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        # （此处对 q, k 施加 RoPE，位置是 start_pos .. end-1）

        # 1) 写入 cache：只写新的这 S 个位置
        self.k_cache[:B, start_pos:end] = k
        self.v_cache[:B, start_pos:end] = v

        # 2) 读出全部历史：[B, S_kv, H_kv, d_h]，S_kv = end
        keys = self.k_cache[:B, :end]
        vals = self.v_cache[:B, :end]

        # 3) 转到 [B, H, S, d_h] 并做 GQA 广播
        q = q.transpose(1, 2)
        keys = self.repeat_kv(keys.transpose(1, 2), self.n_rep)
        vals = self.repeat_kv(vals.transpose(1, 2), self.n_rep)

        # 4) score: [B, H, S, S_kv]。decode 时 S = 1，不需要 mask
        scores = (q @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if S > 1:  # 只有 prefill 才要 causal mask
            mask = torch.full((S, end), float("-inf"), device=x.device)
            mask = mask.triu(diagonal=start_pos + 1)   # 注意偏移 start_pos
            scores = scores + mask

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        out = (probs @ vals).transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)

    @staticmethod
    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0):
    B, prompt_len = prompt_ids.shape
    # 一次 prefill 吃掉整个 prompt
    logits = model(prompt_ids, start_pos=0)[:, -1]          # 只要最后一个位置的 logits
    out = []
    for i in range(max_new_tokens):
        probs = F.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, 1)               # [B, 1]
        out.append(next_tok)
        # 之后每步只喂 1 个 token，start_pos 单调推进
        logits = model(next_tok, start_pos=prompt_len + i)[:, -1]
    return torch.cat(out, dim=1)
```

```python [朴素 cat 版（能写但要说出缺点）]
# 每步 torch.cat 会重新分配并拷贝整个 cache：
# 第 n 步拷贝 O(n)，总共 O(n²) 的搬运量，长序列下非常慢，还会造成显存碎片。
k = torch.cat([k_cache, k_new], dim=1)
v = torch.cat([v_cache, v_new], dim=1)
```

:::

### 关键点

| 点 | prefill | decode |
|---|---|---|
| 输入 S | prompt_len | 1 |
| causal mask | 需要，且 `triu(start_pos + 1)` | 不需要 |
| score shape | `[B, H, S, S]` | `[B, H, 1, S_kv]` |
| 计算性质 | GEMM，compute-bound | GEMV，memory-bound |
| cache 操作 | 写入 S 个位置 | 写 1 个、读 S_kv 个 |

## 面试追问

::: details Q：为什么 decode 不需要 causal mask？
causal mask 的作用是防止位置 $i$ 看到位置 $j > i$。decode 时当前 token 是序列里最新的，cache 里全是它之前的 token，天然满足因果性。加了反而错（会把合法的历史 mask 掉）。但注意 **padding mask 仍可能需要**：batch 内不同序列长度不同时，要 mask 掉 padding 位置。PagedAttention 用变长 + block table 天然避免了 padding。
:::

::: details Q：预分配 cache 有什么问题，PagedAttention 怎么解决？
预分配按 `max_seq` 开，一个实际只生成 100 token 的请求占着 8k 的空间，利用率 20% 不到，而且这块空间不能给别人用。PagedAttention 改成按 16 token 的 block 按需分配，用 block table 做逻辑到物理的映射，浪费上限降到每序列半个 block。详见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)。
:::

::: details Q：RoPE 在 decode 时怎么加？
位置索引是 `start_pos`（当前 token 的绝对位置），不是 0。写错的话表现为生成前几个 token 正常、之后逐渐语无伦次。实现上通常预计算好 `cos/sin` 表，decode 时按 `start_pos` 取一行。另外 **只对 Q 和 K 加 RoPE，V 不加**，因为 RoPE 的相对位置性质来自 $q^\top k$ 的内积。
:::

::: details Q：这段代码怎么改成支持 batch 内序列长度不同？
两条路。一是 padding + attention mask，简单但浪费算力和显存。二是 varlen：把所有序列的 token 拼成一维，用 `cu_seqlens`（累积长度前缀和）标记边界，调 `flash_attn_varlen_func`。生产框架都用第二种，因为 continuous batching 下序列长度必然参差不齐，padding 到最长会浪费掉大部分计算。
:::

## 参考

- [Llama 参考实现的 KV cache](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [FlashAttention varlen 接口](https://github.com/Dao-AILab/flash-attention)

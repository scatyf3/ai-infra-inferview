---
title: Fused QKV Projection with RoPE and KV Cache Update
status: draft
tags: [rope, kv-cache, leetgpu]
difficulty: 3
order: 6.6
related: [/leetgpu/rotary-positional-embedding, /handson/decode-step-kv-cache, /handson/torch-primitives]
stack: []
leetgpu: [115]
---

# Fused QKV Projection with RoPE and KV Cache Update

> [LeetGPU #115](https://leetgpu.com/challenges/fused-qkv-projection-with-rope-and-kv-cache-update) · decode 一步：每个 batch 一个新 token · 在 [Rotary Positional Embedding](./rotary-positional-embedding) 基础上加 QKV 投影和写 cache

## 输入

| 参数 | shape | 说明 |
|---|---|---|
| `x` | `[B, d_model]` | 每个序列这一步的新 token |
| `W_qkv` | `[d_model, (H_q + 2·H_kv)·D]` | Q / K / V 投影拼在一起 |
| `cos_sin_cache` | `[S_max, D]` | 第 p 行：前 D/2 个 cos，后 D/2 个 sin |
| `positions` | `[B]` | 第 b 个序列当前 token 的位置 |
| `K_cache` / `V_cache` | `[B, H_kv, S_max, D]` | 原地写 |
| `Q_out` | `[B, H_q, D]` | 原地写 |

## 思路

1. **投影**：`x @ W_qkv` 得到 `[B, (H_q + 2·H_kv)·D]`，view 成 `[B, H_q + 2·H_kv, D]`，沿 head 维切出 q / k / v
2. **取 cos / sin**：`cos_sin_cache[positions]` 每个 batch 取自己那一行 `[B, D]`，沿最后一维切两半；补 head 维、补到长度 D
3. **RoPE**：只作用在 q 和 k 上，v 不转
4. **写回**：k / v 写进 cache 的 `(b, positions[b])` 位置，q 写进 `Q_out`

## 题解

```python
import torch


# x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out are tensors on the GPU
def solve(
    x: torch.Tensor,              # [B, d_model]
    W_qkv: torch.Tensor,          # [d_model, (H_q + 2 * H_kv) * D]
    cos_sin_cache: torch.Tensor,  # [S_max, D]
    positions: torch.Tensor,      # [B]
    K_cache: torch.Tensor,        # [B, H_kv, S_max, D]
    V_cache: torch.Tensor,
    Q_out: torch.Tensor,          # [B, H_q, D]
    B: int,
    d_model: int,
    H_q: int,
    H_kv: int,
    D: int,
    S_max: int,
):
    # 投影，再拆出 head 维
    qkv = torch.matmul(x, W_qkv)                      # [B, (H_q + 2 * H_kv) * D]
    qkv = qkv.view(B, H_q + 2 * H_kv, D)

    # 每个 batch 取自己位置那一行；前半 cos、后半 sin
    cur_pos = cos_sin_cache[positions, :]             # [B, D]
    half = D // 2
    cos = cur_pos[:, :half]                           # 沿最后一维切
    sin = cur_pos[:, half:]
    cos = torch.cat([cos, cos], dim=-1)[:, None, :]   # [B, 1, D]：补到长度 D，补 head 维
    sin = torch.cat([sin, sin], dim=-1)[:, None, :]

    q = qkv[:, :H_q, :]                               # [B, H_q, D]
    k = qkv[:, H_q:H_q + H_kv, :]                     # [B, H_kv, D]
    v = qkv[:, H_q + H_kv:H_q + 2 * H_kv, :]          # [B, H_kv, D]

    q_rot = torch.cat([-q[:, :, half:], q[:, :, :half]], dim=-1)
    q = q * cos + q_rot * sin
    k_rot = torch.cat([-k[:, :, half:], k[:, :, :half]], dim=-1)
    k = k * cos + k_rot * sin

    # 原地写回；batch 维也用索引张量，和 positions 逐位配对
    batch_idx = torch.arange(B, device=positions.device)   # [0, 1, ..., B-1]
    K_cache[batch_idx, :, positions, :] = k
    V_cache[batch_idx, :, positions, :] = v
    Q_out.copy_(q.view(B, H_q, D))
```

## 写 cache 的索引

想要的语义是循环版：

```python
for b in range(B):
    K_cache[b, :, positions[b], :] = k[b]    # k[b]: [H_kv, D]
```

规则：**`:` 是这一维全取，和别的维无关；只有索引张量之间才逐位配对。**

| 写法 | 写入的 (batch, pos)，B=2 | 选中区域 shape |
|---|---|---|
| `[:, :, positions, :]` | (0,p0) (0,p1) (1,p0) (1,p1)，全组合 | `[B, H_kv, B, D]` |
| `[batch_idx, :, positions, :]` | (0,p0) (1,p1)，逐位配对 | `[B, H_kv, D]` |

- batch 和 pos 有"第 b 个配 `positions[b]`"的对应关系，两维都用索引张量
- head 和 dim 每个 batch 都要全写，没有对应关系，用 `:`
- 两个索引张量中间隔了 `:` 时，配对出来的那一维放到结果最前面，所以是 `[B, H_kv, D]` 而不是 `[H_kv, B, D]`，正好等于 k 的 shape

## 这题的坑

**会报错的**

1. **写 cache 用 `[:, :, positions, :]`**：全组合，选中区域 `[B, H_kv, B, D]`，报 "shape mismatch: value tensor of shape [2, 1, 4] cannot be broadcast to indexing result of shape [2, 1, 2, 4]"。batch 维要换成 `batch_idx`，见 [上面](#写-cache-的索引)。
2. **cos / sin 只有 D/2 长、没有 head 维**：`[B, D/2]` 和 `[B, H, D]` 右对齐后 D/2 对 D，报错；或者恰好某维是 1 被广播，静默错。要 `cat([cos, cos])` 补到 D，再 `[:, None, :]` 补 head 维。
3. **`Q_out.view(H_q, D)`**：漏了 batch 维，元素数对不上。
4. **`torch.arange` 没传 device**：默认在 CPU，和 GPU 上的 `positions` 一起做索引可能报 device 不一致。

**不报错、结果静默错的**

5. **cos / sin 在第 0 维切**：`cur_pos[:half]` 切的是 batch 维。B=1、D=2 时 `cos = cur_pos[1:]` 是 `[0, D]`，`k * cos` 里 1 被广播成 0，k 悄悄变成 `[1, 0, 2]`，直到后面 `view` 才报 "invalid for input of size 0"。要 `cur_pos[:, :half]`。
6. **print 放在 RoPE 之后**：看到 k 是空的以为是切 qkv 切错了，其实切片没问题，是上一条的 cos 把它广播空了。查 shape 要紧跟在产生它的那行后面 print。
7. **cos / sin 顺序**：题面写的是每行前半 cos、后半 sin（和 vLLM 的 `cos_sin_cache` 一样），别凭直觉写成前 sin 后 cos。
8. **公式写反**：`q * sin + q_rot * cos`，应该是原值乘 cos、rotate_half 乘 sin。
9. **`k.view(...)` / `torch.cat([K_cache, k])` 单独一行**：都是 out-of-place，返回值没接住等于没写；`cat` 还会分配新内存，cache 本身不会变。写 cache 要用索引赋值原地写。

其余见 [通用语法坑](./#通用语法坑)。

## 和推理引擎的对应

- vLLM 的 `rotary_embedding` 吃的也是 `positions` + `cos_sin_cache`，按 `[cos | sin]` 两半算，不做 `cat([cos, cos])` 这一步：`[x1 * cos - x2 * sin, x2 * cos + x1 * sin]`。
- 真实 KV cache 是 paged 的，写入走 `reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)`：每个 token 一个 `slot`（物理块号 × block_size + 块内偏移），和这里 `(batch_idx, positions)` 配对索引是同一个思路，只是地址多了一层块表映射。
- 这题拆成了投影、RoPE、写 cache 三步；fused kernel 的价值是 q / k 投影完在寄存器里直接转、直接写 cache，不落中间的 `qkv`。

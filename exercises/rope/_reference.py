"""RoPE 三连的参考实现。做题时别看，check.py 用它出答案、诊断错误。

约定（和 LLaMA / GPT-NeoX 一致）：
- x 的最后一维 D 是偶数，前后两半配对：(x[i], x[i + D/2]) 一起旋转
- 第 k 对的频率 inv_freq[k] = base ** (-2k / D)，位置 p 的转角 p * inv_freq[k]
- 只转 q、k，不转 v
"""
import math

import torch


def rope_cos_sin(positions, D, base=10000.0, dtype=torch.float32):
    inv_freq = base ** (-torch.arange(0, D, 2, dtype=torch.float64, device=positions.device) / D)  # [D/2]
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]                                 # [T, D/2]
    ang = torch.cat([ang, ang], dim=-1)                                                            # [T, D]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, positions, base=10000.0):
    """x: [T, H, D]，positions: [T]"""
    cos, sin = rope_cos_sin(positions, x.shape[-1], base, x.dtype)
    cos, sin = cos[:, None, :], sin[:, None, :]          # [T, 1, D]，对所有 head 广播
    return x * cos + rotate_half(x) * sin


def apply_rope_interleaved(x, positions, base=10000.0):
    """GPT-J 风格：相邻两维 (x[2k], x[2k+1]) 配对。只用来诊断「配对方式不对」。"""
    D = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, D, 2, dtype=torch.float64, device=x.device) / D)
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    cos, sin = ang.cos().to(x.dtype)[:, None, :], ang.sin().to(x.dtype)[:, None, :]
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def attention(q, k, v, causal=True):
    """q: [Tq, H, D]，k / v: [Tk, H, D]；causal 时 query i 看 key j <= i + (Tk - Tq)"""
    Tq, Tk, D = q.shape[0], k.shape[0], q.shape[-1]
    scores = torch.einsum('qhd,khd->hqk', q, k) / math.sqrt(D)
    if causal:
        i = torch.arange(Tq, device=q.device)[:, None] + (Tk - Tq)
        j = torch.arange(Tk, device=q.device)[None, :]
        scores = scores.masked_fill(~(j <= i), float('-inf'))
    return torch.einsum('hqk,khd->qhd', torch.softmax(scores, dim=-1), v)


def p1_solve(q, k, v, positions):
    return attention(apply_rope(q, positions), apply_rope(k, positions), v)


def p2_decode_step(q, k, v, k_cache, v_cache, past_len):
    pos = torch.tensor([past_len], device=q.device)
    q_rot = apply_rope(q[None], pos)[0]
    k_rot = apply_rope(k[None], pos)[0]
    k_new = torch.cat([k_cache, k_rot[None]], dim=0)
    v_new = torch.cat([v_cache, v[None]], dim=0)
    out = attention(q_rot[None], k_new, v_new, causal=False)[0]
    return out, k_new, v_new


def p3_solve(q, k, v, positions):
    group = q.shape[1] // k.shape[1]
    k = apply_rope(k, positions).repeat_interleave(group, dim=1)   # 先在 Hkv 个 head 上转，再复制
    v = v.repeat_interleave(group, dim=1)
    return attention(apply_rope(q, positions), k, v)

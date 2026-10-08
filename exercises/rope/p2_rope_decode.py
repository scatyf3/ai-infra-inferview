"""P2 · decode 一步：RoPE + KV cache。题面见 docs/handson/rope-decode-kv-cache.md

检查：uv run --no-project --with torch python exercises/rope/check.py p2
"""
import math

import torch

from p1_rope_causal import apply_rope  # 复用 P1 写好的 apply_rope


def decode_step(
    q: torch.Tensor,        # [H, D]  新 token，还没转
    k: torch.Tensor,        # [H, D]  新 token，还没转
    v: torch.Tensor,        # [H, D]
    k_cache: torch.Tensor,  # [past_len, H, D]  已经转过
    v_cache: torch.Tensor,  # [past_len, H, D]
    past_len: int,
):
    """返回 (out [H, D], k_cache_new [past_len + 1, H, D], v_cache_new [past_len + 1, H, D])"""
    raise NotImplementedError('decode_step 还没实现')

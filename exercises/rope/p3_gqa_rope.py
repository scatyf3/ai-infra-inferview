"""P3 · GQA + RoPE。题面见 docs/handson/rope-gqa.md

检查：uv run --no-project --with torch python exercises/rope/check.py p3
"""
import math

import torch

from p1_rope_causal import apply_rope  # 复用 P1 写好的 apply_rope


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """q: [T, Hq, D]，k, v: [T, Hkv, D]，Hq 是 Hkv 的整数倍，positions: [T] → causal 输出 [T, Hq, D]"""
    raise NotImplementedError('solve 还没实现')

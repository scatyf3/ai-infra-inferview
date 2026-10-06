"""P1 · RoPE + causal attention。题面见 docs/handson/rope-causal-attention.md

检查：uv run --no-project --with torch python exercises/rope/check.py p1
"""
import math

import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: [T, H, D]（D 为偶数），positions: [T] → 返回同 shape，前后两半配对旋转"""
    raise NotImplementedError('apply_rope 还没实现')


def solve(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """q, k, v: [T, H, D]，positions: [T] → causal attention 输出 [T, H, D]"""
    raise NotImplementedError('solve 还没实现')

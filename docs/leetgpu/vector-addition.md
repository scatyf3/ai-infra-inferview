---
title: Vector Addition
status: draft
tags: [triton, elementwise, leetgpu]
difficulty: 1
order: 0
related: [/handson/vector-add]
stack: []
leetgpu: [1]
---

# Vector Addition

> [LeetGPU #1](https://leetgpu.com/challenges/vector-addition) · triton · 逐行讲解、N = 4 的取值、kernel mindset 分析见 [Vector Add](/handson/vector-add)

## 题解

```python
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) # 必须要 axis=0
    # pid * BLOCK_SIZE is the start point
    # tl.arange(0, BLOCK) is an empty tensor with len=BLOCK
    # [offset,offset+1 ... offset+m]
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 标准操作，mask掉不要的
    mask = offset < n_elements
    a_blk = tl.load(a + offset, mask=mask)
    b_blk = tl.load(b + offset, mask=mask)

    tl.store(c+offset,a_blk+b_blk,mask=mask)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
```

## 这题的坑

- **漏了 mask**：N 不是 `BLOCK_SIZE` 的整数倍时，最后一个 program 会读写越界。
- 其余见 [通用语法坑](./#通用语法坑)。

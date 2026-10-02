---
title: triton 原语
status: draft
tags: [triton, guide, handson]
difficulty: 2
order: 0.2
related: [/handson/torch-primitives, /handson/triton-softmax, /handson/triton-fused-layernorm]
stack: []
---

# triton 原语

在 torch 里我们在乎怎么表述一个抽象的 tensor；在 triton 里，我们在乎怎么把 tensor 切成块、分给各个 program。块内部怎么分给线程，交给编译器。

### 映射：program 而不是 thread

核心区别：**你只写到 block（triton 叫 program）这一级。** CUDA 要手写"第几个线程算第几个元素"；triton 里一个 program 一次处理一整块数据（一个向量或一个 tile），写起来像在操作一个小 tensor。

| | CUDA | triton |
|---|---|---|
| launch | `kernel<<<grid, block>>>(...)` | `kernel[grid](..., BLOCK=1024, num_warps=4)` |
| grid | blockIdx 的范围 | 一样，`grid = (triton.cdiv(n, BLOCK),)` |
| block 内线程 | `blockDim`，你定、你分工 | 没有 thread 这一层；`num_warps` 指定每个 program 几个 warp，分工交给编译器 |
| 我是哪个 block | `blockIdx.x` | `tl.program_id(axis=0)` |
| 我是哪个线程 | `threadIdx.x` | **没有** |
| 本单位算的数据 | `i = blockIdx.x * blockDim.x + threadIdx.x`，标量下标 | `offs = pid * BLOCK + tl.arange(0, BLOCK)`，**下标向量** |
| 越界保护 | `if (i < n)` | `mask = offs < n`，传给 load / store |
| 读写 | `x[i]` | `tl.load(x_ptr + offs, mask=mask)` / `tl.store(...)` |
| shared memory、`__syncthreads`、合并访存 | 手写 | 编译器处理 |

### 最小例子：向量加法

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 我是第几个 program ≈ blockIdx.x
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 本 program 负责的 BLOCK 个下标
    mask = offs < n                              # 最后一块可能越界
    x = tl.load(x_ptr + offs, mask=mask)         # 一次读一整块
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)   # grid 可以依赖 BLOCK
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

- torch tensor 传进 kernel 自动变成指向首元素的指针，`x_ptr + offs` 是一个**指针向量**
- `BLOCK` 必须标 `tl.constexpr`，且是 2 的幂；编译期确定，换值就重新编译一份 kernel
- grid 写成 lambda，launch 时会把 `BLOCK` 等 meta 参数传进去，autotune 换 BLOCK 时 grid 跟着变

### 二维：stride 和广播直接搬过来

triton 不认识 shape，只认指针，地址用 stride 自己算（[torch 原语 · tensor itself](./torch-primitives#tensor-itself)）。

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # 一个 program 处理一行
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * stride_row + cols, num / tl.sum(num, axis=0), mask=mask)

# launch：grid = (n_rows,)，stride_row 传 x.stride(0)
```

二维 tile（比如 matmul）的地址由两个下标向量[广播](./torch-primitives#broadcasting)拼出来：

```python
ptrs = base + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n   # (BM, BN) 的指针矩阵
```

`(BM, 1)` 加 `(1, BN)` 得 `(BM, BN)`，和 torch 的广播规则完全一样。

### 你决定 vs 编译器决定

- **你决定**
  - 一个 program 负责哪块数据：grid 怎么切、`BLOCK` 多大
  - 用几个 warp：`num_warps`
  - 流水线几级：`num_stages`
- **编译器决定**
  - block 内每个线程拿哪几个元素
  - 合并访存、何时放进 shared memory、何时同步

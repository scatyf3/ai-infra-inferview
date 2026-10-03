---
title: Vector Add
status: draft
tags: [triton, elementwise, handson]
difficulty: 1
order: 0.35
related: [/handson/triton_primitives, /handson/kernel-mindset, /leetgpu/vector-addition]
stack: [k-lang]
leetgpu: [1]
---

# Vector Add

> [LeetGPU #1 · Vector Addition](https://leetgpu.com/challenges/vector-addition) · [题解](/leetgpu/vector-addition) · 最简单的 kernel，拿来认识 [triton 编程模型](./triton_primitives)，也是 [kernel mindset](./kernel-mindset) 里 1 → 1 逐元素的代表

## 用 kernel mindset 看

按 [checklist](./kernel-mindset#checklist) 过一遍：

1. **维度角色**：只有一个维度 N，是并行维，切开进 grid；没有归约维
2. **一个 program 的块**：连续的 `BLOCK_SIZE` 个元素
3. **数据复用**：没有，`a[i]`、`b[i]` 只被 `c[i]` 用一次
4. **写冲突**：没有，每个 `c[i]` 只被一个 program 写
5. **瓶颈**：fp32 下每个元素读 8 字节、写 4 字节，只做 1 次加法，算术强度 1/12 FLOP/byte → 纯 memory-bound

所以这道题没有可优化的计算，能做的只有让访存连续、和前后 kernel 融合。

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

- torch tensor 传进 kernel 自动变成指向首元素的指针，`a + offset` 是一个**指针向量**
- `BLOCK_SIZE` 必须标 `tl.constexpr`，且是 2 的幂；编译期确定，换值就重新编译一份 kernel
- 这里 `BLOCK_SIZE` 写死，grid 直接算成元组

## 走一遍：N = 4，BLOCK_SIZE = 2

```python
# a = [1, 2, 3, 4]，b = [10, 20, 30, 40]
# grid = (cdiv(4, 2),) = (2,)，启动 pid = 0、1 两个 program
# 下面是 pid = 0 看到的值

# pid = 0
pid = tl.program_id(axis=0)

# 0 * 2 + [0, 1] = [0, 1]
offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# [0, 1] < 4 = [T, T]
mask = offset < n_elements

# 读 a[0], a[1] -> a_blk = [1, 2]
a_blk = tl.load(a + offset, mask=mask)

# b_blk = [10, 20]
b_blk = tl.load(b + offset, mask=mask)

# a_blk + b_blk = [11, 22]，写进 c[0], c[1]
tl.store(c + offset, a_blk + b_blk, mask=mask)

# pid = 1 同理写 c[2], c[3] = [33, 44]，最后 c = [11, 22, 33, 44]
```

N 不能被 `BLOCK_SIZE` 整除时，mask 才起作用。比如 N = 3：

| pid | offset | mask | 写入 |
|---|---|---|---|
| 0 | `[0, 1]` | `[T, T]` | `c[0], c[1]` |
| 1 | `[2, 3]` | `[T, F]` | 只写 `c[2]`，下标 3 越界不访存 |

## tl.load 读多少个？

`tl.load` 没有长度参数，长度藏在指针里：

```python
a                  # 标量指针，指向 a[0]
offset             # shape (BLOCK_SIZE,) 的下标向量
a + offset         # 广播成 shape (BLOCK_SIZE,) 的指针向量
```

`tl.load` 对每个指针读一次，返回值 shape 和指针 shape 一样。所以：

- **读多少个**：看指针块的 shape，即 `BLOCK_SIZE`，编译期定死
- **哪几个真读**：看 `mask`，运行时定

指针不需要连续（`a + offset * 2` 就是隔一个读一个）。连续的情况编译器能认出来，会合并成宽向量读，不会变慢。

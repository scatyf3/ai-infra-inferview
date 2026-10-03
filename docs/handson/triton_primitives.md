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

<TritonProgramViz mode="vector" />

### 最小例子：向量加法

上面这张表落到代码上，就是 [Vector Add](./vector-add)：题解、N = 4 的逐行取值、`tl.load` 读多少个，都在那一页。


### 二维：stride 和广播直接搬过来

triton 不认识 shape，只认指针，地址用 stride 自己算（[torch 原语 · tensor itself](./torch-primitives#tensor-itself)）。

```python
# 例子：x = [[0., 0., 0.],
#            [1., 2., 3.]]
# shape (2, 3)，行优先连续存，内存里是 [0, 0, 0, 1, 2, 3]
# n_rows = 2，n_cols = 3，stride_row = 3，BLOCK = 4（>= 3 的最小 2 的幂）
# 注释里的值都是 row = 1 这个 program 看到的


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    # 一个 program 处理一行
    # row = 1
    row = tl.program_id(0)

    # 这一行里的列下标；BLOCK 是 2 的幂，比 n_cols 多出来的要 mask 掉
    # cols = [0, 1, 2, 3]
    cols = tl.arange(0, BLOCK)

    # [0, 1, 2, 3] < 3 = [T, T, T, F]
    mask = cols < n_cols

    # 地址 = x_ptr + 1 * 3 + [0, 1, 2, 3] = x_ptr + [3, 4, 5, 6]
    # 即内存里第 3、4、5 个元素（x[1, 0..2]），第 6 个越界被 mask
    # 越界位置填 -inf：max 不会选它，exp(-inf) = 0，sum 也不受影响
    # x = [1., 2., 3., -inf]
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=-float('inf'))

    # 减掉行最大值防止 exp 溢出，softmax 结果不变
    # max = 3 -> x = [-2., -1., 0., -inf]
    x = x - tl.max(x, axis=0)

    # num = [e^-2, e^-1, e^0, 0] = [0.135, 0.368, 1., 0.]
    num = tl.exp(x)

    # sum = 1.503
    # num / sum = [0.090, 0.245, 0.665, 0.]
    # 只写前 3 个：out[1] = [0.090, 0.245, 0.665]
    tl.store(out_ptr + row * stride_row + cols, num / tl.sum(num, axis=0), mask=mask)


def softmax(x):
    # 2 x 3 的空 tensor，和 x 一样连续，所以 out 的 stride_row 也是 3
    out = torch.empty_like(x)

    # n_rows = 2，n_cols = 3
    n_rows, n_cols = x.shape

    # 一整行要装进一个 program，BLOCK 取 >= n_cols 的 2 的幂
    # next_power_of_2(3) = 4
    BLOCK = triton.next_power_of_2(n_cols)

    # grid = (2,)：row 0 写 out[0] = [0.333, 0.333, 0.333]
    #              row 1 写 out[1] = [0.090, 0.245, 0.665]
    # x.stride(0) = 3：跳到下一行要跨 3 个元素
    softmax_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK=BLOCK)

    return out
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

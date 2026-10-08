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

### load / store 速查

```python
x = tl.load(ptr, mask=mask, other=0.0)   # 指针（向量）在前；被 mask 掉的位置取 other
tl.store(ptr, value, mask=mask)          # 也是指针在前、值在后；没有 other，mask 掉的位置不写
tl.atomic_add(ptr, value)                # 多个 program 往同一个地址累加
```

- `other` 不写时，mask 掉的位置是**未定义值**。后面要 reduce（`tl.sum` / `tl.max`）就必须写：sum 填 `0.0`，max 填 `-float('inf')`
- 只 load 后 store、mask 掉的位置不参与计算的，可以不写 `other`
- 标量地址不用 mask：`tl.store(partial_ptr + pid, s)`
- 低精度输入先 `.to(tl.float32)` 再算平方 / 累加；写回低精度输出时 `.to(out_ptr.dtype.element_ty)`

### 标量还是指针

launch 时参数按类型分两种，kernel 里的用法完全不同：

| host 端传的 | kernel 里拿到 | 怎么用 |
|---|---|---|
| `torch.Tensor`（哪怕只有 1 个元素） | 指向首元素的**指针** | `tl.load(p)` / `tl.load(p + offs)` |
| Python `float` / `int` | **标量值** | 直接参与运算 |

**以运行时类型为准，不以签名注解为准。** LeetGPU 签名写着 `gamma: torch.Tensor`，实际传的是 float，`tl.load(gamma)` 报 "Unsupported ptr type triton.language.float32"。拿不准就在 `solve` 里 `print(type(gamma))`。

### 下标怎么算：三种模式

**1. 一维切块**：一个 program 负责连续 BLOCK 个元素

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)   # 先加偏移
mask = offs < n                            # 再算 mask
```

**2. 一行一个 program，整行塞得下**：`grid = (n_rows,)`，`BLOCK = triton.next_power_of_2(n_cols)`

```python
cols = tl.arange(0, BLOCK)
mask = cols < n_cols                       # 列号和列数比
ptrs = x_ptr + row * n_cols + cols         # 行跨度是列数（连续存储时 = stride(0)）
```

**3. 一行一个 program，行太长要循环**：行内按 BLOCK 步长走

```python
acc = 0.0
for start in range(0, n_cols, BLOCK):      # 步长是 BLOCK（块大小），不是块数
    cols = start + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    acc += tl.sum(x * x, axis=0)           # 块内先 reduce 成标量再累加
```

把 `cols` 单独拆出来：mask 用它（`cols < n_cols`），逐通道参数也用它（`w_ptr + cols`），只有数据地址再加行起点。

### 跨 program 归约

program 之间不能同步，整个向量的 reduce 只能：

- **两个 kernel**：第一个每个 program 写一个部分和到 `partial[pid]`，第二个读全部 partial 汇总。确定性，但多一次 launch、HBM 多读一遍
- **atomic**：`tl.atomic_add` 到 `torch.zeros(1)`。写起来短，但浮点加的顺序不固定，结果末位会抖

临时 buffer 在 host 端开：`torch.empty(num_blocks, device=x.device, dtype=torch.float32)`；atomic 的目标必须 `torch.zeros`。

能按行切就按行切（模式 2 / 3），reduce 留在 program 内部，不需要这些。

### 踩过的坑

来自 [RMS Normalization](/leetgpu/rms-normalization)（#50）和 Fused Residual Add + RMSNorm（#83）。

**launch / host 端**

| 写法 | 问题 | 改成 |
|---|---|---|
| `kernel[num_blocks](...)` | grid 必须是元组，报 "object of type 'int' has no len()" | `kernel[(num_blocks,)](...)` |
| `torch.cdiv(N / BLOCK)` | 没有 `torch.cdiv`；参数是两个 | `triton.cdiv(N, BLOCK)` |
| `BLOCK: tl.constexp` | 拼写 | `tl.constexpr` |
| kernel 和 tensor 都叫 `rms` | 后定义的覆盖前者 | 改名；也别用 `sum` 遮内置函数 |
| `BLOCK_SIZE` 随便取 / 没定义 | `tl.arange` 要求 2 的幂 | `min(triton.next_power_of_2(C), 1024)` |

**参数**

| 写法 | 问题 | 改成 |
|---|---|---|
| `tl.load(gamma)`，gamma 实际是 float | 标量不是指针 | 直接用 `gamma` |
| `out / rms + weight`，weight 是 `[C]` tensor | 指针没 load；而且是乘不是加 | `w = tl.load(weight + cols, mask=mask)`，`* w` |

**下标 / mask**

| 写法 | 问题 | 改成 |
|---|---|---|
| `offs = tl.arange(...)`；`mask = offs < N`；再 `offs = pid * BLOCK + offs` | mask 在加偏移之前算，每个 block 几乎全 True，越界读 | 先加偏移再算 mask |
| `pid * N + ...`，数据是 `[N, C]` | 行跨度是 C 不是 N；N=1 的例子测不出来 | `pid * C + cols` |
| `i * chunk_size`，chunk_size = 块数 | 块数 ≠ 块大小，块之间重叠 | `i * BLOCK_SIZE`，或 `range(0, C, BLOCK)` |
| `offset = pid * N, tl.arange(...)` | 逗号变成元组 | `+` |
| `mask = offset < N`（行内） | 列号要和列数比 | `cols < C` |
| `mask = offs < N`（读 partial） | partial 只有 num_blocks 个 | `offs < num_blocks` |
| `range(C / BLOCK)` | `/` 得 float | `tl.cdiv(C, BLOCK)` 或按步长 `range` |

**load / store**

| 写法 | 问题 | 改成 |
|---|---|---|
| `tl.store(y, ptr, mask=...)` | 参数顺序反了 | `tl.store(ptr, y, mask=...)` |
| `tl.store(ptr, y)` 没有 mask | 最后一块越界写 | 加 `mask=mask` |
| `tl.load(...)` 没有 `other`，后面 `tl.sum` | 越界位置是垃圾值，进了平方和 | `other=0.0` |

**reduce / 数学**

| 写法 | 问题 | 改成 |
|---|---|---|
| `sum = 0` 然后循环里 `sum += x * x` | 标量 int 变成向量，循环变量改类型报错 | `acc = 0.0`；`acc += tl.sum(x * x, axis=0)` |
| 每个 block 存 `sqrt(块内平方和 / N + eps)` | $\sqrt{a+b} \ne \sqrt a + \sqrt b$，部分和不能先开根号 | 只存部分平方和，汇总后再开根号 |
| `gamma * rms + beta` | 公式抄错 | `gamma * x_bar + beta` |
| `x ** 2` | 部分版本不支持 / 慢 | `x * x` |
| fused add 先算 rms 再加 residual | rms 是对 `z = x + residual` 算的 | 先 add，再 reduce，再 normalize |

---
title: Reduction
status: draft
tags: [triton, reduce, leetgpu]
difficulty: 2
order: 0.5
related: [/handson/reduce, /handson/kernel-mindset, /handson/vector-add]
stack: [k-lang]
leetgpu: [4]
---

# Reduction

> [LeetGPU #4](https://leetgpu.com/challenges/reduction) · triton · 数组求和，输出一个数 · [kernel mindset](/handson/kernel-mindset) 里 n → 1 的代表，前置是 [Vector Add](/handson/vector-add)

::: tip 已 AC
Tesla T4 · 最好 0.16 ms · 75.9th percentile（2026-10-02），见 [优化记录](#优化记录)
:::

## 用 kernel mindset 看

任意长度的数组要分给多个 program，最后再合成一个数。和 vector add 对比：

| | vector add | reduce |
|---|---|---|
| 维度角色 | N 是并行维 | N 既被切开分给各 program，又要归约成一个数 |
| `other` | 不用，越界的位置不写 | **必须填**：求和填 0，求 max 填 -inf |
| 写冲突 | 没有 | 所有 program 写同一个 output → atomic 或分两遍 |
| 归约分几层 | 无 | 两层：块内用 `tl.sum`（编译器管），跨块用 atomic 或第二遍（你管） |

下面两种写法都用同一个例子：`input = [1, 2, 3, 4, 5, 6, 7]`，`N = 7`，`BLOCK_SIZE = 4`。grid = (2,)，第二块会越界。

## 写法 1：atomic，一遍完成

```python
import torch
import triton
import triton.language as tl


@triton.jit
def reduce_atomic_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    # pid = 1
    pid = tl.program_id(axis=0)

    # 1 * 4 + [0, 1, 2, 3] = [4, 5, 6, 7]
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # [4, 5, 6, 7] < 7 = [T, T, T, F]
    mask = offset < n_elements

    # 求和时越界位置填 0，加上去不影响结果（求 max 时填 -inf）
    # x = [5, 6, 7, 0]
    x = tl.load(input + offset, mask=mask, other=0.)

    # 块内归约：一个 program 内部，编译器负责
    # partial = 18
    partial = tl.sum(x, axis=0)

    # 跨 program 归约：两个 program 都往同一个 output 上加
    # pid 0 加 10，pid 1 加 18，先后顺序不定，最后 output = 28
    tl.atomic_add(output, partial)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    # atomic 是往上加，必须先清零
    output.zero_()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    reduce_atomic_kernel[grid](input, output, N, BLOCK_SIZE)
```

## 写法 2：两遍，结果确定

```python
import torch
import triton
import triton.language as tl


# ---------- 第一次归约：每个 program 把自己那块加成一个数 ----------
@triton.jit
def partial_sum_kernel(input, partials, n_elements, BLOCK_SIZE: tl.constexpr):
    # 我是第几个 program
    # 例：pid = 1
    pid = tl.program_id(axis=0)

    # 本 program 负责的下标
    # 例：[4, 5, 6, 7]
    # offset = pid + tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # ✓


    # 例：[T, T, T, F]
    mask = offset < n_elements

    # 读一整块；越界位置要填一个"加上去不影响结果"的值
    # 例：x = [5, 6, 7, 0]
    x = tl.load(input+offset, mask=mask, other=0)

    # 块内求和，写进 partials 里属于自己的那一格
    # 例：partials[1] = 18
    tl.store(partials+pid, tl.sum(x))





# ---------- 第二次归约：1 个 program 把所有 partials 加成一个数 ----------
@triton.jit
def sum_partials_kernel(partials, output, n, BLOCK_SIZE: tl.constexpr):
    # 累加器：长度 BLOCK_SIZE 的全 0 向量，用 float32
    # acc = tl.range(0,BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # partials 可能比一个块长，按 BLOCK_SIZE 一段一段扫
    # 例：n = 2，只扫一段，start = 0
    for start in range(0, n, BLOCK_SIZE):
        # 例：[0, 1, 2, 3]
        offset = tl.arange(0,BLOCK_SIZE)
        mask =  start + offset < n   # not  offset < n // BLOCK_SIZE
        # 例：读到 [10, 18, 0, 0]，加进 acc
        acc += tl.load(partials+start+offset, mask=mask, other=0)

    # acc 收成一个标量写进 output
    # 例：output = 28
    tl.store(output, tl.sum(acc, axis=0))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024

    # 第一次归约要几个 program
    # 例（BLOCK_SIZE = 4）：num_blocks = 2
    # num_blocks = N//BLOCK_SIZE
    num_blocks = triton.cdiv(N, BLOCK_SIZE)

    # 放部分和的中间 buffer：长度 num_blocks，float32，和 input 在同一个设备上
    # partials = tl.zeros((num_blocks,), dtype=tl.float32)
    partials = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    # host不能用tl

    # 第一次归约：grid 是多少？
    partial_sum_kernel[(num_blocks,)](input, partials, N, BLOCK_SIZE)

    # 第二次归约：只启动 1 个 program，grid 怎么写？
    sum_partials_kernel[(1,)](partials, output, num_blocks, BLOCK_SIZE)
```

- **atomic**：一个 kernel 搞定，最快。但浮点加法不满足结合律，加的顺序每次不同，结果最后几位可能不一样
- **两遍**：多一次 kernel 启动和一次 partials 的读写，但每次结果都一样

## 优化记录

Tesla T4，2026-10-02：

| 版本 | 改动 | 耗时 | percentile |
|---|---|---|---|
| v1 | 写法 2 原样 | 0.22 ms | 43.4th |
| v2 | 第一次归约改成固定数量 program + 跨步循环 | 0.16 ms | 75.9th |

分布图上最快的一批在 0.10 ms 左右。

### 瓶颈：每个 program 干的活太少

reduce 每个元素只做一次加法，纯 memory-bound，上限是把 N 个 float 从显存读一遍（T4 约 320 GB/s）。所以优化方向是让读显存更满，不是少算。

v1 每个 program 只读 1024 个 float（4 KB），读完做一次块内求和、写一个数就结束。N 大时要启动几万个 program，每个都付一遍启动、块内求和、写回的固定开销，读显存的时间占比不高。

### v2：固定 program 数，每个 program 跨步循环

启动"SM 数 × 几"个 program（T4 有 40 个 SM），每个 program 循环读很多块。循环里只做向量累加，最后只做一次块内求和：

```python
@triton.jit
def partial_sum_kernel(input, partials, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # 跨步循环：pid 0 读第 0、P、2P… 块，pid 1 读第 1、P+1… 块
    for start in range(pid * BLOCK_SIZE, n_elements, num_programs * BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(input + offset, mask=offset < n_elements, other=0.)
    tl.store(partials + pid, tl.sum(acc, axis=0))
```

这样 partials 只有一两百个，第二个 kernel 几乎不花时间。

<!-- TODO: 贴提交的 v2 完整代码（program 数、BLOCK_SIZE 取了多少） -->

### 还没试的

- **调 `BLOCK_SIZE` / `num_warps`**：试 `BLOCK_SIZE = 2048 / 4096`、`num_warps = 4 / 8`，手动试几组或用 autotune
- **换成 atomic，省掉第二个 kernel**：总时间降下来之后，一次 kernel 启动（几微秒）的占比就上去了；判题有误差容忍度，atomic 的不确定性一般能过
- `torch.empty` 分配 partials 不用管：PyTorch 有缓存分配器，第二次调用起基本不花时间

## 追问：两个 sum 差不多，为啥非要写两个 kernel？

> 感觉两个 sum 是差不多的，为啥非要写俩 kernel，虽然我理解这是现有的最好 practice。

确实，两个 kernel 做的是同一件事：一堆数加成一个。拆开不是因为运算不同，而是因为**一个 kernel 内部，program 之间没法互相等。**

### 为什么一个 kernel 里做不完

第二步要读所有 program 的部分和，所以得等全部 program 都写完。triton 不提供"全网格一起同步"这种操作：

- grid 可能比 GPU 同时能跑的 program 多，后面的 program 要等前面的跑完才上 SM
- 如果先跑的 program 原地等后跑的，就会死锁

所以 kernel 结束是唯一可靠的"所有人都写完了"的信号。第二个 kernel 本质上是借这个信号做同步，不是为了另一种计算。

### 不想写两个 kernel 的几种办法

**1. 同一个 kernel 启动多次。** partials 本身又是一个待求和的数组，递归调 `partial_sum_kernel` 就行：

```python
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    x, n = input, N
    while n > 1:
        num_blocks = triton.cdiv(n, BLOCK_SIZE)
        partials = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
        partial_sum_kernel[(num_blocks,)](x, partials, n, BLOCK_SIZE)
        x, n = partials, num_blocks
    output.copy_(x)
```

N = 10⁸ 时是 10⁸ → 97657 → 96 → 1，启动 3 次，只要一个 kernel 定义。这就是一棵归约树，每层操作完全一样。

**2. atomic。** 就是写法 1：一个 kernel、一次启动，代价是结果不确定。

**3. 最后一个完成的 program 收尾。** 还是一个 kernel：

- 每个 program 写完自己的部分和后，对一个计数器做 atomic 加 1
- 拿到旧值 = 总数 − 1 的那个 program 知道自己是最后一个，由它把 partials 加起来

只启动一次，加法顺序固定，结果确定。PyTorch 自己的 `sum` 跨 block 归约时就是这个思路。代价是要处理好内存可见性（别的 program 写的 partials，最后那个得确实读得到），在 triton 里写对比前两种麻烦。

**4. N 小时只用一个 program 循环。** 一个 kernel、一个 program，最简单，但只占一个 SM，N 大了就慢。

### 实际库怎么选

- **CUB**（NVIDIA 的 GPU 基础算法库）：N 小时只启动一个 block；N 大时启动两次 kernel
- **PyTorch**：用第 3 种，在一个 kernel 里由最后完成的 block 收尾

所以"两个 kernel"不是唯一的标准答案，只是最好写对的一种。真正的规律是：**跨 program 的归约一定需要一个全局同步点**，可以是 kernel 结束、atomic，或者计数器，你选的是用哪种。

## 这题的坑

- **load 不填 `other`**：mask 掉的位置值未定义，加进 sum 结果就错了。求和填 `0.`，求 max 填 `-inf`。
- **atomic 前没清零**：`atomic_add` 是在 output 原值上加，output 里原来有什么都会算进去。

写两遍版本时踩过的：

- **`offset = pid + tl.arange(...)`**：pid 是第几块，要乘 `BLOCK_SIZE` 才是起点。pid 0 碰巧对，后面每块只往后挪 1 格，读重了
- **`tl.range` 当成 `tl.arange`**：`tl.range` 是写 for 循环用的迭代器；生成下标向量用 `tl.arange`
- **第二次归约的 mask 写成 `offset < n // BLOCK_SIZE`**：`n` 已经是 partials 个数，不用再除；真实下标是 `start + offset`，mask 要和指针用同一个下标
- **host 里用 `tl.zeros`**：报 "Cannot call @triton.jit'd outside of the scope of a kernel"。`tl.*` 只能在 kernel 里用，host 分配显存用 `torch.empty(..., device=input.device)`
- **`num_blocks = N // BLOCK_SIZE`**：向下取整会少最后一块，N < 1024 时直接是 0。用 `triton.cdiv`
- **grid 写成 `[num_blocks]` / `[1]`**：grid 要是元组，`(num_blocks,)` / `(1,)`；`(1)` 就是 int 1
- 其余见 [通用语法坑](./#通用语法坑)。

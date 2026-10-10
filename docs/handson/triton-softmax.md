---
title: Triton 版 Softmax
status: draft
tags: [triton, softmax, handson]
difficulty: 3
order: 7
related: []
stack: [k-lang, k-fused]
leetgpu: [5]
---

# Triton 版 Softmax

> 一行一个 program 的写法

## 一句话结论

row softmax 是「减最大值、exp、除以和」三步，naive 的 PyTorch 实现要读写 x 好几遍；Triton 版一行一个 program，把整行加载进寄存器，`tl.max` 和 `tl.sum` 两次 reduce 后直接写回，HBM 流量降到读一次写一次。这是 Triton 教程的第二课，也是理解 FlashAttention 里 online softmax 的前置。

## 推导

- **数值稳定**：先减行最大值再 exp，防止溢出；推导见 [数值稳定 softmax](/handson/stable-softmax)。
- **融合收益**：naive 版本 5 次读写（max、减、exp、sum、除），fused 版本 2 次；对 memory-bound 的算子就是约 2.5 倍。
- **mask**：行长不是 2 的幂时 `BLOCK = next_power_of_2(N)`，越界位置填 `-inf`，exp 后自然为 0。
- **大行**：单 block 装不下时要分块 + [online softmax](./online-softmax)（维护 running max 和 sum），这就是 FlashAttention 的核心。

### 大 N：先 max 再 sum vs online 合并

LeetGPU #5 是一条长度 N ≤ 500,000 的向量，一个 program 装不下，要跨 program 求全局 max 和 sum。

<!-- TODO: 为什么不能一个 program 包圆（block 上限、SM 利用率、寄存器） -->

| 写法 | kernel 1 | kernel 2 | kernel 3 | 读 x 次数 | 实测（N = 500k） |
|---|---|---|---|---|---|
| 先 max 再 sum | | | | | |
| online 合并 | | | — | | |

<!-- TODO: 为什么每个「要等全局结果」的地方都得切一个 kernel -->

**online 合并公式**：每块先只看自己，求出局部 max 和相对它的局部和：

$$
m_b = \max_{i \in b} x_i, \qquad d_b = \sum_{i \in b} e^{x_i - m_b}
$$

合并时把每块的和从「相对 $m_b$」换算到「相对全局 $M$」，乘一个 $e^{m_b - M}$ 就行：

$$
M = \max_b m_b, \qquad D = \sum_b d_b \, e^{m_b - M} = \sum_b \sum_{i \in b} e^{x_i - m_b} \, e^{m_b - M} = \sum_i e^{x_i - M}
$$

因为 $m_b \le M$，修正因子 $e^{m_b - M} \in (0, 1]$，换算过程本身也不会溢出。

例：$x = [1, 3 \mid 2, 6]$，两块。$(m_0, d_0) = (3,\ e^{-2} + 1)$，$(m_1, d_1) = (6,\ e^{-4} + 1)$。$M = 6$，$D = (e^{-2} + 1)\,e^{-3} + (e^{-4} + 1) = e^{-5} + e^{-3} + e^{-4} + e^{0}$，和直接算 $\sum_i e^{x_i - 6}$ 一样。

**流式写法**：块不是并行算完再合并，而是一块接一块扫的时候，维护一对 running 的 $(m, d)$，每来一块就更新一次：

$$
m' = \max(m,\ m_b), \qquad d' = d \, e^{m - m'} + d_b \, e^{m_b - m'}
$$

初值 $(m, d) = (-\infty, 0)$。这个二元合并满足结合律，所以既能并行地两两合并，也能串行地一块块扫。FlashAttention 用的是串行这种：沿 K/V 扫，同时把输出累加器也乘上 $e^{m - m'}$ 一起修正。

<!-- TODO: 实测差距主要来自哪里（多读一遍 x vs 多一次 launch；2MB 能否放进 L2） -->

## 面试追问

::: details Q：softmax 跨 block 的时候，为什么要 online softmax？两趟不行吗？
<!-- TODO: 两趟可以，代价是什么；什么时候 online 是必须的（FlashAttention） -->
待写。
:::

::: details Q：num_warps 怎么选？
一行 N 个元素，每个线程处理 N / (32 × num_warps) 个；行短时 4 个 warp 够，行长到几万时增加 warp 数让每线程的寄存器压力降下来。Triton 教程里按 BLOCK_SIZE 用启发式（2048 以上 8 warps，4096 以上 16 warps），生产上交给 autotune。
:::

## 手撕

对应 [LeetGPU #5](https://leetgpu.com/challenges)。框架：

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * N + cols, y, mask=mask)
```

大 N 版本：

先 max 再 sum（3 个 kernel），跨块合并有两种做法：

::: code-group

```python [局部结果数组]
@triton.jit
def max_kernel(x_ptr, m_ptr, N, BLOCK: tl.constexpr):
    # 第 1 趟：每块的 max 写进 m[pid]
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))
    tl.store(m_ptr + tl.program_id(0), tl.max(x, axis=0))


@triton.jit
def sum_kernel(x_ptr, m_ptr, d_ptr, N, NB, BLOCK: tl.constexpr, NB_BLOCK: tl.constexpr):
    # 第 2 趟：合并出全局 M，每块的 sum(exp(x - M)) 写进 d[pid]
    i = tl.arange(0, NB_BLOCK)
    M = tl.max(tl.load(m_ptr + i, mask=i < NB, other=-float('inf')), axis=0)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))
    tl.store(d_ptr + tl.program_id(0), tl.sum(tl.exp(x - M), axis=0))


@triton.jit
def norm_kernel(x_ptr, out_ptr, m_ptr, d_ptr, N, NB, BLOCK: tl.constexpr, NB_BLOCK: tl.constexpr):
    # 第 3 趟：合并出全局 M、D（补齐位置：max 填 -inf，sum 填 0），逐元素归一化
    i = tl.arange(0, NB_BLOCK)
    M = tl.max(tl.load(m_ptr + i, mask=i < NB, other=-float('inf')), axis=0)
    D = tl.sum(tl.load(d_ptr + i, mask=i < NB, other=0.0), axis=0)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.exp(x - M) / D, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    NB = triton.cdiv(N, BLOCK)                        # 块数，向上取整才能盖住所有元素
    NB_BLOCK = max(16, triton.next_power_of_2(NB))    # tl.arange 长度必须是 2 的幂
    m = torch.empty(NB, device=input.device, dtype=torch.float32)
    d = torch.empty_like(m)
    grid = (NB,)
    max_kernel[grid](input, m, N, BLOCK=BLOCK)
    sum_kernel[grid](input, m, d, N, NB, BLOCK=BLOCK, NB_BLOCK=NB_BLOCK)
    norm_kernel[grid](input, output, m, d, N, NB, BLOCK=BLOCK, NB_BLOCK=NB_BLOCK)
```

```python [atomic]
@triton.jit
def max_kernel(x_ptr, M_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))
    tl.atomic_max(M_ptr, tl.max(x, axis=0))          # 各块的 max 直接合进全局 M


@triton.jit
def sum_kernel(x_ptr, M_ptr, D_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))
    tl.atomic_add(D_ptr, tl.sum(tl.exp(x - tl.load(M_ptr)), axis=0))   # 各块的 sum 直接加进全局 D


@triton.jit
def norm_kernel(x_ptr, out_ptr, M_ptr, D_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.exp(x - tl.load(M_ptr)) / tl.load(D_ptr), mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    grid = (triton.cdiv(N, BLOCK),)
    M = torch.full((1,), -float('inf'), device=input.device)   # max 的初值：-inf
    D = torch.zeros(1, device=input.device)                    # sum 的初值：0
    max_kernel[grid](input, M, N, BLOCK=BLOCK)
    sum_kernel[grid](input, M, D, N, BLOCK=BLOCK)
    norm_kernel[grid](input, output, M, D, N, BLOCK=BLOCK)
```

:::

| | 局部结果数组 | atomic |
|---|---|---|
| 跨块合并 | 每块写一格，下一趟用 `NB_BLOCK` 读回来再归约 | 每块一次 atomic，直接合进一个标量 |
| 初始化 | 不用（`torch.empty`） | M 填 `-inf`、D 填 0，多 2 次 launch |
| 可复现 | 逐位一致：加法顺序固定 | sum 不一致：`atomic_add` 先后顺序每次不同；max 不受影响 |
| atomic 开销 | — | 每块 1 次，共 NB 次，可忽略；慢的是逐元素 atomic |

<!-- TODO: 两种各提交一次，实测时间填进上面的对比表 -->

online 合并（2 个 kernel）：第 1 趟同时求每块的 max 和 sum，sum 相对这块自己的 max；合并时统一换算到全局 M。

```python
@triton.jit
def partial_kernel(x_ptr, m_ptr, d_ptr, N, BLOCK: tl.constexpr):
    # 第 1 趟：每块的 (m_b, d_b)，d_b = sum(exp(x - m_b))，减的是这块自己的 max
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=-float('inf'))
    m_b = tl.max(x, axis=0)
    tl.store(m_ptr + tl.program_id(0), m_b)
    tl.store(d_ptr + tl.program_id(0), tl.sum(tl.exp(x - m_b), axis=0))


@triton.jit
def norm_kernel(x_ptr, out_ptr, m_ptr, d_ptr, N, NB, BLOCK: tl.constexpr, NB_BLOCK: tl.constexpr):
    # 第 2 趟：合并 M = max(m_b)，D = sum(d_b * exp(m_b - M))，再逐元素归一化
    i = tl.arange(0, NB_BLOCK)
    ms = tl.load(m_ptr + i, mask=i < NB, other=-float('inf'))
    ds = tl.load(d_ptr + i, mask=i < NB, other=0.0)    # 补齐位置：exp(-inf - M) * 0 = 0
    M = tl.max(ms, axis=0)
    D = tl.sum(ds * tl.exp(ms - M), axis=0)            # 和三趟版唯一的区别：乘 exp(m_b - M) 修正
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.exp(x - M) / D, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK = 4096
    NB = triton.cdiv(N, BLOCK)
    NB_BLOCK = max(16, triton.next_power_of_2(NB))
    m = torch.empty(NB, device=input.device, dtype=torch.float32)
    d = torch.empty_like(m)
    grid = (NB,)
    partial_kernel[grid](input, m, d, N, BLOCK=BLOCK)
    norm_kernel[grid](input, output, m, d, N, NB, BLOCK=BLOCK, NB_BLOCK=NB_BLOCK)
```

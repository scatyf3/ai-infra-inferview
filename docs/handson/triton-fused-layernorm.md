---
title: Triton 版 Fused LayerNorm
status: draft
tags: [triton, layernorm, handson]
difficulty: 3
order: 8
related: [/handson/rmsnorm, /handson/triton-softmax, /handson/triton_primitives, /handson/kernel-mindset, /handson/cuda-reduce, /leetgpu/rms-normalization]
stack: [k-fused, k-lang]
leetgpu: [113]
---

# Triton 版 Fused LayerNorm

> forward 与 backward · [LeetGPU #113 Layer Normalization](https://leetgpu.com/challenges)（[题面](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/113_layer_normalization)）

## 一句话结论

LayerNorm 对每行算均值和方差再做 $(x - \mu) / \sqrt{\sigma^2 + \epsilon} \cdot w + b$；fused 的意思是一个 kernel 里读一次 x、在寄存器里算两个统计量、写一次 y，不把 μ、σ² 以外的中间结果落到 HBM。Triton 下一行一个 program，用 `tl.sum` 做行内归约，forward 二十行。方差要用"先减均值再平方"的两遍法（行在寄存器里时第二遍不花访存），不要用 $E[x^2] - E[x]^2$。

## 定义与约定

LayerNorm（[Ba et al., 2016](https://arxiv.org/abs/1607.06450)）。输入 `x` 是 `[N, C]`，N 行（token / 样本），每行 C 个特征，row-major，第 i 行第 j 列在 `x[i * C + j]`。`weight`（w）、`bias`（b）是 `[C]`，所有行共用。每行独立：

$$
\mu_i = \frac{1}{C}\sum_{j} x_{ij}, \qquad
\sigma_i^2 = \frac{1}{C}\sum_{j} (x_{ij} - \mu_i)^2, \qquad
\text{rstd}_i = \frac{1}{\sqrt{\sigma_i^2 + \epsilon}}
$$

$$
\hat{x}_{ij} = (x_{ij} - \mu_i)\,\text{rstd}_i, \qquad y_{ij} = \hat{x}_{ij}\, w_j + b_j
$$

σ² 是有偏方差（除以 C 不是 C − 1），PyTorch 参考里对应 `var(unbiased=False)`。

LeetGPU #113：fp32，1 ≤ N ≤ 65,536，1 ≤ C ≤ 4,096，ε = 1e-5，容差 1e-4。入口 `solve(input, weight, bias, output, N, C, eps)`，`eps` 是 Python float，直接传进 kernel 当标量用，不要 `tl.load`。

## PyTorch 参考

```python
import torch

def layernorm_ref(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True)                     # [N, 1]
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True)            # 有偏方差
    return xc * torch.rsqrt(var + eps) * w + b

x = torch.randn(7, 30); w = torch.rand(30) + 0.5; b = torch.randn(30)
assert torch.allclose(layernorm_ref(x, w, b),
                      torch.nn.functional.layer_norm(x, (30,), w, b, 1e-5), atol=1e-6)

# 题面样例
x = torch.tensor([[1., 2, 3, 4], [-1, 0, 0, 1]])
print(layernorm_ref(x, torch.ones(4), torch.zeros(4)))
# [[-1.3416, -0.4472, 0.4472, 1.3416], [-1.4142, 0.0, 0.0, 1.4142]]
```

## 先算上限：为什么要 fuse

每个元素约 8 FLOP（加、减、平方、加、乘 rstd、乘 w、加 b），fp32 读 4 B 写 4 B，算术强度 ≈ 1 FLOP/B，memory-bound。时间看搬了几遍 `[N, C]`。

LeetGPU 参考实现是 eager PyTorch：

| 算子 | 读 | 写 |
|---|---|---|
| `input.mean(dim=1)` | 1 | 0 |
| `input.var(dim=1)` | 1 | 0 |
| `input - mean` | 1 | 1 |
| `/ torch.sqrt(var + eps)` | 1 | 1 |
| `weight * normalized` | 1 | 1 |
| `+ bias` | 1 | 1 |
| 合计 | 6 | 4 |

fused 只要读 1 写 1。性能测试 N = 65,536、C = 512：一遍是 65,536 × 512 × 4 B = 134 MB，T4（300 GB/s）上

- eager：10 × 134 MB = 1.34 GB → 约 4.5 ms
- fused：2 × 134 MB = 268 MB → 下限 **0.89 ms**

## 数值：方差怎么算

### 一遍法会炸

$\sigma^2 = E[x^2] - E[x]^2$ 只要扫一遍，但两个大数相减会丢掉有效位（catastrophic cancellation）。fp32 只有约 7 位有效数字，均值 1e4、方差 1 的数据，$E[x^2] \approx 10^8$，减掉 $E[x]^2 \approx 10^8$ 后剩下的 1 已经在舍入误差里了：

```python
torch.manual_seed(0)
x = 1e4 + torch.randn(4096)                  # fp32，真实方差 ≈ 1
one_pass = (x * x).mean() - x.mean() ** 2
two_pass = ((x - x.mean()) ** 2).mean()
print(one_pass.item(), two_pass.item(), x.double().var(unbiased=False).item())
# -8.0  0.98656  0.98656
```

一遍法不光不准，还算出了**负的**方差（换 seed 1–4 分别是 −16、−8、−16、0），`sqrt(var + eps)` 直接 NaN。

### 行在寄存器里：两遍法不花访存

C ≤ 4096 时整行一次 load 进寄存器，先 `tl.sum` 出 μ，再在寄存器里算 $\sum (x - \mu)^2$。"两遍"只是两次寄存器上的归约，HBM 只读一次。

### 行放不下：分块 + Welford 合并

C 太大时要分块扫。Triton 官方教程的 forward 是三个循环：扫一遍算 μ、再扫一遍算 σ²、第三遍归一化写出，x 读三遍（[Triton 教程 · Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)）。想少读一遍，就每块算出 (count, mean, M2)（M2 是块内 $\sum (x - \bar{x})^2$），块之间用 Chan 等人的并行合并公式（[Wikipedia: Algorithms for calculating variance § Parallel algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)）：

$$
n = n_a + n_b,\quad \delta = \bar{x}_b - \bar{x}_a,\quad
\bar{x} = \bar{x}_a + \delta \frac{n_b}{n},\quad
M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n}
$$

最后 $\sigma^2 = M_2 / n$。每块内部仍然是"先减块均值再平方"，所以不会出现大数相减。验证：

```python
def merge(a, b):
    na, ma, m2a = a; nb, mb, m2b = b
    n = na + nb; d = mb - ma
    return n, ma + d * nb / n, m2a + m2b + d * d * na * nb / n

def stats(t):
    return t.numel(), t.mean().item(), ((t - t.mean()) ** 2).sum().item()

xs = torch.randn(1000, dtype=torch.float64)
n, m, m2 = merge(stats(xs[:300]), stats(xs[300:]))
assert abs(m - xs.mean().item()) < 1e-12
assert abs(m2 / n - xs.var(unbiased=False).item()) < 1e-12
```

## 手撕：forward

### v1：一行一个 program

```python
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr, C, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)          # 防止 row * C 溢出 int32
    cols = tl.arange(0, BLOCK)                   # BLOCK = next_power_of_2(C) >= C
    mask = cols < C

    x = tl.load(x_ptr + row * C + cols, mask=mask, other=0.).to(tl.float32)
    mean = tl.sum(x, axis=0) / C                 # 越界位置填的 0 不影响和；分母是 C 不是 BLOCK
    xc = tl.where(mask, x - mean, 0.)            # 越界位置 0 - mean ≠ 0，必须清掉，否则进方差
    var = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.)
    b = tl.load(b_ptr + cols, mask=mask, other=0.)
    tl.store(y_ptr + row * C + cols, xc * rstd * w + b, mask=mask)
    # 反向要用的两个统计量，每行 2 个 float，相比 y 可以忽略
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def layernorm_fwd(x, w, b, y, eps):
    N, C = x.shape
    BLOCK = triton.next_power_of_2(C)
    num_warps = min(max(BLOCK // 256, 1), 8)     # Triton 教程的启发式
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    rstd = torch.empty(N, device=x.device, dtype=torch.float32)
    layernorm_fwd_kernel[(N,)](x, w, b, y, mean, rstd, C, eps, BLOCK=BLOCK, num_warps=num_warps)
    return mean, rstd


# input, weight, bias, output are tensors on the GPU
def solve(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output: torch.Tensor,
          N: int, C: int, eps: float):
    layernorm_fwd(input.view(N, C), weight, bias, output.view(N, C), eps)
```

逐项检查：

- **C = 1**：BLOCK = 1，mean = x，xc = 0，var = 0，y = 0 · rstd · w + b = b，和参考一致（题目的第一个测试 `[[3.0]]`、bias 0.5 → 0.5）。
- **全 0 行**：var = 0，rstd = 1/√ε ≈ 316，xc = 0，y = b，不会出 NaN。
- **C = 30（非 2 的幂）**：BLOCK = 32，后两个位置 mask 掉；`xc` 那一行的 `tl.where` 就是为它写的。
- **规模**：C ≤ 4096 时 BLOCK ≤ 4096，num_warps = 8 时每线程 16 个 fp32，寄存器够用。

### v2：小 C 时一个 program 处理多行

性能测试 C = 512：v1 有 65,536 个 program，每个只处理 2 KB 数据，num_warps = 2（64 个线程，每线程 8 个数）。每个 program 都要付一遍固定开销：算地址、两次 `tl.sum` 的 warp 间同步、load w 和 b。可以让一个 program 处理 BLOCK_M 行，把这些开销摊掉，w、b 也只 load 一次。下标用行 `[:, None]`、列 `[None, :]` 广播成二维：

```python
@triton.jit
def layernorm_fwd_rows_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, C, eps,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)            # [BLOCK_M]
    cols = tl.arange(0, BLOCK_N)                            # [BLOCK_N]
    row_ok = rows < N
    col_ok = cols < C
    mask = row_ok[:, None] & col_ok[None, :]                # [BLOCK_M, BLOCK_N]
    offs = rows.to(tl.int64)[:, None] * C + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.).to(tl.float32)
    mean = tl.sum(x, axis=1) / C                            # [BLOCK_M]，每行一个
    xc = tl.where(mask, x - mean[:, None], 0.)
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=col_ok, other=0.)        # 整个 program 只 load 一次
    b = tl.load(b_ptr + cols, mask=col_ok, other=0.)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(y_ptr + offs, y, mask=mask)


def solve(input, weight, bias, output, N, C, eps):
    BLOCK_N = triton.next_power_of_2(C)
    BLOCK_M = max(1, min(16, 4096 // BLOCK_N))              # 每个 program 约 4096 个元素
    num_warps = min(max(BLOCK_M * BLOCK_N // 256, 1), 8)
    grid = (triton.cdiv(N, BLOCK_M),)
    layernorm_fwd_rows_kernel[grid](input, weight, bias, output, N, C, eps,
                                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
```

C = 512 时 BLOCK_M = 8，program 数从 65,536 降到 8,192。越界的行（`rows >= N`）x 全是 0，mean = 0、var = 0，算出来的值被 store 的 mask 丢掉，不会写出去。这一版能快多少要实测，我没有跑过；BLOCK_M 和 num_warps 最好交给 `triton.autotune`。

对拍（需要 GPU）：

```python
for N, C in [(1, 1), (2, 2), (4, 4), (7, 30), (15, 100), (25, 255), (512, 768), (4, 4096), (65536, 512)]:
    x = torch.empty(N, C, device="cuda").uniform_(-10, 10)
    w = torch.empty(C, device="cuda").uniform_(0.5, 2.0)
    b = torch.empty(C, device="cuda").uniform_(-1, 1)
    out = torch.empty_like(x)
    solve(x, w, b, out, N, C, 1e-5)
    assert torch.allclose(out, layernorm_ref(x, w, b, 1e-5), rtol=1e-4, atol=1e-4), (N, C)
```

## 反向

### 公式

记上游梯度 $dy = \partial L / \partial y$，一行内（下标 j 是列）：

$$
c_1 = \frac{1}{C}\sum_j \hat{x}_j (w_j\, dy_j), \qquad
c_2 = \frac{1}{C}\sum_j w_j\, dy_j
$$

$$
dx_j = \text{rstd} \cdot \big(w_j\, dy_j - (\hat{x}_j c_1 + c_2)\big), \qquad
dw_j = \sum_{\text{rows}} dy_j\, \hat{x}_j, \qquad
db_j = \sum_{\text{rows}} dy_j
$$

和 Triton 教程里的写法一致。结构上分两类：

- **dx**：每行两个行内归约（c₁、c₂），和 forward 一样一行一个 program 就能做完
- **dw、db**：**跨行**归约，所有行加到同一个 `[C]` 向量上，是写冲突的地方

用 autograd 验证公式：

```python
def layernorm_bwd_ref(x, w, dy, eps=1e-5):
    C = x.shape[-1]
    mu = x.mean(-1, keepdim=True)
    rstd = torch.rsqrt(((x - mu) ** 2).mean(-1, keepdim=True) + eps)
    xhat = (x - mu) * rstd
    wdy = w * dy
    c1 = (xhat * wdy).sum(-1, keepdim=True) / C
    c2 = wdy.sum(-1, keepdim=True) / C
    dx = (wdy - (xhat * c1 + c2)) * rstd
    return dx, (dy * xhat).sum(0), dy.sum(0)

x = torch.randn(6, 20, dtype=torch.float64, requires_grad=True)
w = torch.randn(20, dtype=torch.float64, requires_grad=True)
b = torch.randn(20, dtype=torch.float64, requires_grad=True)
dy = torch.randn(6, 20, dtype=torch.float64)
torch.nn.functional.layer_norm(x, (20,), w, b, 1e-5).backward(dy)
dx, dw, db = layernorm_bwd_ref(x.detach(), w.detach(), dy)
assert torch.allclose(dx, x.grad) and torch.allclose(dw, w.grad) and torch.allclose(db, b.grad)
```

### dw、db 的跨行归约

N 行都往同一个 `dw[C]` 上加，三种做法（和 [CUDA Reduce](/handson/cuda-reduce) 里 grid 级归约是同一个问题）：

| 做法 | 说明 |
|---|---|
| 每行 `atomic_add` 到 `dw` | N 次 atomic 打在同一组地址上，严重争用；结果不确定 |
| Triton 教程：GROUP_SIZE_M 个 buffer + 锁 | 行按 `row % GROUP_SIZE_M` 分组，同组共用一个 partial buffer，用自旋锁互斥写；第二个 kernel 把 GROUP_SIZE_M 个 buffer 加起来 |
| 下面的写法：每个 program 负责连续若干行 | program 在寄存器里累加自己那几行的 dw、db，最后写到 `partial[pid]`，不需要锁；第二步 `partial.sum(0)` |

第三种最好写，而且确定（每个 program 加的顺序固定）：

```python
@triton.jit
def layernorm_bwd_kernel(x_ptr, dy_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                         dw_part_ptr, db_part_ptr, N, C, ROWS_PER_PROG,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < C
    w = tl.load(w_ptr + cols, mask=mask, other=0.).to(tl.float32)
    dw_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK,), dtype=tl.float32)

    row_start = pid * ROWS_PER_PROG
    row_end = tl.minimum(row_start + ROWS_PER_PROG, N)
    for row in range(row_start, row_end):
        offs = row.to(tl.int64) * C + cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.).to(tl.float32)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)

        xhat = tl.where(mask, (x - mean) * rstd, 0.)
        wdy = w * dy                                   # 越界位置 w = 0、dy = 0
        c1 = tl.sum(xhat * wdy, axis=0) / C
        c2 = tl.sum(wdy, axis=0) / C
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(dx_ptr + offs, dx, mask=mask)

        dw_acc += dy * xhat                            # 跨行部分先在寄存器里累加
        db_acc += dy

    tl.store(dw_part_ptr + pid * C + cols, dw_acc, mask=mask)
    tl.store(db_part_ptr + pid * C + cols, db_acc, mask=mask)


def layernorm_bwd(x, w, mean, rstd, dy):
    N, C = x.shape
    BLOCK = triton.next_power_of_2(C)
    num_warps = min(max(BLOCK // 256, 1), 8)
    G = min(N, 256)                                    # program 数，几倍 SM 数即可
    rows_per_prog = triton.cdiv(N, G)
    G = triton.cdiv(N, rows_per_prog)                  # 去掉分不到行的 program
    dx = torch.empty_like(x)
    dw_part = torch.empty(G, C, device=x.device, dtype=torch.float32)
    db_part = torch.empty_like(dw_part)
    layernorm_bwd_kernel[(G,)](x, dy, w, mean, rstd, dx, dw_part, db_part, N, C, rows_per_prog,
                               BLOCK=BLOCK, num_warps=num_warps)
    return dx, dw_part.sum(0).to(w.dtype), db_part.sum(0).to(w.dtype)
```

对拍：

```python
x = torch.randn(1000, 768, device="cuda", requires_grad=True)
w = torch.randn(768, device="cuda", requires_grad=True)
b = torch.randn(768, device="cuda", requires_grad=True)
dy = torch.randn(1000, 768, device="cuda")
torch.nn.functional.layer_norm(x, (768,), w, b, 1e-5).backward(dy)

y = torch.empty_like(x)
mean, rstd = layernorm_fwd(x.detach(), w.detach(), b.detach(), y, 1e-5)
dx, dw, db = layernorm_bwd(x.detach(), w.detach(), mean, rstd, dy)
assert torch.allclose(dx, x.grad, atol=1e-4)
assert torch.allclose(dw, w.grad, atol=1e-3) and torch.allclose(db, b.grad, atol=1e-3)
```

反向的访存：读 x、dy，写 dx，各一遍；partial 是 G × C 个数（256 × 768 × 4 B ≈ 0.8 MB），相对 `[N, C]` 很小。forward 存下来的 mean、rstd 让反向不用重算统计量。

## 面试追问

::: details Q：hidden 维超过一个 program 能放下的大小怎么办？
分块循环。最简单是 Triton 教程的三遍：一遍算 μ，一遍算 σ²，一遍归一化写回，x 读三遍。用 Welford / Chan 合并可以把前两遍并成一遍：每块算 (n, mean, M2)，块之间按合并公式更新，x 读两遍。都比非融合版本少写中间结果。
:::

::: details Q：为什么不用 E[x²] − E[x]² 一遍算方差？
两个大数相减丢有效位。均值 1e4、方差 1 的 fp32 数据，一遍法算出 −8.0（负方差，开根号就是 NaN），两遍法 0.987（上面的例子）。LLM 激活里有幅值远大于其他维度的 outlier 特征（[Dettmers et al., 2022, LLM.int8()](https://arxiv.org/abs/2208.07339)），$E[x^2]$ 和 $E[x]^2$ 的量级被抬高，相减丢掉的有效位也随之变多。行在寄存器里时两遍法不多花访存，没理由用一遍法。
:::

::: details Q：LayerNorm 反向里 dw、db 为什么麻烦？
dx 只依赖本行，一行一个 program 就行；dw、db 是所有行的和，N 个 program 写同一个 `[C]` 向量。直接 atomic 争用严重、结果不确定。Triton 教程用分组 buffer + 锁 + 第二个 kernel；更简单的是每个 program 处理连续若干行，先在寄存器里累加，写各自的 partial，再 `sum(0)`。
:::

## 参考

- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016
- [Triton 教程 · Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)（forward 三遍循环、反向公式、分组 buffer + 锁的 dw/db 归约、`num_warps` 启发式）
- [Wikipedia: Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)（Welford 与 Chan 并行合并公式）

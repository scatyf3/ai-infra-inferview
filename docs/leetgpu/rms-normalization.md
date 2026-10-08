---
title: RMS Normalization
status: draft
tags: [rmsnorm, triton, leetgpu]
familiarity: 2
difficulty: 3
order: 5
related: [/handson/rmsnorm]
stack: [k-fused]
leetgpu: [50]
---

# RMS Normalization

> [LeetGPU #50](https://leetgpu.com/challenges/rms-normalization) · 1D 输入 `[N]`，标量 γ / β · Triton · 概念见 [RMSNorm](/handson/rmsnorm)

$$
\text{rms} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon}, \qquad
y_i = \gamma \frac{x_i}{\text{rms}} + \beta
$$

## 思路

每个 $y_i$ 都依赖全局的 rms，所以至少两轮：一轮 reduce 求 $\sum x_i^2$，一轮 pointwise。
N 可能远大于一个 block，block 之间不能同步，于是拆成两个 kernel：

1. `sum_kernel`：每个 block 算自己那段的平方和，写进 `sum[pid]`
2. `normalize`：每个 block 先把 `sum` 整个读进来求总和、算 rms，再处理自己那段

注意这题和 LLM 里的 RMSNorm 不是一回事，见 [下面](#和-llm-里的-rmsnorm-的区别)。

## 题解

```python
import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(input, N, sum, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 先加偏移
    mask = offset < N                                      # 再算 mask
    x = tl.load(input + offset, mask=mask, other=0)
    tl.store(sum + pid, tl.sum(x * x, axis=0))             # 只存部分平方和，不开根号


@triton.jit
def normalize(input, output, N, sum, num_blocks, gamma, beta, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # 汇总所有 block 的部分和；要求 num_blocks <= BLOCK_SIZE
    offset_sum = tl.arange(0, BLOCK_SIZE)
    mask_sum = offset_sum < num_blocks
    rms = tl.sqrt(tl.sum(tl.load(sum + offset_sum, mask=mask_sum, other=0), axis=0) / N + eps)
    # pointwise
    offset_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_x = offset_x < N
    x_bar = tl.load(input + offset_x, mask=mask_x, other=0) / rms
    y = gamma * x_bar + beta                               # gamma / beta 实际传进来是 float
    tl.store(output + offset_x, y, mask=mask_x)            # 先指针后值


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    sum = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    sum_kernel[(num_blocks,)](input, N, sum, BLOCK_SIZE)
    normalize[(num_blocks,)](input, output, N, sum, num_blocks, gamma, beta, eps, BLOCK_SIZE)
```

## 这题的坑

**会报错的**

1. **grid 传 int**：`kernel[num_blocks](...)` 报 "object of type 'int' has no len()"。grid 是元组：`kernel[(num_blocks,)]`。
2. **`torch.cdiv(N / BLOCK_SIZE)`**：没有 `torch.cdiv`，用 `triton.cdiv(N, BLOCK_SIZE)`，两个参数。
3. **对标量 `tl.load`**：签名写着 `gamma: torch.Tensor`，实际传进来是 Python float，`tl.load(gamma)` 报 "Unsupported ptr type triton.language.float32"。规则：kernel 收到 tensor 是指针，要 load；收到 float / int 是值，直接用。拿不准就在 `solve` 里 `print(type(gamma))`。
4. **`tl.store(y, ptr)`**：参数顺序是 `tl.store(ptr, value, mask=...)`。

**不报错、结果静默错的**

5. **mask 在加 pid 偏移之前算**：`mask = tl.arange(0, BLOCK_SIZE) < N` 对每个 block 几乎全 True，最后一个 block 越界读。
6. **每个 block 先开根号再合并**：$\sqrt{a+b} \ne \sqrt{a}+\sqrt{b}$。block 里只存部分平方和，开根号放到汇总之后。
7. **读部分和时 mask 用 `< N`**：`sum` 只有 `num_blocks` 个元素，要 `< num_blocks`，否则读进越界垃圾。
8. **公式抄错**：`gamma * rms + beta`，应该是 `gamma * x_bar + beta`。

## 和 LLM 里的 RMSNorm 的区别

LLM 里的 RMSNorm 是**每个 token 单独归一化**，沿 hidden 维 reduce，不是把所有元素当成一个大向量。

- 输入 `x`：`[batch, seq_len, hidden]`；vLLM / SGLang 里先拍平成 `[num_tokens, hidden]`
- `weight`（γ）：`[hidden]`，逐通道；**没有 β**
- 每个 token 一个 rms，shape `[num_tokens, 1]`

$$
y_{t,j} = \frac{x_{t,j}}{\sqrt{\frac{1}{H}\sum_{k=1}^{H} x_{t,k}^2 + \epsilon}} \cdot w_j
$$

| | LeetGPU #50 | LLM 里的 RMSNorm |
|---|---|---|
| 输入 | `[N]` 一个大向量 | `[T, H]`，H 一般 4k–8k |
| reduce 范围 | 全部元素，跨 block | 每行内部 |
| γ / β | 标量，有 β | 向量 `[H]`，无 β |
| kernel | 两个，HBM 读两遍 | 一个，一行一个 program，HBM 读一遍 |

PyTorch 参考（HF LLaMA 的写法）：

```python
def rmsnorm(x, weight, eps=1e-6):              # x: [T, H], weight: [H]
    dtype = x.dtype
    x = x.float()                               # 统计量在 fp32 里算
    var = x.pow(2).mean(dim=-1, keepdim=True)   # [T, 1]
    x = x * torch.rsqrt(var + eps)
    return x.to(dtype) * weight                 # 先转回原 dtype 再乘 weight
```

Triton 写法：grid `(T,)`，一个 program 处理一行，`BLOCK_SIZE = triton.next_power_of_2(H)`，整行 load 进来后 `tl.sum(x * x)` 再做 pointwise。全在片上，不需要 partial buffer 也不需要 atomic，比本题还简单。

推理引擎里更常见的是 **fused residual add + RMSNorm**：`residual = x + residual; y = rmsnorm(residual)` 合成一个 kernel，少一遍 HBM 读写（vLLM 的 `fused_add_rms_norm`）。对应 LeetGPU #83，可以接着做。

## 局限和改法

- `normalize` 用一个 `tl.arange(0, BLOCK_SIZE)` 读部分和，要求 `num_blocks <= BLOCK_SIZE`，即 $N \le 1024^2$。更大的 N：host 端 `sum.sum()` 后把 total 传进去，或者第一个 kernel 改成 `tl.atomic_add` 到一个 `torch.zeros(1)`。
- `atomic_add` 版本更短，但浮点加的顺序不固定，结果不确定（末位抖动）；当前写法每个 block 写自己的格子、固定顺序求和，是确定的。
- 输入如果是 fp16 / bf16，load 后先 `.to(tl.float32)` 再平方累加。
- `sum` 这个名字遮住了 Python 内置 `sum`，现在不出错，以后在 `solve` 里想用内置 `sum()` 时会坏，可以改名 `partial`。

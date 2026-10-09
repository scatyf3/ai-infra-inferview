---
title: RMSNorm
status: draft
tags: [rmsnorm, handson]
difficulty: 2
order: 4
related: [/leetgpu/rms-normalization, /leetgpu/fused-rms-norm, /handson/triton-fused-layernorm, /handson/triton-softmax, /handson/kernel-mindset, /handson/cuda-reduce]
stack: [k-fused]
leetgpu: [50, 83]
---

# RMSNorm

> 与 LayerNorm 的区别、fused 实现 · [LeetGPU #50 RMS Normalization](https://leetgpu.com/challenges)（题解：[LeetGPU · RMS Normalization](/leetgpu/rms-normalization)）· [LeetGPU #83 Fused Residual Add and RMS Norm](https://leetgpu.com/challenges)（[题面](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm)，笔记：[LeetGPU · Fused RMS Norm](/leetgpu/fused-rms-norm)）

## 一句话结论

RMSNorm 去掉了 LayerNorm 的减均值，只用均方根归一化再乘可学习的逐通道权重 w：$y = x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot w$。少一个统计量、没有偏置 β，LLaMA 系列之后的主流开源模型都用它。它是 memory-bound 的：一个 kernel 里读一遍 x、算 rms、写一遍 y，别让 x²、mean 这些中间结果落到 HBM；推理框架再把前面的残差加也融合进来。

## 定义

一个 token 的隐藏向量 $x \in \mathbb{R}^H$（H 是 hidden size，LLaMA-7B 是 4096），权重 $w \in \mathbb{R}^H$，$\epsilon$ 是防止除 0 的小常数（HF `LlamaConfig` 默认 `rms_norm_eps=1e-6`，LeetGPU 用 1e-5）：

$$
\text{rms}(x) = \sqrt{\frac{1}{H}\sum_{k=1}^{H} x_k^2 + \epsilon}, \qquad
y_j = \frac{x_j}{\text{rms}(x)} \cdot w_j
$$

LLM 里输入是 `[T, H]`（T 是这一批的 token 数，batch 和 seq 拍平），**每一行（每个 token）独立归一化**，沿 H 归约。row-major 存储，第 t 行第 j 列在 `x[t * H + j]`。

LeetGPU 两题的形状：

| | #50 | #83 |
|---|---|---|
| 输入 | 一维 `[N]`，整个向量算一个 rms | `x`、`residual` 都是 `[N, C]`，每行一个 rms |
| 参数 | 标量 γ、β：$y = \gamma \hat{x} + \beta$ | 逐通道 `weight[C]`，无 β |
| 难点 | 归约跨 block，要两个 kernel | 残差加和 norm 融合，一行一个 program |

#50 是"全局归约 + 逐元素"，和 LLM 里的 RMSNorm 不是一回事，题解和区别都在 [LeetGPU · RMS Normalization](/leetgpu/rms-normalization)。本页讲 LLM 里的逐行版本，也就是 #83。

## 和 LayerNorm 的区别

LayerNorm（[Ba et al., 2016](https://arxiv.org/abs/1607.06450)）：

$$
\mu = \frac{1}{H}\sum_k x_k, \quad \sigma^2 = \frac{1}{H}\sum_k (x_k - \mu)^2, \quad y_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} w_j + b_j
$$

| | LayerNorm | RMSNorm |
|---|---|---|
| 统计量 | μ 和 σ²（两个归约；两遍法要读两次） | 只有 $\sum x^2$（一个归约） |
| 不变性 | 对输入平移和缩放都不变 | 只对缩放不变 |
| 参数 | w、b | 只有 w |

RMSNorm 的论文（[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)）的论点是 LayerNorm 起作用的主要是缩放不变性，去均值（re-centering）可以省掉；他们报告在不同模型上运行时间减少 7%–64%，效果相当（作者自测）。LLaMA 在每个 transformer 子层的**输入**上做 RMSNorm（pre-norm）（[Touvron et al., 2023](https://arxiv.org/abs/2302.13971) §2.2）。LayerNorm 的 kernel 写法见 [Triton 版 Fused LayerNorm](/handson/triton-fused-layernorm)。

## PyTorch 参考实现

```python
import torch

def rmsnorm_ref(x, w, eps=1e-6):
    # x: [T, H]，w: [H]
    xf = x.float()                                           # 统计量用 fp32 算
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)   # [T, 1]，rstd = 1 / rms
    return (xf * rstd).to(x.dtype) * w                       # HF LLaMA 的顺序：先转回原 dtype 再乘 w

def fused_add_rmsnorm_ref(x, residual, w, eps=1e-6):
    z = x + residual                                         # 新的残差流，下一层还要用
    return rmsnorm_ref(z, w, eps), z
```

和 PyTorch 自带的 `torch.nn.functional.rms_norm` 对一下：

```python
x, w = torch.randn(8, 4096), torch.rand(4096) + 0.5
assert torch.allclose(rmsnorm_ref(x, w, 1e-6),
                      torch.nn.functional.rms_norm(x, (4096,), w, 1e-6), atol=1e-5)

# LeetGPU #83 的样例
x = torch.tensor([[1.0, 0.0, -1.0, 2.0]]); r = torch.tensor([[0.5, 1.5, 0.5, -0.5]])
y, _ = fused_add_rmsnorm_ref(x, r, torch.ones(4), 1e-5)
# y ≈ [1.1339, 1.1339, -0.3780, 1.1339]
```

**数值**：bf16 只有 8 位尾数，H = 4096 个平方加在一起误差很大，所以 load 后先 cast 到 fp32 再平方累加。ε 加在根号里面。

## 为什么要 fuse：数字

每行 H 个元素，做的事是平方（H 次乘）、累加（H 次加）、乘 rstd（H 次）、乘 w（H 次），约 4H FLOP。bf16 下读 2H 字节、写 2H 字节，w 每行都一样，留在 cache 里。算术强度约 4H / 4H = **1 FLOP/B**，远低于 ridge point，纯 memory-bound，时间 ≈ 搬的字节 / 带宽。

所以只数"整张 `[T, H]` 张量被读写了几遍"。上面的 `rmsnorm_ref` 用 eager PyTorch 跑，每个算子一个 kernel、结果都写回 HBM：

| 算子 | 读 | 写 |
|---|---|---|
| `x.float()` | 1 | 1 |
| `pow(2)` | 1 | 1 |
| `mean(-1)` | 1 | 0（只写 T 个数） |
| `xf * rstd` | 1 | 1 |
| `.to(dtype)` | 1 | 1 |
| `* w` | 1 | 1 |
| 合计 | 6 | 5 |

fused kernel 只要读 1 遍、写 1 遍。按 11 遍 vs 2 遍粗算（忽略 fp32 中间结果比 bf16 大一倍），差 5 倍多。`torch.compile` 能自动把这串融成一个 kernel，手写 kernel 的意义在于还能把**前后的算子**一起融进来。

### 残差加 + RMSNorm

pre-norm 的 decoder 层里，每个子层前都是：

```python
residual = x + residual          # 上一个子层的输出加回残差流
h = rmsnorm(residual, w)         # 归一化后送进 attention / MLP
```

数 HBM 读写（单位：一张 `[T, H]` 张量）：

| 写法 | 读 | 写 | 合计 |
|---|---|---|---|
| 分开两个 kernel | x、residual、新 residual（norm 再读一遍）= 3 | 新 residual、h = 2 | 5 |
| 融合，新 residual 写回（vLLM 的做法） | x、residual = 2 | 新 residual、h = 2 | 4 |
| 融合，不需要写回 residual（LeetGPU #83） | 2 | h = 1 | 3 |

推理框架里下一个子层还要用新的残差，所以必须写回，融合省的是 5 → 4，即 20% 的流量。vLLM 的 `fused_add_rms_norm` 就是这么做的：原地把 `input + residual` 写回 `residual`，再把归一化结果写回 `input`（[vLLM `layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu)）。它的通用版第二遍是从 `residual` 重新读 z（大概率命中 cache），不是从寄存器里拿。

LeetGPU #83 的性能测试 N = C = 4096、fp32，一张张量 4096 × 4096 × 4 B = 67 MB。T4（300 GB/s）上：

- 分开：5 × 67 MB = 336 MB → 1.12 ms
- 融合：3 × 67 MB = 201 MB → **0.67 ms**

## 手撕：Triton 版（LeetGPU #83）

kernel mindset 的标签（见 [kernel mindset](/handson/kernel-mindset)）：N（token）是并行维，进 grid，一行一个 program；C 是归约维，在 program 内部 `tl.sum`。C 最大 65,536，一行不一定能整个放进寄存器，所以写两版：

- **整行版**：C ≤ 16384 时一次 load 整行。阈值照 Triton LayerNorm 教程的 `MAX_FUSED_SIZE = 65536 // element_size`（fp32 就是 16384），`num_warps` 也照它的启发式 `min(max(BLOCK // 256, 1), 8)`（[Triton 教程 · Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)）。
- **分块版**：更长的行分两遍扫：第一遍累加平方和，第二遍重新读 x、residual 算输出。多读一遍 x、residual，但不用写中间结果。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_row_kernel(x_ptr, r_ptr, w_ptr, out_ptr, C, eps, BLOCK: tl.constexpr):
    # 一个 program 处理一行；row 转 int64，防止 row * C 超过 int32
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < C
    offs = row * C + cols

    # 越界位置填 0：z 也是 0，不影响平方和
    x = tl.load(x_ptr + offs, mask=mask, other=0.).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.).to(tl.float32)
    z = x + r                                   # 残差加，留在寄存器里，不落 HBM

    ms = tl.sum(z * z, axis=0) / C              # 除以真实的 C，不是 BLOCK
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.)
    tl.store(out_ptr + offs, z * rstd * w, mask=mask)


@triton.jit
def add_rmsnorm_chunked_kernel(x_ptr, r_ptr, w_ptr, out_ptr, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * C

    # 第一遍：按 BLOCK 一段段扫，向量累加平方和，最后一次性 tl.sum
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, C, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < C
        z = tl.load(x_ptr + base + cols, mask=mask, other=0.).to(tl.float32) \
          + tl.load(r_ptr + base + cols, mask=mask, other=0.).to(tl.float32)
        acc += z * z
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / C + eps)

    # 第二遍：重新算 z，归一化后写出
    for start in range(0, C, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < C
        z = tl.load(x_ptr + base + cols, mask=mask, other=0.).to(tl.float32) \
          + tl.load(r_ptr + base + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.)
        tl.store(out_ptr + base + cols, z * rstd * w, mask=mask)


# x, residual, weight, out are tensors on the GPU
def solve(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
          N: int, C: int, eps: float):
    MAX_FUSED = 65536 // x.element_size()       # fp32: 16384
    BLOCK = triton.next_power_of_2(C)
    grid = (N,)                                 # 一维 grid 放在 axis 0，上限 2^31 - 1
    if BLOCK <= MAX_FUSED:
        num_warps = min(max(BLOCK // 256, 1), 8)
        add_rmsnorm_row_kernel[grid](x, residual, weight, out, C, eps, BLOCK=BLOCK, num_warps=num_warps)
    else:
        add_rmsnorm_chunked_kernel[grid](x, residual, weight, out, C, eps, BLOCK=4096, num_warps=8)
```

和参考对拍（需要 GPU）：

```python
for N, C in [(1, 1), (1, 4), (30, 100), (100, 255), (128, 512), (4096, 4096), (4, 65536)]:
    x = torch.randn(N, C, device="cuda"); r = torch.randn(N, C, device="cuda")
    w = torch.empty(C, device="cuda").uniform_(0.5, 1.5)
    out = torch.empty_like(x)
    solve(x, r, w, out, N, C, 1e-5)
    ref, _ = fused_add_rmsnorm_ref(x, r, w, 1e-5)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5), (N, C)   # #83 的容差
```

这题的坑：

- **除以 BLOCK**：`tl.sum(z * z) / BLOCK` 在 C 不是 2 的幂时错，分母是真实列数 C。
- **mask 掉的位置不填 0**：`other` 不写时越界值未定义，加进平方和就错了。
- **每行一个 program 会不会 program 太多**：N = 65,536 个 program 没问题，硬件按 wave 调度；N 在 axis 0，上限 2³¹ − 1，axis 1、2 才只有 65,535。详见 [LeetGPU · Fused RMS Norm](/leetgpu/fused-rms-norm)。

## 反向

训练时也要 fuse 反向。记 $r = \text{rms}(x)$，上游梯度 $g = \partial L / \partial y$。由 $\partial r / \partial x_k = x_k / (H r)$：

$$
\frac{\partial L}{\partial x_k} = \frac{g_k w_k}{r} - \frac{x_k}{H r^3} \sum_{j} g_j w_j x_j,
\qquad
\frac{\partial L}{\partial w_j} = \sum_{\text{rows}} \frac{g_j x_j}{r}
$$

dx 是行内的又一次归约（$\sum_j g_j w_j x_j$），dw 是**跨行**归约，和 LayerNorm 反向的结构一样，跨行那部分的写法见 [Triton 版 Fused LayerNorm](/handson/triton-fused-layernorm#反向)。用 autograd 验证公式：

```python
def rmsnorm_bwd(x, w, g, eps=1e-6):
    H = x.shape[-1]
    r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)       # [T, 1]
    gw = g * w
    dx = gw / r - x * (gw * x).sum(-1, keepdim=True) / (H * r ** 3)
    dw = (g * x / r).sum(0)
    return dx, dw

x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
w = torch.randn(16, dtype=torch.float64, requires_grad=True)
g = torch.randn(5, 16, dtype=torch.float64)
rmsnorm_ref(x, w).backward(g)
dx, dw = rmsnorm_bwd(x.detach(), w.detach(), g)
assert torch.allclose(dx, x.grad) and torch.allclose(dw, w.grad)
```

## 面试追问

::: details Q：为什么推理框架把残差加和 RMSNorm 融合成一个 kernel？
两者都是 memory-bound 的逐元素 / 逐行操作。分开写时，残差和要写到 HBM 再被 norm 读回来。融合后读 x、residual 各一次，写新 residual 和 norm 结果各一次：按整张 `[T, H]` 张量计是 5 遍降到 4 遍，省 20%；如果新 residual 不用写回（LeetGPU #83），是 5 遍降到 3 遍，省 40%。每层两次 norm，decode 下每个 token 都要过所有层，累积起来可观。
:::

::: details Q：hidden size 很大、一行放不进一个 program 怎么办？
分块扫两遍：第一遍累加 $\sum x^2$，第二遍重新读 x 做归一化，见上面的 `add_rmsnorm_chunked_kernel`。代价是 x 读两遍，但第二遍很可能命中 L2。另一种是把一行拆给多个 program，但那样平方和要跨 program 归约，需要第二个 kernel 或 atomic，见 [CUDA Reduce](/handson/cuda-reduce) 里 grid 级归约的三种方式，通常不划算。
:::

::: details Q：RMSNorm 的 ε 放在根号里面和外面有区别吗？
有。$x / (\sqrt{\text{ms}} + \epsilon)$ 和 $x / \sqrt{\text{ms} + \epsilon}$ 在 ms 很小时差别大，加载预训练权重时必须和训练时一致。HF LLaMA 和 LeetGPU 都是放在根号里面（`rsqrt(mean + eps)`）。
:::

## 参考

- Biao Zhang, Rico Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), NeurIPS 2019
- Jimmy Lei Ba et al., [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016
- Hugo Touvron et al., [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971), 2023
- [vLLM `csrc/libtorch_stable/layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu)（`rms_norm_kernel`、`fused_add_rms_norm_kernel`）
- [Triton 教程 · Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)

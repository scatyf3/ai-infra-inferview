---
title: SwiGLU MLP Block
status: draft
tags: [swiglu, ffn, matmul, leetgpu]
difficulty: 2
order: 6.8
related: [/leetgpu/swish-gated-linear-unit]
stack: []
leetgpu: [84]
---

# SwiGLU MLP Block

> [LeetGPU #84](https://leetgpu.com/challenges/swiglu-mlp-block) · `x: [M, d_model]` · torch · 激活本身见 [Swish-Gated Linear Unit](./swish-gated-linear-unit)

$$
y = \big(\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big)\, W_{\text{down}}
$$

| | shape |
|---|---|
| `x` | `[M, d_model]` |
| `W_gate`, `W_up` | `[d_model, d_ffn]` |
| `W_down` | `[d_ffn, d_model]` |
| 中间结果 | `[M, d_ffn]` |
| `output` | `[M, d_model]` |

Llama 系的 FFN 就是这个（Llama 论文 arXiv 2302.13971 采用 Shazeer 2020 的 SwiGLU）。

## 题解

```python
# * 是逐元素乘，torch.matmul / @ 是矩阵乘
def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    z = torch.matmul(x, W_gate)              # [M, d_ffn]
    z = z * torch.sigmoid(z)                 # SiLU
    up = torch.matmul(x, W_up)               # [M, d_ffn]
    res = torch.matmul(z * up, W_down)       # 逐元素乘后再投影回 [M, d_model]
    output.copy_(res)
```

## 这题的坑

1. **`*` 和 `@` 混用**：gate 和 up 之间是逐元素乘，三次投影才是 matmul。`z @ up` 会报 shape 不匹配（`[M, d_ffn] @ [M, d_ffn]`），`z * W_down` 则是广播错。
2. **权重方向**：本题给的是 `[in, out]`，直接 `x @ W`。HF 里 `nn.Linear.weight` 存的是 `[out, in]`，那边是 `x @ W.T`。照着 HF 的习惯加 `.T` 在方阵时不报错、静默错。
3. **SiLU 只作用在 gate 上**：up 分支不过激活。

## 数一下

- FLOPs：三个 GEMM，每个 $2 M\, d_{\text{model}}\, d_{\text{ffn}}$，共 $6 M\, d_{\text{model}}\, d_{\text{ffn}}$；SiLU 和逐元素乘是 $O(M d_{\text{ffn}})$，可忽略。
- 参数：$3\, d_{\text{model}}\, d_{\text{ffn}}$。为了和两矩阵的 GELU FFN（$d_{\text{ffn}} = 4d$）参数量持平，SwiGLU 取 $d_{\text{ffn}} \approx \tfrac{2}{3} \cdot 4d$，再取整到 256 的倍数。例：Llama-2-7B，$d = 4096$，$\tfrac{8}{3} \cdot 4096 \approx 10923 \to 11008$；Llama-3-8B 额外放大到 $14336 = 3.5d$。
- 显存：上面的写法会产生 `z`、`sigmoid(z)`、`z * sigmoid(z)`、`up`、`z * up` 好几个 `[M, d_ffn]` 临时张量。用 `F.silu` 能省一个；真要省就得 fuse。

## 推理引擎里怎么做

1. **gate、up 合成一个 GEMM**：权重拼成 `[d_model, 2·d_ffn]`，一次 matmul 出 `[M, 2·d_ffn]`，x 只读一遍，GEMM 也更大更好打满。
2. **SiLU × up 做成一个逐元素 kernel**（vLLM `SiluAndMul`），就是 [#54](./swish-gated-linear-unit) 那道题，或者直接做成 GEMM 的 epilogue。
3. **down 投影**再一个 GEMM。TP 下 gate/up 按列切、down 按行切，整个 FFN 只需一次 all-reduce（Megatron-LM，arXiv 1909.08053 第 3 节）。

```python
def swiglu_mlp_fused(x, W_gate_up, W_down):  # W_gate_up: [d_model, 2*d_ffn]
    gu = x @ W_gate_up                        # [M, 2*d_ffn]
    g, u = gu.chunk(2, dim=-1)
    return (F.silu(g) * u) @ W_down
```

---
title: 梯度记号：公式里的偏导与代码里的 dX
---

# 梯度记号：公式里的偏导与代码里的 dX

$L$ 是最终的 loss（标量）。公式里写偏导 $\frac{\partial L}{\partial X}$；代码里这些梯度是普通变量，惯例命名为 `dX`。$\odot$ 是逐元素乘（对应位置相乘，代码里的 `*`），$AB$ 是矩阵乘（代码里的 `@`）：$\begin{bmatrix}1&2\end{bmatrix} \odot \begin{bmatrix}3&4\end{bmatrix} = \begin{bmatrix}3&8\end{bmatrix}$。

## 不同场合的写法

| 场合 | 写法 | 出处 |
|---|---|---|
| 数学推导 | $\frac{\partial L}{\partial h}$，或 $\nabla_h L$ | 大多数教材 |
| 自动微分 | $\bar h$（adjoint），JAX 里叫 cotangent | Griewank & Walther，*Evaluating Derivatives* |
| 经典 backprop | $\delta^{(l)}$，第 $l$ 层的误差 | Rumelhart et al. 1986 |
| 代码 | `dh`、`grad_h`；PyTorch `backward(ctx, grad_output)` | CS231n、PyTorch、Triton / CUDA kernel |

## 公式与代码变量的对应

一个算子 $h = f(a, b)$ 的 backward：

| 公式 | 代码变量 | 意思 | 谁给 |
|---|---|---|---|
| $\frac{\partial L}{\partial h}$ | `dh` | loss 对输出的梯度 | 上游（后一层的 backward）传下来 |
| $\frac{\partial h}{\partial b}$ | — | 本算子的局部导数 | 自己推 |
| $\frac{\partial L}{\partial b}$ | `db` | loss 对输入的梯度 | 本算子算出来，传给前一层 |

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial b}
$$

逐元素算子的 $\frac{\partial h}{\partial b}$ 是对角 Jacobian，乘法写成 $\odot$ 即可。

## 链式法则

高中版：$y = f(g(x))$ 时 $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$。例：$y = (3x+1)^2$，$\frac{dy}{dx} = 2(3x+1) \cdot 3$。

backward 就是从 $L$ 往回，一层层乘上局部导数：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

一个变量走了几条路，梯度就把几条路**相加**。例：残差 $y = x + f(x)$，$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} + \frac{\partial L}{\partial y} \cdot f'(x)$；FFN 里 $x$ 同时算出 $a$ 和 $b$，$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial a} W_{\text{gate}}^\top + \frac{\partial L}{\partial b} W_{\text{up}}^\top$。

## LLM 里常见函数的导数

记 $g = \frac{\partial L}{\partial y}$（上游传来的）。「存什么」是 forward 要留给 backward 的张量，决定激活显存。

| forward | backward | 存什么 |
|---|---|---|
| 加法 $y = a + b$ | $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial b} = g$ | 无 |
| 逐元素乘 $y = a \odot b$ | $\frac{\partial L}{\partial a} = g \odot b$，$\frac{\partial L}{\partial b} = g \odot a$ | $a, b$ |
| 矩阵乘 $Y = XW$ | $\frac{\partial L}{\partial X} = g\,W^\top$，$\frac{\partial L}{\partial W} = X^\top g$ | $X$（$W$ 本来就在） |
| $\exp(x)$ / $\log(x)$ | $g \odot e^x$ / $g \odot \frac{1}{x}$ | $e^x$ / $x$ |
| ReLU | $g \odot \mathbb{1}[x > 0]$ | $x$（或一个 mask） |
| sigmoid $\sigma(x)$ | $g \odot \sigma(x)(1 - \sigma(x))$ | $\sigma(x)$ |
| SiLU $x\,\sigma(x)$ | $g \odot \sigma(x)\big(1 + x(1 - \sigma(x))\big)$ | $x$ |
| GELU $x\,\Phi(x)$ | $g \odot \big(\Phi(x) + x\,\varphi(x)\big)$，$\Phi$、$\varphi$ 是标准正态的 CDF、PDF | $x$ |
| softmax $p = \text{softmax}(z)$（按行） | $\frac{\partial L}{\partial z} = p \odot \big(g - \textstyle\sum_j g_j p_j\big)$ | $p$ |
| softmax + cross-entropy，标签 $y$ | $\frac{\partial L}{\partial z} = p - \text{onehot}(y)$ | $p$（或只存 logsumexp） |
| RMSNorm $y = \hat x \odot \gamma$，$\hat x = x / r$，$r = \sqrt{\text{mean}(x^2) + \epsilon}$ | 令 $u = g \odot \gamma$：$\frac{\partial L}{\partial x} = \frac{1}{r}\big(u - \hat x \cdot \text{mean}(u \odot \hat x)\big)$；$\frac{\partial L}{\partial \gamma} = \sum_{\text{token}} g \odot \hat x$ | $x$、$r$ |

三个常考的：

1. **矩阵乘**：形状对上就不会写错。$X$ 是 $[T, k]$、$W$ 是 $[k, m]$、$g$ 是 $[T, m]$，能凑出 $[T, k]$ 的只有 $g\,W^\top$，能凑出 $[k, m]$ 的只有 $X^\top g$。这就是 [训练算力账](/train/compute-mfu) 里 backward 是 forward 两倍 FLOPs 的来由。
2. **softmax + cross-entropy**：$p = [0.7, 0.2, 0.1]$、标签是第 0 类时，$\frac{\partial L}{\partial z} = [-0.3, 0.2, 0.1]$。合在一起求导比分开简单，fused cross-entropy kernel 就是这么算的。
3. **softmax**：attention 的 backward 里 $\frac{\partial L}{\partial S} = P \odot \big(\frac{\partial L}{\partial P} - \text{rowsum}(\frac{\partial L}{\partial P} \odot P)\big)$，FlashAttention-2 先算的 $D = \text{rowsum}(dO \odot O)$ 就是括号里那一项（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691) Algorithm 2）。

## 例：SwiGLU

$h = \text{silu}(a) \odot b$，$a = xW_{\text{gate}}$，$b = xW_{\text{up}}$，$\text{silu}(a) = a\,\sigma(a)$。

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial h} \odot \text{silu}(a), \qquad
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial h} \odot b \odot \text{silu}'(a)
$$

$$
\text{silu}'(a) = \sigma(a) + a\,\sigma(a)(1-\sigma(a)) = \sigma(a)\big(1 + a(1-\sigma(a))\big)
$$

验算：$a = 2$、$b = 3$、$\frac{\partial L}{\partial h} = 5$ 时，$\frac{\partial L}{\partial b} = 5 \times 1.76 = 8.8$；把 $b$ 改成 $3.01$，$h$ 增 $0.0176$，$L$ 增 $0.088$，比值也是 $8.8$。

对应的代码（变量名就是上表的 `dX`）：

```python
def swiglu_backward(a, b, dh):
    sig = torch.sigmoid(a)
    db = dh * a * sig                          # ∂L/∂b
    da = dh * b * sig * (1 + a * (1 - sig))    # ∂L/∂a
    return da, db
```

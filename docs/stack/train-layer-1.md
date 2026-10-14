---
title: "训练栈第 1 层：kernel"
---

# 训练栈第 1 层：kernel

训练的 kernel 比推理多一半：每个 forward 算子都有对应的 backward，以最简单的SwiGLU为例：

FFN 的中间一步是 $h = \text{silu}(a) \odot b$，其中 $a = xW_{\text{gate}}$，$b = xW_{\text{up}}$，$\text{silu}(a) = a \cdot \sigma(a)$。

backward 拿到上游传来的 $\frac{\partial L}{\partial h}$，要算出 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$（记号、链式法则和常见函数的导数见 [梯度记号](/train/notation)）：

<svg viewBox="0 0 720 215" width="100%" style="max-width:720px;display:block;margin:1em auto;color:var(--vp-c-text-1);font-size:15px" role="img" aria-label="SwiGLU 的 forward 和 backward 数据流">
  <defs>
    <marker id="swiglu-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="70" y="40" width="160" height="100" rx="6"/>
    <rect x="470" y="40" width="160" height="100" rx="6"/>
    <line x1="20" y1="70" x2="68" y2="70" marker-end="url(#swiglu-arrow)"/>
    <line x1="20" y1="110" x2="68" y2="110" marker-end="url(#swiglu-arrow)"/>
    <line x1="230" y1="90" x2="278" y2="90" marker-end="url(#swiglu-arrow)"/>
    <line x1="715" y1="90" x2="632" y2="90" marker-end="url(#swiglu-arrow)"/>
    <line x1="470" y1="70" x2="412" y2="70" marker-end="url(#swiglu-arrow)"/>
    <line x1="470" y1="110" x2="412" y2="110" marker-end="url(#swiglu-arrow)"/>
    <path d="M150,140 C150,200 550,200 550,142" stroke-dasharray="5 4" stroke-width="1.2" marker-end="url(#swiglu-arrow)"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="150" y="28">①</text>
    <text x="150" y="85">SwiGLU</text>
    <text x="150" y="110" opacity="0.65">fwd</text>
    <text x="550" y="85">SwiGLU</text>
    <text x="550" y="110" opacity="0.65">bwd</text>
    <text x="36" y="62">a</text>
    <text x="36" y="102">b</text>
    <text x="294" y="95" text-anchor="start">h</text>
    <text x="675" y="80">∂L/∂h</text>
    <text x="405" y="75" text-anchor="end">② ∂L/∂a</text>
    <text x="405" y="115" text-anchor="end">③ ∂L/∂b</text>
    <text x="350" y="210" opacity="0.65" font-size="13">存 a, b 给 backward</text>
  </g>
</svg>

*① 是 forward；② ③ 分别是下式的 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$。*

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial h} \odot \text{silu}(a), \qquad \frac{\partial L}{\partial a} = \frac{\partial L}{\partial h} \odot b \odot \sigma(a)\,\big(1 + a\,(1 - \sigma(a))\big)
$$

$\frac{\partial L}{\partial a}$ 里最后那一串是 silu 的导数：$\frac{d}{da}\big[a\,\sigma(a)\big] = \sigma(a) + a\,\sigma(a)(1-\sigma(a))$。

这样我们有如下 kernel：
1. 一个 Linear 层是三个 GEMM（forward、对输入求梯度、对权重求梯度），是训练 FLOPs 的主体。
2. FlashAttention 的 backward 不存 $s \times s$ 的注意力矩阵，只存每行的 logsumexp，按块重算 softmax。
3. 其余逐元素（RoPE、SwiGLU）或按行归约（RMSNorm、cross-entropy）的算子是 memory-bound，靠融合省掉访存；上面的 SwiGLU backward 就属于这一类。cross-entropy 的 logits 是最大的单个激活，fused linear cross-entropy 分块算，不物化完整 logits。

时间主要花在第 1、2 类上。

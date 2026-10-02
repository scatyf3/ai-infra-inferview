---
title: Fused Ops
---

# Fused Ops

LayerNorm、激活函数、残差加、RoPE 这类逐元素或按行的算子 FLOPs 很少，时间几乎全花在读写 HBM 和启动 kernel 上。把几个串联的算子融合进一个 kernel，中间结果不落显存，省下的就是这部分开销。

- **典型组合**：residual add + RMSNorm、SiLU × gate（SwiGLU）、QKV 投影后接 RoPE。
- **epilogue 融合**：把 bias、激活、量化直接放进 GEMM 的收尾阶段。
- **谁来融合**：手写 Triton / CUDA，或者交给 torch.compile 的 Inductor 自动生成。

**延伸阅读**：[Triton tutorial: Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

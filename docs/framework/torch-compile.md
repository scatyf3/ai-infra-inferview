---
title: torch.compile：Dynamo / AOTAutograd / Inductor
status: todo
tags: [torch-compile, dynamo, inductor]
difficulty: 4
order: 2
related: []
stack: [f-graph]
---

# torch.compile：Dynamo / AOTAutograd / Inductor

> dynamo 抓图、graph break、inductor 生成什么

## 一句话结论

`torch.compile` 分三段：Dynamo 在字节码层面抓 Python 函数的计算图（遇到不能抓的就 graph break 切开），AOTAutograd 把前向图和反向图一起 trace 成 ATen 算子图，Inductor 把算子图做融合后生成 Triton（GPU）或 C++（CPU）kernel。推理框架用它主要拿融合的小算子和去掉 Python 开销。

## 推导

- **Dynamo 抓图**：挂在 CPython 的 frame evaluation 钩子上，符号执行字节码，把 tensor 操作记进 FX graph，把依赖的 Python 对象状态记成 guard；下次调用先查 guard，命中就直接跑编译好的图。
- **graph break**：遇到数据依赖的控制流、`print`、不支持的 C 扩展调用时切断图，前后各编一段；break 太多就退化成 eager，用 `TORCH_LOGS=graph_breaks` 查。
- **Inductor 生成什么**：对 pointwise / reduction 算子做 fusion，生成 Triton kernel 并 autotune；matmul 默认调 cuBLAS，可选 `max-autotune` 用 Triton matmul 比一比。
- **动态 shape**：`dynamic=True` 让 Dynamo 用符号维度，避免每个 shape 重新编译；推理里 batch 维常标为动态，hidden 维固定。

## 面试追问

::: details Q：vLLM 里 torch.compile 和 CUDA Graph 分别解决什么？能同时用吗？
compile 负责减少 kernel 数量（融合 RMSNorm、残差、激活等），CUDA Graph 负责消除剩下那些 kernel 的 launch 开销。两者正交：先 compile 出融合后的 kernel，再把这组 kernel capture 成 graph；vLLM V1 默认两者都开，见 [CUDA Graph 与 fusion](/gpu/cuda-graph-fusion)。
:::

## 手撕

常见题：给一段含 `if x.sum() > 0` 的代码，指出 graph break 的位置和改法。

## 参考

- [PyTorch 2 论文：TorchDynamo and TorchInductor](https://pytorch.org/assets/pytorch2-2.pdf)
- [torch.compile 文档](https://pytorch.org/docs/stable/torch.compiler.html)

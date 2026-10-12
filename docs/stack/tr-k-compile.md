---
title: "torch.compile"
---

# torch.compile

torch.compile 用 Dynamo 抓图、AOTAutograd 同时生成 forward 和 backward 图、Inductor 把逐元素算子融合成 Triton kernel。训练里常按 Transformer block 编译，收益主要来自融合小算子。

- **注意**：动态形状和 graph break 会让编译失效或反复重编。

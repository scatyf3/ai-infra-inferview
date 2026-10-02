---
title: "第 3 层：kernel → forward"
---

# 第 3 层：kernel → forward

把一个个 kernel 串成一次完整的前向。这一层定义模型结构，把调度器给出的一批请求整理成张量输入，并且让整条 kernel 序列跑得没有空隙：消除 CPU 侧的启动开销，减少中间结果写回显存。

- **模型定义**：结构正确，并且接入 KV cache、TP 和量化。
- **输入准备**：把变长请求拼平，生成 position 和 slot mapping。
- **执行效率**：CUDA graph 去掉启动开销，编译器做跨算子融合。

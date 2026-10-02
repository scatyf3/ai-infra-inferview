---
title: 手撕高频
---

# 手撕高频

比 LeetCode 更常考。每一题都要能白板写出、讲清 shape、指出数值稳定性和性能考量。

::: tip 面试官真正在考什么
面试官真正在考的是：你是否真的写过这些东西，能否边写边解释每一行为什么这样写。
:::

## 进度

左边方块是**熟练度**（各题 frontmatter 的 `familiarity`，0 最熟 → 4 不会，不写 = 未评，阶梯和隔壁 leetcode 看板同一套），小标签直达 LeetGPU 对应的题（`leetgpu: [题号]`）。练完一遍改一下 `familiarity` 就会自动更新。

<HandsonProgress />

## 本领域主题

- [手写 MHA / GQA Forward](./mha-gqa-forward) — shape 与 mask 处理
- [带 KV Cache 的 Decode Step](./decode-step-kv-cache) — 增量计算与 cache 更新
- [数值稳定 Softmax](./stable-softmax) — max-subtraction、online softmax
- [RMSNorm](./rmsnorm) — 与 LayerNorm 的区别、fused 实现
- [Top-p 采样](./top-p-sampling) — 与 top-k / temperature 组合
- [Simple Beam Search](./beam-search) — 长度惩罚与 early stopping
- [Triton 版 Softmax](./triton-softmax) — 一行一个 program 的写法
- [Triton 版 Fused LayerNorm](./triton-fused-layernorm) — forward 与 backward
- [CUDA Reduce](./cuda-reduce) — warp shuffle、shared memory 分层归约
- [CUDA Tiled Matmul](./cuda-tiled-matmul) — shared memory tiling、bank conflict

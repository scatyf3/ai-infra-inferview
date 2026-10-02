---
title: 中文标题（术语保留英文）
status: todo            # todo | draft | reviewed
tags: []
difficulty: 3           # 1–5
order: 99
related: []             # 绝对路径，不含 base，如 /inference/memory-accounting
stack: []               # 分层图位置：小主题 id（如 kv-paged）或整层号（0–8），见 docs/.vitepress/layers.ts；空 = 不在图上
# 手撕题才需要下面两项：
# familiarity: 3        # 0 | 1 | 1.5 | 2 | 3 | 3.5 | 4（0 最熟），不写 = 未评
# leetgpu: [5]          # LeetGPU 题号，见 src/data/leetgpu-challenges.json
---

# {{ $frontmatter.title }}

## 一句话结论

30 秒内能说完的答案。

## 推导

shape → 访存量 (bytes) 和计算量 (FLOPs) → arithmetic intensity → roofline 落在哪一侧 → 所以优化手段只能是 X。

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

## 面试追问

::: details Q1：追问示例
答案。
:::

## 手撕

::: code-group

```python [python]
# reference implementation
```

:::

## 参考

- 论文 / 博客 / 源码链接

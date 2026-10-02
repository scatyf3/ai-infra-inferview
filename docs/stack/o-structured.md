---
title: Structured Output
---

# Structured Output

让模型的输出严格符合 JSON schema、正则表达式或某种语法。做法是每一步根据当前状态算出哪些 token 合法，把其余 token 的 logits 屏蔽掉。难点在速度：词表有十几万个 token，每一步都要判断一遍。

- **有限状态机**：正则和简单 schema 可以编译成 FSM，预先算好每个状态下的合法 token 集合（Outlines 的做法）。
- **上下文无关文法**：嵌套 JSON 需要下推自动机。XGrammar 把大部分 token 的判断提前算好，运行时只检查少数依赖上下文的 token。
- **和推理重叠**：mask 在 CPU 上计算，和 GPU 的 forward 并行进行。

**延伸阅读**：[Outlines](https://arxiv.org/abs/2307.09702) · [XGrammar](https://arxiv.org/abs/2411.15100)

---
title: Continuous Batching
---

# Continuous Batching

静态 batching 要等一整批请求都生成完才接新请求，短请求只能陪着长请求空转。continuous batching（iteration-level scheduling）每一步都重新组 batch：结束的请求立刻退出，新请求立刻加入，GPU 一直保持满载。

- **前提**：请求长度各不相同，attention 要能处理拼平的变长序列，不再 padding 成矩形。
- **收益**：吞吐通常是静态 batching 的数倍，已经是现代推理引擎的默认做法。
- **约束**：每一步能加入多少新请求，受 KV cache 剩余空间和 token 预算限制。

**延伸阅读**：[Orca (OSDI '22)](https://www.usenix.org/conference/osdi22/presentation/yu)

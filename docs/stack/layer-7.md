---
title: "第 7 层：输出"
---

# 第 7 层：输出

从最后一层的 hidden state 到返回给用户的文本。计算量不大，但细节很多：logits 要按每个请求的参数改写，采样要保证分布正确，结构化输出要在每一步限定合法 token，文本要能流式地正确拼出来。这一层的开销大多在 CPU 上，处理不好会拖慢整个引擎循环。

- **logits → token**：logits processor、采样、约束解码。
- **token → 文本**：增量 detokenize、停止条件。

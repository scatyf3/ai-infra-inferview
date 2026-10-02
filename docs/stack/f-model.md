---
title: 模型定义
---

# 模型定义

把论文里的结构翻译成一组模块：embedding → N 层 decoder（attention 和 MLP，各自带 norm 和残差）→ final norm → LM head。推理框架里的模型定义和训练版不同：attention 要接入 KV cache 和 paged 布局，线性层要换成支持 TP 切分和量化的版本。

- **attention 变体**：MHA / MQA / GQA / MLA 主要差在 KV head 的数量，直接决定 KV cache 有多大。
- **MoE**：MLP 换成 router 加多个 expert，每个 token 只激活 top-k 个。
- **位置编码**：RoPE 作用在 Q、K 上；长上下文扩展靠调整它的频率。

**延伸阅读**：[GQA](https://arxiv.org/abs/2305.13245)

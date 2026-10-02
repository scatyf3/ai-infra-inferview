---
title: Tokenize 与 Chat Template
---

# Tokenize 与 Chat Template

请求进来后的第一步。chat template 把 messages 列表按模型训练时的格式拼成一段文本（角色标记、特殊 token、工具调用格式），再由 tokenizer 切成 token id。模板用错不会报错，只会让模型效果悄悄变差。

- **模板**：以 Jinja 模板的形式存在 tokenizer_config.json 里，每个模型都不一样。
- **开销**：长 prompt 的 tokenize 可能要几毫秒到几十毫秒，要放在引擎主循环之外并行处理。
- **多模态**：图片、音频在这一步预处理成 patch 或特征，并在 token 序列里插入占位符。

**延伸阅读**：[HF: Chat templates](https://huggingface.co/docs/transformers/chat_templating)

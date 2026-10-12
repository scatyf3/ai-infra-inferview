---
title: "token 存储格式"
---

# token 存储格式

语料离线 tokenize 成扁平的二进制 token 流（Megatron 的 .bin）加每篇文档的偏移索引（.idx），训练时 mmap 按样本号随机读。词表超过 65535 时 token id 要用 4 字节存。

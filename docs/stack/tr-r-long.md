---
title: "长上下文"
---

# 长上下文

序列变长时 attention FLOPs 和激活都涨，所以预训练主要用短序列，最后分几个阶段扩到 128K。显存靠 CP 切序列，位置编码靠调大 RoPE base 或 PI / YaRN 改频率。

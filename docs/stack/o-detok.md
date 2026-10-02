---
title: Detokenize
---

# Detokenize

把生成的 token id 变回文本。流式输出时不能对每个 token 单独解码：一个汉字或 emoji 可能被拆成好几个字节级 token，单独解码会得到乱码；SentencePiece 类分词器对前导空格的处理也依赖上下文。

- **增量解码**：记住已经输出的文本和一小段前文 token，每次解码一个窗口再取差值；遇到不完整的 UTF-8 序列就先不输出。
- **停止条件**：stop string 要在文本层面匹配，可能跨越好几个 token，命中后要把它截掉。
- **放在哪里**：detokenize 是 CPU 工作，通常放在单独的进程里，避免拖慢引擎主循环。

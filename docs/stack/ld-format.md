---
title: 权重格式
---

# 权重格式

权重文件本质上是一个「张量名 → dtype、shape、字节」的字典。主流格式是 safetensors：文件头是一段 JSON，记录每个张量的偏移，正文是裸字节，可以 mmap 零拷贝读取，也没有 pickle 那种任意代码执行的风险。GGUF 是 llama.cpp 生态的单文件格式，把量化参数和 tokenizer 一起打包。

- **量化 checkpoint**：GPTQ / AWQ 存的是打包后的 int4 加上 scale / zero-point，加载时要按对应方案解包。
- **分片**：大模型拆成多个 shard，用 index.json 记录每个张量在哪个文件里。
- **名字映射**：HF 的参数名和推理框架内部的模块名往往对不上，加载时要重命名和合并，比如把 q / k / v 拼成 qkv_proj。

**延伸阅读**：[safetensors](https://github.com/huggingface/safetensors)

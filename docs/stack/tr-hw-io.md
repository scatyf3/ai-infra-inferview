---
title: "存储与 IO"
---

# 存储与 IO

训练读的数据很少，写的 checkpoint 很大：每参数 12 字节（fp32 主权重和 Adam 状态），405B 约 4.9 TB。写盘时训练是否要停，决定了容错的代价。

- **checkpoint**：所有 rank 并行写各自的分片，或者先拷到 CPU 内存再异步写。
- **数据**：mmap 读预 tokenize 的二进制文件，靠 page cache 共享。

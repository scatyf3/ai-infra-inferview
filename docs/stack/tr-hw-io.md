---
title: "存储与 IO"
---

# 存储与 IO

主要是写weight checkpoint是bound，而非dataloading

1. weight的数量级：每参数 12 字节（fp32 主权重和 Adam 状态），405B 约 4.9 TB。
2. data的数量级：每卡 $0.41 \times 989\text{T} / (6 \times 405\text{B}) \approx 167$ token/s，× 16384 卡 × 4 B/token ≈ **11 MB/s**，一块 SSD 就够；整个训练集 15.6T token ≈ 62 TB。多模态读原始图片视频时另说


怎么做
- **checkpoint**：所有 rank 并行写各自的分片，或者先拷到 CPU 内存再异步写。405B 切到 16384 卡每卡约 300 MB；PyTorch 用 `torch.distributed.checkpoint.async_save`。
- **数据**：mmap 读预 tokenize 的二进制文件，靠 page cache 共享。

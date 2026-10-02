---
title: 为 kernel 重排权重
---

# 为 kernel 重排权重

磁盘上的布局追求通用和可移植，kernel 想要的布局追求访存合并和适配 Tensor Core 指令。所以加载完常有一步 repack：把 int4 权重按 kernel 的 tile 顺序重新排列、把 scale 交织进去，或者转置成 kernel 需要的主序。这一步只做一次，之后每次 forward 都受益。

- **Marlin**：一个 W4A16 kernel，把 int4 权重预先重排成每个 warp 拿来就能用的顺序。
- **融合投影**：把 q / k / v、gate / up 的权重拼成一个大矩阵，一次 GEMM 代替好几次。
- **代价**：重排后的格式和 kernel 绑死，换 kernel 就要重新排。

**延伸阅读**：[Marlin](https://github.com/IST-DASLab/marlin)

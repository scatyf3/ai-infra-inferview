---
title: CUDA Graph / torch.compile
---

# CUDA Graph / torch.compile

decode 一步可能要发射上千个小 kernel，每个 kernel 的启动开销是微秒级，累加起来能和 GPU 实际计算时间相当。CUDA graph 把整段 kernel 序列录制下来一次性重放，消掉 CPU 侧的启动开销；torch.compile 则在图层面做算子融合，生成 Triton kernel。

- **CUDA graph 的约束**：shape 和内存地址必须固定，所以按几档 batch size 分别捕获，每次把输入拷进固定的 buffer。
- **torch.compile**：Dynamo 负责抓图，Inductor 负责生成 kernel；graph break 会让收益打折扣。
- **组合使用**：常见做法是用 compile 做融合，再用 CUDA graph 包住整条 decode 路径。

**延伸阅读**：[torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

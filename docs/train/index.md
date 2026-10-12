---
title: 训练系统
---

# 训练系统

从「一次训练要多少卡、多少天」算起，到并行怎么配、精度怎么降、loss 炸了怎么救、机器坏了怎么续。SFT 和 RL 的算法与架构在 [Post-train](/posttrain/)，这里放它们的提速。

::: tip 面试官真正在考什么
能不能把一次训练的账算清楚：总 FLOPs、MFU、显存、通信、故障，各自卡在哪；以及看到一条坏曲线时，先查哪里。
:::

## 先算账

- [训练算力账：6ND、MFU 与训练时长](./compute-mfu) — 每 token 6N 的来由、MFU vs HFU、Chinchilla
- [训练显存账](/posttrain/training-memory) — 每参数 16 字节、激活公式、recompute
- [一个训练 step：AdamW、学习率调度与梯度裁剪](./optimizer-lr) — 一步里按顺序发生了什么

## 并行与精度

- [ZeRO / FSDP](/parallel/zero-fsdp) — 去掉数据并行的冗余
- [Pipeline 并行调度](./pipeline-schedules) — GPipe、1F1B、Interleaved、Zero Bubble、DualPipe
- [FSDP2 与 DTensor](./fsdp2-dtensor) — DeviceMesh、placement 与集合通信、fully_shard 的 hook 和预取
- [并行配置怎么选](./parallel-config) — 从显存约束到 MFU 调优
- [FP8 训练](./fp8-training) — 哪些 GEMM 用 FP8、scale 怎么定、DeepSeek-V3 的细粒度量化
- [长上下文训练](./long-context) — Context Parallel、RoPE 外推、分阶段扩长
- [MoE 训练](./moe-training) — 算力按激活、显存按总量；负载均衡、router 稳定性、为什么 MFU 更低

## 跑得稳、跑得久

- [训练稳定性](./stability) — loss spike 从哪来、怎么防、怎么救
- [大规模训练的容错](./fault-tolerance) — 故障率、checkpoint 间隔、快速恢复
- [分布式训练排障](./debugging) — hang、OOM、NaN、慢，各先查什么
- [预训练数据管线](./data-pipeline) — 离线 tokenize、混合采样、可确定性恢复

## SFT 与 RL 提速

- [SFT 训练提速](./efficient-sft) — padding、logits 显存、融合 kernel，按什么顺序优化
- [RL 训练提速](./efficient-rl) — 一步拆成几段，每段怎么压

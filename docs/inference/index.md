---
title: 推理系统核心
---

# 推理系统核心

这一部分按[首页](/)的推理栈分层组织：先学几本跨层通用的账（算力、带宽、显存、延迟怎么算），再从下往上看每一层的核心机制。在首页的推理栈图上点某个格子，也能看到挂在那一层的所有文章。

几本账是后面所有章节的工具：判断一个优化为什么有效，就是看它改变了 FLOPs、访存字节、显存占用中的哪一项，以及这一项是不是瓶颈。所以建议先读第一组。

::: tip 面试官真正在考什么
能否用一套一致的账（显存、带宽、FLOPs）解释每个优化为什么有效、在什么条件下失效，而不是背名词。
:::

## 先算账：roofline / 显存 / 指标

跨层通用，后面每一章都会用到。

- [Prefill vs Decode 与 Roofline](./prefill-decode-roofline)：给定 shape，算出 FLOPs 和访存字节，判断是 compute-bound 还是 memory-bound
- [显存账：权重 / KV cache / 激活](./memory-accounting)：口算「70B bf16 + 8k context + batch 32 要几张卡」
- [指标与 Benchmark：TTFT / TPOT / ITL / Goodput](./metrics-benchmark)：每个指标由哪段时间组成，怎么压测才可信

## L1–2 权重与 kernel

权重以什么格式进显存，算子怎么把带宽和 Tensor Core 用满。硬件本身（L0）见 [GPU 架构](/gpu/gpu-architecture) 和 [显存层次](/stack/hw-mem)；GEMM、Triton 等通用算子见 [GPU / 算子](/gpu/)。

- [量化：GPTQ / AWQ / SmoothQuant / FP8](./quantization)：W4A16 和 W8A8 各省的是什么，per-group 和 per-tensor 怎么选
- [GPTQ 推导](./quantization-gptq)：逐列量化、用二阶信息补偿误差
- [FlashAttention v1 / v2 / v3](./flash-attention)：tiling + online softmax 省的是 HBM 访问，不是 FLOPs

## L3 模型前向

模型结构上的选择如何改变每一步的计算量和访存量。把 kernel 串成一次前向的工程（model runner、CUDA Graph、torch.compile）见 [框架内功](/framework/)。

- [Attention 变体：MHA / MQA / GQA / MLA](./attention-variants)：KV head 数怎么决定 KV 大小和 decode 带宽
- [Speculative Decoding](./speculative-decoding)：用闲置算力一次验证多个 token，EAGLE / Medusa / MTP 的区别
- [投机采样的数学](./speculative-sampling-math)：为什么输出分布和 target 单独采样完全一样

## L4 KV cache

推理时唯一随并发和上下文长度线性增长的状态。

- [KV Cache 与 PagedAttention](./kv-cache-paged-attention)：分块管理、prefix caching / RadixAttention、抢占、KV 量化

## L5–6 调度与分离

每一步让哪些请求上 GPU，prefill 和 decode 怎么互不干扰。多卡并行本身（TP / PP / EP）见 [并行与通信](/parallel/)，服务层（L7–8）见 [Serving 层](/framework/serving-layer)。

- [Continuous Batching、Chunked Prefill 与 PD 分离](./batching-scheduling)：token 预算怎么分、抢占时踢谁、什么时候值得把 prefill 和 decode 拆到不同的卡上

## 系统案例

把上面各层组合起来的完整系统。

- [Omni Serving：多模态输入输出的推理系统](./omni-serving)：Thinker-Talker 流水线、EPD 分离、首包延迟与 RTF
- [多模态输入：一张图、一段视频、一段音频变成多少 token](./multimodal-encoder)：动态分辨率、M-RoPE、ViT 算力账、编码器的调度和缓存
- [语音输出：codec token、Talker 与首包延迟](./speech-output)：RVQ 与码率、多码本怎么少走几步、用 Qwen3-Omni 的数字拆首包和 RTF

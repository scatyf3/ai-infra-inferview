---
title: vLLM 版本演进
status: draft
tags: [vllm, release-history, architecture, scheduler]
difficulty: 3
order: 7
related: [/framework/vllm-v1-architecture, /framework/request-lifecycle, /inference/kv-cache-paged-attention, /inference/batching-scheduling]
stack: []
---

# vLLM 版本演进

按照observation => 如何解决这个问题的mindset看下演进历史，感觉会很有趣

## 演进主线

每个版本固定讲四件事：**Observation**（观察到什么瓶颈）→ **场景**（优化谁）→ **做法**（怎么做）→ 关联页面。引号里是 release note 或官方博客原文。所有版本的一行摘要见页末速查表。

### 2023 · 显存与 batching

**v0.1（2023-06-20）PagedAttention**

- **Observation**：博客原文，现有系统 =="waste 60% – 80% of memory due to fragmentation and over-reservation"=={面试常问：浪费来自三部分——按 max_len 预留（reservation）、内部碎片（internal）、外部碎片（external）。PagedAttention 只剩最后一个 block 的内部碎片。}。KV cache 按 `max_len` 连续预分配，每个请求都占最坏情况的显存，batch 做不大。
- **场景**：单卡高并发 serving，尤其 parallel sampling 和 beam search 这类多序列共享 prompt 的场景。
- **做法**：KV 切成固定大小 block，block table 把逻辑连续映射到物理离散，浪费降到 "under 4%"；相同前缀的 block 用 refcount 共享，写时 copy-on-write。注意**continuous batching 不是 0.1.0 就有的**：0.1.3 的 release note 才写 "vLLM now uses a TGI-style continuous batching"，之前是整批调度。
- 关联：[KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)、[Continuous Batching](/inference/batching-scheduling)

**v0.2（2023-09-28）量化与 CUDA graph**

- **Observation**：权重是显存大头，7B fp16 就要 14 GiB；decode 每步几百次 kernel launch，小 batch 下 CPU launch 开销和 GPU 计算同量级。
- **场景**：显存受限的单卡部署，以及小 batch 低延迟 decode。
- **做法**：AWQ (0.2.0) 和 GPTQ W4A16 (0.2.6) 把权重压到 1/4；0.2.6 "Fast model execution with CUDA/HIP graph" 把 decode 整图捕获重放；0.2.1 PagedAttention V2 kernel 端到端延迟降 20%；0.2.5 Mixtral 用 expert parallelism。
- 关联：[量化](/inference/quantization)、[CUDA Graph 与算子融合](/gpu/cuda-graph-fusion)

### 2024 上半年 · 功能扩张

这半年每个版本都在加能力，但每个能力都是独立的 flag、独立的代码路径（prefix caching 和 chunked prefill 到 0.6.0 才能同时开）。这是 V1 重写的直接原因。

**v0.3（2024-01-31）multi-LoRA 与 prefix caching**

- **Observation**：多租户场景每个微调模型各占一份完整权重；长 system prompt 每个请求都重新 prefill 一遍。
- **场景**：多 LoRA adapter 共享一个 base model；共享长前缀的对话和 RAG。
- **做法**："Experimental multi-lora support"（base 权重一份，adapter 按请求动态切换）；"Experimental prefix caching support"（手动声明前缀复用 block）；FP8 KV cache 把每 token 的 KV 减半。spec decode 的第一块代码（rejection sampler）也在这版合入。
- 关联：[KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)

**v0.4（2024-03-30）automatic prefix caching、chunked prefill、spec decode**

- **Observation**：0.4.2 原文，chunked prefill "improves inter-token latency in high load scenario"。高负载下一个长 prompt 的 prefill 要几百毫秒，期间所有 decode 请求都在等，ITL 出现尖刺。
- **场景**：长短请求混合、TTFT 和 ITL 都有 SLA 的在线服务。
- **做法**：0.4.0 automatic prefix caching 用 hash 自动识别可复用 block（`--enable-prefix-caching`，注意它不在 0.3.3 而在 0.4.0）；0.4.2 chunked prefill 把 prompt 切块与 decode 混批并 "prioritizes decode"；spec decode 加 ngram proposer 和 logprobs；FlashInfer 成为可选 attention backend；0.4.3 MultiprocessingGPUExecutor 让单机 TP 不再依赖 Ray。
- 关联：[Chunked Prefill](/inference/batching-scheduling)、[Speculative Decoding](/inference/speculative-decoding)

**v0.5（2024-06-11）FP8、PP，以及 CPU 优化的开端**

- **Observation**：单机 8 卡 TP 放不下更大的模型，TP 跨机通信太贵；另一端，API server 和引擎跑在同一个进程里争 GIL，小模型高 QPS 时互相拖累。
- **场景**：70B+ 多机部署；单卡小模型高 QPS。
- **做法**：0.5.0 FP8 权重 "1.5x boost"；0.5.1 pipeline parallelism；**0.5.4 用 zeromq 把 HTTP 处理和推理循环分成两个进程**，"20% speedup over time to first token and 2x speedup over inter token latency"；**0.5.5 multi-step scheduling**（`--num-scheduler-steps 8`）一次调度跑多步，8B/30B QPS +20%。这两项常被归到 v0.6.0，实际先在 0.5.x 落地。
- 关联：[并行总览](/parallel/parallelism-overview)、[一个 Request 的全链路](/framework/request-lifecycle)

### 2024 下半年 · CPU overhead

**v0.6（2024-09-04）2.7x 吞吐、5x TPOT**

- **Observation**：博客原文，Llama 3 8B 在单张 H100 上，"The HTTP API server takes 33% of the total execution time. 29% of the total execution time is spent on scheduling... only 38% of the time was spent on the actual GPU execution"。博客还点明了背景：vLLM 过去针对 "relatively large models on GPUs with limited memory" 优化，H100 普及后 GPU 太快，Python 成了瓶颈。
- **场景**：快 GPU + 小模型。GPU 每步不到 10 ms，Python 调度、detokenize、HTTP 与之同量级。8B 是 2.7x，70B 只有 1.8x：模型越大 CPU 占比越低。
- **做法**：三件事打包。API server 与引擎进程分离（ZMQ，消除 GIL 争抢）；multi-step scheduling 一次调度 N 步摊薄 CPU 开销；async output processor 把输出结构构造与下一步 GPU 执行 overlap（+12%）。另有 object caching、non-blocking H2D、简单采样 fast path。0.6.2 MQLLMEngine 再 +30%；0.6.3 block manager V2 和 multi-step 默认开；0.6.4 release note 出现 "Significant progress in V1 engine core refactor"。
- 关联：[一个 Request 的全链路](/framework/request-lifecycle)、[LLM Serving 系统设计](/basics/system-design-llm-serving)

### 2025 上半年 · V1 重写与 DeepSeek

**v0.7（2025-01-27）V1 alpha**

- **Observation**：V1 博客原文，"Features were often developed independently, making it difficult to combine them effectively and cleanly. Over time, technical debt accumulated." 具体是 prefill/decode 双路径、各 feature 各自 flag、prefix caching 因 CPU 开销默认关、CPU 开销日益突出。
- **场景**：所有文本和多模态模型。文本模型 vs V0 "up to 1.7x higher throughput"（不开 multi-step 的情况下），VLM 提升更大。
- **做法**：
  1. 独立 `EngineCore` 进程跑执行循环，前端的 tokenize / detokenize / HTTP 与之并行；
  2. 调度器不再区分 prefill/decode，用 `{request_id: num_tokens}` 的 token budget 统一表达 chunked prefill、prefix caching、spec decode；
  3. 零开销 prefix caching：常数时间 evict、减少 Python 对象创建，0% 命中率时损耗 "less than 1%"，因此可以默认开；
  4. TP worker 缓存请求状态，每步只传 diff；persistent batch 避免重建输入；
  5. torch.compile 全面集成 + piecewise CUDA graph（attention 之外的部分捕获成图）；
  6. 多模态预处理独立进程 + encoder cache。

  同期 0.7.1 MLA kernel 让 DeepSeek 生成吞吐 ~3x、token 容量 ~10x；0.7.2 KV entry 对齐 256 B 再 +43%；0.7.3 MTP 低 QPS 下 1.69x。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)、[Attention 变体：MLA](/inference/attention-variants)

**v0.8（2025-03-18）V1 成为默认**

- **Observation**：V1 补齐 structured outputs、LoRA、PP、ngram spec decode 等 V0 特性后可以翻转默认；DeepSeek R1 的 MoE + MLA 把 EP/DP 和专用 attention kernel 变成刚需。
- **场景**：DeepSeek R1 级 MoE 的多卡部署；需要 KV 跨实例传输的 PD 分离和多轮缓存。
- **做法**："enabled V1 engine by default"（`VLLM_USE_V1=0` 回退）；FlashMLA 集成，MLA 进入 V1 并支持 chunked prefill；EP/TP MoE + DP attention；0.8.3 V1 原生 sliding window + hybrid memory allocator；0.8.5 KV Connector API V1、LMCache connector、EAGLE-3。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)、[Speculative Decoding](/inference/speculative-decoding)

**v0.9（2025-05-15）大规模推理基础设施**

- **Observation**：大规模 MoE 单机放不下也跑不满，需要 expert 跨机分布、attention 多副本、prefill/decode 分池这一整套基础设施。
- **场景**：多机 DeepSeek 部署；PD 分离。
- **做法**："Initial DP, EP, PD support for large scale inference"：EP 用 PPLX kernel 并为 DeepEP 做准备，DP 解耦引擎进程管理与通信，PD 集成 NIXL 并支持多个 KV connector；V1 full CUDA graph；0.9.2 EPLB、原生 xPyD P2P NCCL 传输。**0.9.2 声明 "the last version where V0 engine code and features stay intact"**。
- 关联：[并行总览](/parallel/parallelism-overview)、[PD 分离](/inference/batching-scheduling)

### 2025 下半年 · V0 移除、async scheduling、DBO

**v0.10（2025-07-24）async scheduling 实验版**

- **Observation**：V1 里 scheduler（CPU）和 model runner（GPU）仍是串行的，每步 GPU 跑完要等 CPU 调度下一步，调度期间 GPU 空闲。
- **场景**：decode 密集、高 QPS 的中小模型，每步 CPU 开销占比高。
- **做法**：`--async-scheduling` "overlap engine core scheduling with GPU runner"：GPU 执行第 N 步时 CPU 先调度第 N+1 步（假定每请求产 1 token），采样结果异步回填。同时 "begins the cleanup of V0 engine codebase"，删除 V0 CPU/XPU/TPU 后端和 spec decode worker。0.10.2 加 Decode Context Parallel for MLA 和 aarch64/GB200。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)、[一个 Request 的全链路](/framework/request-lifecycle)

**v0.11（2025-10-02）V0 移除完成，DBO**

- **Observation**：DP+EP 的 MoE 部署里 DeepEP all-to-all 通信占比高，计算等通信；fine-grained MoE 的 piecewise graph 仍有大量小 kernel launch；多轮对话 KV 放不进 GPU。
- **场景**：DeepSeek 类 MoE 的大规模 DP+EP 部署；长多轮会话。
- **做法**："V1 is the only engine in the codebase now"；CUDA graph 默认改为 FULL_AND_PIECEWISE，"particularly fine-grained MoEs"；**Dual-Batch Overlap** 把 batch 拆成两个 microbatch，一个做 expert 通信时另一个做计算；CPU KV offloading 带 LRU 管理；DeepGEMM 默认开。0.11.1 "Robust async scheduling"，修复与 chunked prefill、structured outputs、MTP、DeepEP/DCP 组合时的正确性问题（0.10.2 和 0.11.0 在抢占等场景会输出乱码）。
- 关联：[CUDA Graph 与算子融合](/gpu/cuda-graph-fusion)、[KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)

### 2026 · Model Runner V2、sparse MLA、分层 KV

**v0.12（2025-12-03）Model Runner V2 实验版**

- **Observation**：release note 列出的 V1 model runner 痛点：persistent batch 需要 "reordering" 与复杂 bookkeeping；CPU 侧 block table 随 `max_model_len` 和 `num_kv_groups` 扩展性差；sampler 有 "-1 temperature hack"。
- **场景**：长上下文、多 KV group 的 hybrid 模型、DP、结构化输出与 spec decode 混合的负载。
- **做法**："Complete refactoring of model execution pipeline"：移除 persistent batch 的重排逻辑，block table 常驻 GPU，Triton 原生 sampler，简化 DP 和 CUDA graph 路径。默认化走了九个月：0.22 Qwen3 dense → 0.23 Llama/Mistral → 0.24 量化模型 → 0.25 所有 dense → 0.29 所有模型，MRV1 目标 v0.32 删除。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)

**v0.14（2026-01-20）async scheduling 默认、gRPC**

- **Observation**：async scheduling 自 0.10 实验、0.11.1 修复后，收益已不依赖用户手动调参；REST + JSON 在高并发短请求下序列化和连接开销可观。
- **场景**：默认配置下的所有部署；对协议开销敏感的内部服务。
- **做法**："Async scheduling is now enabled by default"（`--no-async-scheduling` 关闭），并兼容 spec decode 与 structured outputs；gRPC entrypoint 提供二进制协议和 HTTP/2 多路复用（0.18 并入 `vllm serve --grpc`）；`--max-model-len auto` 按剩余显存自动定上下文长度。
- 关联：[一个 Request 的全链路](/framework/request-lifecycle)、[LLM Serving 系统设计](/basics/system-design-llm-serving)

**v0.20 → v0.28（2026-04 → 08）DeepSeek V4 sparse MLA**

- **Observation**：DeepSeek V4 的 DSA（sparse attention）+ MTP + MegaMoE 让 attention 不再是「对全部 KV 做 dense 计算」，现有 MLA 路径和 KV 布局不够用；超长上下文下 KV 容量仍是硬约束。
- **场景**：sparse-MLA MoE 的超长上下文 decode。
- **做法**：0.20 初始支持，FlashAttention 4 成为 MLA prefill 默认，TurboQuant 2-bit KV cache 容量 4x；0.22 独立 `deepseek_v4` 包与融合 kernel（MegaMoE、indexer、sparse MLA）、NVFP4 MoE、MTP；0.23 sparse MLA metadata 与 V3.2 解耦、TRTLLM-gen kernel；0.28 sparse MLA 对 decode / MTP / DSpark 端到端打通。0.27 起 Kimi K3 走了同一条路（AttnRes、FlashKDA、DCP）。
- 关联：[Attention 变体](/inference/attention-variants)、[显存账](/inference/memory-accounting)

**v0.22 → v0.30（2026-05 → 09）分层 KV offloading**

- **Observation**：0.11 的 CPU offloading 只有一层，主机内存满了就没有下一级；多轮对话、长 system prompt 和 PD 分离场景的 KV 总量远超单机 GPU + CPU 内存。
- **场景**：多轮 agent 会话、共享长前缀的 RAG、跨实例复用 KV 的 PD 分离集群。
- **做法**：0.22 "A new multi-tier KV cache offloading framework"，"extends offloading beyond CPU memory"，首发文件系统二级 tier 和 Mooncake 磁盘 offloading；0.23 object-store tier、HMA 对 connector 默认开、per-request offloading 策略；0.28 disk offloading、树外 secondary tier manager；0.30 HiSparse 把 sparse-MLA 的 KV 页溢出到 pinned host memory，top-k miss 由每请求 GPU hot buffer 服务。
- 关联：[KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)、[PD 分离](/inference/batching-scheduling)

**v0.25 / v0.29 / v0.30 执行层收尾**

- **v0.25（2026-07-11）**："Model Runner V2 is now the default for all dense models"；"PagedAttention has been removed"，"The legacy attention implementation is deleted now that V1/MRv2 backends are the standard path"。PagedAttention 的思想留在 block table 里，kernel 由 FlashAttention / FlashInfer / Triton 等 backend 实现，自研 kernel 只剩 V0 遗留路径在用，所以删。
- **v0.29（2026-09-09）**："Model Runner V2 is now the default for all models"，MRV1 标记 deprecated，"targeting v0.32 for its removal"；batch-sharded sampling 把每步 logits 内存降到 1/TP；新增 `--max-num-queued-reqs` 准入控制。
- **v0.30（2026-09-22）Fast Start**：Observation 是引擎每次重启都要从磁盘重载权重、再做一遍量化和 TP 切分，RL rollout 和滚动升级会频繁重建引擎。做法是 "a persistent per-GPU weight-cache daemon holds post-quantized, TP-sharded weights in GPU memory so restarting engines map them over CUDA IPC with `--load-format ipc_cache`"。同版 CUDA graph 捕获期间冻结 gc，H200 上引擎初始化从 28.9 s 降到 8.2 s。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)、[给 vLLM / SGLang 加一个新模型](/framework/add-model-vllm-sglang)

## 交互

按类别筛选，看一条线上的瓶颈怎么转移；点任意版本展开 observation / 场景 / 做法和 release note 链接。

<ReleaseTimeline />

## 全版本速查表

30 个 minor 版本的一行摘要。带 ★ 的是里程碑，点任意行展开。

<ReleaseTimeline view="table" :show-eras="false" />

## 面试追问

::: details Q：为什么 V1 要重写而不是在 V0 上修？
V0 的问题不是某个点慢，而是**组合爆炸**。prefill 和 decode 是两条代码路径，chunked prefill、prefix caching、spec decode、LoRA、多模态各自加 flag，两两组合都要单独处理（prefix caching 和 chunked prefill 到 0.6.0 才能同时开）。multi-step scheduling 又在这之上加了一层「一次调度跑 N 步」的状态。V1 的解法是把调度抽象降到最低：每个请求只有一个数「这一步给它多少 token」，prefill、decode、chunk、spec decode 的 draft 都是这个数的不同取值，组合自然成立。再把 EngineCore 拆成独立进程、prefix caching 做成零开销默认开、attention 之外的部分用 piecewise CUDA graph，CPU 开销问题顺带解决。修 V0 做不到这些，因为它们都是结构性的。
:::

::: details Q：multi-step scheduling 和 async scheduling 有什么区别？
两者都是为了让 GPU 不等 CPU，但粒度不同。**multi-step**（0.5.5，V0）是一次调度后连跑 N 步 forward，CPU 开销摊到 N 步上，代价是这 N 步里 batch 不能变，新请求要等，生成的 token 也成批返回，ITL 变得不均匀，而且和 chunked prefill、spec decode 组合很差。**async scheduling**（0.10 实验、0.14 默认，V1）是单步粒度：GPU 跑第 N 步时 CPU 调度第 N+1 步，假定每个 running 请求产 1 个 token，采样结果回来后再修正 stop 条件和 KV 分配。每步都能重组 batch，所以不牺牲 ITL 均匀性，也能和 chunked prefill、spec decode（0.19 zero-bubble）组合。它的难点是正确性：0.10.2 和 0.11.0 在抢占等边界场景会输出乱码，0.11.1 才修稳。
:::

::: details Q：prefix caching 在 V0 为什么默认关，V1 为什么默认开？
V0 的 prefix caching 要给每个 block 算 hash、维护 evictor，这些都是 Python 层的工作，命中率低时纯属额外开销，所以默认关，只在明确有共享前缀时手动开。V1 博客的说法是把 hash 计算并入正常的 block 分配路径、evict 做成常数时间、减少 Python 对象创建，结果是 0% 命中率时吞吐损耗 "less than 1%"，高命中率时提升数倍。开关的收益曲线变成「几乎没有下行」，就可以默认开。这也是 V1 的通用思路：一个 feature 要默认开，前提是它在不适用的场景下接近零成本。
:::

::: details Q：v0.6 的 2.7x 为什么主要来自 CPU 侧？这个结论能推广吗？
博客的 profile 很直接：8B 模型在 H100 上，HTTP 33%、调度 29%、GPU 38%。8B decode 一步在 H100 上不到 10 ms，Python 的调度循环、detokenize、JSON 序列化加起来也是这个量级，GPU 一大半时间在等。所以优化手段全在 CPU 侧：进程分离、multi-step、async output processing。推广时要看 GPU 每步时长和 CPU 每步时长的比值：70B 只有 1.8x，因为 GPU 每步变长了；MoE 大模型或者 batch 很大时 CPU 占比更低。反过来，GPU 越快（H100 → B200）、模型越小，CPU 侧优化越重要，这也是 V1、async scheduling、Model Runner V2 一直在打 CPU 开销的原因。
:::

## 参考

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention（2023-06-20）](https://vllm.ai/blog/2023-06-20-vllm)
- [vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction（2024-09-05）](https://vllm.ai/blog/2024-09-05-perf-update)
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture（2025-01-27）](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
- [vLLM V1 prefix caching 设计文档](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)
- [GitHub Releases · vllm-project/vllm](https://github.com/vllm-project/vllm/releases)（每个版本的一手 release note）
- 数据文件：`src/data/vllm-releases.json`，追加新版本只改它

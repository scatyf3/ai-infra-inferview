

我是个想找北美ai infra为主的人，我需要准备一个知识库应付面试

tech stack
1. md知识文档
2. 一点web前端，展示知识领域的overview
3. 一些可视化小插件，展示tensor shape之类的
4. 部署到github.io的静态网站

我想写的知识包括

### 1. 推理系统核心

这里的语言是tensor怎么算，shape，计算量，对吗？
>shape → 访存量 (bytes) 和计算量 (FLOPs) → arithmetic intensity → roofline 落在哪一侧 → 所以优化手段只能是 X

- **prefill vs decode：compute-bound vs memory-bound、arithmetic intensity、roofline**
- **显存账**：权重/KV cache/激活的估算公式，能口算「70B bf16 + 8k context + batch 32 要多少卡」
- **KV cache**：布局、PagedAttention、prefix caching / RadixAttention、KV 量化
- **Attention 变体**：MHA/MQA/GQA/MLA 的 KV 大小与 decode 带宽影响
- **FlashAttention v1/v2/v3**：tiling + online softmax，为什么省的是 HBM 访问而不是 FLOPs；v2 改了什么切分
- **continuous batching、chunked prefill、PD 分离**、scheduler 抢占（swap vs recompute）
- speculative decoding：draft-target、acceptance rate、EAGLE/Medusa/MTP 的区别（你自带优势，会被追问到底）
- 量化：GPTQ/AWQ/SmoothQuant、W4A16 vs W8A8、FP8、per-group vs per-tensor、为什么 decode 场景 weight-only 就够
- 指标：TTFT / TPOT / ITL / goodput，怎么做 benchmark，SLA 下怎么调 batch

### 2. 并行与通信

- **TP/PP/DP/EP/SP/CP 各自切什么、通信量、通信插在哪一层**
- Megatron 的 column/row parallel 组合为什么能只要两次 all-reduce
- ZeRO 1/2/3、FSDP 的通信-显存 tradeoff
- 集合通信原语 + ring all-reduce 带宽公式；NVLink/PCIe/IB 数量级；NCCL 基本概念
- MoE：路由、all-to-all、专家负载不均、EP 与 TP 混用
- 计算通信 overlap 的几种做法

### 3. GPU / 算子（够用即可，别装深）

- SM / warp / shared memory / bank conflict / coalescing / occupancy
- Tensor Core、GEMM tiling、为什么 GEMV 打不满
- 手写过的：reduce、softmax（数值稳定）、RMSNorm、fused bias+gelu
- Triton 编程模型、autotune、什么时候写 Triton 什么时候直接调 cuBLAS/CUTLASS
- CUDA Graph、kernel fusion 在框架侧的落地
- profiling：nsys / ncu / torch profiler 看哪几个指标

### 4. 框架内功（你的差异化，最该下功夫）

- **PyTorch**：dispatcher、autograd 机制、custom op 注册、caching allocator 与碎片
- **torch.compile**：dynamo 抓图、graph break、inductor 生成什么、AOTAutograd
- **给 vLLM/SGLang 加一个新模型的完整流程**：config 映射、weight loading、model runner、attention backend 接入、tokenizer/chat template
- vLLM V1 架构：EngineCore / scheduler / worker / executor 怎么分层；SGLang 的 RadixAttention + 前端语言
- 服务层：异步请求、batching 队列、streaming、多副本路由
- 能讲清「一个 request 从 HTTP 进来到第一个 token 出去」的全链路

### 5. Post-train / efficient

- SFT：packing、loss mask、长上下文
- LoRA/QLoRA 原理 + 显存账；多 LoRA serving
- **RLHF 全家桶**：PPO 四模型流程、KL 约束、DPO / GRPO 的推导动机与区别
- RL infra：rollout 与 training 分离、权重同步、veRL/OpenRLHF 架构
- 混合精度、grad checkpointing、优化器状态显存账
- 蒸馏、剪枝、稀疏（你的 KV pruning 归这里，准备好 motivation 和 ablation）

### 6. 基础不能挂

- Python：GIL、asyncio、多进程 vs 多线程、CPython 对象模型
- C++：RAII、智能指针、移动语义、虚函数表（kernel/framework 岗常问）
- OS：进程线程、虚拟内存、锁与原子操作；网络：TCP、RPC、gRPC
- 系统设计：**设计一个 LLM 推理服务**（几乎必出）

### 手撕高频（比 LC 更常考）

手写 MHA/GQA forward、带 KV cache 的 decode step、数值稳定 softmax、RMSNorm、top-p 采样、simple beam search；Triton 版 softmax / fused layernorm；CUDA reduce + tiled matmul。
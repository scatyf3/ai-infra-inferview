---
title: SGLang-Omni：多阶段 omni / 语音 serving
status: draft
tags: [sglang-omni, omni, tts, asr, serving, release-history]
difficulty: 4
order: 8
related: [/inference/omni-serving, /inference/speech-output, /inference/omni-duplex, /inference/multimodal-encoder, /framework/vllm-release-history, /framework/vllm-v1-architecture]
stack: []
---

# SGLang-Omni：多阶段 omni / 语音 serving

> 三件事：它怎么把一个 omni 模型拆成 stage 来 serve；和纯 LLM serving 比多了哪些优化 trick；从 2026-01 第一个 commit 到 v0.1.8 的版本演进。
> 数字都来自仓库文档、PR 正文或 LMSYS 博客，多数是 PR 作者在特定硬件和并发下的单次测量，不是官方 benchmark。面试时要说「PR 报告值」。

## 一句话结论

[SGLang-Omni](https://github.com/sgl-project/sglang-omni) 是 sgl-project 做的 **多阶段 serving runtime**，serve 三类模型：omni（Qwen3-Omni、Ming-Omni、MiniCPM-o）、TTS（Higgs、Qwen3-TTS、MOSS-TTS、S2-Pro 等十几个）、ASR（Qwen3-ASR、Whisper 等）。它把模型拆成 **stage 图**：AR stage（thinker、talker、TTS 主干）复用 SGLang 的调度器和 ModelRunner，其他 stage（预处理、encoder、vocoder）用轻量调度器；stage 之间是 **ZMQ 控制面 + 按拓扑自动选的数据面**。

和纯 LLM serving 最大的不同是**瓶颈在 host，不在 GPU**：模型小（0.6–4B），batch 又被首包延迟卡小，GPU 大部分时间在等 CPU。所以大部分优化都在砍 host 开销（CUDA graph 覆盖多码本内循环、one-step lookahead、合并 D2H、拆进程躲 GIL、同卡多副本 + MPS），再加上语音特有的几件事：首包、播放不卡顿、参考音色缓存。

## 1. 定位：为什么不直接用 SGLang

2026-03 的 README 原文："Omni models ... typically consist of multiple stages, making SGLang's LLM-specific architecture no longer suitable."

SGLang 的 scheduler 只会调度**一个**自回归模型。Qwen3-Omni 一个请求要过 encoder → thinker → talker → code2wav 四种计算形态（各级是什么负载见 [Omni Serving](/inference/omni-serving)），单个 TTS 也至少是「AR 主干 + vocoder」两级。所以 SGLang-Omni 只管 stage 图、stage 生命周期、stage 间传输和 OpenAI 兼容 API；AR stage 内部的 continuous batching、radix cache、CUDA graph 直接复用 SGLang。

| | SGLang | SGLang-Omni |
|---|---|---|
| 调度单位 | 一个 AR 模型的 batch | 每个 stage 一个调度循环 |
| 一个请求经过 | tokenizer → scheduler → detokenizer | preprocessing → encoder → thinker → talker → vocoder（一张 DAG） |
| 进程 / GPU | TP/DP/PP 切同一个模型 | 每个 stage 选进程、选卡、选副本数 |
| 延迟指标 | TTFT / TPOT | TTFA（首包音频）、RTF、播放 underrun |
| 输出 | token 流 | 文本流 + PCM 流（24 kHz 等） |

为什么不用 Ray（RFC 原文）："Ray's overhead — extra runtime, object-store semantics, and operational footprint — is significant for our shape: a small fixed number of stages on a small fixed number of GPUs, with no autoscaling."

## 2. 架构：一个请求怎么走

### 2.1 分层

```text
HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelRunner -> model forward
```

| 层 | 职责 | 感知模型吗 |
|---|---|---|
| HTTP API | OpenAI 兼容：`/v1/chat/completions`、`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/realtime`（WebSocket） | 否 |
| Client | `GenerateRequest` → `OmniRequest`；聚合文本、音频、流 | 否 |
| Coordinator | 请求生命周期：准入（`max_in_flight`）、送入 entry stage、收 terminal 结果、多 terminal 合并、abort 广播 | 否 |
| Stage | **IO 壳**：收控制消息、从数据面取 payload、fan-in、把活塞进 `scheduler.inbox`，从 outbox 取结果往下游发 | 否 |
| Scheduler | 每个 stage 一个执行循环 | 部分 |
| ModelRunner | forward 准备、分派、输出抽取 | 是 |

文档里的不变量："`Stage` does not branch on scheduler type"。所有 scheduler 只暴露 `inbox` / `outbox` 两个队列，消息类型只有 `new_request` / `stream_chunk` / `stream_done` 进，`result` / `stream` / `error` 出。

### 2.2 每个 stage 选一种 scheduler

| Scheduler | 用在 | 要点 |
|---|---|---|
| `OmniScheduler` | AR stage：thinker、talker、TTS 主干、ASR decoder | 复用 SGLang Scheduler 的 batch 选择、KV、radix cache、prefill/decode 调度 |
| `SimpleScheduler` | 预处理、encoder | 没有 KV，`inbox.get() → compute → outbox.put()`，可选局部攒批 |
| `StreamingVocoderBase` / `Code2WavScheduler` | 流式 vocoder | 攒够帧就解码，跨请求合批 |
| `StreamingDetokenizeScheduler` | Qwen3-Omni 的 `decode` stage | token id 流 → UTF-8 安全的文本 delta |
| `SessionScheduler` | realtime / 全双工 | 请求是 open / append / close，跨会话批量 append |

**OmniScheduler 怎么复用 SGLang**：不是继承，是借方法。模块 docstring 原文："Uses SGLang's batch selection and result processing logic via unbound method calls on the upstream `Scheduler` class. No inheritance." 实现上 `__getattr__` 去 SGLang 的 `Scheduler` 类上找同名方法，再绑定到自己身上。

- **复用**：`get_next_batch_to_run()`、`process_batch_result()`，也就是 KV 管理、prefill/decode 调度、tree cache。
- **覆写**：`recv_requests()`（从 inbox 取，并把 stream chunk 分给对应请求）、`run_batch()`（交给 Omni 自己的 ModelRunner）、`send_to_tokenizer()`（no-op）。
- **不用**：SGLang 的 ZMQ 通道、tokenizer、grammar、spec decoding、PP、overlap scheduler。

overlap scheduler 不能用的原因：chunked prefill 中间块的计数（`Req.inflight_middle_chunks`）在 overlap 下会滞后一轮。替代方案是自己写的 one-step lookahead（见 3.3）。

### 2.3 Qwen3-Omni 的 7 个 stage

文档和 RFC 还写着 8 个 stage，但 #1548 已经去掉了 `mm_aggregate` 这一跳，代码里语音模式是 7 个：

```text
                 [preprocessing]  CPU：tokenize、HF processor、读媒体
          ┌─────────┼──────────┐
          ▼         ▼          │ (route_fn：只发给有输入的 encoder)
 [image_encoder] [audio_encoder]   GPU0，各自进程，SimpleScheduler + 输出缓存
          └────┬────┘          │
               ▼ fan-in         │
           [thinker]  GPU0：OmniScheduler + SGLang（radix cache、CUDA graph、chunked prefill）
               │ 每个 token 一条 stream：8 字节 token id，inline 在控制消息里
       ┌───────┴────────┐
       ▼                ▼
   [decode] CPU     [talker_ar]  GPU1：MoE 主干出第 0 层码本 + code predictor 补 15 层
   文本 SSE              │ codec 帧：前 10 帧逐帧发，之后 10 帧一组
                         ▼ 跨卡 CUDA IPC
                    [code2wav]  GPU0：10 帧/窗 + 25 帧左上下文，CUDA graph，跨请求合批
                         └─► 24 kHz PCM
```

几个和直觉不一样的点：

1. **code predictor 不是独立 stage**。它和 talker 是「每一步都要互相等」的关系（talker 出第 0 层码本 → predictor 补齐 16 层 → 码本 embedding 之和作为 talker 下一步输入），拆开就是每帧一次跨 stage 往返。所以放在同一个 `talker_ar` stage 里，配置成独立 stage 会直接报错 "Qwen code predictor is part of talker_ar and must not be a stage"。
2. **thinker → talker 只传 token id**，不传 hidden state。#1574 先把每 token 的传输量从 8 KB 砍到 4 KB，#2175 发现 talker 其实是拿 token id 过自己的 text embedding 和 projection 重建输入，传过去的 hidden 从来没被读，于是只传 token id。8 字节小于 16 KiB 的 inline 阈值，直接塞进控制消息，不走数据面。
3. **默认放置**：talker 独占 GPU1，code2wav 蹭 thinker 的 GPU0。依据是 c8 实测 talker 中位 GPU 利用率约 86%，code2wav 只有 9–13%。talker 独占一卡后 TTFA p50 从 1.43–1.71 s 降到 0.99–1.20 s；再给 code2wav 单独一卡，"giving the vocoder a dedicated GPU bought nothing"。
4. **单卡也能跑**（`--colocate`）：这时 code2wav 并进 talker 的进程，跑在优先级更高的 CUDA stream 上，用 talker 的 codes-ready event 排序（#2374）。原因是一张卡上多个进程只能时间片轮转，code2wav 等不起。

TTS 的图简单得多：`preprocessing → tts_engine → vocoder` 三段。

### 2.4 控制面和数据面分离

RFC 原文："The split between control and data planes is the core architectural decision."

- **控制面**：ZMQ IPC socket + msgpack，只传 `Submit` / `DataReady` / `Complete` / `Abort` 这类小消息。abort 走 PUB/SUB 广播给所有 stage。
- **数据面**：张量。发送方把 payload 里的 tensor 抽出来拼成一个 `uint8` buffer，其余部分 pickle 成 header；控制消息里只带 `DataRef`（buffer 位置、每个 tensor 的 shape / dtype / offset）。

传输方式由 `CommRouter` **按拓扑自动选**，没有用户可选的 backend：

| 优先级 | 条件 | 传输 | 特点 |
|---|---|---|---|
| 1 | 同进程 | `local_object` | 传 Python 引用，真零拷贝，接收方只读 |
| 2 | 同卡、不同进程 | direct CUDA IPC | 接收方直接映射发送方显存，无 relay buffer、无 ACK |
| 3 | 同节点、跨卡 | pooled CUDA IPC | 发送方有一个 GPU slot 池（64 KiB 粒度），接收方 P2P / NVLink 拷走后 ACK 释放 |
| 4 | 同节点 CPU 数据 | SHM | 命名共享内存 |
| 5 | 跨节点 | Mooncake（RDMA） | 管道已铺好，配置入口还没开放 |
| — | CPU 张量且 ≤16 KiB | inline | 直接塞进控制消息，无 relay、无 ACK |

早期的 NCCL、NIXL relay 还在注册表里，但 router 不会选它们。

两个实测数字：

- #496 引入「同进程传引用、同卡走 CUDA IPC」后，text-only 同进程 "-33% mean latency"。
- **direct CUDA IPC 对小 payload 反而慢**：258 KB 的 audio encoder 输出每次要在接收侧 `cudaIpcOpenMemHandle`，还和同卡的 thinker prefill 抢，实测 13.4 ms mean / 69 ms p95，走 SHM 只要 1.26 ms。所以 direct IPC 加了大小门槛，只有图像、视频 encoder 的大输出走 IPC。

### 2.5 进程拓扑是一份配置

| 机制 | 配置 | 改变什么 |
|---|---|---|
| process 分组 | `StageConfig.process` | 同名 stage 在同一进程（共享 GIL、CUDA context，可传引用）；TP stage 每 rank 一个进程 |
| process replica | `processes.<name>.num_replicas` | 只复制瓶颈进程（比如 talker），按请求 sticky 绑定副本 |
| placement | `stage.gpu` / `replica_devices` | 每个副本在哪张卡 |
| CUDA MPS | `mps: off / auto / on` | 同卡多进程的 kernel 能并发，而不是时间片轮转 |
| 权重共享 | `--weight-share on` | 同卡副本通过 CUDA IPC 共用一份只读权重 |

拆不拆进程是个 trade-off，文档原话：拆开 "can improve throughput by overlapping vocoder scheduling and GPU work with generation, but it also changes IPC and serialization paths, can increase idle VRAM, and may duplicate process-local caches"。具体例子见 3.4。

横向扩展靠 router（先有 Python 版，v0.1.4 起加了 Rust 版）："Each worker is a complete Omni V1 HTTP server. The router does not load model weights or split a single request across workers." 它按请求需要的能力（chat、speech、audio_input……）路由，上传的音色是 worker 本地状态，相关请求钉在那个 worker 上。

## 3. 和纯 LLM serving 不一样的优化 trick

### 3.1 根因：小模型 + 延迟封顶的 batch = host-bound

纯 LLM serving 的常规解法是「加大 batch 把 GPU 喂满」。语音模型做不到：

1. **模型小**：talker / TTS 主干 0.6–4B，每步 GPU 计算只有几毫秒。
2. **batch 被首包延迟卡住**：batch 越大，每个请求首包越晚。
3. **每步 host 活多**：多码本采样、码本拼装、vocoder 调度、chunk 组装、HTTP 流式，每步都要 CPU 参与。

证据（issue #907，H100）：Qwen3-ASR 单副本 GPU timeline **94.3% idle**。把 SM 频率降到 0.455×，吞吐只掉到 0.90×；把 host CPU 限到约 1/4，吞吐掉到 0.31×。Higgs TTS 单副本调优后 SM Active 约 29%。

**为什么 host 开销在语音里被放大**：设一段语音时长 $D$ 秒，codec 帧率 $f$ 帧/秒，每帧一个 AR step，则步数 $N = D \cdot f$。每步多 $\delta$ 的 host 开销，端到端多 $N\delta$。

例：$D = 10$ s，$f = 25$ 帧/s（Higgs 的帧率），$N = 250$ 步；$\delta = 0.1$ ms 时多 $250 \times 0.1 = 25$ ms。LMSYS 博客原文："A tiny 0.1ms overhead per step inflates end-to-end latency by nearly 25ms"。而且一「步」里不是一个 token：Qwen3-TTS 的 talker 每帧还要让 code predictor 再自回归 15 次，单步内就是十几次小 forward，天然 launch-bound。

总收益（LMSYS / TTS 优化长文，1×H100，SeedTTS EN 1088 条，3 次均值；vanilla 是关掉 CUDA graph 和 async decode）：

| 模型 / 模式 | 并发 | QPS vanilla → 优化 | RTF vanilla → 优化 | 首包 ms |
|---|---|---|---|---|
| Higgs 流式 | 4 | 2.41 → 5.93（2.46×） | 0.393 → 0.163 | 166 → 109 |
| Higgs 流式 | 16 | 7.08 → 14.63（2.07×） | 0.533 → 0.261 | 214 → 176 |
| MOSS-TTS-Local 流式 | 2 | 0.82 → 2.78（3.40×） | 0.561 → 0.165 | 257 → 67 |
| MOSS-TTS-Local 非流式 | 16 | 4.67 → 10.88（2.33×） | 0.781 → 0.347 | — |

RTF = 生成耗时 / 生成音频时长，< 1 才能实时播放（定义见 [语音输出](/inference/speech-output)）。

### 3.2 流式接力：把首包压到第一帧

纯 LLM 的首 token 只取决于一次 prefill；omni 的首包是一条链：encoder + thinker 首 token + talker 首帧 + vocoder 首窗。每一级都要「上游出一点、下游就开始」。

| trick | 做法 | 数字 |
|---|---|---|
| stage 间 token 级流式 | `stream_to` 让上游每步推 chunk；下游可以在 fan-in 的 payload 到齐前先缓存 stream（`can_accept_stream_before_payload`） | MOSS-TTS-Local 接流式后 c1 首包 0.70 s → 0.13 s（#753） |
| talker partial start | talker 收到 ≥5 个 thinker token 就建请求、开始 prefill，不等 thinker 说完 | 作者估计省 100–200 ms（#475，无 A/B 表） |
| codec 帧合并但保留前缀 | talker 前 10 帧逐帧发，之后 10 帧一组 | 见下面的算例 |
| 首块 ramp | Qwen3-TTS 首块 1 帧，然后 2、4 帧，再到稳态块宽 | H100 1 RPS，首包 p50 90.1 → 36.0 ms（#1928） |
| 首窗优先 | code2wav 每轮先解所有请求的第一个窗口，原文 "a first window carries its request's time to first audio" | — |
| 编码器批窗口懒启动 | 只有第二个请求到了才开始计 50 ms 攒批窗口 | 之前空闲时白等 50 ms，约占 speech TTFT 的 1/3（#1628） |

**为什么帧合并要保留前缀**：设 talker 每步间隔 $d$ ms。一律 10 帧一组的话，发送方要多扣住最新一行（判断它是不是 EOS），第一组要到第 11 步才发出；逐帧发的话第 2 步就能发第 1 帧。差 $9d$。$d = 15$ ms 时就是 135 ms。（这是输入就绪时间的估算，不是实测 TTFA。）

**开头静音也算首包**。Qwen3-TTS 只给 x-vector 的克隆请求，talker 第 0 帧常采样到一个「室内底噪」码，然后静音 4–7 帧。修法是前 2 帧把静音码的 logit 置 $-\infty$；静音码集合是启动时用模型自己的 codec 编码白噪声、粉噪声、棕噪声推出来的。1.7B 的中位起音 495 → 155 ms，WER 0.854% → 0.904%（#2357）。这引出两个指标的区别：**first playable**（第一个 PCM 包到达）和 **audible first audio**（第一个听得见的样本）。

### 3.3 多码本 AR：CUDA graph 要包住整个帧内循环

三种多码本结构，serving 难点不同：

| 结构 | 代表 | 每帧怎么出码本 | 难点 |
|---|---|---|---|
| 主干 + code predictor | Qwen3-Omni talker、Qwen3-TTS | 主干出第 0 层，5 层小 transformer 自回归补 15 层 | 每帧 15 次小 forward |
| 主干 + frame-local transformer | MOSS-TTS-Local（Qwen3-4B + 1 层 local，12 层 RVQ） | 1 + 12 个 micro-step | 每帧 13 次采样 |
| delay pattern | Higgs（8 码本）、ZONOS2 | 码本在时间上错位，一步出全部 | 首块要多等「码本数」行 |

对应的 trick：

1. **整链 CUDA graph**。SGLang 的 decode graph 只包主干一步；这里要把帧内循环也包进去。
   - Qwen3-TTS 把 15 次子迭代（lm head、采样、codec embedding、predictor）捕获成一张图：PR 标题写 "~1.9x at c8, ~3x at c32"（#1134）。
   - MOSS-TTS-Local 的 1 + 12 micro-step：eager 约 22 ms/帧，replay 约 4 ms（#728）。
   - 要能 capture，采样器状态必须张量化成 `[max_bs, ...]` 的固定地址 buffer，Python 分支改写成 `torch.where`。Higgs 这样改完 c32 吞吐 +69%（#503）。
2. **采样融合**。MOSS-TTS-Local 每帧 13 次 seeded 采样融成一个 Triton kernel：kernel 本身 8.21×（bs 8），端到端 3–6%（#1666）。MOSS-TTS-delay 的 32 个 audio LM head 融成一个 stacked GEMM（#1665）。
3. **one-step lookahead async decode**。把一步拆成 `execute_launch`（入队 forward、GPU 上采样、异步 D2H 到 pinned buffer）和 `execute_resolve`（等 event、做 host 收尾），循环变成 launch(N) → resolve(N−1)，CPU 收尾和下一步 GPU 计算重叠。Higgs c16 吞吐 +12.7%、RTF p99 −39.1%（#590）。难点在停止边界：bs=1 时 lag 一步可能漏掉 EOS 一直跑到 4096 帧，所以 launch 时要在 GPU 上快照 stop id（#758）。
4. **prefill 也要图化**。TTS / ASR 的 prompt 很短，prefill 是 launch-bound：Qwen3-Omni c1 一个 17 token 的 prompt，prefill 76.3 ms，设备计算只有 3–4 ms。breakable prefill graph 按 token 数分桶，把每层 attention 之外的部分捕获成图。thinker text TTFT p50 降 27–55%（#1381）。分桶要从 1 开始：实测 40.6% 的 prefill 恰好 1 个 token（前缀命中后只重算最后一个 token）。
5. **prefill admission 攒批**。TTS 请求短、到达零散，每次 prefill 的调度固定开销和 batch 大小无关。等 K 个请求或最老的等满 T ms 再放行（decode 空闲时立即放行）。Higgs c96 下 48.5 → 63.8 QPS（#1071）。

### 3.4 Vocoder：纯 LLM 没有的一级

| trick | 问题 | 做法 / 数字 |
|---|---|---|
| 按精确长度合批 | 非因果 codec 对 padding 敏感，pad 了会污染短样本 | 只把长度完全相同的窗口 stack；Higgs c16 吞吐 +7.8%（#574） |
| 窗口形状对齐 CUDA graph | Qwen3-Omni code2wav 窗口 = 25 帧左上下文 + 10 帧新帧，可变窗口匹配不上 graph | talker 按 10 帧对齐发送 → 窗口长度只有 10 / 20 / 30 / 35 几种，全部预先 capture；同长度窗口一次 replay，最多 8 行 |
| 有状态增量 codec | 每块都带左上下文重算，"The streaming vocoder recomputed left context on every chunk" | Qwen3-TTS 每个流只解新帧、状态放预分配 arena：kernel 数 1147 → 356，播放 underrun 的流 20.9% → 0.6–2.3%（#1997） |
| 异步 D2H | 每个窗口的波形读回阻塞调度线程 | depth-2 流水：异步拷到 pinned slot、每 slot 一个 event；调度线程每窗占用 p50 8.98 → 0.65 ms（#1567） |
| 按播放 deadline 排序 | 高并发时有的流快播完了 | 最快到期的先解（#2400）；vocoder worker 从 1 个加到 2 个，underrun 23.5% → 10.3%（#1930） |
| 拆进程躲 GIL | vocoder 和 AR 调度器抢 GIL，"prefill steps stretch +38% while it is active" | Higgs 拆进程后 c96 下 63.8 → 83.8 QPS（#1071）。**但同一改动在 c16、无 MPS 时 9.3 → 4.7 req/s**（两个 CUDA context 只能时间片轮转），最后又移回去 |

最后一行是这一节最值得讲的：**拆进程的收益取决于并发点和有没有 MPS**，benchmark 的工作点必须和 CI / 生产一致。

### 3.5 Encoder 侧

纯文本 serving 没有 encoder；omni 和 ASR 的 encoder 会卡住 decode。

- **pre-LM encoder service**：Qwen3-ASR 原来在调度线程上跑 encoder + prefill，每次准入让整个 running batch 停 28–46 ms。改成在 request build 阶段用专用线程、专用 CUDA stream 跑 audio tower，embedding 齐了才准入（#1341）。
- **encoder CUDA graph**：Qwen3-Omni audio tower 32 层发约 460 个 kernel，GPU 活只有 1.75 ms。把 `cu_seqlens.tolist()` 提到层外后整层栈 capture，11–13 ms → 2.2–2.6 ms（#1628）。
- **ViT 的 Conv3d 换 Linear**：patch embed 的 Conv3d 满足 kernel = stride，不滑动，等价于 reshape + Linear，快 10–15×（#132）。
- **encoder 攒批不一定赚**：Higgs 的 audio encoder 攒批反而 −18% / −25%（c16 / c32），因为 encoder 和 AR 主干同卡抢算力；AR 之后的 vocoder 攒批才是净赚（#610）。

### 3.6 语音特有的缓存

1. **参考音色编码缓存**。声音克隆每个请求都要把参考音频过 codec / speaker encoder（MOSS 约 0.25 GPU 秒一次），同一音色会被大量复用。`ReferenceEncodeService` 是按字节限额的 LRU，同 key 并发请求只算一次（single-flight），失败不缓存。key 必须包含模型版本、encoder 配置和**参考内容本身**，远程 URL 不能只按 URL 缓存。MOSS-TTS-Local 把缓存从 256 条加到 1024 条后，c16 6.8 → 8.9 QPS（+32%，#778）。它用精确匹配而不是 radix：参考音频之间没有前缀关系。
2. **radix key 必须来自内容**。图片、音频在 prompt 里展开成一串相同的占位 token，只看 token id 的话两段不同音频会命中同一个 KV 前缀，结果是「一个用户的声音漏到另一个用户」。#260 用内容 hash 生成 pad value 替换占位 token；规则（#937）："every input or piece of state that can affect the KV state ... must contribute a fingerprint to `Req.extra_key`"。
3. **GPU radix hash**（MOSS-TTS-Local 专用）。它每生成一帧就往 KV 链追加一个 radix token，key 要 hash 整行 13 个通道。原来每帧 `row.numpy().tobytes()` + blake2b，等于每帧一次 GPU→CPU 同步，挡住了 CUDA graph 和 lookahead。改成一个 Triton kernel 在 GPU 上算多项式 hash：$h \leftarrow (h \cdot B + x_c) \bmod M$，$M = 2^{31} - 1$，$B = 10^9 + 7$，$x_c$ 是第 $c$ 个通道的 id。碰撞只影响缓存命中，不影响输出。

### 3.7 同卡多副本 + MPS：在 LLM 里是反模式，在 TTS 里是 1.4–2.1×

LLM 的直觉是「一张卡一个 engine，把 batch 做大」。TTS 反过来：一张卡上起 2–3 个完整副本，开 CUDA MPS。

为什么有用（`mps_dp.md` 原话）："Each replica is a full serving process with its own cores, so it adds an independent host chain. Without MPS the processes only time-slice; MPS lets their kernels co-reside on the SMs and use the idle compute." host 是瓶颈，多一个进程就多一条 host 链；MPS 让几个进程的小 kernel 同时占 SM。

数字（H100，Higgs）：单副本 21.7–22.1 qps；DP2 + MPS 31.5–37.7 qps（1.4–1.7×）；DP3 + MPS 39.9–46.9 qps（1.8–2.1×）。经验法则："colocate replicas when a tuned single replica holds roughly ≤60% SM-active under its latency SLO and N× the footprint fits"。

几个坑：

1. **显存不是按比例分的**。SGLang 的 `mem_fraction_static` 是「加载权重后剩余显存」的比例。顺序启动三个 0.27 的副本，KV 分别拿到 97,503 / 53,149 / 20,961 token。所以要用绝对值：`max_total_tokens` 或 `engine.kv_cache_bytes`（#1616）。同样的问题在多 stage 里也有：stage 依次加载，「剩余显存」对每个 stage 都不一样。
2. **权重共享只省显存，不加吞吐**。follower 通过 CUDA IPC 映射 leader 的权重；MOSS-TTS-delay 原来只能放 1 个副本，共享后放 2 个，+32%。固定副本数时共享与否吞吐持平，原话 "a capacity mechanism, not a throughput optimization"。
3. **MPS 不稳定**：有 `cudaErrorMpsRpcFailure`；没 attach 上 MPS 的进程会静默退化成时间片，所以 #1635 要求验证失败就 "fails closed instead of silently time-slicing"。
4. **不是所有模型都赚**：Qwen3-TTS 1.7B 在一张 H100 上，调优后的单实例峰值吞吐反超同卡 MPS-DP2，首包延迟还好几倍，CI 已不再推荐同卡共置。

### 3.8 指标：从 TTFT / TPOT 变成 TTFA / RTF / underrun

| 指标 | 定义 | 纯 LLM 对应 |
|---|---|---|
| TTFA / TTFP | 客户端发请求到第一个 PCM 包到达 | TTFT |
| audible first audio | 第一个听得见的样本（5 ms 帧峰值 ≥ −34 dBFS） | 无 |
| RTF | 生成耗时 / 音频时长；ASR 里是处理耗时 / 输入音频时长 | TPOT 的倒数意义 |
| inter-chunk | 相邻音频 chunk 到达间隔 | ITL |
| underrun / c50 | 模拟播放：deadline = 首 chunk 到达 + 已收音频时长，chunk 晚于 deadline 就是卡顿；c50 = 最大卡顿 ≤ 50 ms 的流占比 | 无 |

算例：一个流首 chunk 在 $t=0.1$ s 到达，含 0.5 s 音频；第二个 chunk 必须在 $0.1 + 0.5 = 0.6$ s 前到。若它 0.65 s 才到，这次 underrun 50 ms。

benchmark 噪声比 LLM 大（采样温度、runaway、输出长度都会变）。RFC #1018 测过 A/A 噪声带：speech c8 11.18%、c16 13.78%，原话是 "every apparent c8/c16 speech win flipped sign" after swapping launch order。所以「吞吐 +5%」需要换启动顺序、配对多次才可信，而且比吞吐必须同时看平均音频时长。

### 3.9 负结果：哪些看起来该赚但没赚

| 改动 | 结果 | 原因 |
|---|---|---|
| Higgs prefill CUDA graph | 53.7 vs 53.2 QPS，持平 | 瓶颈在调度循环，不在 prefill kernel |
| Higgs audio encoder 攒批 | −18% / −25% | 和 AR 同卡抢算力 |
| MOSS-TTS-Local async decode | ≈ sync | resolve 只有 0.06 ms，host step 有 10.4 ms，lookahead 只能藏住前者 |
| Qwen3-TTS talker 开 mixed chunk | 20 rps 首包 40 ms → 2.7 s | issue #1754 只列为放弃项，没写根因；对比 thinker 开 mixed chunk 是 +14% 吞吐 |
| Higgs vocoder 拆进程（c16、无 MPS） | 9.3 → 4.7 req/s | 两个 CUDA context 只能时间片轮转 |
| 神经 codec 编码批处理 | `encode(A)` 和 `encode([A,B])[0]` 约 5.8% token 不同 | BF16 GEMM 的 M 维变了，cuBLAS 换了 kernel 和累加顺序，RVQ 硬量化把误差放大；所以批量编码的结果不能放心缓存 |

## 4. 版本演进

### 演进主线

同 [vLLM 版本演进](/framework/vllm-release-history) 一样按 **Observation → 场景 → 做法** 讲。来源说明：GitHub release notes 基本是自动生成的 PR 列表，只有 0.1.1 有正文，所以下面的 observation 和数字大多取自 PR 标题和 squash commit 正文。0.1.0 的 release notes 还漏掉了 #143–#203，那一段用 git log 补成 dev·03。

所有版本 pin 一个固定的 SGLang（原则 "Pin, don't track"）：0.1.0 用 0.5.12.post1，0.1.1–0.1.3 用 0.5.16，之后逐版升到 0.1.8 的 0.5.21。

### 2026 H1 · 起步与 V1 重写

**dev·01（2026-01）骨架**

- **Observation**：SGLang 只会调度一个 AR 模型，omni 的 encoder、talker、vocoder 放不进去。
- **做法**：第一个 commit 就是 "add control panel and data panel on SHM"：ZMQ 传小消息，张量写共享内存。1 月里 relay 接上 NIXL（RDMA）和 NCCL，并加了 credits 做流控；01-27 接入 Qwen3-Omni encoder（#28）。2 月 worker 才支持并发处理请求（#42，之前是串行），加了 Mooncake relay。

**dev·03（2026-03）接入 SGLang、跑通语音输出**

- **Observation**：自研 AR runtime 没有 radix cache、CUDA graph、continuous batching；语音输出要等 thinker 说完。
- **做法**：03-04 thinker 换成 SGLang 后端（#120/#123）；03-09 第一个 TTS 是 FishAudio S2-Pro（#143）；03-17 一天合入 stage 间 token 级流式（#167）和 Qwen3-Omni talker + code2wav（#168/#169）。04-15 的 RFC #294 定下迁移原则："every intermediate state has a working server"。

**V1（2026-05）重写，V0 退役**

- **Observation**：V0 的 talker 一次只服务一个请求（"V0 talker was c=1 serial"），每个模型自带一套 scheduler。
- **做法**：#334 引入 V1，链路变成 Coordinator → Stage → OmniScheduler（借 SGLang Scheduler 的方法）→ ModelRunner；talker MoE 改走 FusedMoE 以便 piecewise CUDA graph（#384），talker、thinker CUDA graph 默认开；05-14 删掉 V0（#435）。同一台 H200 上 "SeedTTS talker: ~22-26x faster latency"，"MMSU thinker: ~1.4-1.5x faster"（#411）。
- 关联：[vLLM V1 架构](/framework/vllm-v1-architecture)，两者都是「推倒重来一次统一抽象」。

**0.1.0（2026-06-14）首个 tag**

- **Observation**：V1 之后 stage 间开销成了主要矛盾：speech stage 不允许同卡，"left Talker on H200 at <2% utilization long-term"；同进程、同卡 stage 之间的张量也一律走 relay；`stream=true` 实际上是 no-op。
- **做法**：colocation 按 per-stage 显存预算记账（#430）；stage → 进程 → GPU 的拓扑（#461）；拓扑感知传输，"-33% mean latency"（#496）；真流式（#406）；talker partial start；one-step lookahead async decode（#590）。

### 2026 夏 · 产品化与 host 开销

**0.1.1（2026-08-08）首个 PyPI 版本，262 个 commit**

- **Observation**：GPU 计算已经不是瓶颈，瓶颈换成了 host：视频场景 "latency is dominated by inter-stage *delivery* of the large video embeddings, not compute"；同节点的 GPU 张量还要绕道 host 内存；vocoder 和调度器抢 GIL。
- **做法**：
  1. 同节点 NVLink CUDA-IPC 数据面 + 集中式 transport router（#869）；
  2. Lazy TensorRef，用到才物化，视频请求 "latency_mean_s 160-179s -> 36-37s (~4.5x)"（#808）；
  3. Higgs 三招：prefill admission 攒批、vocoder 拆进程、compile codec decode，合计 "48.5 -> 99.3 QPS (+105%)"（#1071）；
  4. 同卡 DP + MPS + CUDA IPC 权重共享（#986/#1124）；
  5. breakable prefill CUDA graph（#1364）；
  6. OpenAI speech API、上传音色、RL 权重同步和 rollout 接口。

**v0.1.2 / v0.1.3（2026-08）去 host sync**

- thinker prefill 去掉每请求的 `mask.any()` / `.item()` host 同步，改成每 batch 一次 `index_copy_`（#1161）。
- code2wav 窗口按 chunk 对齐以命中 CUDA graph（#1237）；默认放到 thinker 的卡上，让 talker 独占一卡（#1235）。
- v0.1.3 的 code2wav 异步 D2H：调度线程每窗占用 −93%（#1567）。

### 2026 年 9 月 · 部署拓扑与多硬件

**v0.1.4（2026-09-01）process replica、原生 MPS**

- **Observation**：一个 stage 成为瓶颈时只能复制整个服务；同卡多进程默认时间片轮转。
- **做法**：只复制瓶颈进程（#1175）；原生 CUDA MPS，attach 失败时 "fails closed instead of silently time-slicing"（#1635）；Rust router 起步；去掉 `mm_aggregate` 一跳（#1548）；平台加 ROCm、MUSA、Apple MLX。

**v0.1.5（2026-09-10）显存字节预算、增量 codec**

- **Observation**：`mem_fraction_static` 是 "a fraction of remaining VRAM after weights load"，多 stage 依次加载时语义不清；流式 vocoder 每块重算左上下文导致 underrun。
- **做法**：per-stage 绝对字节预算 + 起副本前 preflight（#1616）；Qwen3-TTS 有状态增量 codec（#1997）；预处理、vocoder 拆进程（#1923 "+23.0% at cap 64"，#1924 "about +70% single-replica throughput"）；talker KV pool 按 talker 自己的 20 层算，不再按 thinker 的 48 层（#1910）。

**v0.1.6（2026-09-17）最小 PD、空闲时阻塞**

- 与模型无关的 KV page handoff（#1773）。作者注明 "GPU inference was not run"，目前只是框架层的最小实现，不要说成已可用于生产。
- 调度器空闲时阻塞等 inbox，不再轮询：之前 "An idle scheduler wakes about a thousand times a second"，Apple M4 上空闲的 ASR worker 占一个核的 4.11%（#1809）。

### 2026 秋 · 会话与全双工

**v0.1.7（2026-09-28）session、torch.compile 默认开**

- pipeline 有了 session 生命周期，并桥接到 SGLang 的 streaming session（#2035/#2069）。
- code2wav 并进 talker 进程、走优先级 stream（#2374）；talker 交接只传 token id（#2175）；torch.compile 默认开（#2212）。

**v0.1.8（2026-10-12）原生全双工**

- **Observation**：全双工对话里很多会话同时追加小片段，逐个处理成不了批；多个 stage 进程在一台机器上争 CPU、跨 NUMA；冷启动的第一批请求要承担 JIT 和 graph capture 的开销。
- **做法**：MiniCPM-o 原生全双工（#2377），"Batch session appends across sessions"（#2535）；每个 stage 进程绑定到它 GPU 附近的 CPU（#2636）；预热完成才报 `/health` ready（#2686）；bs=1 也默认走 lookahead（#2638）；一批 abort / 失败路径一致性修复；Voxt（Mac 本地语音输入）换成原生 C++ MLX ASR 运行时。
- 关联：[全双工语音服务](/inference/omni-duplex)

### 交互

按类别筛选看一条线的演进；点任意版本展开 observation / 场景 / 做法和来源链接。

<ReleaseTimeline dataset="sglang-omni" />

### 全版本速查表

<ReleaseTimeline dataset="sglang-omni" view="table" :show-eras="false" />

### 六条主线

1. **数据面从必经之路降为 fallback**：relay → 同进程传引用、同卡 IPC（#496）→ NVLink CUDA IPC + 集中路由（#869）→ 懒物化（#808）→ token id inline（#2175）。
2. **显存语义**：per-stage 比例（#430）→ 绝对字节预算（#1616）→ talker KV 按自己的层数算（#1910）。
3. **GIL**：拆进程（#1071、#1923、#1924）换来跨进程搬运和多一个 CUDA context，再用 CUDA IPC 和 MPS 补回来；反方向把 code2wav 并回 talker 进程（#2374）。
4. **小模型吃不满大卡**：同卡 DP + MPS（#986 → #1635），再加权重共享（#1124 → #1892）。
5. **流式指标是 TTFA 和 underrun**：partial start → 首块 ramp → 增量 codec → 异步 D2H → 窗口拷完即发。
6. **CUDA graph 覆盖面扩大**：decode → talker MoE 走 FusedMoE → breakable prefill → vocoder、encoder、code predictor 整链 → torch.compile 默认开。

## 5. 和 vLLM-Omni 比

两者思路相同：any-to-any 模型拆成 stage 图，AR stage 各自复用自家 LLM 引擎，stage 间用 connector / relay 传数据（vLLM-Omni 的做法见 [Omni Serving](/inference/omni-serving) 第 5 节）。差异主要在实现取舍：

| | SGLang-Omni | vLLM-Omni |
|---|---|---|
| 复用 LLM 引擎的方式 | `OmniScheduler` 借 SGLang Scheduler 的方法，不继承 | 每个 AR stage 是一个 vLLM 引擎 |
| stage 间传输 | 按拓扑自动选：传引用 / CUDA IPC / SHM / Mooncake | 每条边可选 connector：共享内存 / Mooncake |
| 级间推进 | stage 间 token 级流式 + talker partial start | `batch` 或 `async_chunk` 两种模式 |
| 同卡多副本 | 一等公民：process replica + MPS + 权重共享 | 技术报告里有单卡融合单 stage 的做法 |

**没有同硬件的正面对比**。仓库里没有任何对 vLLM-Omni 的 benchmark；两边的论文和报告也只和 HF Transformers 或自己比。唯一的第三方数字来自 [M\* 论文](https://arxiv.org/abs/2606.12688)：Qwen3-Omni TTS，2×H200，B=16 时 M\* 吞吐是 vLLM-Omni 的 2.7×、SGLang-Omni 的 4.0×，归因于 "Both vLLM-Omni and SGLang-Omni require separate processes for each stage"，而 M\* 把整个 talker（含 MTP 循环）捕获成一张图、talker 和 code2wav 同进程。被测的 SGLang-Omni 版本很可能早于 #2374（code2wav 并入 talker 进程，2026-09），这个差距现在未必成立。

## 面试追问

::: details Q：omni / TTS serving 为什么是 host-bound，LLM serving 不是？
三个原因叠加。第一，模型小（0.6–4B），每步 GPU 计算只有几毫秒。第二，batch 被首包延迟卡住，不能像 LLM 那样把 batch 做到几百。第三，每步 host 活多：多码本采样、码本拼装、vocoder 调度、chunk 组装、流式 HTTP。LLM 的大模型、大 batch 让每步 GPU 时间远大于 host 时间，语音反过来。证据是 #907：Qwen3-ASR 单副本 GPU 94.3% 时间空闲，降 SM 频率到 0.455× 吞吐只掉 10%，限 CPU 到 1/4 吞吐掉 69%。所以优化方向全是砍 host：CUDA graph 包住帧内循环、lookahead 让 CPU 收尾和下一步 GPU 重叠、合并 D2H、拆进程躲 GIL、同卡多副本。
:::

::: details Q：为什么同卡起多个副本 + MPS 对 TTS 有用，对 LLM 是反模式？
对 LLM，加大 batch 就能把 GEMM 推到 SM 饱和，多进程只会争用，而且权重太大放不下多份。TTS 的瓶颈是单进程的 host 链（调度、采样、vocoder、HTTP 都串行在一个解释器里），GPU 只有约 29% 在忙。多起一个副本就多一条独立的 host 链；没有 MPS 时多进程只能时间片轮转，开了 MPS 几个进程的小 kernel 能同时占 SM。Higgs 实测 DP2 1.4–1.7×、DP3 1.8–2.1×。条件是单副本在延迟 SLO 下 SM-active ≲ 60%，且 N 份权重和 KV 放得下。坑是 `mem_fraction_static` 按剩余显存算，顺序启动的副本 KV 不一样大，要用绝对 token 数或字节数。
:::

::: details Q：Qwen3-Omni 的 code predictor 为什么不拆成独立 stage？thinker 和 talker 之间传什么？
code predictor 和 talker 每一帧都要互相等：talker 出第 0 层码本，predictor 补齐其余 15 层，16 层码本的 embedding 之和又是 talker 下一步的输入。拆成两个 stage 就是每帧一次跨 stage 往返，所以放在同一个 `talker_ar` stage 里，配成独立 stage 会报错。thinker → talker 现在只传 token id（8 字节，inline 在控制消息里）：talker 用自己的 text embedding 和 projection 重建输入，以前传的 hidden state 实际没人读，#2175 删掉了。
:::

::: details Q：stage 之间的张量怎么传？为什么不全用 CUDA IPC？
控制面 ZMQ 只传小消息，张量走数据面，路由器按拓扑自动选：同进程传 Python 引用（零拷贝）、同卡跨进程 direct CUDA IPC、同节点跨卡 pooled CUDA IPC（slot 池，P2P / NVLink 拷贝后 ACK）、CPU 数据走 SHM、跨节点走 Mooncake；≤16 KiB 的 CPU 张量直接 inline 进控制消息。不全用 CUDA IPC 是因为接收侧每次 `cudaIpcOpenMemHandle` 有固定开销，还会和同卡计算争用：258 KB 的 audio encoder 输出走 IPC 平均 13.4 ms，走 SHM 只要 1.26 ms。所以 IPC 只给图像、视频这类大张量用。
:::

::: details Q：流式语音怎么把首包做小，又不卡顿？
首包看链路上每一级能多早开始：talker partial start（thinker 出 5 个 token 就开始）、codec 前 10 帧逐帧发、首块 ramp（1 → 2 → 4 帧）、vocoder 首窗优先且同步解码。不卡顿看稳态：后续块要宽，摊薄 vocoder 的固定开销；有状态增量 codec 不重算左上下文；异步 D2H 不阻塞调度线程；按播放 deadline 排序。指标上要同时看 TTFA 和 underrun（c50），只看吞吐和 RTF 会漏掉卡顿。首块小和后续块宽是一对 trade-off：Qwen3-TTS 增量 codec 让 c8 TTFA p95 1.342 → 0.249 s，但吞吐 −14%。
:::

::: details Q：语音场景的 prefix / radix cache 有什么坑？
三个。一是占位 token：图片、音频在 prompt 里展开成一串相同的占位 id，只按 token id 建 key 会让两段不同音频命中同一 KV，等于把一个用户的声音漏给另一个用户；要用内容 hash 做 pad value，或者把所有影响 KV 的输入拌进 `extra_key`。二是多码本生成行：MOSS-TTS-Local 每帧追加的 radix key 要 hash 整行 13 个通道，原来在 CPU 上算就要每帧同步一次，改成 GPU 上的 Triton hash。三是参考音色：参考音频之间没有前缀关系，用精确匹配的 LRU + single-flight，而不是 radix。
:::

## 参考

- [GitHub · sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)（README、`docs/design/refactor_rfc.md`、`docs/developer_reference/*`、`docs/basic_usage/mps_dp.md`）· [文档](https://sgl-project.github.io/sglang-omni/) · [Releases](https://github.com/sgl-project/sglang-omni/releases)
- [LMSYS：Higgs Audio v3 TTS on SGLang-Omni（2026-06-04）](https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)
- [LMSYS：MOSS-TTS Local v1.5（2026-06-17）](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15/)
- [Awesome-ML-SYS-Tutorial · sglang-omni](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/sglang-omni)（TTS 优化长文 `tts-optimization.md`、TTS 架构重构 `tts-refactor.md`）
- [Issue #907：host-bound profiling](https://github.com/sgl-project/sglang-omni/issues/907) · [Issue #1018：Qwen3-Omni 性能 RFC](https://github.com/sgl-project/sglang-omni/issues/1018)
- [M\*: A Modular, Extensible, Serving System for Multimodal Models](https://arxiv.org/abs/2606.12688)
- [vLLM-Omni 论文](https://arxiv.org/abs/2602.02204) · [vLLM-Omni 技术报告](https://arxiv.org/abs/2610.09307)
- 数据文件：`src/data/sglang-omni-releases.json`，追加新版本只改它

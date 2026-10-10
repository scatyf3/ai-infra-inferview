---
title: 全双工语音服务：会话、打断与容量
status: draft
tags: [omni, full-duplex, barge-in, vad, realtime, session]
difficulty: 4
order: 13
related: [/inference/omni-serving, /inference/speech-output, /inference/multimodal-encoder, /inference/metrics-benchmark]
stack: []
---

# 全双工语音服务：会话、打断与容量

> 从「一问一答的请求」变成「一直开着的会话」之后，serving 系统多了哪些事：每个时间步都要出声、用户插话时怎么取消和截断历史、会话怎么准入和回收、一张卡能撑多少路

## 一句话结论

半双工（按轮次）的语音服务，本质还是「请求进、音频流出」；全双工的模型一直在听、一直在出（说话或静音），serving 的单位从**请求**变成**会话**。三件事因此变难：

1. **实时步长**：Moshi 每 80 ms 必须出一帧，所有并发会话一起跑的一个 batch 步必须在 80 ms 内完成。
2. **打断**：用户插话时要取消正在生成的回答，并且**把对话历史截断到用户实际听到的位置**，而不是生成到的位置。生成通常比播放快一倍，两者能差好几秒。
3. **会话管理**：会话长时间占着 KV 和显存，要有准入上限、空闲回收、断线重连。

## 推导

### 1. 三种交互形态

| 形态 | 例子 | 怎么判断「轮到谁说」 | 一步的时长 |
|---|---|---|---|
| 级联：STT → LLM → TTS | Kyutai Unmute | VAD（语音活动检测）判断用户说完 | 各模块各自的 |
| 端到端、按轮次 | Qwen3-Omni 的 `/v1/realtime` | VAD 判断用户说完，再生成整轮回答 | |
| 端到端、全双工 | Moshi、MiniCPM-o 4.5 | 模型自己决定每一步说还是不说 | Moshi 80 ms；MiniCPM-o 4.5 1 s |

vLLM-Omni 的技术报告明确说，Qwen3-Omni 的 `/v1/realtime` 是「实时聊天 API，不是全双工」：一轮一轮来，没有播放确认，也不处理双方重叠说话（[arXiv 2610.09307](https://arxiv.org/abs/2610.09307) §4.2.6）。

**全双工在操作上意味着什么**：模型按固定节拍运行，每一拍都吃进这一拍的用户音频、吐出这一拍的输出，用户没说话时它也在跑。[Moshi](https://arxiv.org/abs/2410.00037) 的原话是「一直在听，一直在出声，要么是话，要么是静音」。它每一步（80 ms）生成 17 个 token：1 个文本 token（inner monologue，多数是 PAD）、自己的 8 个码本、用户的 8 个码本。MiniCPM-o 4.5 节拍粗一些，每秒决定一次说不说。

### 2. 实时步长：一个 batch 步必须在一拍之内

设节拍 $\Delta$（Moshi 是 80 ms），$B$ 个会话一起跑一个 batch 步的耗时是 $t(B)$。要实时就得

$$
t(B) \le \Delta
$$

能撑的并发就是满足这个式子的最大 $B$。和文本 serving 的区别：文本里 TPOT 超了只是慢一点；这里超了，播放就会断，用户的音频也会积压。

**估一下量级**（示意，忽略 KV 和 depth transformer）：Moshi 的主干是 7B，bf16 权重 14 GB；decode 是 memory-bound，H100 带宽 3.35 TB/s，读一遍权重约 4.2 ms，远小于 80 ms。所以单看主干，一个 batch 步里能塞下很多会话。真正吃时间的是：

1. 每帧还要在 depth transformer 上串行走 8 步（每个码本一步），每步算得很少，kernel 启动开销占大头，要靠 CUDA graph。
2. 每个会话的 KV 随对话变长。Moshi 的上下文 4096 步，按 12.5 步/秒只够 $4096 / 12.5 = 328$ 秒，约 5.5 分钟。

**公开的容量数字**（作者自测）：

| 系统 | 硬件 | 并发 |
|---|---|---|
| Kyutai DSM-ASR（流式语音识别） | 1 × H100 | 约 400 路实时流 |
| Kyutai Unmute（级联语音对话） | 1 × L40S | 64 路，速度是实时的 3 倍 |
| vLLM-Omni 部署 MiniCPM-o 4.5 全双工 | 1 张卡 | 配置上限 16 个会话（不是实测容量） |

**注意 RTF 的方向**：Qwen3-Omni 和 vLLM-Omni 的 RTF 是「生成耗时 ÷ 音频时长」，越小越好，必须小于 1；Kyutai 的论文里写的「RTF」是「音频时长 ÷ 耗时」，越大越好，必须大于 1。DSM-ASR 在 batch 64 时「RTF 3.5」，意思是比实时快 3.5 倍，吞吐 = 3.5 × 64 = 224 路实时流的量。读数字前先看定义。

### 3. 打断：取消、截断、防串轮

用户在助手说话时插话（barge-in），要做这几件事，以 [OpenAI Realtime API](https://developers.openai.com/api/reference/resources/realtime/client-events) 的事件为例：

1. **检测到用户开口**：服务端 VAD 发出 `input_audio_buffer.speech_started`。默认参数：能量阈值 0.5，说话前补 300 ms 的音频（`prefix_padding_ms`），静音 500 ms 判为说完（`silence_duration_ms`）。
2. **停止播放**：客户端立刻停下正在播的音频。
3. **取消生成**：`response.cancel`，服务端返回状态为 cancelled 的 `response.done`。
4. **截断历史**：`conversation.item.truncate`，参数 `audio_end_ms` 是**实际播放到的毫秒数**。服务端把这条助手消息的音频截到这里，并删掉用户没听到的那部分文字。

**为什么第 4 步不能省**：生成比播放快。设 RTF = 0.47，用户在播放到 1.5 s 时插话，这时已经生成了约 $1.5 / 0.47 = 3.2$ 秒的音频（示意，忽略首包延迟）。如果不截断，对话历史里助手「说过」3.2 秒的内容，用户只听到 1.5 秒，下一轮模型会以为用户听过后面那半段。文字也要按比例截：这条回答的文字有 60 个 token、音频共 6 秒，播放到 1.5 秒，保留约 $60 \times 1.5 / 6 = 15$ 个 token。vLLM-Omni 就是按比例截文字的，不按词对齐。

**谁知道播放到哪了**：WebSocket 接入时播放在客户端，只有客户端知道，要由客户端发 truncate；WebRTC / SIP 接入时服务端自己跟踪播放进度，会自动截掉没播的部分。vLLM-Omni 的全双工 API 加了一个 `playback.ack` 事件，客户端定期报告 `played_ms`。

**防串轮（epoch fence）**：取消之后，上一轮的生成可能还有几个分片在路上（各 stage 之间是流水的）。vLLM-Omni 给每个会话维护一个 epoch 号，每次取消、清空、打断都加一，所有输出和命令都带着 epoch，旧 epoch 的直接丢弃，迟到的旧分片就不会混进新回答里。

**全双工模型自己判断要不要停**：Freeze-Omni 在每个输入 chunk 的最后一帧上接一个三分类的状态头：0 继续听；1 用户打断了，开始新的生成；2 用户说完了。它报告从打断到出第一段音频平均 745 ms（P90 1020 ms），加上网络等实际约 1.2 s（[arXiv 2411.00774](https://arxiv.org/abs/2411.00774)，作者自测）。vLLM-Omni 把这个决策做成 `overlap.decision` 事件，取值 `drop` / `listen` / `barge_in`；能不能打断取决于模型，MiniCPM-o 4.5 支持，有的模型不支持。

### 4. 流式输入：边听边 prefill

用户还在说时，音频按 chunk 送进来。要边听边算，编码器就不能对整段音频做全局 attention：

1. Qwen2.5-Omni 的音频编码器按 **2 秒一块**做 attention，每来一块编码一块，接着 chunked prefill 进 Thinker。
2. Qwen3-Omni 的 AuT 用 **1 到 8 秒的动态窗口**，在实时性和离线任务的效果之间折中。
3. 级联方案里 STT 本身就是流式的：Kyutai 的 STT 有 0.5 s 和 2.5 s 两种延迟的模型。

用户说完的那一刻，大部分音频已经编码、prefill 完了，首包延迟里就只剩最后一块。这是把 [首包延迟](/inference/speech-output) 里「预处理 + 编码」那一项压下去的主要办法。

### 5. 会话管理

会话是长连接，一直占着 KV 和显存，所以：

| 问题 | vLLM-Omni 的做法 |
|---|---|
| 准入 | 部署时配一个会话上限（`duplex_session.max_sessions`），超了直接拒绝（`resource_exhausted`） |
| 空闲回收 | 空闲超时（示例配置 300 s）后回收 |
| 断线 | 留 30 s 的重连窗口，客户端凭 `resume_token` 和最后收到的事件序号续上 |
| 输出积压 | 每个会话未发出的输出最多 2 MiB、512 个事件 |
| 路由 | 会话绑定到某个副本（sticky），KV 留在本地 |

MiniCPM-o 4.5 的示例部署把三个 stage 放在同一张卡上，按显存比例切：LLM 55%、Talker 15%、Code2Wav 18%，会话上限 16。

## 面试追问

::: details Q：全双工和「VAD + 按轮次」有什么本质区别？
按轮次时，什么时候轮到模型说由外部的 VAD 决定，模型只在用户说完后生成一整轮。全双工时模型按固定节拍一直在跑，每一拍都同时吃进用户的音频、输出自己的音频（可能是静音），什么时候说、什么时候停由模型自己决定，所以能接话、能被打断、能边听边应声。代价是 serving 上每个会话每一拍都要算一次，没人说话时也在占算力。
:::

::: details Q：用户打断时，为什么要按「播放到哪」截断历史，而不是「生成到哪」？
生成比播放快：RTF 0.47 时，播放 1.5 秒的时候已经生成了约 3 秒。没播出来的那部分用户没听到，如果留在历史里，下一轮模型会以为用户听过，回答就接不上。OpenAI Realtime API 的 `conversation.item.truncate` 带 `audio_end_ms`，截掉没播的音频，同时删掉对应的文字。
:::

::: details Q：一张卡能撑多少路全双工会话，怎么估？
先定节拍 $\Delta$（Moshi 80 ms），再测 batch 为 $B$ 时一步的耗时 $t(B)$，满足 $t(B) \le \Delta$ 的最大 $B$ 就是上限，还要留出余量给抖动。主干 decode 是 memory-bound，读一遍 7B 权重约 4 ms，不是瓶颈；瓶颈常在每帧内串行的小模型步（depth transformer、MTP）和随对话增长的 KV。另外每个会话长时间占显存，所以还要按显存定一个会话上限。
:::

::: details Q：看到两个系统的 RTF，一个 0.5、一个 3，哪个快？
先看定义。Qwen / vLLM-Omni 的 RTF 是耗时 ÷ 音频时长，0.5 表示比实时快一倍；Kyutai 的论文写的是音频时长 ÷ 耗时，3 表示比实时快 3 倍。换成同一个定义再比。
:::

## 手撕

**打断处理**（服务端，一个会话）：

```python
class Session:
    def __init__(self):
        self.epoch = 0
        self.history = []                  # [(role, text, audio_ms)]

    def on_output(self, chunk):
        if chunk.epoch != self.epoch:      # 上一轮迟到的分片，丢掉
            return
        send_to_client(chunk)

    def on_barge_in(self, played_ms):
        self.epoch += 1                    # 之后旧 epoch 的输出一律作废
        cancel_generation()                # 各 stage 停掉这一轮
        role, text, audio_ms = self.history[-1]
        assert role == "assistant"
        keep = played_ms / audio_ms        # 按播放比例截文字
        tokens = tokenize(text)
        self.history[-1] = (role, detokenize(tokens[: int(len(tokens) * keep)]), played_ms)
```

**容量估算**：

```python
def max_sessions(step_ms, tick_ms=80, headroom=0.8):
    # step_ms: 函数 B -> 一个 batch 步的耗时（ms），实测得到
    b = 1
    while step_ms(b + 1) <= tick_ms * headroom:
        b += 1
    return b

def generated_ahead_s(played_s, rtf):
    return played_s / rtf                  # 播放到 played_s 时已生成的音频秒数（忽略首包）
```

常见题：解释全双工和按轮次的区别；设计打断的处理流程（检测、取消、截断、防串轮）；给定节拍和 batch 步耗时估算会话容量；比较不同来源的 RTF 数字。

## 参考

- [vLLM-Omni 技术报告](https://arxiv.org/abs/2610.09307) · [vLLM-Omni 全双工 API 文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/full_duplex_api/) · [vLLM-Omni 论文](https://arxiv.org/abs/2602.02204)
- [Moshi](https://arxiv.org/abs/2410.00037) · [Moshi GitHub](https://github.com/kyutai-labs/moshi)
- [Kyutai DSM（Delayed Streams Modeling）](https://arxiv.org/abs/2509.08753) · [Kyutai STT](https://kyutai.org/stt) · [Unmute](https://github.com/kyutai-labs/unmute)
- [Freeze-Omni](https://arxiv.org/abs/2411.00774)
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)
- [OpenAI Realtime API：client events](https://developers.openai.com/api/reference/resources/realtime/client-events) · [VAD 指南](https://developers.openai.com/api/docs/guides/realtime-vad)
- [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) · [Qwen3-Omni](https://arxiv.org/abs/2509.17765)

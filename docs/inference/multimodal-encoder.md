---
title: 多模态输入：一张图、一段视频、一段音频变成多少 token
status: draft
tags: [multimodal, vit, encoder, m-rope, epd, omni]
difficulty: 3
order: 11
related: [/inference/omni-serving, /inference/speech-output, /inference/batching-scheduling, /inference/kv-cache-paged-attention, /inference/prefill-decode-roofline]
stack: []
---

# 多模态输入：一张图、一段视频、一段音频变成多少 token

> 多模态请求的显存和延迟账，都从「这段输入变成了多少 token」算起。再算编码器本身的 FLOPs、位置编码怎么给，最后看 serving 系统怎么调度、缓存、拆分编码器

## 一句话结论

图像按 patch 切开，Qwen2-VL / 2.5-VL 每 $28 \times 28$ 像素一个 token，一张 1080p 图约 2.7k token；视频每两帧合成一组，2 fps 下一分钟 360p 视频约 1.8 万 token；音频每 40 ms 一个 token（Qwen2.5-Omni）或每 80 ms 一个（Qwen3-Omni），一分钟 750–1500 个。编码器（ViT）参数不大，但 token 多：一张 1080p 图过一遍 632M 参数的 ViT 约 16 TFLOP，是 7B LLM 对同样 token 做 prefill 的 40%。serving 上的要点：编码器是双向 attention，一张图不能切开分步算；按内容 hash 缓存编码结果；prefix caching 的 block hash 要拌进图像的 hash；图像多时把编码器拆到单独的卡上（EPD）。

## 推导

### 1. 图像：每 28 × 28 像素一个 token

以 [Qwen2-VL](https://arxiv.org/abs/2409.12191) / [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) 为例：

1. ViT 把图切成 $14 \times 14$ 的 patch，每个 patch 一个向量。
2. ViT 之后，把相邻的 $2 \times 2$ 个 patch 拼起来、过一个 MLP，压成一个 token 送进 LLM。

所以每个 LLM token 对应 $28 \times 28$ 像素。图先缩放到长宽都是 28 的倍数（各自取最近的倍数），token 数是

$$
N_{\text{img}} = \frac{H'}{28} \cdot \frac{W'}{28}
$$

$H', W'$ 是缩放后的高和宽。总像素被限制在 `min_pixels` 到 `max_pixels` 之间，默认是 $4 \times 28^2$ 到 $16384 \times 28^2$，即每张图 4 到 16384 个 token；超出就按比例缩小。前后再加 `<|vision_start|>`、`<|vision_end|>` 两个特殊 token。

| 输入 | 缩放后 | token 数 |
|---|---|---|
| 224 × 224 | 224 × 224 | 64（论文写 66，含两个特殊 token） |
| 640 × 360 | 644 × 364 | 299 |
| 1280 × 720 | 1288 × 728 | 1196 |
| 1920 × 1080 | 1932 × 1092 | 2691 |
| 3840 × 2160 | 3836 × 2156 | 10549 |

对比其他做法：

1. **LLaVA-1.5**：固定缩放到 $336 \times 336$，CLIP ViT-L/14 切出 $(336/14)^2 = 576$ 个 token，不管原图多大。
2. **LLaVA-NeXT**：把图切成几个 336 的 tile 再加一张缩略图。一张方图映射到 $672 \times 672$ 时是 2304 + 576 + 48（每行一个换行 token）= 2928 个 token。常说的「2880」是 $5 \times 576$，没算换行 token。
3. **Qwen3-VL**：patch 改成 16，合并后每 $32 \times 32$ 像素一个 token，1080p 是 $60 \times 34 = 2040$ 个。

动态分辨率的好处是小图少花 token、大图保留细节；代价是 token 数随输入变化很大，serving 系统没法按固定长度预留资源。

### 2. 视频：两帧一组

1. 每秒采 2 帧（Qwen2-VL 的默认）。
2. 相邻两帧合成一组：ViT 的 patch 在时间维上也取 2（3D patch，$2 \times 14 \times 14$）。单张图被当成两帧相同的图。
3. 每组的 token 数和一张图一样，按分辨率算。

所以 2 fps 下**每秒视频约等于一帧的 token 数**。例：360p（$640 \times 360$）每组 299 个 token，一分钟 60 组，共 **17,940** 个 token。各模型都给视频设了上限：Qwen2-VL 每段视频最多 16384 个 token；Qwen2.5-VL 评测时最多 768 帧、24576 个 token。帧多了就只能降分辨率或降帧率。

**时间怎么告诉模型**：

1. Qwen2.5-VL：位置编码的时间分量和**绝对时间**对齐，帧率不同的视频，同一秒对应的时间 ID 间隔相同。
2. Qwen3-VL：改成在每组前面插一段文本时间戳，比如 `<3.0 seconds>`。理由是长视频的时间 ID 会变得又大又稀疏。代价是多了几个文本 token。

### 3. 音频：每 40 ms 或 80 ms 一个 token

Whisper 系编码器的一条链：

1. 16 kHz 采样，每 10 ms 算一帧 mel 频谱（窗口 25 ms）：**100 帧/秒**。
2. 编码器开头两层卷积，第二层 stride 2：**50 帧/秒**（20 ms 一帧）。Whisper 一次处理 30 s，正好 1500 帧。
3. Qwen2-Audio / Qwen2.5-Omni 再加一个 stride 2 的池化：**25 token/s**（40 ms 一个）。
4. Qwen3-Omni 的 AuT 编码器用卷积下采样 8 倍：**12.5 token/s**（80 ms 一个）。

| 编码器 | token/s | 30 s | 1 分钟 |
|---|---|---|---|
| Whisper（无池化） | 50 | 1500 | 3000 |
| Qwen2-Audio / Qwen2.5-Omni | 25 | 750 | 1500 |
| Qwen3-Omni AuT | 12.5 | 375 | 750 |

token 率减半，同样长的音频 prefill 和 KV 都减半。语音输出那一侧的 token 率见 [语音输出](/inference/speech-output)。

### 4. 位置编码：M-RoPE

文本的 RoPE 只有一个位置 ID。图像有高和宽，视频再加时间，[Qwen2-VL](https://arxiv.org/abs/2409.12191) 的 M-RoPE 把 RoPE 的频率维分成三段，分别用时间、高、宽三个 ID 旋转。Qwen2-VL-7B 的 head 维度 128，有 64 对频率，按 `mrope_section = [16, 24, 24]` 分给时间、高、宽。

规则：

1. 文本 token：三个 ID 相同，退化成普通 RoPE。
2. 图像 token：时间 ID 都一样，高、宽 ID 按它在网格里的行、列。
3. 视频 token：时间 ID 按第几组。
4. 图像之后的下一个文本 token，从「前面所有 ID 的最大值 + 1」接着编。

**例子**：3 个文本 token，接一张合并后 $2 \times 2$ 的图（4 个 token），再接 1 个文本 token。

| token | 时间 | 高 | 宽 |
|---|---|---|---|
| 文本 0, 1, 2 | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 |
| 图像 (0,0) (0,1) (1,0) (1,1) | 3, 3, 3, 3 | 3, 3, 4, 4 | 3, 4, 3, 4 |
| 文本 | 5 | 5 | 5 |

图像占了 4 个 token，但位置 ID 只往前走了 2（从 3 到 4）。所以多模态序列的位置 ID 比 token 数小，KV 的长度仍按 token 数算。

后续的变化：Qwen2.5-Omni 的 TMRoPE 规定一个时间 ID 对应 40 ms，让音频和视频帧在时间上对齐（Qwen3-Omni 是 80 ms）；Qwen3-VL 把三段改成在频率维上交错排列（`[24, 20, 20]`，交错），因为原来的分段让某个轴只拿到高频或只拿到低频。

### 5. 编码器的算力账

ViT 的前向和 prefill 一样是 compute-bound，按 [Roofline](/inference/prefill-decode-roofline) 的规则数：线性层每个 patch $2P$ FLOP，attention 每层 $4N^2 d$（$N$ 个 patch，$d$ 是 hidden）。

Qwen2.5-VL 的 ViT（3B、7B、72B 共用同一个）：32 层，hidden 1280，主体约 632M 参数，加上合并用的 MLP 共约 675M。为了控制 attention 的 $N^2$，32 层里只有 4 层是全局 attention，其余 28 层只在 $112 \times 112$ 像素（$8 \times 8 = 64$ 个 patch）的窗口里做。

**算一张 1920 × 1080 的图**：合并前有 $2691 \times 4 = 10764$ 个 patch。

1. 线性层：$2 \times 632\text{M} \times 10764 = 13.6$ TFLOP。
2. 4 层全局 attention：$4 \times 4 \times 10764^2 \times 1280 = 2.4$ TFLOP。
3. 28 层窗口 attention：每个 patch 只看 64 个，$28 \times 4 \times 10764 \times 64 \times 1280 = 0.1$ TFLOP。
4. 合计约 **16 TFLOP**。

对比 LLM 对这 2691 个 token 做 prefill（$2P$ 每 token）：

| LLM | prefill FLOPs | ViT 占比 |
|---|---|---|
| 7.6B | 41 TFLOP | 约 40% |
| 72B | 388 TFLOP | 约 4% |

ViT 大小不随 LLM 变，所以小模型上编码器是不可忽略的一大块。按 H100 bf16 峰值 989 TFLOP/s、50% 利用率算，编码这张图约 32 ms。如果 28 层也用全局 attention，attention 那项会变成 $32 \times 4 \times 10764^2 \times 1280 = 19$ TFLOP，比线性层还大。

### 6. Serving 上的几件事

**(a) 一张图不能切开算**。chunked prefill 可以把一段长 prompt 切成几步算，因为 LLM 是因果 attention，前面的 token 不依赖后面的。ViT 是双向 attention，一张图的所有 patch 必须一起算。vLLM 的调度器每步有一个编码器预算（等于 `max_num_batched_tokens`，并保证至少放得下最大的一张图）：这一步要算到某张图的占位 token 时，编码器预算够就把整张图编码；不够就只调度到这张图之前的文本 token 为止。

**(b) 编码结果缓存**。按图像内容的 hash 缓存编码器输出，多轮对话里反复带着的同一张图只编码一次，多个请求也能共享。vLLM 里还有一层预处理缓存（`mm_processor_cache_gb`，默认 4 GiB），缓存的是缩放、切 patch 之后的输入，不是编码结果。

**(c) prefix caching 要拌进图像的 hash**。图像在 prompt 里展开成一串相同的占位 token，只看 token id 的话两张不同的图会算出相同的 block hash，错误命中。vLLM 给每个和图像重叠的 block 的 hash 加上 `("mm", 图像 hash, 偏移)` 这个额外的 key，见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)。

**(d) ViT 用数据并行**。ViT 只有几亿参数，对它做 TP，每层的 all-reduce 不划算。vLLM 的 `mm_encoder_tp_mode="data"` 让每张卡各自跑一份完整的 ViT、分不同的图，LLM 照样用 TP。

**(e) EPD 分离**。PD 分离把 prefill 和 decode 拆开（见 [Continuous Batching 与 PD 分离](/inference/batching-scheduling)），EPD 再把编码（Encode）拆到单独的卡上，编码结果通过网络传给 prefill 实例：

1. 好处：一张大图的编码不再卡住同一组卡上其他请求的 decode；编码器可以单独扩缩容。
2. [EPD 论文](https://arxiv.org/abs/2501.05460)（作者自测）：TTFT 最多降 71%，batch 最多大 22 倍。
3. SGLang 的测试（作者自测，Qwen3-VL-235B，8 张 H20，每个请求约 4 张 1080p 图）：1 QPS 下延迟比不拆低 6–8 倍；博客也提醒，图像少的负载拆开反而会增加 TTFT，因为多了一次传输。
4. 开关：vLLM 是 `--ec-transfer-config`，SGLang 是 `--encoder-only` / `--language-only` / `--encoder-urls`。

## 面试追问

::: details Q：一张 1080p 图在 Qwen2.5-VL 里变成多少 token？怎么算的？
每 $28 \times 28$ 像素一个 token：ViT 的 patch 是 14，之后 $2 \times 2$ 合并。1920 和 1080 各自缩放到最近的 28 的倍数（1932、1092），$69 \times 39 = 2691$ 个，再加两个特殊 token。总像素超过上限（默认 $16384 \times 28^2$）会先按比例缩小。
:::

::: details Q：多模态请求的 chunked prefill 有什么特殊之处？
图像的占位 token 可以和文本一起被切进不同的 step，但编码器本身不能切：ViT 是双向 attention，一张图的 patch 要一起算。所以调度器在要处理某张图的 token 时，必须一次给够编码这张图的预算；预算不够，这一步就只处理到图像前面的文本为止。编码结果缓存起来，后面几步 prefill 直接用。
:::

::: details Q：为什么小模型上编码器更显得重？
ViT 大小是固定的（Qwen2.5-VL 的 3B、7B、72B 共用约 675M 的 ViT），而且它看的是合并前的 patch，数量是 LLM token 的 4 倍。一张 1080p 图，ViT 约 16 TFLOP；7.6B 的 LLM 对同样的 2691 个 token 做 prefill 约 41 TFLOP，编码器占了四成；换成 72B，只占 4%。
:::

::: details Q：EPD 分离什么时候值得做？
图像多、图像大时：编码耗时长，和 decode 抢同一组卡会拉高其他请求的 TPOT，拆开后两边互不干扰，还能单独扩缩编码器。图像少的负载反而不划算：每个请求多一次跨机传编码结果，TTFT 可能变长。
:::

## 手撕

**算图像、视频、音频的 token 数**（Qwen2.5-VL / Qwen2.5-Omni 的规则）：

```python
import math

def smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:                       # 太大：按比例缩小
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:                     # 太小：按比例放大
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb

def image_tokens(h, w):
    hb, wb = smart_resize(h, w)
    return (hb // 28) * (wb // 28)

def video_tokens(h, w, seconds, fps=2):
    groups = math.ceil(seconds * fps / 2)          # 两帧一组
    return groups * image_tokens(h, w)

def audio_tokens(seconds, hz=25):                  # Qwen2.5-Omni 25 Hz，Qwen3-Omni 12.5 Hz
    return math.ceil(seconds * hz)

image_tokens(1080, 1920)        # 2691
video_tokens(360, 640, 60)      # 17940
audio_tokens(60)                # 1500
```

**M-RoPE 的位置 ID**（文本 + 一张图，图已合并成 `gh × gw` 的网格）：

```python
def mrope_ids(n_text_before, gh, gw, n_text_after):
    t, h, w = [], [], []
    for i in range(n_text_before):                 # 文本：三个 ID 相同
        t.append(i); h.append(i); w.append(i)
    s = n_text_before
    for r in range(gh):                            # 图像：时间不变，高宽按网格
        for c in range(gw):
            t.append(s); h.append(s + r); w.append(s + c)
    nxt = max(t + h + w) + 1                       # 接着最大 ID 往后编
    for i in range(n_text_after):
        t.append(nxt + i); h.append(nxt + i); w.append(nxt + i)
    return t, h, w
```

常见题：给一张图的分辨率算 token 数；算一段视频 / 音频的 token 数；解释 M-RoPE 怎么给图像编位置；估算 ViT 编码一张图的 FLOPs 并和 LLM prefill 比较。

## 参考

- [Qwen2-VL](https://arxiv.org/abs/2409.12191) · [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) · [Qwen3-VL](https://arxiv.org/abs/2511.21631)
- [Whisper](https://arxiv.org/abs/2212.04356) · [Qwen2-Audio](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) · [Qwen3-Omni](https://arxiv.org/abs/2509.17765)
- [LLaVA-1.5](https://arxiv.org/abs/2310.03744) · [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- [EPD Disaggregation](https://arxiv.org/abs/2501.05460) · [vLLM disaggregated encoder](https://docs.vllm.ai/en/latest/features/disagg_encoder.html) · [SGLang EPD 博客](https://lmsys.org/blog/2026-01-12-epd)

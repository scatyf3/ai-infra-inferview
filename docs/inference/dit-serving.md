---
title: 图像与视频生成：DiT 推理与 Serving
status: draft
tags: [omni, diffusion, dit, video-generation, teacache, sequence-parallel]
difficulty: 4
order: 14
related: [/inference/omni-serving, /inference/prefill-decode-roofline, /inference/flash-attention, /parallel/parallelism-overview]
stack: []
---

# 图像与视频生成：DiT 推理与 Serving

> any-to-any 模型的输出侧还有图像和视频。它们由扩散 transformer（DiT）生成：一张图、一段视频变成多少 token，一次生成要多少 FLOPs，为什么和 LLM serving 完全不是一回事，以及少走几步、步间缓存、序列并行这些加速手段

## 一句话结论

DiT 生成一段视频，相当于对**几万个 token 做一次没有因果 mask 的 prefill**，再重复「采样步数 × 2（CFG）」次。Wan2.1-14B 生成 720p、5 秒的视频，序列有 75,600 个 token，一次前向约 6.5 PFLOP，其中 attention 占七成；50 步、每步两次前向，共约 0.65 EFLOP，一张 H100 按 50% 利用率要 20 多分钟。它是纯 compute-bound 的，没有 KV cache、没有 decode。所以加速手段和 LLM 不同：**少走几步**（步数蒸馏）、**相邻步之间复用计算**（TeaCache 等缓存）、**把一个请求拆到多张卡上**（CFG 并行、序列并行）、**让 attention 更便宜**（稀疏、量化）。serving 上，形状不同的请求很难拼进一个 batch。

## 推导

### 1. 从像素到 token：VAE 压缩 + patchify

DiT 不直接处理像素：

1. **VAE** 把图像 / 视频压成 latent。图像通常空间上缩小 8 倍；视频 VAE 时间上再缩小 4 倍（第一帧只做空间压缩）。latent 有 16 个通道。
2. **patchify** 把 latent 里相邻的 $2 \times 2$ 合成一个 token。

所以空间上每 $16 \times 16$ 像素一个 token。

**FLUX.1，1024 × 1024 的图**：latent $128 \times 128$，patchify 后 $64 \times 64 =$ **4096** 个图像 token，再和最多 512 个文本 token 一起做 attention，共约 4608 个。

**Wan2.1，视频**（[arXiv 2503.20314](https://arxiv.org/abs/2503.20314) §4.1–4.2）：$T$ 帧、$H \times W$ 的视频，token 数是

$$
L = \Big(1 + \frac{T - 1}{4}\Big) \cdot \frac{H}{16} \cdot \frac{W}{16}
$$

| 分辨率 | 帧数（16 fps 约 5 秒） | token 数 |
|---|---|---|
| 832 × 480 | 81 | $21 \times 30 \times 52 = 32{,}760$ |
| 1280 × 720 | 81 | $21 \times 45 \times 80 =$ **75,600** |

Wan2.2 的 5B 模型换了压缩更狠的 VAE，加上 patchify 后是时间 4 倍、空间 $32 \times 32$，同样的视频 token 数少到约 1/4。压 VAE 是减少 token 数最直接的办法。

### 2. 一次生成的算力账

DiT 的每一步，是对**所有** token 做一次完整的 transformer 前向，双向 attention、没有 KV cache。按 [Roofline](/inference/prefill-decode-roofline) 的规则数：线性层每个 token $2P$ FLOP，attention 每层 $4L^2 d$。

**Wan2.1-14B，720p 视频**（$d = 5120$，FFN 13824，40 层，$L = 75{,}600$）：

1. 每层每个 token 碰到的权重：self-attention $4d^2$、cross-attention 的 Q 和 O $2d^2$、FFN $2 \cdot d \cdot 13824$，共约 299M；40 层约 12B。
2. 线性层：$2 \times 12\text{B} \times 75{,}600 = 1.8$ PFLOP。
3. self-attention：$4 \times 75{,}600^2 \times 5120 \times 40 = 4.7$ PFLOP。
4. 和 512 个文本 token 的 cross-attention：约 0.03 PFLOP，可以忽略。
5. 一次前向约 **6.5 PFLOP**，attention 占 72%。

一次生成：50 步 × 2（CFG：每步有条件、无条件各算一次）× 6.5 PFLOP ≈ **0.65 EFLOP**。一张 H100（bf16 峰值 989 TFLOP/s）按 50% 利用率算约 **22 分钟**。实测佐证：HunyuanVideo 720p 5 秒（约 11.5 万 token）在 H100 上用 FA3 跑 945 秒，其中 attention 占 800 秒（[STA 论文](https://arxiv.org/abs/2502.04507)，作者自测）。

**FLUX.1-dev，1024² 的图**（12B，约 4608 个 token）：每步约 0.125 PFLOP，attention 只占 12%；diffusers 默认 28 步。FLUX-dev 把引导强度做成了模型的一个输入（guidance distillation），不需要每步多算一次无条件分支，共约 3.5 PFLOP，同样按 50% 利用率算约 7 秒。

**图像和视频的差别**：图像几千个 token，线性层为主；视频几万到十几万个 token，attention 的 $L^2$ 项压倒一切。所以视频生成的加速重点在 attention（稀疏、量化、序列并行）。

### 3. 和 LLM serving 有什么不同

| | LLM | DiT |
|---|---|---|
| 一个请求的计算 | 一次 prefill + 很多步小 decode | 固定步数，每步都是一次完整的大前向 |
| 瓶颈 | decode 是 memory-bound | 全程 compute-bound |
| KV cache | 有，是显存大头 | 没有（每步输入都变了） |
| batch | continuous batching，长短请求随便拼 | 形状、CFG、LoRA 不同的请求很难拼 |
| 延迟 | TTFT 毫秒级、TPOT 几十毫秒 | 秒到几十分钟 |

batch 带来的收益也小：一个视频请求本身就有几万个 token，单个请求已经能把 GPU 算满。所以视频生成的 serving 更关心**怎么把一个请求拆到多张卡上**，而不是怎么拼 batch。

### 4. 加速：少走几步

**步数蒸馏**：训练一个学生模型，用 1–4 步达到老师 50 步的效果。

1. LCM：2–4 步，CFG 也蒸馏进了模型。4 步时 FID 11.1（作者自测）。
2. FLUX.1-schnell：12B，1–4 步，不用 CFG。

步数从 50 降到 4，计算量直接少一个数量级以上；代价是质量和多样性通常有损，而且要专门训练。

### 5. 加速：相邻步之间复用

相邻两个去噪步的输入只差一点噪声，transformer 的输出也很接近。**缓存**就是在输出变化小的步上，直接复用上一步的结果，跳过这一步的大部分计算。

**TeaCache**（[arXiv 2411.19108](https://arxiv.org/abs/2411.19108)）：要判断这一步能不能跳，就得知道输出变了多少，可算输出又正是要省的那部分计算。TeaCache 拿一个便宜的代理量来估：被时间步 embedding 调制后的输入，相对上一步变了多少（相对 L1 距离），再用一个多项式校正成「输出变化」的估计。把每一步的估计值累加起来，超过阈值 $\delta$ 才真正算一次、把累加清零，否则复用缓存。$\delta = 0.1$ 偏质量、0.2 偏速度。作者自测 Open-Sora-Plan 上 4.4 倍加速，画质指标几乎不变。

其他做法：FasterCache 复用 attention 的输出，并缓存「有条件减无条件」的差值来省 CFG 的一半计算（作者自测 1.6–1.7 倍）；cache-dit 把多种缓存策略做成一个库，接进了 vLLM-Omni 和 SGLang。vLLM-Omni 的测试（作者自测，Qwen-Image 1024²，H200）：20.0 s，用 TeaCache 降到 10.5 s（1.91 倍），用 cache-dit 降到 10.8 s（1.85 倍）。

### 6. 加速：一个请求拆到多张卡

1. **CFG 并行**：每步的有条件、无条件两次前向放到两张卡上同时算，每步只需要一次 all-gather 把结果合起来。两张卡，接近两倍。
2. **序列并行**：视频的 $L$ 太大，一张卡算不动 attention，就沿序列维切到多张卡上。
   - Ulysses：用 all-to-all 把「按序列切」换成「按 head 切」，每张卡拿一部分 head、看完整的序列。并行度不能超过 head 数（Wan2.1-14B 有 40 个 head）。
   - Ring：K、V 在卡之间沿环传递，每张卡用自己的 Q 和流过来的 K、V 分块算，靠 [online softmax](/handson/online-softmax) 合并。没有 head 数的限制。
   - USP：两者组合成二维网格。Wan2.1 用外层 Ring、内层 Ulysses，作者报告 256K token 时通信开销从单用 Ulysses 的 10% 以上降到 1% 以下。
3. **PipeFusion**：把图像按 patch 切开流水，用上一步的旧激活补齐没算到的部分（利用的还是相邻步很像），通信量和层数无关。步数很少的蒸馏模型上效果差，因为相邻步差别变大了。

xDiT 把这几种并行组合在一起，作者自测 CogVideoX-5B 在 12 张 L40 上从 5 分多钟降到 52 秒。

### 7. 加速：让 attention 更便宜

视频的 attention 占七八成，所以：

1. **稀疏**：视频里一个位置主要和时空上相邻的位置相关。Sliding Tile Attention 按 3D tile 做滑动窗口，tile 要么全算、要么全跳，对 GPU 友好；HunyuanVideo 上不微调 1.89 倍，微调后 3.53 倍（945 s → 268 s，作者自测）。
2. **量化**：SageAttention 把 Q、K 量化到 INT8（K 先减去均值，压掉离群值），$PV$ 仍用 FP16，作者自测比 FlashAttention2 快约 2.1 倍，端到端指标掉约 0.2%。Wan2.1 的报告里 8-bit FlashAttention 带来 1.27 倍以上，FP8 GEMM 1.13 倍。

### 8. 统一模型：自回归 + 扩散

any-to-any 模型要同时「看懂」和「画出来」，常见的接法：

| 模型 | 理解 | 生成 | serving 上的含义 |
|---|---|---|---|
| [BAGEL](https://arxiv.org/abs/2505.14683) | 理解专家（自回归） | 生成专家（rectified flow，在 VAE latent 上去噪） | 两个专家共享 attention，7B 激活、14B 总参数；生成时只缓存干净 token 的 KV |
| Janus | SigLIP 编码 | VQ token，纯自回归逐个生成 | 生成就是 LLM decode，可以直接用 LLM 引擎 |
| Show-o | 自回归文本 | 离散 token 的 masked diffusion，并行解码 | 一个模型两种 attention 模式 |

vLLM-Omni 把 BAGEL 拆成两个 stage：stage 0 用 vLLM 引擎跑自回归的理解部分，stage 1 用扩散引擎跑去噪，两者之间通过 connector 传 KV cache。作者自测 1024² 下文生图从 23.1 s 降到 9.6 s（2.40 倍），图生图从 41.4 s 降到 11.1 s（3.72 倍）。

### 9. serving 上怎么 batch

vLLM-Omni 的扩散引擎有两种 batching：

1. **请求级**：形状、引导强度、输出张数、LoRA 都一样的请求才能拼成一个 batch，一起从头跑到尾。
2. **步级**（实验性）：不同请求可以处在不同的步、总步数也可以不同，每个调度周期把能拼的请求一起跑一个去噪步，再切回各自的请求。思路和 LLM 的 continuous batching 一样，但形状、CFG、LoRA 仍要兼容。按先来先服务准入，队首有个不兼容的请求就会堵住后面的。

SGLang Diffusion 在 SGLang 的调度器上做了类似的引擎，作者自测比 diffusers 快 1.2–5.9 倍。

## 面试追问

::: details Q：生成一段 720p、5 秒的视频，算力大概多少？
Wan2.1 的 VAE 时间压 4 倍、空间压 8 倍，patchify 再 $2 \times 2$，81 帧 1280 × 720 变成 $21 \times 45 \times 80 = 75{,}600$ 个 token。14B 模型一次前向：线性层 $2 \times 12\text{B} \times 75{,}600 \approx 1.8$ PFLOP，attention $4L^2d \times 40$ 层 $\approx 4.7$ PFLOP，共约 6.5 PFLOP。50 步、CFG 两次前向，约 0.65 EFLOP，一张 H100 按 50% 利用率二十多分钟。
:::

::: details Q：DiT 为什么没有 KV cache？
LLM 能缓存 KV，是因为前面 token 的 K、V 算完就不变了。DiT 每一步去噪都会更新所有 token（整段 latent 都在变），所以每一步的 K、V 都不一样，没有可以复用的。能复用的是「相邻步之间输出很像」这一点，TeaCache 这类方法用的就是它，但那是近似，不像 KV cache 那样精确。
:::

::: details Q：视频生成为什么优先做序列并行，而不是拼更大的 batch？
一个视频请求就有几万到十几万个 token，单个请求已经能把 GPU 算满，拼 batch 收益很小，而且形状不同的请求还拼不到一起。真正的问题是单个请求太慢、显存放不下 $L^2$ 的 attention，所以要把一个请求沿序列切到多张卡上（Ulysses / Ring / USP），再加上 CFG 并行。
:::

::: details Q：TeaCache 为什么能跳步，跳错了会怎样？
相邻去噪步的输入只差一点噪声，输出也很接近。TeaCache 用被时间步调制后的输入变化量来估计输出变化，累计变化小于阈值就复用上一步的输出。跳得太多，误差会累积，画面出现细节丢失或伪影；阈值就是速度和质量之间的旋钮。
:::

## 手撕

**视频 token 数和一次生成的 FLOPs**：

```python
def wan_tokens(frames, h, w, t_stride=4, s_stride=16):
    return (1 + (frames - 1) // t_stride) * (h // s_stride) * (w // s_stride)

def dit_flops(L, params, d, layers, steps=50, cfg=2):
    per_forward = 2 * params * L + 4 * L * L * d * layers
    return per_forward * steps * cfg

L = wan_tokens(81, 720, 1280)                    # 75600
dit_flops(L, 12e9, 5120, 40)                     # ≈ 6.5e17
```

**TeaCache 式的跳步**：

```python
def denoise(model, x, timesteps, threshold=0.15):
    acc, prev_inp, cached = 0.0, None, None
    for t in timesteps:
        inp = model.modulated_input(x, t)              # 便宜：只过第一层前的调制
        if prev_inp is not None:
            rel = (inp - prev_inp).abs().mean() / prev_inp.abs().mean()
            acc += rescale(rel)                        # 多项式校正成输出变化的估计
        prev_inp = inp
        if cached is None or acc >= threshold:
            cached = model.blocks(inp, t)              # 真正算一次
            acc = 0.0
        x = scheduler_step(x, model.head(cached), t)   # 否则复用上一次的输出
    return x
```

常见题：给分辨率和帧数算视频 token 数；估算一次视频生成的 FLOPs，判断 attention 和线性层谁占大头；DiT 和 LLM serving 的区别；序列并行的 Ulysses 和 Ring 有什么不同。

## 参考

- [Wan2.1 技术报告](https://arxiv.org/abs/2503.20314) · [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) · [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [LCM](https://arxiv.org/abs/2310.04378)
- [TeaCache](https://arxiv.org/abs/2411.19108) · [FasterCache](https://arxiv.org/abs/2410.19355) · [cache-dit](https://github.com/vipshop/cache-dit)
- [xDiT](https://arxiv.org/abs/2411.01738) · [PipeFusion](https://arxiv.org/abs/2405.14430) · [USP](https://arxiv.org/abs/2405.07719)
- [Sliding Tile Attention](https://arxiv.org/abs/2502.04507) · [SageAttention](https://arxiv.org/abs/2410.02367)
- [BAGEL](https://arxiv.org/abs/2505.14683) · [Janus](https://arxiv.org/abs/2410.13848) · [Show-o](https://arxiv.org/abs/2408.12528)
- [vLLM-Omni](https://arxiv.org/abs/2602.02204) · [vLLM-Omni 扩散缓存博客](https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration) · [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion)

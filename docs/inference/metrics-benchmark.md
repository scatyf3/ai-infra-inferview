---
title: 指标与 Benchmark：TTFT / TPOT / ITL / Goodput
status: draft
tags: [metrics, benchmark, sla]
difficulty: 2
order: 9
related: [/inference/prefill-decode-roofline, /inference/batching-scheduling, /framework/request-lifecycle, /inference/memory-accounting, /gpu/profiling]
stack: [sv-sla]
---

# 指标与 Benchmark：TTFT / TPOT / ITL / Goodput

> 每个指标怎么从时间戳算出来；怎么做 benchmark；SLA 下怎么反推 batch

## 一句话结论

几个指标各管一段：TTFT 看排队和 prefill，TPOT / ITL 看 decode 每步的快慢和抖动，吞吐看 GPU 利用率，goodput 是满足 SLA 的请求吞吐，在线服务真正要优化的是它。benchmark 要固定输入输出长度分布、按泊松到达逐档扫请求速率、报 P50/P99 和 goodput 曲线，而不是只报一个最大吞吐。TTFT 主要由 prefill 的算力决定，TPOT 主要由 decode 每步要读的字节数决定，两者都能用 [roofline](/inference/prefill-decode-roofline) 先估个下界。

## 推导

### 一个请求的时间线

以流式返回（SSE）为准，客户端记下这些时间戳：

| 符号 | 含义 |
|---|---|
| $t_0$ | 客户端发出请求 |
| $t_1, t_2, \ldots, t_n$ | 收到第 $1, 2, \ldots, n$ 个输出 token |
| $n$ | 这个请求的输出 token 数 |

由此定义（单位一般是 ms）：

- **TTFT**（time to first token）$= t_1 - t_0$。包含排队、tokenize、prefill、第一次采样、网络。各段的拆分见 [请求生命周期](/framework/request-lifecycle)。
- **ITL**（inter-token latency）：相邻两个 token 的间隔 $t_k - t_{k-1}$，$k = 2 \ldots n$。每个请求贡献 $n - 1$ 个样本。
- **TPOT**（time per output token）：一个请求除首 token 外的平均间隔 $\dfrac{t_n - t_1}{n - 1}$。每个请求贡献 1 个样本。
- **E2E latency** $= t_n - t_0$，于是 $\text{E2E} = \text{TTFT} + (n - 1) \cdot \text{TPOT}$。常见的写法「TTFT + TPOT × 输出长度」差一个 token，$n$ 大时可忽略。

vLLM 的压测代码就是这么算的：`tpot = (latency - ttft) / (output_len - 1)`，ITL 则是把所有请求的每个间隔放进一个大列表（[vllm/benchmarks/serve.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)）。不同工具的定义不完全一样，比如有的工具把 ITL 和 TPOT 当同义词、有的把 TTFT 也算进 ITL，NVIDIA 的文档专门提醒只有定义对齐的结果才能比较（[NIM LLM Benchmarking: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)）。

TPOT 和 ITL 的区别在于加权方式：

1. TPOT 的均值是先对每个请求求平均、再在请求之间平均，每个请求权重相同。
2. ITL 的均值把所有间隔混在一起，输出越长的请求贡献的样本越多。
3. ITL 的分布能看到单次卡顿。一个长 prompt 的 prefill 插进来，正在 decode 的请求会出现一个几百 ms 的间隔，平均到 TPOT 里几乎看不出来，但 ITL 的 P99 会很难看。chunked prefill 主要改善的就是它，见 [batching 与调度](/inference/batching-scheduling)。
4. speculative decoding 或服务端把多个 token 打包成一个 SSE chunk 时，一次收到多个 token，ITL 会出现大量接近 0 的样本和一些大间隔，ITL 的分布会变成两个峰。这时 TPOT 更能代表用户感受到的速度。

### 吞吐

压测窗口从第一个请求发出到最后一个响应结束，记为 $T_{\text{wall}}$：

- **请求吞吐** = 完成的请求数 ÷ $T_{\text{wall}}$（req/s）
- **输出 token 吞吐** = 所有输出 token 数 ÷ $T_{\text{wall}}$（tok/s）。有的工具还报包含输入 token 的总吞吐，它对 prompt 长度很敏感，跨负载比较时意义不大。
- **每用户速度** $\approx 1 / \text{TPOT}$。TPOT 20 ms 就是每个用户看到约 50 tok/s。

系统吞吐和每用户速度是一对取舍：batch 越大，系统 tok/s 越高，每个用户越慢。所以报告里常把它们画成一条曲线（横轴每用户 tok/s，纵轴每卡总 tok/s），而不是给单个数字。

### 分位数

P99 的意思是 99% 的请求低于这个值。SLA 一般写成分位数（如 TTFT P99 < 2 s），因为排队让延迟分布有很长的尾巴，平均值会掩盖它。

样本量要够：1000 个请求里 P99 只由最慢的约 10 个决定，换个随机种子就会跳。要报 P99 至少跑几千个请求，P99.9 要上万。

### goodput

goodput 只把满足 SLO 的请求算进吞吐（[DistServe, Zhong et al., 2024](https://arxiv.org/abs/2401.09670)）。一个请求是好请求，当且仅当它的每一项指标都达标：

$$
\text{good}(r) = [\text{TTFT}_r \le T_{\text{ttft}}] \wedge [\text{TPOT}_r \le T_{\text{tpot}}] \wedge \cdots
$$

两种常见口径：

1. **一次压测里的 goodput** = 好请求数 ÷ $T_{\text{wall}}$。vLLM 的 `--goodput ttft:1000 tpot:50` 就是这个，值以 ms 为单位（[serve.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)）。
2. **系统的 goodput**：在 SLO 达成率不低于某个目标（如 90% 的请求达标）的前提下，系统能承受的最大请求速率。DistServe 用的是这个口径，它是对一整条「请求速率 → 达成率」曲线取的值。

为什么要盯 goodput：把 batch 加大，tok/s 往往上涨，但每步变慢，TPOT 超标的请求变多，显存紧张时抢占增加、TTFT 长尾恶化。只看吞吐会把「牺牲一部分请求换总量」的改动当成正收益。

### 指标对应到 roofline

用 [roofline 页](/inference/prefill-decode-roofline) 的符号，估一个下界。例子：Llama-3-70B（$P = 70 \times 10^9$，KV/token $= 320$ KiB），bf16，8 × H100 SXM 做 TP=8。总算力 $8 \times 989$ TFLOP/s，总带宽 $8 \times 3.35 = 26.8$ TB/s。先忽略 TP 的 all-reduce（它的开销见 [计算通信 Overlap](/parallel/comm-overlap)）。

**TTFT ≈ 排队 + prefill 时间**。prefill 是 compute-bound：

$$
t_{\text{prefill}} \approx \frac{2 P S}{n_{\text{gpu}} \cdot \text{peak} \cdot \text{MFU}}
$$

$S = 2048$：$2 \times 70\text{e}9 \times 2048 = 287$ TFLOP；MFU 取 50%，$287\text{T} / (8 \times 989\text{T} \times 0.5) \approx 72$ ms。prompt 翻倍，TTFT 至少翻倍（长 prompt 时 attention 的 $S^2$ 项还会让它涨得更快）。

**TPOT ≈ 一个 decode step 的时间**。decode 是 memory-bound，每步要把权重和所有请求的 KV 各读一遍：

$$
\text{TPOT}(B) \approx \frac{P b_w + B \cdot S_{ctx} \cdot \text{KV/token}}{\text{BW}_{\text{total}} \cdot \text{MBU}}
$$

MBU 是实际带宽占峰值的比例，取 70%。$S_{ctx} = 4096$：

| $B$ | 每步读的字节 | TPOT（MBU 70%） | 系统吞吐 $B / \text{TPOT}$ |
|---|---|---|---|
| 1 | 140 GB + 1.3 GB | 7.5 ms | 133 tok/s |
| 64 | 140 GB + 86 GB | 12.0 ms | 5300 tok/s |

batch 从 1 到 64，TPOT 只慢了 60%，吞吐涨了 40 倍：权重这 140 GB 是所有请求分摊的。

**在 TPOT 的 SLO 下反推 batch**。令 $\text{TPOT}(B) \le T_{\text{tpot}}$，解出

$$
B_{\max} = \frac{T_{\text{tpot}} \cdot \text{BW}_{\text{total}} \cdot \text{MBU} - P b_w}{S_{ctx} \cdot \text{KV/token}}
$$

$T_{\text{tpot}} = 20$ ms：$(0.02 \times 26.8\text{T} \times 0.7 - 140\text{G}) / (4096 \times 327680) \approx 175$。再和显存约束比：$8 \times 80$ GB 减去 140 GB 权重，剩约 500 GB，按每请求 $4096 \times 320$ KiB $= 1.34$ GB 能放约 370 个请求（没扣激活和碎片，详见 [显存账](/inference/memory-accounting)）。所以这里先撞上的是 TPOT 的 SLO，不是显存。

**从 batch 到能扛的请求速率**。Little 定律（[Little, 1961](https://doi.org/10.1287/opre.9.3.383)）：系统里平均同时在跑的请求数 = 到达速率 × 平均停留时间，$L = \lambda W$。输出 256 token、TPOT 20 ms、TTFT 100 ms 时 $W \approx 0.1 + 255 \times 0.02 = 5.2$ s，并发上限 175，于是 $\lambda_{\max} \approx 175 / 5.2 \approx 34$ req/s。请求速率超过它，队列只增不减，TTFT 会一路涨上去。压测曲线上 TTFT 突然变陡的位置，就在这个速率附近。

这些估算没算 TP 通信、调度开销、prefill 和 decode 混跑的干扰，实测通常更差，但它告诉你该去哪找瓶颈：TTFT 远高于 $t_{\text{prefill}}$ 说明在排队；TPOT 远高于带宽下界说明 MBU 低，该看 kernel 或 CPU 调度，方法见 [Profiling](/gpu/profiling)。

### 怎么做 benchmark

1. **固定负载**：输入输出长度用真实 trace 或 ShareGPT 这类对话数据的分布。固定长度会让所有请求同时结束，batch 的行为不真实。要控制输出长度时用 `ignore_eos`，否则模型提前停止，输出长度就不是你设定的值。
2. **发请求的方式**分两种：
   - **开环**（open-loop）：按给定速率发，不管前面的请求回没回来。相邻请求的间隔服从指数分布、均值 $1/\lambda$，就是泊松到达。vLLM 的 `--request-rate` 就是这样，`--burstiness` 不等于 1 时改用 gamma 分布，小于 1 更扎堆、大于 1 更均匀（[serve.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)）。开环才能测出过载时的排队。
   - **闭环**（closed-loop）：固定 $C$ 个并发，一个回来再发下一个（vLLM 的 `--max-concurrency`）。系统再慢请求也不会堆积，所以看不到排队爆炸，适合测「给定并发下的延迟」。
3. **逐档扫**：请求速率从低到高取 6–10 档，每档先 warmup，再跑够几千个请求。
4. **报告**：每档的 P50 / P90 / P99 的 TTFT、TPOT、ITL，吞吐，goodput；画速率对延迟的曲线，标出 SLO 线。
5. **常见坑**：
   - 客户端成了瓶颈。客户端要解析大量流式响应，CPU 打满时测到的延迟里混着客户端自己的排队。看客户端 CPU，或和服务端自己记的指标对一下。
   - prefix cache。随机 prompt 几乎不命中，真实对话的多轮历史大量命中，TTFT 能差几倍。要么关掉，要么在报告里写明命中率。
   - 只报 `request_rate=inf`（所有请求在 0 时刻一起发）测出来的最大吞吐，这个数没有延迟含义。

一个具体的命令（参数名以所用 vLLM 版本的 `--help` 为准）：

```bash
vllm bench serve --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset-name random --input-len 1024 --output-len 256 --ignore-eos \
  --num-prompts 2000 --request-rate 8 --burstiness 1.0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:1000 tpot:50
```

## 面试追问

::: details Q：加大 max_num_seqs 之后吞吐涨了但 goodput 反而降了，为什么？
batch 变大后每步 decode 要多读 $\Delta B \cdot S_{ctx} \cdot \text{KV/token}$ 字节，TPOT 线性变长，超过 SLA 的请求变多，这些请求虽然算了但不计入 goodput；同时 KV 占用变多，抢占增加，TTFT 的长尾恶化。用上面的 $B_{\max}$ 公式可以先估出 SLO 允许的 batch 上限。
:::

::: details Q：TTFT 很高，怎么判断是排队还是 prefill 慢？
先用 $2PS / (\text{peak} \cdot \text{MFU})$ 估 prefill 本身要多久。低负载（请求速率很小）时测一次 TTFT，那时几乎不排队，应该接近这个估算；高负载时 TTFT 远高于它，多出来的就是排队。服务端日志里通常也有 queue time。排队多就要准入控制或扩容，prefill 慢才去看 kernel 和 chunk 大小。
:::

::: details Q：TPOT 和 ITL 的 P99 差很多，说明什么？
TPOT 是每个请求的平均，ITL 是单次间隔。ITL P99 高而 TPOT P99 正常，说明有偶发的长间隔被平均掉了，典型原因是长 prompt 的 prefill 插进来卡住了 decode，或者抢占后的 recompute。上 chunked prefill、调小 chunk、或做 PD 分离。
:::

::: details Q：为什么压测要用泊松到达，而不是匀速发？
平均速率相同时，泊松到达会扎堆，短时间内来一批请求，更接近真实流量，也更容易暴露排队和抢占。匀速发会低估尾延迟。代码就两行：`gaps = np.random.exponential(1 / qps, n); send_at = np.cumsum(gaps)`。
:::

## 手撕

常见题：设计一个压测方案并说明报告哪些图；给一组时间戳算 TTFT / TPOT / ITL；给模型和卡算 TPOT 下界和 SLO 下的 batch 上限。定义见 [SLA](/stack/sv-sla)。

从时间戳算指标（和 vLLM 的口径一致）：

```python
import numpy as np

def summarize(reqs, slo_ttft=1.0, slo_tpot=0.05, pcts=(50, 90, 99)):
    """reqs: 每个请求一个 dict，t0 是发送时刻，ts 是每个输出 token 的到达时刻（秒）。"""
    ttft, tpot, itl, e2e, good = [], [], [], [], 0
    for r in reqs:
        t0, ts = r["t0"], np.asarray(r["ts"])
        n = len(ts)
        ttft.append(ts[0] - t0)
        e2e.append(ts[-1] - t0)
        tp = (ts[-1] - ts[0]) / (n - 1) if n > 1 else 0.0
        if n > 1:
            tpot.append(tp)
        itl.extend(np.diff(ts))                    # 每个请求贡献 n-1 个样本
        good += (ts[0] - t0 <= slo_ttft) and (tp <= slo_tpot)
    wall = max(r["ts"][-1] for r in reqs) - min(r["t0"] for r in reqs)
    out_tok = sum(len(r["ts"]) for r in reqs)
    pct = lambda x: {p: float(np.percentile(x, p)) for p in pcts}
    return {
        "ttft": pct(ttft), "tpot": pct(tpot), "itl": pct(itl), "e2e": pct(e2e),
        "req_per_s": len(reqs) / wall,
        "out_tok_per_s": out_tok / wall,
        "goodput_req_per_s": good / wall,
    }
```

## 参考

- [vllm/benchmarks/serve.py](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)：`vllm bench serve` 的实现，TPOT / ITL / goodput 的计算和到达过程
- [DistServe](https://arxiv.org/abs/2401.09670)（Zhong et al., 2024）：goodput 的定义；[作者博客](https://hao-ai-lab.github.io/blogs/distserve)
- [NVIDIA NIM LLM Benchmarking: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)：各指标定义及不同工具的口径差异
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)（Agrawal et al., 2024）：chunked prefill 对 TTFT / ITL 尾延迟的取舍
- [Etalon](https://arxiv.org/abs/2407.07000)（Agrawal et al., 2024）：讨论 TTFT / TPOT 这类指标在流式场景下的不足，提出 fluidity-index
- [A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383)（Little, 1961）

---
title: 计算通信 Overlap
status: draft
tags: [overlap]
difficulty: 3
order: 6
related: [/parallel/collective-comm, /parallel/megatron-tp, /parallel/zero-fsdp, /parallel/moe-ep, /parallel/parallelism-overview, /gpu/profiling]
stack: [d-comm]
---

# 计算通信 Overlap

> 几种做法：async collective、按层预取、拆块流水、kernel 级融合；每种都先用 α-β 模型算一下值不值得

## 一句话结论

通信和计算用的是不同的硬件（NVLink / 网卡 vs tensor core），只要有一段计算**不依赖**这次通信的结果，就能让两者同时跑，耗时从 $T_{\text{comp}} + T_{\text{comm}}$ 降到接近 $\max(T_{\text{comp}}, T_{\text{comm}})$。难度取决于依赖关系：DP 的梯度同步和 FSDP 的参数预取天然有独立的计算可做，开个异步就行；TP 的 all-reduce 在关键路径上，下一步计算要等它的结果，只能把 GEMM 和通信一起切块做流水（collective matmul / async-TP），或者融合成一个 kernel；MoE 的 all-to-all 靠两个 micro-batch 交替。overlap 不是白送的：通信 kernel 会占 SM、抢 HBM 带宽，切块会让 GEMM 变小变慢，所以先算一下通信和计算的比例，再决定做不做。

## 推导

### 符号和基本模型

| 符号 | 含义 |
|---|---|
| $T_{\text{comp}}, T_{\text{comm}}$ | 一段计算、一次通信各自单独跑的时间 |
| $N$ | 参与通信的卡数 |
| $\text{BW}$ | 单方向链路带宽。H100 NVLink 标称 900 GB/s 是双向，单方向 450 GB/s；IB NDR 每卡约 50 GB/s（见 [集合通信](/parallel/collective-comm#拓扑-节点内-nvlink-跨节点-ib)） |
| $\alpha$ | 每发一次消息的固定延迟，下面沿用集合通信页的假设值 5 µs |
| peak · MFU | 实际算力。H100 bf16 dense 989 TFLOP/s，MFU 取 50%，即 494 TFLOP/s |
| $b$ | 每个元素的字节数，bf16 = 2 |

通信时间用 α-β 模型（推导见 [集合通信](/parallel/collective-comm#α-β-模型-带宽项与延迟项)），消息大小 $M$ 字节：

- ring all-reduce：$2(N-1)\alpha + \dfrac{2(N-1)}{N} \cdot \dfrac{M}{\text{BW}}$
- all-gather 或 reduce-scatter（$M$ 是完整结果的大小）：$(N-1)\alpha + \dfrac{N-1}{N} \cdot \dfrac{M}{\text{BW}}$

三种情况的总耗时：

$$
T_{\text{串行}} = T_{\text{comp}} + T_{\text{comm}}, \qquad
T_{\text{理想重叠}} = \max(T_{\text{comp}}, T_{\text{comm}}), \qquad
T_{\text{暴露}} = T_{\text{实际}} - T_{\text{comp}}
$$

「暴露的通信」（exposed communication）就是没被藏住、让 GPU 计算单元空等的那部分，nsys 时间线上看 NCCL kernel 和计算 kernel 有没有并排就能看出来（见 [Profiling](/gpu/profiling)）。

### 能重叠的两个前提

**前提 1：有不依赖这次通信的计算。** 这是最根本的。按依赖关系分三类：

1. 通信结果**很晚才用**：DP 的梯度 all-reduce 要等到 optimizer step 才用，反向的其他层可以先算。
2. 通信结果**下一层才用**：FSDP 第 $\ell+1$ 层的参数 all-gather，在算第 $\ell$ 层时就能提前发起。
3. 通信结果**下一步马上用**：TP 的 all-reduce 输出就是下一个算子的输入。没有现成的独立计算，必须自己切出来，即把数据切成块，让第 $i$ 块的通信和第 $i+1$ 块的计算重叠。

**前提 2：资源不冲突。** 通信不是「免费的后台任务」：

- **占 SM**。NCCL 每个 channel 跑一个 CUDA block（[NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)，`NCCL_MAX_CTAS` 可以限制数量），这些 SM 就不能跑 GEMM。DeepSeek-V3 的跨节点 all-to-all 专门优化到只用 20 个 SM（H800 共 132 个）就能打满 IB 和 NVLink（[DeepSeek-V3 技术报告 §3.2.2](https://arxiv.org/abs/2412.19437)），也就是说仍有约 15% 的 SM 让给了通信。
- **抢 HBM 带宽和 L2**。通信 kernel 要从 HBM 读发送缓冲、写接收缓冲。
- **launch 顺序**。在 Hopper 及更早的卡上，Megatron 要求 TP / SP / CP 时设 `CUDA_DEVICE_MAX_CONNECTIONS=1`，而 FSDP 反而要求它大于 1（[Megatron-LM arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) 里的检查和警告）。常见的解释是：只有一条硬件队列时，kernel 按发起顺序上 GPU，先发起的通信 kernel 能先拿到 SM，不会被后面占满所有 SM 的大 GEMM 挡住；多条队列时，不同 stream 的 kernel 才能真正并发。

### 拆块流水的通用公式

前提 1 的第 3 类要靠拆块。把计算和通信各切成 $k$ 块，第 $i$ 块算完立刻发、同时算第 $i+1$ 块。每块计算 $T_{\text{comp}}/k$、通信 $T_{\text{comm}}/k$。第一块的计算和最后一块的通信无法被藏住，中间的 $k-1$ 段每段取两者中较慢的：

$$
T(k) = \frac{T_{\text{comp}}}{k} + (k-1)\max\!\left(\frac{T_{\text{comp}}}{k}, \frac{T_{\text{comm}}}{k}\right) + \frac{T_{\text{comm}}}{k}
= \max(T_{\text{comp}}, T_{\text{comm}}) + \frac{\min(T_{\text{comp}}, T_{\text{comm}})}{k}
$$

$k$ 越大越接近理想重叠。但每块都要多付一次延迟 $\alpha$，GEMM 变小后效率还会下降（最后一波 tile 填不满 SM，即 [wave quantization](/gpu/tensor-core-gemm#wave-quantization-tile-数和-sm-数对不齐)），所以实际是

$$
T(k) \approx \max(T_{\text{comp}}, T_{\text{comm}}) + \frac{\min(T_{\text{comp}}, T_{\text{comm}})}{k} + k\,\alpha' + \text{GEMM 变小的损失}
$$

$\alpha'$ 是每块额外的固定开销。只看前两项随 $k$ 变化的部分，$\frac{\min}{k} + k\alpha'$ 在 $k^* = \sqrt{\min(T_{\text{comp}}, T_{\text{comm}}) / \alpha'}$ 处最小。所以 $k$ 通常取 TP 的卡数 $N$ 或者个位数，不会切得很碎。

### DP：梯度 all-reduce 和反向重叠

反向从最后一层往前算，最后一层的梯度最先出来。DDP 把梯度按反向的顺序装进若干 bucket（默认每个 25 MiB，[PyTorch DDP 源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)），一个 bucket 装满就立刻异步发起 all-reduce，同时继续算更前面的层（[Li et al., 2020](https://arxiv.org/abs/2006.15704)）。

算一笔：7B 模型、bf16 梯度 14 GB、单节点 8 卡 NVLink，每卡每步 32768 个 token。

- 通信：$\frac{2 \times 7}{8} \times 14\text{ GB} / 450\text{ GB/s} \approx 54$ ms，分成约 530 个 bucket。
- 反向计算：约 $4 P \cdot T_{\text{tok}}$ FLOP（前向 $2P$ 的两倍），$4 \times 7\text{e}9 \times 32768 / 494\text{T} \approx 1.9$ s。

通信只有计算的 3%，很容易藏住。藏不住的是**最后一个 bucket**：它装的是第一层的梯度，反向算完时它才开始通信，后面已经没有计算可以重叠。bucket 越大，这段尾巴越长；越小，每个 bucket 的 $\alpha$ 开销越多。

### FSDP / ZeRO-3：按层预取参数

ZeRO-3 每层计算前要 all-gather 完整参数（见 [ZeRO / FSDP](/parallel/zero-fsdp)）。第 $\ell+1$ 层的参数和第 $\ell$ 层的计算无关，所以算第 $\ell$ 层时就发起第 $\ell+1$ 层的 all-gather：

```python
def fsdp_forward(layers, x):
    h = all_gather(layers[0].shard, async_op=True)          # 先发起第 0 层
    for l in range(len(layers)):
        w = h.wait()                                         # 等第 l 层参数到位
        if l + 1 < len(layers):
            h = all_gather(layers[l + 1].shard, async_op=True)   # 预取下一层，和本层计算重叠
        x = layers[l](x, w)
        free(w)                                              # 用完就丢，只留 1/N 的分片
    return x
```

`async_op=True` 时通信在单独的 CUDA stream 上跑，`wait()` 是让计算 stream 等通信 stream，CPU 不会阻塞。反向同理，算第 $\ell$ 层梯度时预取第 $\ell-1$ 层的参数，同时把第 $\ell+1$ 层的梯度 reduce-scatter 出去。PyTorch FSDP2 用 `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` 显式指定预取哪些层（[_fully_shard.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py)），FSDP 论文里也讨论了预取和限制同时在途的 all-gather 数量以控制显存（[Zhao et al., 2023](https://arxiv.org/abs/2304.11277)）。

**什么时候藏得住**。一层参数 $P_\ell$，每卡每步 $T_{\text{tok}}$ 个 token：

$$
\underbrace{\frac{2 P_\ell T_{\text{tok}}}{\text{peak} \cdot \text{MFU}}}_{\text{本层前向计算}} \ \ge\ \underbrace{\frac{N-1}{N} \cdot \frac{P_\ell\, b}{\text{BW}}}_{\text{下一层 all-gather}}
\iff
T_{\text{tok}} \ \ge\ \frac{N-1}{N} \cdot \frac{b \cdot \text{peak} \cdot \text{MFU}}{2\,\text{BW}}
$$

$P_\ell$ 消掉了，条件只和每卡 token 数、硬件有关。代入 $N = 8$、bf16：NVLink 450 GB/s 时 $T_{\text{tok}} \ge 961$；IB 50 GB/s 时 $T_{\text{tok}} \ge 8654$。所以跨节点做 FSDP 要求每卡 batch 足够大，否则用 HSDP 把 all-gather 限制在节点内。代价是显存：预取时同时有两层完整参数在卡上。

例：7B、32 层，每层 2.2 亿参数，8 卡 NVLink all-gather 一层约 0.85 ms，32768 token 的前向计算约 29 ms，轻松藏住。

### TP / SP：通信在关键路径上

Megatron TP 每层前向两次 all-reduce（attention 输出、MLP 输出），下一个算子直接要用结果（见 [Megatron TP](/parallel/megatron-tp)）。先算比例：每次 all-reduce 的消息是一份完整激活 $T_{\text{tok}} \cdot d \cdot b$，一层的权重约 $12 d^2$，每卡算 $1/N$：

$$
\frac{T_{\text{comm}}}{T_{\text{comp}}}
= \frac{2 \cdot \frac{2(N-1)}{N} \cdot \frac{T_{\text{tok}} d b}{\text{BW}}}{\frac{2 \cdot 12 d^2 \cdot T_{\text{tok}}}{N \cdot \text{peak} \cdot \text{MFU}}}
= \frac{(N-1)\, b \cdot \text{peak} \cdot \text{MFU}}{6\, d \cdot \text{BW}}
$$

$T_{\text{tok}}$ 消掉了：TP 的通信占比和 batch 无关，只和 hidden 维 $d$、TP 度 $N$、硬件比值有关。代入 70B 的 $d = 8192$、$N = 8$、NVLink：$\frac{7 \times 2 \times 494\text{T}}{6 \times 8192 \times 450\text{G}} \approx 0.31$。按 70B 的实际层参数（约 8.75 亿）和 32768 token 直接算，两次 all-reduce 约 4.2 ms，前向计算约 14.5 ms，比例 0.29，对得上。串行时有约 22% 的时间在等通信（$4.2 / 18.7$）。$d$ 越大占比越低，这也是 TP 适合大模型的原因；跨节点 BW 降到 1/9，占比就到 2.8，完全不可接受。

SP（sequence parallel）把 all-reduce 换成 reduce-scatter + all-gather，总量不变（[Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)）。拆开之后正好可以分别和相邻的 GEMM 融合成流水：

- **all-gather + GEMM**（进入 column-parallel 层）：每卡持有序列的 $1/N$ 段输入 $X_r$，要算 $[X_0; \ldots; X_{N-1}]\, W_r$。不必等全部拼齐，环形传递，手上有哪段就先算哪段。
- **GEMM + reduce-scatter**（离开 row-parallel 层）：每卡算出部分和，按输出行块切开，一块算完就发给下一张卡累加。

```python
def allgather_matmul(x_local, w_local, rank, N):
    """x_local: [T/N, d] 本卡的序列分片；w_local: [d, h/N] column-parallel 权重分片。"""
    out = [None] * N
    cur = x_local
    for s in range(N):
        src = (rank + s) % N                          # cur 原本属于哪张卡
        if s < N - 1:
            nxt = recv_async(from_rank=(rank + 1) % N)
            send_async(cur, to_rank=(rank - 1) % N)
        out[src] = cur @ w_local                      # 和上面的收发同时进行
        if s < N - 1:
            cur = nxt.wait()
    return concat(out)                                # [T, h/N]，等于先 all-gather 再乘


def matmul_reducescatter(x_local, w_local, rank, N):
    """x_local: [T, d/N]；w_local: [d/N, h] row-parallel 权重分片。返回输出的第 rank 个行块（已求和）。"""
    rows = split(range(x_local.shape[0]), N)          # 输出按行（token）切成 N 块
    acc = None
    for s in range(N):
        c = (rank + s + 1) % N                        # 本步负责的输出行块，最后一步 c == rank
        part = x_local[rows[c]] @ w_local
        if s > 0:
            part += acc_from_next.wait()              # 下一张卡传来的同一行块的累加值
        if s < N - 1:
            send_async(part, to_rank=(rank - 1) % N)
            acc_from_next = recv_async(from_rank=(rank + 1) % N)
        acc = part
    return acc
```

（两个函数的逻辑用 numpy 模拟 4 个 rank 验证过，结果分别等于「先 all-gather 再乘」和「先乘再 reduce-scatter」。）

这个思路最早在 TPU 上系统化，叫 collective matmul（[Wang et al., ASPLOS 2023](https://doi.org/10.1145/3567955.3567959)）。GPU 上的实现：

- **Transformer Engine / Megatron `--tp-comm-overlap`**：用 userbuffers 做 P2P 环形交换，要求 TP 组在单节点内、NVLink 互联（[TE 示例](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/comm_gemm_overlap)）。
- **PyTorch async-TP**：用 SymmetricMemory 直接做 P2P 拷贝，连续数据走 copy engine，不占 SM；两条 stream 交替做计算和通信，减轻 GEMM 切块带来的 wave quantization 损失。作者自测：64 张 H100 上 Llama3 70B 前向最多快约 20%，端到端约 8%；限制是只支持节点内、需要 NVSwitch 全互联，对推理这种小矩阵还不理想（[PyTorch 论坛](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)）。
- **kernel 级融合**：把通信塞进 GEMM kernel，按 tile 粒度算完就发，不再切成 $N$ 个独立的小 GEMM。Flux 作者自测：最多重叠 96% 的通信，128 卡训练比 Megatron-LM 最多快 1.24 倍，8 卡推理 prefill / decode 比 vLLM 最多快 1.66 / 1.30 倍（[Chang et al., 2024](https://arxiv.org/abs/2406.06858)）；T3 用硬件跟踪 GEMM 的输出写入来自动触发通信（[Pati et al., 2024](https://arxiv.org/abs/2401.16677)）。

### 推理 decode：TP all-reduce 是延迟问题

decode 时 batch 64、$d = 8192$、bf16，一次 all-reduce 只有 1 MiB。8 卡 ring：延迟项 $14 \times 5\,\mu s = 70\,\mu s$，带宽项约 $4\,\mu s$，合计约 74 µs。而 70B 每卡每层读权重只要 $875\text{M} \times 2 / 8 / 3.35\text{ TB/s} \approx 65\,\mu s$。一层两次 all-reduce 比计算本身还长。

这时 overlap 的意义不大，因为没有足够的计算可以藏，主要靠**减少延迟**：

1. **one-shot all-reduce**：每张卡通过 NVLink P2P 直接读其他 $N-1$ 张卡的缓冲区，在本地求和，只需一步。时间约 $\alpha + (N-1) M / \text{BW} = 5 + 7 \times 1\text{ MiB} / 450\text{ GB/s} \approx 21\,\mu s$，比 ring 的 74 µs 少很多。代价是总流量变大，所以只适合小消息。vLLM 的 [custom all-reduce](https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh) 就是这类实现。
2. **NVLS**：让 NVSwitch 做归约（见 [集合通信](/parallel/collective-comm)）。
3. **拆 batch 交错**：把一个 batch 拆成两半，一半做 all-reduce 时另一半做 GEMM。NanoFlow 把这个思路推广到单卡内让计算、访存、通信三类算子并发（[Zhu et al., 2024](https://arxiv.org/abs/2408.12757)）。代价是每半 batch 的 GEMV 都要把权重整读一遍，decode 是 memory-bound，读权重的总量翻倍。所以 batch 很大、计算时间够长时才划算。

### MoE：all-to-all 和双 micro-batch

每个 MoE 层两次 all-to-all（dispatch、combine），通信量和计算量的估算、DeepEP 的两类 kernel、two-batch overlap 的实测数据都在 [MoE 与 EP](/parallel/moe-ep#第-6-步-deepep-两类-kernel-与-two-batch-overlap)，这里只补和 overlap 相关的结论：

- DeepSeek-V3 训练时跨节点 EP 带来的计算 : 通信约为 1 : 1，DualPipe 把一对前向和反向 chunk 交错，让一个的 all-to-all 和另一个的计算重叠，同时减少流水线 bubble（[DeepSeek-V3 技术报告 §3.2.1](https://arxiv.org/abs/2412.19437)）。
- 推理时的 two-batch overlap 同样受上面 decode 那段的限制：每卡 token 太少时计算太短，藏不住通信，切成两半还让每半的 GEMM 更小，反而变慢。

### PP 和 CP

**PP** 的 stage 间 P2P 只传一份激活，和其他 micro-batch 的计算天然重叠，主要问题是 bubble 而不是通信，见 [并行总览](/parallel/parallelism-overview)。

**CP / Ring Attention**：每卡持有 $S/N$ 个 token 的 Q、K、V，K/V 块沿环传递，算当前块时收下一块（[Liu et al., 2023](https://arxiv.org/abs/2310.01889)）。每一步，计算是本地 Q 和一个 K/V 块的 attention，$4 (S/N)^2 d$ FLOP；通信是一个 K 块加一个 V 块，$2 (S/N)\, d_{kv}\, b$ 字节（MHA 时 $d_{kv} = d$）。要藏住：

$$
\frac{4 (S/N)^2 d}{\text{peak} \cdot \text{MFU}} \ge \frac{2 (S/N)\, d_{kv}\, b}{\text{BW}}
\iff
\frac{S}{N} \ge \frac{d_{kv}}{d} \cdot \frac{b \cdot \text{peak} \cdot \text{MFU}}{2\,\text{BW}}
$$

MHA、bf16 时：NVLink 下每卡至少约 1100 个 token，IB 下约 9900 个。GQA 的 $d_{kv} / d = H_{kv} / H$ 更小，门槛按比例降低。这和 Ring Attention 论文里「块大小要大于 算力 ÷ 带宽」的结论是同一件事。

### 总结

| 场景 | 依赖关系 | 做法 | 藏得住的条件 |
|---|---|---|---|
| DP 梯度 | 结果很晚才用 | bucket + 异步 all-reduce | 几乎总能；最后一个 bucket 暴露 |
| FSDP 参数 | 下一层才用 | 预取下一层 | $T_{\text{tok}} \ge \frac{N-1}{N} \cdot \frac{b \cdot \text{peak} \cdot \text{MFU}}{2\text{BW}}$ |
| TP / SP | 马上用 | AG+GEMM、GEMM+RS 拆块或融合 | 通信占比 $\frac{(N-1) b \cdot \text{peak} \cdot \text{MFU}}{6 d \cdot \text{BW}}$ 越小越容易 |
| decode TP | 马上用、消息小 | 降延迟为主（one-shot、NVLS），大 batch 再拆 batch | 延迟项主导，overlap 收益有限 |
| MoE EP | 马上用 | 双 micro-batch、DualPipe | 每卡 token 足够多 |
| CP | 下一块才用 | 环形预取 K/V | $S/N \ge \frac{d_{kv}}{d} \cdot \frac{b \cdot \text{peak} \cdot \text{MFU}}{2\text{BW}}$ |

## 面试追问

::: details Q：为什么 overlap 之后 GEMM 本身可能变慢？
三个原因。一是 NCCL 每个 channel 占一个 CUDA block，这些 SM 不能跑 GEMM；二是通信也读写 HBM 和 L2，和 GEMM 抢带宽；三是为了重叠把 GEMM 切小，最后一波 tile 填不满 SM（wave quantization）。所以净收益要实测：通信远小于计算时（比如 DP 梯度只占 3%），不 overlap 也只损失 3%，而强行拆块可能得不偿失。async-TP 用 copy engine 做 P2P 拷贝，就是为了不占 SM。
:::

::: details Q：`dist.all_reduce(t, async_op=True)` 之后立刻 `wait()`，有 overlap 吗？
没有。`async_op=True` 只是把通信放到 NCCL 的 stream 上并返回 handle，`wait()` 让当前计算 stream 等它完成。中间没有插入独立的计算，就和同步调用一样。overlap 要求「发起」和「wait」之间有不依赖通信结果的 kernel。
:::

::: details Q：TP 的通信占比为什么和 batch 无关？
每次 all-reduce 的字节数和每卡的 GEMM FLOPs 都正比于 token 数 $T_{\text{tok}}$，相除就消掉了，剩下 $\frac{(N-1) b \cdot \text{peak} \cdot \text{MFU}}{6 d \cdot \text{BW}}$。所以加大 batch 救不了跨节点 TP，只有加大 $d$、减小 $N$ 或提高带宽有用。相比之下 FSDP 通信的是参数，和 batch 无关，计算正比于 batch，所以加大每卡 batch 就能藏住。
:::

::: details Q：画出 TP 一层里 GEMM 和 all-reduce 的依赖，哪些能重叠？
attention 块：QKV GEMM（column）→ attention → O-proj GEMM（row）→ all-reduce → 残差 + norm；MLP 块同理。前向里每个 all-reduce 的输出都马上被下一个算子用，没有天然可重叠的计算，只能拆块（AG+GEMM / GEMM+RS）或拆 batch。反向里，激活梯度的 all-reduce 和同一层的权重梯度 GEMM（$dW = X^\top dY$）互不依赖，可以重叠。
:::

## 手撕

常见题：给定模型和互联，估算 TP / FSDP 的通信占比，判断能否藏住；写 all-gather + GEMM 的环形流水；解释 `async_op` 的语义。通信量的算法见 [集合通信](/parallel/collective-comm)。

用 α-β 模型估算一层的暴露通信：

```python
def ring_ar(M, N, BW, alpha=5e-6):
    return 2 * (N - 1) * alpha + 2 * (N - 1) / N * M / BW

def ag(M, N, BW, alpha=5e-6):
    return (N - 1) * alpha + (N - 1) / N * M / BW

def pipelined(t_comp, t_comm, k, alpha_chunk=5e-6):
    """切 k 块流水，不含 GEMM 变小的损失。"""
    return max(t_comp, t_comm) + min(t_comp, t_comm) / k + k * alpha_chunk

PEAK = 989e12 * 0.5                      # bf16 dense × MFU
# TP=8，70B 一层前向，32768 token
d, N, T, P_layer = 8192, 8, 32768, 875e6
t_comm = 2 * ring_ar(T * d * 2, N, 450e9)          # ≈ 4.2 ms
t_comp = 2 * P_layer * T / (N * PEAK)              # ≈ 14.5 ms
print(t_comp + t_comm, [pipelined(t_comp, t_comm, k) for k in (2, 4, 8)])
# 串行 ≈ 18.8 ms；k=2/4/8 时 ≈ 16.7 / 15.6 / 15.1 ms

# FSDP 一层，7B/32 层，8 卡：NVLink vs IB 的 all-gather
P_layer = 7e9 / 32
print(ag(P_layer * 2, 8, 450e9), ag(P_layer * 2, 8, 50e9), 2 * P_layer * T / PEAK)
# ≈ 0.89 ms / 7.7 ms / 29 ms：NVLink 和 IB 下都能藏住
```

## 参考

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)（Korthikanti et al., 2022）：sequence parallel
- [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://doi.org/10.1145/3567955.3567959)（Wang et al., ASPLOS 2023）：collective matmul
- [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858)（Chang et al., 2024）
- [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](https://arxiv.org/abs/2401.16677)（Pati et al., 2024）
- [Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)（PyTorch 论坛，torchtitan）
- [Transformer Engine comm_gemm_overlap 示例](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/comm_gemm_overlap)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)（Li et al., 2020）：DDP 的 bucket 与 overlap
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)（Zhao et al., 2023）
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：DualPipe、20 SM 的 all-to-all
- [DeepEP](https://github.com/deepseek-ai/DeepEP)：MoE all-to-all 通信库
- [NanoFlow](https://arxiv.org/abs/2408.12757)（Zhu et al., 2024）：推理时单卡内的算子级并发
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)（Liu et al., 2023）
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)：channel 与 CUDA block、`NCCL_MAX_CTAS`

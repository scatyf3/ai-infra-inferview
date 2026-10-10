---
title: RL 权重同步：从训练引擎到推理引擎
status: draft
tags: [rl-infra, weight-sync, reshard, checkpoint-engine, cuda-ipc]
difficulty: 4
order: 4.3
related: [/posttrain/rl-infra, /posttrain/rl-async-rollout, /parallel/zero-fsdp, /parallel/megatron-tp]
stack: []
---

# RL 权重同步：从训练引擎到推理引擎

> 每个 RL step 训练完，推理引擎要换上新权重。要搬多少字节、带宽下界是多少、为什么不是一次 memcpy（切分不同、名字和布局不同、可能还要量化）、同卡和跨卡各走什么路径、怎么做快

## 一句话结论

一次同步要把整个模型（70B bf16 是 140 GB）从训练引擎的切分方式，换成推理引擎的切分方式和参数布局，再送进推理引擎。带宽下界很好算：字节数 ÷ 链路带宽，70B 走一张 400 Gb/s 网卡要 2.8 s。实际耗时通常是下界的几倍，慢在：训练侧要先 all-gather 出完整张量、改名字和布局（QKV、gate/up 融合，Megatron 的交错布局）、可能还要量化成 fp8；每个张量单独发一次的调用开销；推理侧加载后的后处理。优化手段都是工程上的：**分桶**（把小张量拼成几百 MB 一块）、**流水**（gather、传输、加载重叠）、**同卡走 CUDA IPC 不拷贝**、**跨卡用 NCCL 广播**。同步完别忘了**清掉推理侧的 prefix cache**，那些 KV 是旧权重算的。

## 推导

### 1. 要搬多少字节，最快多久

带宽下界：

$$
t_{\text{sync}} \ge \frac{P \cdot b_w}{\text{BW}}
$$

$P$ 是参数量，$b_w$ 是每个参数的字节数（bf16 = 2，fp8 = 1），BW 是这条路径的带宽。几条常见链路（H100 一代，单方向）：

| 链路 | 单方向带宽 |
|---|---|
| NVLink 4（H100 SXM，节点内） | 约 450 GB/s（官方写 900 GB/s 是两个方向的总和） |
| PCIe Gen5 x16（GPU ↔ CPU） | 64 GB/s |
| 一张 InfiniBand NDR 网卡（400 Gb/s） | 50 GB/s |
| DGX H100 一个节点 8 张网卡 | 约 400 GB/s |

代入（单位：秒）：

| 模型 | 字节数 | 一张网卡 | PCIe | 8 张网卡 | NVLink |
|---|---|---|---|---|---|
| 7B bf16 | 14 GB | 0.28 | 0.22 | 0.04 | 0.03 |
| 70B bf16 | 140 GB | 2.8 | 2.2 | 0.35 | 0.31 |
| Qwen3-235B bf16 | 470 GB | 9.4 | 7.3 | 1.2 | 1.0 |
| Kimi-K2 1T fp8 | 约 1 TB | 20 | 16 | 2.5 | 2.2 |

公开的实测（作者自测）：

1. checkpoint-engine 把 Qwen3-235B 更新到 8 张 H800（TP8）：6.2 s；Kimi-K2 更新到 256 张 H20（16 个实例 × TP16）：16 s（[README](https://github.com/MoonshotAI/checkpoint-engine)）。
2. verl 用 NCCL 把 Qwen3-30B-A3B 同步给 30 个 rollout 实例：约 7 s，8.25 GB/s（4 × 8 张 H100，400G IB）。
3. slime 同步 GLM-4.5（355B-A32B）bf16：48 s；加上 fp8 分块量化：100 s（v0.1.0 博客）。

235B 的下界约 1 s，实测 6 s；355B 量化后 100 s。差出来的部分就是下面几节要讲的。

和一个 RL step 比：推理类任务一个 step 往往几分钟，同步几秒到几十秒还能接受。但 [PipelineRL](/posttrain/rl-async-rollout) 这种每个优化器 step 都同步一次的设计，同步耗时就直接卡住吞吐了。

### 2. 为什么不是一次 memcpy

**(a) 切分方式不同**。训练侧是 FSDP（每张卡存每个参数的一片）或 Megatron（TP × PP × EP 切），推理侧是另一套 TP（通常更小）。所以要先在训练侧把张量凑完整（all-gather），再按推理侧的切分发过去，推理侧每个 TP rank 只取自己那一片。

verl 现在的做法是逐个张量 all-gather（对 FSDP 的 DTensor 调 `full_tensor()`），而不是一次 gather 整层，峰值显存只多一个张量。

**(b) 名字和布局不同**。推理引擎为了少发几次 kernel，会把几个矩阵拼成一个：

| 训练侧 / HF 的名字 | vLLM 里 |
|---|---|
| `q_proj`、`k_proj`、`v_proj` | `qkv_proj`（拼成一个） |
| `gate_proj`、`up_proj` | `gate_up_proj` |

发送方通常按 HF 的名字逐个发，由推理侧的 `load_weights` 按映射表（`packed_modules_mapping`）拼进融合后的张量里。

Megatron 更麻烦：它的 QKV 不是 `[全部 q | 全部 k | 全部 v]`，而是**按 GQA 组交错**。设 8 个 Q head、2 个 KV head（2 组，每组 4 个 Q head），按 head 排的顺序是

```
Megatron linear_qkv:  q0 q1 q2 q3 k0 v0 | q4 q5 q6 q7 k1 v1     每组：4 个 q、1 个 k、1 个 v
HF / vLLM:            q0 q1 q2 q3 q4 q5 q6 q7 | k0 k1 | v0 v1
```

转换时先 view 成 `(组数, 每组 q 数 + 2, head_dim, hidden)`，沿第 1 维切成 `[4, 1, 1]` 三份，再分别拼回去。SwiGLU 的 `linear_fc1` 同理：TP 切分后每个分片里是 `[gate_i; up_i]`，要先按分片重组再拼。词表还可能为了整除 TP 而补过 padding，要去掉。slime 给每个模型族写了一个 Megatron → HF 的转换器。

**(c) MoE**。专家按 EP 切在不同的卡上，要先在 EP 组内 all-gather。slime 先发非专家参数，再发专家参数。

**(d) 精度可能不同**。训练用 bf16，rollout 想用 fp8 加速，就要在同步时量化：slime 在训练侧转成 HF 格式后做分块 fp8 量化（输出 `weight` 和 `weight_scale_inv`），verl 也有 on-the-fly 的 128 × 128 分块量化。量化让推理侧和训练侧的概率差得更多，要配合 [训推不一致](/posttrain/rl-train-infer-mismatch) 里的修正。

**(e) 推理侧的后处理**。有些量化 kernel 需要把权重重排成特定布局（比如 Marlin），在 `process_weights_after_loading` 里做。它应该在所有张量都到齐之后做一次，而不是每来一个张量做一次。SGLang 现在要求权重更新包在 `begin_weight_update` / `end_weight_update` 之间，就是为了这个。

### 3. 两条路径：同卡和跨卡

**同卡（colocate）：CUDA IPC，不拷贝**

训练和推理在同一组卡上轮流跑时，新权重已经在 GPU 显存里了。训练侧把张量导出成 **CUDA IPC handle**（一个能让另一个进程直接映射同一块显存的句柄），把 handle 发给推理进程，推理进程直接从这块显存读进自己的参数里。走的是同卡显存拷贝，不过网络。

这里的难点是显存：训练状态（每参数约 16 字节，见 [训练显存账](/posttrain/training-memory)）和推理的权重、KV cache 要轮流占同一块显存。vLLM 的 sleep mode 就是为此设计的：

| 级别 | 权重 | KV cache |
|---|---|---|
| level 1 | 卸到 CPU 内存 | 丢掉 |
| level 2 | 丢掉 | 丢掉 |

RL 用 level 2：反正醒来要换新权重，没必要备份旧的。顺序是 `sleep(level=2)` → 训练 → `wake_up(tags=["weights"])` → 灌新权重 → `wake_up(tags=["kv_cache"])`。先只唤醒权重、灌完再分配 KV，避免两者同时占满显存而 OOM。SGLang 对应的是 `release_memory_occupation` / `resume_memory_occupation`。

**跨卡（分离部署）：NCCL 广播**

训练和推理在不同的卡上时：

1. 训练侧每个 PP stage 选一张卡（DP = 0、TP = 0 的那个 rank），和所有推理卡建一个 NCCL 通信组。
2. 训练侧 gather 出完整张量，`broadcast` 给组里所有推理卡。
3. 推理侧每个 TP rank 只把属于自己的那一片拷进参数。

SGLang 的接口是 `init_weights_update_group` + `update_weights_from_distributed`，vLLM 现在有原生的 weight transfer API（`init_weight_transfer_engine` → `start_weight_update` → `update_weights` → `finish_weight_update`），后端可选 NCCL、IPC 等。OpenRLHF 早期的实现是逐个参数广播，不分桶。

还有一条最简单的路径：训练侧把 checkpoint 写盘，推理侧从盘上重新加载（`update_weights_from_disk`）。慢，但不需要两边建通信组，调试方便。

### 4. 怎么做快

**(a) 分桶**。模型有几千个张量，每个张量单独发一次，NCCL 的启动开销和一次 RPC 的往返就会累积起来。把小张量拼成几百 MB 一块再发。一篇工程博客记录过 Qwen3-30B-A3B 在 8 张 H100 上 colocate 同步的优化过程：分桶后调用次数从约 2000 次降到约 120 次，耗时从 50 s 降到 30 s，512 MB 的桶最好；再把桶里的张量拍平成一块连续内存，降到 20 s；优化推理侧的 `load_weights` 后到 7 s（[He, RL weight sync](https://hebiao064.github.io/rl-weight-sync)，作者注明这些延迟是按一系列 PR 模拟的）。slime 默认桶大小 512 MiB，verl 默认 2 GB。

**(b) 流水**。把一次同步拆成几个阶段，相邻的桶在不同阶段上重叠。checkpoint-engine 的三段：

```
桶 k:    H2D ─> broadcast ─> 推理引擎拷走自己那份
桶 k+1:         H2D       ─> broadcast ─> 拷走
桶 k+2:                      H2D       ─> broadcast ─> 拷走
```

用两块显存缓冲轮流用（ping-pong），一块在广播时另一块在做 H2D。

**(c) 全量广播 vs 只发需要的**。理论上每张推理卡只需要自己那一片，按需发送的字节数最少。Kimi K2 的报告反而选了**把全部参数广播给整个集群**，传的字节数是最优方案的好几倍，但训练和推理的切分完全解耦，同步协调的开销更小，实测更快（K2 报告 §3.3.2，作者自述全量更新不到 30 s）。

**(d) colocate 下的 reshard 不多占显存**。HybridFlow（verl 的论文）的 3D-HybridEngine 让推理侧的 TP 分组和训练侧的分片对齐：每张卡上推理需要的那片和它训练时持有的那片重叠，只需在一个小组内 all-gather。设训练是 TP = $t$、PP = $p$，推理是 TP = $t_g$、PP = $p_g$，模型大小 $M$：

| | 通信量 | 峰值显存 |
|---|---|---|
| DeepSpeed-Chat | $\frac{tpd - 1}{tpd} M$ | $M$ |
| HybridFlow | $\frac{tp - t_g p_g}{t_g p_g\, tp} M$ | $\frac{M}{t_g p_g}$ |

例：训练 TP = 8、推理 TP = 2（$p = p_g = d = 1$）。DeepSpeed-Chat 每张卡要 gather 出完整模型，通信 $0.875M$、峰值 $M$；HybridFlow 通信 $0.375M$、峰值 $0.5M$。作者自测 70B 上切换时间减少 89%。

### 5. 推理侧的收尾

1. **先停生成**。SGLang 的 `pause_generation` 有三种模式：`abort` 丢掉正在写的请求，`retract` 把它们退回队列，`in_place` 原地暂停、换完权重继续写（这就是 [PipelineRL](/posttrain/rl-async-rollout) 的 in-flight 更新）。
2. **清掉 prefix cache**。缓存的 KV 是旧权重算的，新请求如果命中这些 KV，就是在用旧策略的中间结果，悄悄引入 off-policy。slime 的流程里有一步 `flush_cache`。
3. **继续生成**（`continue_generation`）。

## 面试追问

::: details Q：估算一下 70B 模型一次权重同步要多久？
bf16 是 140 GB。跨节点走一张 400 Gb/s（50 GB/s）的网卡，下界 2.8 s；一个节点 8 张网卡并行约 0.35 s。实际要加上训练侧 all-gather、改布局、可能的 fp8 量化、分桶调度和推理侧加载，通常是下界的几倍。再和一个 step 的时长比：推理任务一个 step 几分钟，几秒的同步不是瓶颈；每个优化器 step 都同步（in-flight 更新）时才需要压到很低。
:::

::: details Q：为什么不能直接把训练侧的 state_dict 拷给 vLLM？
三个不匹配：切分不同（FSDP 分片或 Megatron TP/PP/EP，对推理侧的另一种 TP），所以要先 gather 再重切；名字和布局不同（vLLM 把 q/k/v、gate/up 拼成一个矩阵，Megatron 的 QKV 按 GQA 组交错），要转换；精度可能不同（rollout 用 fp8），要量化。还有推理侧的后处理（比如量化 kernel 的权重重排）要在所有张量到齐后做一次。
:::

::: details Q：同步完权重，推理引擎还有什么要做的？
清掉 prefix cache。缓存的 KV 是旧权重算出来的，新请求命中它们等于用旧策略的中间结果继续生成。colocate 时还要按顺序唤醒：先唤醒权重、灌入新权重，再分配 KV cache，避免显存同时被占满。
:::

::: details Q：分桶为什么能快那么多？
模型有几千个张量，大部分很小（norm、bias）。逐个发送时，每次都有 NCCL 调用、RPC 往返、推理侧一次加载的固定开销，加起来远大于传数据本身的时间。拼成几百 MB 一块，调用次数从几千次降到一百多次，固定开销被摊薄；桶也不能太大，否则占显存、也没法和下一个桶流水重叠。
:::

## 手撕

**估算同步时间**：

```python
def sync_time_lower_bound(params_b, bytes_per_param, bw_gb_s):
    # params_b: 参数量（十亿）；bw_gb_s: 链路单方向带宽（GB/s）
    return params_b * bytes_per_param / bw_gb_s

sync_time_lower_bound(70, 2, 50)    # 2.8 s，一张 400 Gb/s 网卡
```

**分桶广播**（发送端是训练侧 rank 0，接收端是所有推理卡；`named_full_params` 已经 gather 好、改好名字）：

```python
import torch, torch.distributed as dist

def send_weights(named_full_params, group, bucket_bytes=512 << 20):
    bucket, size = [], 0
    for name, t in named_full_params:            # 逐个张量产生，峰值显存只多一个张量
        bucket.append((name, t)); size += t.numel() * t.element_size()
        if size >= bucket_bytes:
            flush(bucket, group); bucket, size = [], 0
    if bucket:
        flush(bucket, group)

def flush(bucket, group):
    # 示意：假设一个桶里的张量 dtype 相同，否则按字节切回去时要注意对齐
    meta = [(n, t.shape, t.dtype) for n, t in bucket]
    send_meta(meta)                              # 名字、形状、dtype 走控制面（Ray / RPC）
    flat = torch.cat([t.reshape(-1).view(torch.uint8) for _, t in bucket])
    dist.broadcast(flat, src=0, group=group)     # 一次广播整个桶

def recv_bucket(model, meta, group, device):
    total = sum(torch.Size(s).numel() * torch.empty(0, dtype=d).element_size() for _, s, d in meta)
    flat = torch.empty(total, dtype=torch.uint8, device=device)
    dist.broadcast(flat, src=0, group=group)
    off, tensors = 0, []
    for name, shape, dtype in meta:
        n = torch.Size(shape).numel() * torch.empty(0, dtype=dtype).element_size()
        tensors.append((name, flat[off:off + n].view(dtype).view(shape))); off += n
    model.load_weights(tensors)                  # 推理侧按名字拼进 qkv_proj 等，只取自己的 TP 分片
```

常见题：给定模型大小和链路算同步下界；解释 Megatron 到 vLLM 的布局转换；colocate 下的显存腾挪顺序；同步后为什么要清 prefix cache。

## 参考

- [checkpoint-engine（Moonshot）](https://github.com/MoonshotAI/checkpoint-engine) · [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)
- [HybridFlow（verl）](https://arxiv.org/abs/2409.19256)
- [SGLang for RL](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/sglang_for_rl.mdx)
- [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) · [vLLM weight transfer](https://github.com/vllm-project/vllm/tree/main/docs/training/weight_transfer)
- [slime](https://github.com/THUDM/slime) · [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [He：RL weight sync 优化记录](https://hebiao064.github.io/rl-weight-sync)
- [LMSYS：FP8 RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [NVIDIA Hopper 架构](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

---
title: 一个训练 step 的时间账：同步点与暴露的通信
status: draft
tags: [training-step, profiling, optimizer, overlap, mfu]
difficulty: 3
order: 2.5
related: [/train/optimizer-lr, /train/compute-mfu, /parallel/comm-overlap, /train/data-pipeline, /train/debugging, /train/fault-tolerance]
stack: []
train: [tr-st-opt, tr-st-accum]
---

# 一个训练 step 的时间账：同步点与暴露的通信

> 一个 step 的时间花在哪，除了 forward / backward kernel 以外还有哪些地方会拖慢训练

一个 step 的顺序是：取 batch → forward → backward → 梯度同步 → 裁剪 → 优化器更新 → 调学习率 → 清梯度。forward 和 backward 的 kernel 在 [训练栈第 1 层](/stack/train-layer-1) 讲过，它们占了 step 的大头。其余环节单看都不贵，问题在两件事：**制造同步点**，以及**让通信暴露在关键路径上**。

## 时间大概花在哪

dense LLM 预训练、H100、3D / FSDP 并行、算子调得不错时的量级：

| 环节 | 占 step 时间 | 说明 |
|---|---|---|
| forward | ~25–30% | 每 token $2N$ FLOPs |
| backward | ~50–60% | $4N$；开整层重算再加一次 forward，选择性重算加得少 |
| 暴露的通信 | ~5–15% | 没被 backward 盖住的 all-reduce / reduce-scatter，以及 TP 通信 |
| PP bubble | 0–10% | interleaved 1F1B 下占比 $\frac{p-1}{vm}$，见 [训练算力账](/train/compute-mfu) |
| 裁剪 + 优化器 | ~1–3% | 用 fused / foreach 实现的前提下 |
| 取 batch、调 LR、清梯度 | ≈0 | 做对了的话；做错了就是每步几十 ms 的空洞 |

整体效率的参照：Llama 3 405B 的 MFU 是 38–43%（Llama 3 paper, Table 4），剩下的时间丢在通信、bubble 和非 matmul 算子上。自己的配置怎么分布，用 `torch.profiler` 抓一两个 step，看三样：kernel 之间的空洞、NCCL 和计算的重叠、step 开头和结尾的同步点（读 trace 的方法见 [训练调试](/train/debugging)）。

## 优化器为什么便宜

AdamW 是纯逐元素运算，瓶颈在访存。每个参数读 fp32 主权重、梯度、$m$、$v$（16 B），写回主权重、$m$、$v$（12 B），合计约 28 B。

- 7B 模型不分片：$7\text{e}9 \times 28\text{B} / 3.35\,\text{TB/s} \approx 60$ ms。ZeRO-1、dp = 8 时每卡只更新 $1/8$，约 7.5 ms。
- 同一个 7B，global batch 4M token、64 张 H100，每卡要算 $6 \times 7\text{e}9 \times 4\text{e}6 / 64 \approx 2.6\text{e}15$ FLOPs，按 400 TFLOPS 有效算力约 6.6 s。

所以优化器只占 step 的 1% 上下。前提是用 multi-tensor 实现：一个 Llama 7B 有约 290 个参数张量，逐个参数循环更新每个要发好几个小 kernel，加起来几千次 launch，CPU 开销可能比计算本身还大。PyTorch 的 AdamW 在 CUDA 上默认走 `foreach`（一组张量一次 launch），`fused=True` 更进一步把整个更新融成一个 kernel。

Muon 例外：Newton–Schulz 要整块矩阵，和 ZeRO-1 按元素切分冲突，需要先在 DP 组内 gather，见 [优化器与学习率](/train/optimizer-lr) 第 7 节的 Distributed Muon。

## 各环节的优化点

**取 batch**
- 多 worker 预取，`pin_memory=True` 配合 `.cuda(non_blocking=True)`，H2D 拷贝走 DMA、和计算重叠；packing 放到离线做。
- profiler 里 step 开头 GPU 有空洞，多半是在线 tokenize 或远端存储读不动。判断和修法见 [数据管线](/train/data-pipeline)。

**梯度同步**（最值得抠）
- 梯度分桶，和 backward 重叠（DDP 默认 25 MB 一桶；FSDP 是 reduce-scatter 加 all-gather 预取）。
- 最后一个桶装的是第一层的梯度，backward 算完它才开始通信，天然盖不住；桶越大这段尾巴越长，越小每桶的启动开销越多（[计算通信 overlap](/parallel/comm-overlap)）。
- 梯度累积时前 $K-1$ 个 micro-batch 用 `no_sync()`，否则通信量多 $K$ 倍。

**裁剪**
- 计算量很小，但要算**全局** grad norm：所有 rank 做一次标量 all-reduce，所有 rank 必须用同一个缩放系数。这是每步一次的全局屏障，straggler 在这里暴露。
- 用 `_foreach_norm` 一类的 multi-tensor 实现；不要每步对 norm 调 `.item()`，那是一次 host 同步。

**调学习率**
- CPU 上的一个标量，计算可以忽略。
- 坑：lr 是 Python float 且每步在变，被 `torch.compile` 编译的 optimizer step 会把它当常量，每步触发重编译；CUDA graph 里它也被固化。把 lr 包成 tensor（optimizer 的 `capturable=True`），每步只改 tensor 的值。

**清梯度**
- `zero_grad(set_to_none=True)`（PyTorch 2.0 起默认）：直接释放梯度张量，省一次 memset kernel，也更早还回显存。

## step 外面、但也算进 step 时间的

- **日志里的 `loss.item()`**：每步一次 host 同步，CPU 本来提前好几个 kernel 发射，同步后队列被清空，GPU 要空等下一个 kernel 到达。改成在 GPU 上累加、每 N 步取一次（代价的估算见 [训练调试](/train/debugging) 第 4 节）。
- **Python GC**：各 rank 在不同时刻触发 GC，随机某个 rank 的某一步变慢，其他 rank 在下一个 collective 里陪等。修法是关掉自动 GC，所有 rank 每隔固定步数一起 `gc.collect()`。
- **checkpoint**：同步写会让整个集群停下几十秒到上百秒；异步 checkpoint 先拷到 CPU 内存再后台写盘，阻塞降到秒级（[容错](/train/fault-tolerance)）。

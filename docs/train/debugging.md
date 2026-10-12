---
title: 分布式训练排障：hang、OOM、NaN、慢，各先查什么
status: draft
tags: [debugging, nccl, hang, oom, nan, straggler, flight-recorder]
difficulty: 3
order: 7.5
related: [/train/fault-tolerance, /train/stability, /gpu/profiling, /posttrain/training-memory, /framework/pytorch-internals, /train/compute-mfu, /train/efficient-sft, /posttrain/rl-train-infer-mismatch, /parallel/comm-overlap]
stack: []
---

# 分布式训练排障：hang、OOM、NaN、慢，各先查什么

> 先按现象分类，再跑对应的第一个命令；大多数问题在第一步就能缩小到一个 rank、一个 step、一层

## 一句话结论

四类问题各有一个核心原理。**hang**：collective 按调用顺序配对，某个 rank 少调、多调或形状不同，其他 rank 就一直等，用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 和 flight recorder 找出是哪个 rank、哪一次调用。**OOM**：先分第一步就 OOM（账算错）还是跑一阵才 OOM（长序列、碎片或泄漏），看每步开始时 `memory_allocated` 是否单调上涨。**NaN**：最常见的是 0/0（整个 micro-batch 的 loss mask 全 0）、$0 \times \log 0$、fp16 溢出，用逐层 forward hook 找第一个输出非有限值的 module。**慢**：先用 $6N$ 算期望吞吐，再 profile 一个 step，分清 GPU 忙但慢、GPU 空等、通信没藏住、straggler 四种；同步训练里所有 rank 的 step time 一样长，找 straggler 要比**计算时间**而不是 step time。

## 推导

### 0. 记号与总表

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $n$ | rank 数 | 16 |
| $T_{\text{step}}$ | 一步的墙钟时间 | 500 ms |
| $k$ | 每步强制 CPU–GPU 同步的次数 | 1 / 16 / 512 |
| $t_{\text{gap}}$ | 一次同步后，CPU 从返回到下一个 kernel 到达 GPU 的时间（这段 GPU 空着） | 1 ms（假设） |
| $c_r$ | rank $r$ 每步的**计算**时间（不含等 collective） | 400 ms |
| $\theta$ | 判为 straggler 的阈值：比中位数慢多少 | 10% |
| $\Delta M$ | 每步开始时 `memory_allocated` 的增量 | 0.5 GB |

GB = $10^9$ 字节；PyTorch 报错里的 GiB = $2^{30}$ 字节。

| 现象 | 最可能的原因 | 第一个要跑的命令 / 看的指标 |
|---|---|---|
| 卡住不动，10 分钟后 NCCL timeout | 某 rank 少调 / 多调了一次 collective，或形状不一致 | 复现时开 `TORCH_DISTRIBUTED_DEBUG=DETAIL`；线上开 flight recorder，`torchfrtrace` 分析 dump |
| 卡住，想知道每个进程停在哪行 | 同上，或死循环、等数据 | 每个 rank `py-spy dump --pid <pid>` |
| 第一步就 OOM | 显存账算错 | 对照 [训练显存账](/posttrain/training-memory) 的公式，看 `max_memory_allocated` |
| 跑了几百步才 OOM | 碰到最长序列、碎片、泄漏 | 每步开始打 `memory_allocated` 和 `memory_reserved`；录 memory snapshot |
| loss 突然 NaN | 0/0、$\log 0$、fp16 溢出、坏数据、lr 太大 | 逐层 forward hook 找第一个非有限输出；打印这一步的有效 token 数 |
| MFU 一直偏低 | kernel 慢、GPU 空等、通信暴露 | 和 $6N$ 算出的期望比，然后 [profile](/gpu/profiling) 一个 step |
| 吞吐突然掉了 | 某节点降频 / 掉卡、网卡降速、写 checkpoint、Python GC | 每个 rank 的计算时间 $c_r$ 对比中位数 |
| 换并行配置后 loss 曲线不重合 | bf16 舍入，或真的有 bug | 切到 fp32 比第 0 步；逐阶段二分 |

loss spike 本身（不是 NaN）的根因和处理见 [训练稳定性](/train/stability)；硬件故障、checkpoint 间隔、SDC 见 [大规模训练的容错](/train/fault-tolerance)。

### 1. Hang

**原理**：NCCL 不检查「这次 all_reduce 是不是代码里同一行」，它只按每个 rank 上的**调用顺序**配对：rank 0 的第 $j$ 次 collective 和 rank 3 的第 $j$ 次配成一组。要求所有 rank 以相同顺序调用同类型、同形状的操作。一旦某个 rank 多一次或少一次，后面全部错位：

1. 错位的两次形状相同：不卡，**静默算错**（把不同 step 的梯度加在一起），直到后面某次形状或类型不一致才卡住。
2. 形状或类型不同：直接卡住，或者读写越界。
3. 某个 rank 已经退出循环：其他 rank 在下一次 collective 里永远等。

常见触发：

1. **数据相关的分支**：`if torch.isnan(loss): continue` 只在看到 NaN 的那个 rank 上生效；MoE 某个 rank 没有 token 发给某专家就跳过了 all-to-all。
2. **只在 rank 0 上做带 collective 的事**：`if rank == 0: model.state_dict()` 在 FSDP 下要 all-gather 参数，`if rank == 0: evaluate(model)` 的 forward 也要 all-gather，rank 0 自己等不到其他人。
3. **某 rank 早退出**：各 rank 的 batch 数不一样，数据少的那个先跑完循环。
4. **NCCL / 网络本身**：初始化时连不上 master（地址、端口、防火墙），或者训练中某块网卡故障。

**一个最小例子**：4 个 rank，rank 3 在 step 1 看到 NaN 自己跳过了这一步。CPU 上用 gloo 能复现（`timeout` 设成 5 秒）：

```python
for step in range(3):
    grad = torch.full((4,), float(step))
    loss = torch.tensor(float("nan") if (rank == 3 and step == 1) else 1.0)
    if torch.isnan(loss):          # 只有 rank 3 看到了 NaN，自己跳过这一步
        continue
    dist.all_reduce(grad)          # 梯度同步
    print(f"rank {rank} step {step} grad {grad[0].item()}")
dist.barrier()
```

实测输出（PyTorch 2.5.1，gloo）：

```text
rank 0 step 1 grad 5.0      ← 应该是 1+1+1+1 = 4：rank 3 的 step 2 梯度（=2）被加进来了
rank 3 step 2 grad 5.0      ← rank 3 的 step 2 和别人的 step 1 配成了一对
RuntimeError: ... Timed out waiting 5000ms for recv operation to complete
```

1. 形状一样，第一次错位没有报错，梯度已经错了：$1 + 1 + 1 + 2 = 5$。
2. rank 3 进了 `barrier`，rank 0–2 还在第 3 次 `all_reduce` 里，这次类型对不上才卡住。
3. 修法：跳不跳过这一步要所有 rank 一起决定，先把「本 rank 有没有 NaN」的标志 all-reduce（取 max），再统一 `continue`。

**工具，按顺序**：

1. **超时与 watchdog**：`init_process_group` 的 `timeout` 对 NCCL 默认 10 分钟，到时间后 watchdog 中止 collective、进程退出（[torch.distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)）。行为由 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 控制，默认 3：直接拆掉进程（[NCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)）。超时只说明「卡了」，不说明是谁。
2. **`TORCH_DISTRIBUTED_DEBUG=DETAIL`**：每次 collective 前先检查所有 rank 的操作类型和张量形状是否一致，有额外开销，用于复现（文档：「may impact the application performance」）。上面的例子开了它，错位那一刻直接报：

   ```text
   RuntimeError: Detected mismatch between collectives on ranks. Rank 3 is running collective:
   CollectiveFingerPrint(SequenceNumber=2OpType=BARRIER), but Rank 0 is running collective: ...
   ```

   它在形状不同时立刻报（另一个 2 rank 的实测：`Tensor shapes: 2vs 4`）；形状完全相同的错位它也抓不到，要等到类型不同的那次。
3. **`NCCL_DEBUG=INFO`**：NCCL 自己的日志，看初始化、拓扑和走的是哪种网络（IB 还是 socket）。卡在初始化、或者怀疑网络时看它。
4. **flight recorder**：生产作业不能开 DETAIL，用它。每个 rank 在内存环形缓冲区里记最近的 collective：进程组、collective 序号、操作名（如 `nccl:all_reduce`）、输入输出形状、状态（scheduled / started / completed）、Python 调用栈（[Flight Recorder 教程](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)）。
   - `TORCH_NCCL_TRACE_BUFFER_SIZE=2000`：缓冲区条数，大于 0 才开启（教程推荐 2000）。
   - `TORCH_NCCL_DUMP_ON_TIMEOUT=1`：超时时每个 rank 写一个文件。
   - `TORCH_FR_DUMP_TEMP_FILE`：文件路径前缀，默认 `/tmp/nccl_trace_rank_`（较早版本的变量名是 `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`）。
   - 分析：把所有 rank 的文件放到一个目录，`torchfrtrace <dir>`（或源码里的 `fr_trace.py`）。它按 collective 序号对齐各 rank，报出第一个「不是所有 rank 都加入」的 collective 和缺席的 rank。
5. **`py-spy dump --pid <pid>`**：打印进程当前的 Python 调用栈，每个 rank 跑一次。注意 NCCL 是异步的：`all_reduce` 只是把 kernel 排进 stream 就返回，CPU 会继续往下走，**卡在下一个同步点**（`loss.item()`、`synchronize`）。所以 py-spy 显示停在 `.item()` 不代表问题在那，要以 flight recorder 的序号为准。

**用 flight recorder 定位上面的例子**（NCCL 下，示意）：

| rank | 序号 0 | 序号 1 | 序号 2 | 序号 3 |
|---|---|---|---|---|
| 0–2 | all_reduce [4] completed | all_reduce [4] completed | all_reduce [4] started | barrier |
| 3 | all_reduce [4] completed | all_reduce [4] completed | barrier（等不到人） | — |

1. 从小到大找第一个各 rank 不一致的序号：序号 2，rank 3 是 barrier，其他是 all_reduce。
2. 看各 rank 每个调用位置的次数：rank 0–2 在 `train.py` 的 `all_reduce(grad)` 那行有 3 条，rank 3 只有 2 条，说明 rank 3 少走了一次。
3. 回到代码，找这一行之前、只在部分 rank 上成立的分支：`continue`。

### 2. OOM

**先区分发生在什么时候**：

| 时机 | 原因 | 怎么确认 |
|---|---|---|
| 第一步就 OOM | 静态账算错：模型状态、激活、logits 超了 | 按 [训练显存账](/posttrain/training-memory) 的通用公式重算；注意 Adam 的 $m$、$v$ 在第一次 `optimizer.step()` 才分配，峰值在第一步末尾 |
| 跑一阵才 OOM，之前峰值平稳 | 碰到了最长的那条序列 | OOM 那一步的 token 数比平时大；开训前先用最长的 batch 跑一步 |
| 跑一阵才 OOM，reserved 远大于 allocated | 碎片：变长的块把 segment 切碎 | 报错里「reserved by PyTorch but unallocated」很大；原理见 [PyTorch 内部机制](/framework/pytorch-internals) 第 4 节 |
| 每步开始时 allocated 单调上涨 | 泄漏：有张量被一直引用着 | 每步开始打 `torch.cuda.memory_allocated()`，画出来是一条斜线 |

**泄漏**最常见的两种（[PyTorch FAQ](https://docs.pytorch.org/docs/stable/notes/faq.html)）：

1. `total_loss += loss`：`loss` 带着 autograd 历史，累加让历史跨 step 一直被引用。改成 `total_loss += loss.detach()`（留在 GPU 上，见第 4 节为什么不每步 `.item()`）。
2. 把带梯度的张量存进 list：比如 eval 时没包 `torch.no_grad()`，`outputs.append(logits)`，每个 `logits` 都拉着整张图的激活。

**算一笔**：每步开始时 allocated 从 60 GB 起，每步涨 $\Delta M = 0.5$ GB，卡上可用 80 GB：$(80 - 60)/0.5 = 40$ 步后 OOM。泄漏的特征就是「OOM 的步数和每步涨多少能对上」，而且和那一步的数据无关。

**工具**：

1. `torch.cuda.max_memory_allocated()` vs `max_memory_reserved()`：前者是张量实际占用的峰值，用来和公式比；后者减前者是缓存加碎片。每次读之前用 `torch.cuda.reset_peak_memory_stats()` 清零。
2. `torch.cuda.memory_summary()`：allocator 的统计表，看大块池的空闲块分布。
3. memory snapshot（[Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)）：

   ```python
   torch.cuda.memory._record_memory_history(max_entries=100_000)   # 开始记录每次分配和释放
   for _ in range(5):
       train_step()
   torch.cuda.memory._dump_snapshot("snapshot.pickle")
   torch.cuda.memory._record_memory_history(enabled=None)          # 停止记录
   ```

   把 `snapshot.pickle` 拖进 <https://pytorch.org/memory_viz>（在浏览器本地解析，不上传）。Active Memory Timeline 画出每个活着的张量随时间的变化，鼠标悬停显示它分配时的调用栈：泄漏是一条每步都往上叠的带子，点开就是那行代码；碎片看 Allocator State History，OOM 的那次事件会被标出来。

**旋钮顺序**（从免费到贵）：

1. logits：词表大时先上 fused linear cross-entropy，不改数学、不多算（[SFT 提速](/train/efficient-sft) 第 3 节）。
2. 碎片：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，micro-batch 按 token 数组（形状稳定）。
3. 减 micro-batch 的 token 数，用 gradient accumulation 保持 global batch。只要每个 micro-batch 的 GEMM 还是 compute-bound，每 token 速度不变（[SFT 提速](/train/efficient-sft) 第 5 节）。
4. 逐层开 recompute：每多 checkpoint 一层，这一层只留输入、其余激活扔掉，代价是 backward 时多算这一层的一次 forward（每层省多少、多算多少见 [SFT 提速](/train/efficient-sft) 第 5 节）。
5. 更多切分：ZeRO-2 → FSDP，或加 TP / CP（[ZeRO / FSDP](/parallel/zero-fsdp)）。
6. CPU offload：最后才用，PCIe 带宽比 HBM 低一两个数量级。

这正好是 [SFT 提速](/train/efficient-sft) 里「省下的显存先关 recompute、再加 micro-batch」的反方向。

### 3. NaN / Inf

**来源**，每种给一个能自己算的例子：

1. **0/0：整个 micro-batch 没有一个计 loss 的 token**。SFT 里 loss mask 只在回答上为 1；一个 micro-batch 恰好全是 prompt（长 prompt 被截断、回答全被切掉），按 micro-batch 平均就是
   $$\mathcal{L} = \frac{\sum_i m_i \ell_i}{\sum_i m_i} = \frac{0}{0} = \text{NaN}$$
   $m_i \in \{0,1\}$ 是第 $i$ 个 token 的 loss mask，$\ell_i$ 是它的交叉熵。PyTorch 里 `F.cross_entropy` 的 target 全是 `ignore_index`、`reduction="mean"` 时也返回 `nan`（实测）。一个 rank 的 NaN 梯度经过 all-reduce 后，所有 rank 的梯度都是 NaN。正确做法见 [SFT](/posttrain/sft) 第 3 节：分母用整个优化步、跨 DP all-reduce 后的有效 token 总数 $N$，每个 micro-batch 用 $\sum_i m_i\ell_i / N$ 反传。这样空的 micro-batch 贡献 0，不会除零。
2. **$0 \times \log 0$**：先 `softmax` 再 `log`。logits $[0, -200]$，第二项 $e^{-200} \approx 1.4\times10^{-87}$，比 fp32 能表示的最小正数（非正规数 $\approx 1.4\times10^{-45}$）还小，下溢成 0，`log` 得 $-\infty$；再乘一个 0（mask 或概率）：$0 \times (-\infty) = \text{NaN}$。改用 `log_softmax`，直接得到 $-200$。
3. **fp16 溢出**：fp16 最大 65504。attention 的 $q$、$k$ 是 128 维、每个元素 25，点积 $128 \times 25^2 = 80000 > 65504$，在 fp16 里变成 inf；softmax 先减这一行的最大值，$\infty - \infty = \text{NaN}$，整行都是 NaN（实测）。先乘后除 $\sqrt{d_h}$ 的写法更容易溢出（$d_h$ 是 head 维度）。动态 loss scaling 会跳过梯度有 inf 的步，scale 一直在减半说明溢出频繁（[训练显存账](/posttrain/training-memory)）。bf16 范围和 fp32 一样，没有这个问题。
4. **坏数据**：输入里本身有 NaN（比如图像预处理时除以的 std 是 0）。token id 越界则是另一种表现：GPU 上触发 device-side assert，但 kernel 是异步的，报错出现在后面别的行；设 `CUDA_LAUNCH_BLOCKING=1` 让报错停在真正出错的那行。
5. **lr 太大 / 梯度爆炸**：grad norm 先涨、loss 再炸，属于稳定性问题，见 [训练稳定性](/train/stability)。

**工具，按顺序**：

1. **先固定复现**：从出事前最近的 checkpoint、同样的数据位置重跑这一步（数据位置怎么恢复见 [预训练数据管线](/train/data-pipeline) 第 6 节）。能复现，才能用下面的慢工具；复现不了而且只在某台机器上出现，怀疑 SDC（[容错](/train/fault-tolerance) 第 6 节）。
2. **打印这一步的统计**：每个 rank 的有效 token 数（为 0 就是第 1 种）、loss 是不是在 all-reduce **之前**就 NaN（区分是哪个 rank 先坏）。
3. **逐层 forward hook**（手撕 (a)）：每个 module 的 forward 结束时检查输出 `torch.isfinite`，第一个不通过的就是源头；同时看它的输入是否正常，输入已经坏了说明问题在它前面的非 module 代码或数据里。
4. **`torch.autograd.detect_anomaly()`**：forward 时记下每个算子的调用栈，backward 里哪个算子产生 NaN 就报错并打印它 forward 时的位置（实测报错：`Function 'SqrtBackward0' returned nan values in its 0th output`）。很慢，只在复现时开。forward 正常、backward 才 NaN 的情况用它。
5. **`TORCH_NCCL_NAN_CHECK=1`**：NCCL collective 的输入里有 NaN 就报错（[NCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)），能指出是哪个 rank 把 NaN 带进了 all-reduce。

### 4. 慢

**第一步：算期望**（[训练算力账](/train/compute-mfu)）：

$$
T_{\text{gpu}} = \frac{\text{MFU} \cdot \text{peak}}{6N}
$$

$T_{\text{gpu}}$ 是每卡 token/s，$N$ 是参数量，peak 是单卡峰值（H100 bf16 dense 989 TFLOP/s）。例：8B、MFU 40% 时期望 $0.4 \times 9.89\times10^{14} / (6 \times 8\times10^9) \approx 8200$ token/s/GPU。实测 4000，就有一半时间去向不明。

**第二步：profile 一个 step**（[Profiling](/gpu/profiling)），把时间线归到四类：

| 时间线上的样子 | 类别 | 下一步 |
|---|---|---|
| GPU 一直有 kernel，但吞吐低 | kernel 本身慢：非 GEMM 算子多、GEMM 太小、没用上 FlashAttention | 汇总表按 GPU 时间排序，融合小算子（[SFT 提速](/train/efficient-sft) 第 4 节） |
| GPU 行有空隙，CPU 卡在 `cudaStreamSynchronize` 或 dataloader | GPU 空等：数据、Python、同步点 | 去掉每步的 `.item()`，查 `next(data_iter)` 的耗时（[数据管线](/train/data-pipeline) 第 7 节） |
| NCCL kernel 单独跑，计算 stream 空着 | 通信没藏住 | 调 bucket / 预取，换并行方式（[计算通信 overlap](/parallel/comm-overlap)） |
| 本 rank 的计算很快，但在 collective 里等很久 | 别的 rank 是 straggler | 对比各 rank 的计算时间，见下面 |

**同步点的代价**：`loss.item()`、`print(loss)`、`if loss > x:`、`.cpu()` 都要等 GPU 把队列里的活干完、把结果拷回 CPU。平时 CPU 提前好几个 kernel 发射，GPU 不用等；一同步，队列被清空，CPU 返回后要花 $t_{\text{gap}}$ 才能让下一个 kernel 到达 GPU，这段 GPU 空着。每步损失约

$$
\frac{k \cdot t_{\text{gap}}}{T_{\text{step}}}
$$

$T_{\text{step}} = 500$ ms、$t_{\text{gap}} = 1$ ms（假设）：

1. 每步一次 `loss.item()`：$k = 1$，0.2%，可以接受。
2. 每个 micro-batch 打一次 loss，16 个 micro-batch：$k = 16$，3.2%。
3. MoE 每层每个 micro-batch 打一次 aux loss，32 层 × 16：$k = 512$，512 ms，step 时间翻倍。

而且 $t_{\text{gap}}$ 不是常数：同步之后 CPU 和 GPU 齐步走，后面一串短 kernel 的发射时间都会暴露，直到 CPU 重新领先。做法是把要记录的量 `detach()` 后在 GPU 上累加，每 $n$ 步 `.item()` 一次。

**突然变慢**：MFU 原来正常，某一刻起掉了。同步训练里一个 rank 慢，所有 rank 的 step time 都一起变长，所以**比 step time 找不到人**。要比的是每个 rank 的计算时间 $c_r$（两次 collective 之间的 GPU 时间，用 CUDA event 量），或者反过来比每个 rank 在 collective 里等了多久：straggler 等得最少，因为别人都在等它。

| 原因 | 特征 | 确认 |
|---|---|---|
| GPU 降频 / 坏卡 | 固定某个 rank 的 $c_r$ 一直偏大 | `nvidia-smi -q -d PERFORMANCE` 看降频原因（新版叫 Clocks Event Reasons），`-d TEMPERATURE` 看温度 |
| 网卡降速 | 计算时间正常，某节点的 collective 时间变长 | `ibstat` 看端口的 Rate 是不是比别的节点低 |
| 写 checkpoint | 每隔固定步数变慢一次，和 checkpoint 间隔对得上 | 改异步 checkpoint（[容错](/train/fault-tolerance) 第 4 节） |
| Python GC | 随机的 rank、随机的步偶发变慢 | 关自动 GC，所有 rank 每隔固定步数一起 `gc.collect()` |

ByteDance 对其 LLM 训练集群 5 个月的追踪（[Lin et al. 2025](https://arxiv.org/abs/2505.05713)，作者自测）：42.5% 的作业出现 straggler（慢于无 straggler 情形 10% 以上），浪费了 10.4% 的 GPU 小时。主因不是坏机器：单个 worker 造成一半以上慢速的作业只占 straggler 作业的 1.7%；更多来自 PP 各 stage 切得不均、各 rank 序列长度不均（21.4% 的作业，平均慢 1.34 倍），以及 Python GC（一个 128 DP rank 的作业改成每 500 步统一 GC 后快 12.6%）。

### 5. 数值对不上

**场景**：TP 从 1 改成 2，或者换了 attention kernel，loss 曲线和原来不重合。

**为什么一定会有差**：浮点加法不满足结合律，切分方式变了，归约顺序就变了。bf16 里 $[256, 1, 1, 1, 1]$ 从左往右加得 256（每次 $256 + 1$ 都舍入回 256），先加四个 1 再加 256 得 260（实测）。原理和量级见 [训推不一致](/posttrain/rl-train-infer-mismatch) 第 3 节。

**多大算正常**：用自己的噪声带当标准，不用一个固定阈值。

1. 噪声带：原配置换两个随机种子各跑一次，两条曲线之间的差就是正常波动。新配置的曲线落在带里就算对上。
2. 第 0 步：同一份权重、同一个 batch 只做 forward，两种配置的 loss 差异应在 bf16 舍入量级。bf16 有效位 8 位，单次舍入的相对误差最多 $2^{-8} \approx 0.39\%$。
3. 判别舍入还是 bug：两种配置都切到 fp32 再比第 0 步。fp32 的 ulp 比 bf16 小 $2^{23}/2^{7} = 2^{16}$ 倍，舍入造成的差会缩小几个数量级；差还在，就是 bug。

**二分定位**：每次只改一个维度（只改 TP，不同时改 PP），用同一份权重、同一个 batch：

1. **forward**：比 loss。对不上就用 forward hook 记下每层输出，比相对误差，找第一个跳出舍入量级的层。
2. **backward**：比每个参数的梯度（把分片 gather 成完整张量再比）。forward 对、梯度不对，查梯度的归约：loss 在 DP 上被平均了两次或没平均、sequence parallel 下 LayerNorm 权重的梯度没在 TP 组内 all-reduce、PP 下首尾两个 stage 共享的 embedding 梯度没同步。
3. **optimizer step**：比更新后的参数。前两步都对、这里不对，查梯度裁剪的 global norm 是否跨所有分片算（只在本 rank 上算 norm 会导致每个 rank 裁剪系数不同）。

## 面试追问

::: details Q：训练卡住了，没有任何报错，你第一步做什么？
先确定是不是所有 rank 都卡在 collective：每个 rank `py-spy dump` 看 Python 栈，注意 NCCL 是异步的，栈停在 `.item()` 或 `synchronize` 也可能是前面的 collective 没完成。生产环境靠 flight recorder：开 `TORCH_NCCL_TRACE_BUFFER_SIZE` 和 `TORCH_NCCL_DUMP_ON_TIMEOUT`，超时后用 `torchfrtrace` 对齐各 rank 的 collective 序号，找第一个不是所有 rank 都加入的那次，再看缺席 rank 的调用栈。能复现的话开 `TORCH_DISTRIBUTED_DEBUG=DETAIL`，第一次类型或形状不一致就会报出来。
:::

::: details Q：为什么 rank 0 上保存 FSDP 的 checkpoint 会卡住？
FSDP 的参数分片在各个 rank 上，拿完整的 `state_dict` 要 all-gather，这是 collective，所有 rank 都得调用。写成 `if rank == 0: model.state_dict()`，rank 0 进了 all-gather，其他 rank 没进，rank 0 等到超时。要么所有 rank 都调用、只让 rank 0 写文件，要么用分布式 checkpoint 让每个 rank 写自己的分片（[容错](/train/fault-tolerance) 第 4 节）。
:::

::: details Q：某个 rank 遇到 NaN 就跳过这一步，有什么问题？
其他 rank 照常 all-reduce，这个 rank 少调一次，从此所有 collective 错位。形状相同时不报错，梯度被静默地加错（上面的例子 4 变成 5）；到了形状或类型不同的那一次才卡住。跳过与否必须全体一致：把本 rank 的 NaN 标志 all-reduce 取 max，再一起跳过。
:::

::: details Q：训练跑了 2000 步 OOM，怎么判断是泄漏还是碎片？
每步开始时打 `memory_allocated` 和 `memory_reserved`。allocated 单调上涨是泄漏，录 memory snapshot，在 Active Memory Timeline 里找每步往上叠的那条，悬停看分配位置；allocated 平稳、reserved 远大于 allocated、报错里「reserved but unallocated」很大是碎片，开 `expandable_segments`、按 token 数组 batch 让形状稳定。两者都不是，看 OOM 那一步的 token 数是不是最长的。
:::

::: details Q：SFT 训练偶尔 loss 变 NaN，又不是每次都复现，可能是什么？
先查那一步每个 micro-batch 的有效 token 数。全是 prompt 的 micro-batch 在「按 micro-batch 平均」的写法下是 0/0；它和数据顺序有关，所以看起来随机。改成用整个优化步的全局有效 token 数做分母，空的 micro-batch 贡献 0。如果有效 token 数正常，再用逐层 hook 找第一个非有限输出。
:::

::: details Q：同步训练里所有 rank 的 step time 都一样，怎么找出慢的那个？
step time 一样是因为大家都在 collective 里等最慢的那个。要量每个 rank 两次 collective 之间的计算时间（CUDA event），或者量每个 rank 在 collective 里等待的时间：straggler 计算时间最长、等待时间最短。每个 rank 先对多步取中位数（排除偶发的 GC），再和所有 rank 的中位数比，超过阈值的就是慢节点。
:::

::: details Q：换了 TP 配置，loss 第 0 步差了 0.3%，算 bug 吗？
不一定。bf16 单次舍入相对误差最多约 0.39%，切分变了归约顺序就变，有差是正常的。判别方法：两种配置都改成 fp32 再比第 0 步，舍入误差会缩小几个数量级；还差 0.3% 就是 bug。之后按 forward → backward → optimizer step 二分，每次只改一个并行维度。
:::

## 手撕

(a) 逐层 forward hook，打印第一个输出含 NaN / Inf 的 module 名；(b) 给定各 rank 每步的计算时间，用中位数加阈值找 straggler。CPU 上能跑：

```python
import random
import statistics
import torch
import torch.nn as nn


# ---- (a) 找第一个输出 NaN / Inf 的 module ----
def install_nan_hooks(model):
    """每个 module 的 forward 结束时检查输出；第一次发现非有限值就抛异常，停在那里"""
    def bad(xs):
        xs = xs if isinstance(xs, (tuple, list)) else (xs,)
        return any(torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all()
                   for t in xs)

    def make_hook(name):
        def hook(mod, inp, out):
            if bad(out):
                where = "输入已经坏了，查它之前的代码或数据" if bad(inp) else "输入正常，就是它产生的"
                raise FloatingPointError(f"first non-finite: {name} ({type(mod).__name__})，{where}")
        return hook

    # 子 module 比父 module 先结束 forward，所以第一个触发的就是最内层的
    return [m.register_forward_hook(make_hook(n)) for n, m in model.named_modules() if n]


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm, self.fc1, self.act, self.fc2 = nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


torch.manual_seed(0)
model = nn.Sequential(Block(8), Block(8), nn.Linear(8, 4))
handles = install_nan_hooks(model)

with torch.no_grad():
    model[1].fc1.weight[0, 0] = float("inf")       # 模拟一次坏的更新
for x in [torch.randn(2, 8), torch.tensor([[float("nan")] + [0.0] * 7])]:
    try:
        model(x)
    except FloatingPointError as e:
        print(e)
        with torch.no_grad():
            model[1].fc1.weight[0, 0] = 0.0          # 第二个例子：权重修好，数据里有 NaN
for h in handles:
    h.remove()
# first non-finite: 1.fc1 (Linear)，输入正常，就是它产生的
# first non-finite: 0.norm (LayerNorm)，输入已经坏了，查它之前的代码或数据


# ---- (b) 从每个 rank 的计算时间里找 straggler ----
def find_stragglers(times, thr=0.10):
    """times: {rank: [每步计算时间 ms]}，不含等 collective 的时间。
    每个 rank 取自己的中位数（抗偶发的 GC），再和所有 rank 的中位数比，超出 thr 的算慢"""
    per_rank = {r: statistics.median(ts) for r, ts in times.items()}
    base = statistics.median(per_rank.values())
    slow = {r: round(t / base, 3) for r, t in per_rank.items() if t > base * (1 + thr)}
    return base, dict(sorted(slow.items(), key=lambda kv: -kv[1]))


random.seed(0)
times = {r: [400 + random.gauss(0, 5) for _ in range(20)] for r in range(16)}
times[11] = [t * 1.25 for t in times[11]]          # rank 11 降频，每步慢 25%
times[3][7] += 800                                 # rank 3 某一步 GC，偶发一次
base, slow = find_stragglers(times)
print(f"中位数 {base:.1f} ms，慢的 rank：{slow}")
# 中位数 399.3 ms，慢的 rank：{11: 1.254}
```

要点：

1. hook 只能看到 module 的边界。NaN 若产生在父 module 的 `forward` 里两个子 module 之间的函数调用上（比如一个裸的 `torch.log`），会被报成下一个子 module「输入已经坏了」，这时去看它前面那几行。
2. 用中位数而不是均值：rank 3 那一步多了 800 ms，均值会被拉高 40 ms，中位数不受影响；偶发的慢交给 GC 那一行的处理，持续的慢才是要换掉的节点。
3. 常见变体：给每个 rank 在 collective 里的等待时间，找等待最短的；或者要求同时报出「偶发慢」的 rank（比如超过阈值的步数占比大于 20%）。

## 参考

- [torch.distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)：`init_process_group` 的 timeout、`TORCH_DISTRIBUTED_DEBUG`
- [PyTorch NCCL 环境变量](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)：`TORCH_NCCL_ASYNC_ERROR_HANDLING`、`TORCH_NCCL_DUMP_ON_TIMEOUT`、`TORCH_NCCL_NAN_CHECK`
- [Flight Recorder for Debugging Stuck Jobs](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)：环境变量、`torchfrtrace`
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)：`_record_memory_history`、`_dump_snapshot`、memory_viz
- [PyTorch FAQ](https://docs.pytorch.org/docs/stable/notes/faq.html)：「Don't accumulate history across your training loop」
- [torch.autograd.detect_anomaly](https://docs.pytorch.org/docs/stable/autograd.html#anomaly-detection)
- [py-spy](https://github.com/benfred/py-spy)
- [Understanding Stragglers in Large Model Training Using What-if Analysis](https://arxiv.org/abs/2505.05713)：straggler 的比例与成因
- [NCCL 环境变量（NVIDIA）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)：`NCCL_DEBUG`

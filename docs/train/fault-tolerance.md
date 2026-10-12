---
title: 大规模训练的容错：故障率、Checkpoint 间隔与快速恢复
status: draft
tags: [fault-tolerance, checkpoint, reliability, dcp, mtbf]
difficulty: 3
order: 7
related: [/posttrain/training-memory, /parallel/zero-fsdp, /parallel/parallelism-overview, /parallel/collective-comm]
stack: []
train: [tr-j-ft, tr-hw-io]
---

# 大规模训练的容错：故障率、Checkpoint 间隔与快速恢复

> 卡越多，作业挂得越频繁；容错就是在「写 checkpoint 的开销」和「挂了丢掉的工作」之间找平衡，再把写入时间和重启时间压到最小

## 一句话结论

同步训练里任何一张卡出事，整个作业都要停，所以作业的 MTBF 是单卡 MTBF 除以卡数：Llama 3 405B 在约 16K 张 H100 上 54 天遇到 419 次非计划中断，平均约 3 小时一次。checkpoint 是 fp32 主权重加 Adam 状态，每参数 12 字节，405B 约 4.9 TB。最优间隔用 Young 公式 $\tau^* = \sqrt{2\delta M}$，浪费的时间比例约 $\sqrt{2\delta/M} + R/M$。要提高有效训练时间比例（ETTR），就压写入阻塞 $\delta$（异步、分片并行写、去重）和重启耗时 $R$（内存 checkpoint、热备节点、快速检测）。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $n$ | 作业用的 GPU 数（$n_{\text{gpu}}$） | 16384 |
| $M_{\text{gpu}}$ | 单卡 MTBF：平均每张卡多久出一次能打断作业的故障 | 50,000 小时 |
| $M$ | 作业 MTBF：整个作业平均多久被打断一次 | 3.1 小时 |
| $\delta$ | 写一次 checkpoint 让训练停下来的时间 | 100 s（同步写）、5 s（异步） |
| $\tau$ | 两次 checkpoint 之间的训练时间 | 待求 |
| $R$ | 一次故障后的重启耗时：检测、换机、拉起进程、加载 checkpoint，直到恢复训练 | 600 s |
| $\Psi$ | 参数量 | $405\times10^9$ |
| $B_{\text{fs}}$ | 文件系统总写带宽 | 50 GB/s（假设） |
| ETTR | 有效训练时间比例：真正推进训练的时间 / 墙钟时间 | |

GB = $10^9$ 字节，TB = $10^{12}$ 字节。

### 1. 故障有多频繁

**推导**：每张卡的故障相互独立，单卡故障率 $\lambda = 1/M_{\text{gpu}}$（按指数分布，即单位时间出事的概率恒定）。同步训练里任何一张卡出事作业就停，作业的故障率是 $n$ 张卡之和：

$$
\lambda_{\text{job}} = n\lambda \quad\Rightarrow\quad M = \frac{M_{\text{gpu}}}{n}
$$

卡数翻倍，作业 MTBF 减半。这里的「单卡」是把所有原因（GPU、网络、主机、软件）都摊到每张卡上的等效值。

**Llama 3 405B 的数据**（[arXiv 2407.21783](https://arxiv.org/abs/2407.21783) §3.3.4、Table 5）：

1. 预训练中的一段 54 天，共 466 次作业中断：47 次计划内（自动维护等），419 次非计划。
2. 非计划中断里约 78% 是确认或疑似的硬件问题，GPU 相关的占 58.7%。
3. 最大的几类：GPU 故障 30.1%、HBM3 显存 17.2%、软件 bug 12.9%、网络交换机 / 线缆 8.4%、主机维护 7.6%。还有 NCCL watchdog 超时 7 次、silent data corruption 6 次。
4. 训练用到 16K 张 H100；整个过程只有 3 次需要大量人工介入，有效训练时间高于 90%（作者自测）。

（Table 5 里「Faulty GPU」一行写的是 148 次、30.1%，148/419 是 35.3%，两者对不上；GPU 各行的百分比加起来正好是 58.7%，以百分比为准。）

**算一笔**：

1. 作业 MTBF：$54 \times 24 / 419 = 3.09$ 小时。
2. 反推单卡等效 MTBF：$3.09 \times 16384 \approx 50{,}700$ 小时，约 5.8 年。单张卡其实很可靠，是卡数把它放大了。
3. 同样的卡，1024 卡的作业：$50{,}700 / 1024 = 49.5$ 小时，约两天一次。10 万卡：约 30 分钟一次。

### 2. Checkpoint 有多大、写多久

**存什么**：恢复训练需要的全部状态。

| 东西 | 精度 | 每参数字节 |
|---|---|---|
| 主权重 | fp32 | 4 |
| Adam $m$ | fp32 | 4 |
| Adam $v$ | fp32 | 4 |
| **合计** | | **12** |

1. bf16 参数不用存：从 fp32 主权重 cast 一下就得到。
2. 梯度不用存：一步用完就扔，恢复后从下一步重新算。
3. 另外还有几 KB 到几 MB 的小东西：step 数、学习率调度器状态、RNG 状态、数据加载器读到哪了。
4. 每参数 16 字节的来历见 [训练显存账](/posttrain/training-memory)，这里比显存少了 bf16 参数和梯度的 4 字节。

**大小**：

| 模型 | $12\Psi$ | 按 $B_{\text{fs}} = 50$ GB/s 写 |
|---|---|---|
| 70B | 840 GB | 16.8 s |
| 405B | 4.86 TB | 97 s |

$\delta = 12\Psi / B_{\text{fs}}$ 的前提是所有 rank 一起写、把文件系统的总带宽用满（第 4 节）。如果先 gather 到 rank 0 再由它一个人写，带宽只有单节点的，405B 按单节点 5 GB/s（假设）要 16 分钟。

### 3. 最优间隔：Young 公式

**浪费的时间**：每隔 $\tau$ 的训练写一次 checkpoint，停 $\delta$。故障在一个间隔内均匀发生，平均丢掉半个间隔 $\tau/2$ 的工作，再花 $R$ 重启。单位时间内浪费的比例：

$$
W(\tau) = \underbrace{\frac{\delta}{\tau}}_{\text{写 checkpoint}} + \underbrace{\frac{\tau/2}{M}}_{\text{丢掉的工作}} + \underbrace{\frac{R}{M}}_{\text{重启}}
$$

1. 第一项：$\tau$ 越小写得越勤，开销越大。
2. 第二项：$\tau$ 越大，每次故障丢得越多；单位时间平均发生 $1/M$ 次故障。
3. 第三项：和 $\tau$ 无关，只能靠缩短 $R$。

对 $\tau$ 求导令其为 0：$-\delta/\tau^2 + 1/(2M) = 0$，得

$$
\tau^* = \sqrt{2\delta M}
$$

这是 Young（1974）的一阶近似，前提是 $\delta \ll \tau \ll M$；Daly（2006）给了更高阶的修正。最优点上前两项相等，都是 $\sqrt{\delta/(2M)}$，于是

$$
\text{ETTR} \approx 1 - W(\tau^*) = 1 - \sqrt{\frac{2\delta}{M}} - \frac{R}{M}
$$

ETTR（Effective Training Time Ratio）这个名字来自 Meta 对其研究集群可靠性的分析（[arXiv 2410.21680](https://arxiv.org/abs/2410.21680)）。

**代入**（$M_{\text{gpu}} = 50{,}700$ 小时，见第 1 节）：

| $n$ | $\delta$ | $R$ | $M$ | $\tau^*$ | ETTR |
|---|---|---|---|---|---|
| 16384 | 100 s（同步写） | 600 s | 3.09 h | 24.9 min | 0.812 |
| 16384 | 5 s（异步） | 600 s | 3.09 h | 5.6 min | 0.916 |
| 16384 | 5 s | 120 s | 3.09 h | 5.6 min | 0.959 |
| 1024 | 100 s | 600 s | 49.5 h | 99.5 min | 0.963 |

1. 1024 卡时同步写也有 96%；16K 卡同步写只剩 81%，浪费的 19% 里写 checkpoint 和丢工作各占约 6.7%、重启占 5.4%。
2. $\delta$ 从 100 s 降到 5 s，$\sqrt{2\delta/M}$ 从 13.4% 降到 3.0%。
3. 再把 $R$ 从 10 分钟降到 2 分钟，ETTR 到 96%。Llama 3 报告高于 90% 的有效训练时间，就是这两件事都要做。

### 4. 降低 $\delta$

**1. 异步 checkpoint**：拆成两段（[PyTorch 博客](https://pytorch.org/blog/reducing-checkpointing-times/)）：

1. 阻塞段：每个 rank 把自己的分片从 GPU 拷到 CPU 内存（D2H），拷完训练就继续。
2. 后台段：CPU 线程把内存里的副本序列化、写盘。

训练只停第一段，$\delta$ 变成「每卡分片字节 / D2H 带宽」加固定开销。IBM 实测 7B 的停顿从平均 148.8 s 降到 6.3 s（作者自测）。

约束：

1. 后台写盘没完成前不能开始下一次 checkpoint，所以 $\tau$ 要大于后台写盘时间（405B 是 97 s）。
2. CPU 内存要放得下一份分片。
3. checkpoint 线程的 collective 要用单独的 process group，否则和训练的 collective 抢着等会死锁（同一篇博客）。

API：`torch.distributed.checkpoint.async_save`，默认 staging 到 CPU 内存，然后在后台线程里 save（[DCP 文档](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)）。

**2. 分片并行写**：PyTorch DCP（Distributed Checkpoint）让每个 rank 只写自己持有的分片，一次 checkpoint 产生多个文件、每个 rank 至少一个，外加一个全局的 `.metadata` 文件记录每个张量的全局形状和各分片的位置（DCP 文档）。所有节点同时写，带宽是文件系统的总带宽，这就是第 2 节 $\delta = 12\Psi / B_{\text{fs}}$ 的前提。

**3. 去重**：数据并行（DP）的副本持有同样的张量，只需要一个写。例：405B，TP 8 × PP 16 × DP 128。

1. ZeRO-1 / FSDP 已经把优化器状态在 DP 维切开，这部分天然没有重复。
2. bf16 权重（如果要存）在 128 个 DP 副本上各有一份：每份都写是 $810 \text{ GB} \times 128 = 104$ TB，去重后只写 810 GB。
3. DCP 在生成保存计划时，把多个 rank 都有的同一个张量只分给一个 rank 写（`DefaultSavePlanner` 的 `dedup_replicated_tensors`）。

### 5. 降低恢复时间 $R$

$R$ 由几段组成，按顺序：

1. 检测：发现作业挂了（第 6 节）。只靠 NCCL 超时的话，默认要等 10 分钟。
2. 换机：把坏节点踢掉，换上好的。
3. 拉起：所有进程重启，重建 NCCL communicator。
4. 加载：读 checkpoint，405B 从文件系统读 4.86 TB 要以分钟计。
5. 恢复数据位置：数据加载器回到 checkpoint 时读到的位置。

**1. 内存 / 对等节点 checkpoint**：Gemini（[SOSP 2023](https://doi.org/10.1145/3600006.3613145)，Wang et al.）把 checkpoint 存在 CPU 内存里：每台机器存自己的，再给几台对等机器各存一份副本。

1. 一台机器坏了，它的 checkpoint 还在对等机器的内存里，换上来的机器从对方内存拉，不用读远端存储。
2. 内存带宽高，可以做到每个 iteration 都 checkpoint，$\tau$ 极小，丢的工作几乎为 0。
3. 同一组机器同时坏、内存里的副本都没了，才退回远端存储里的 checkpoint。
4. 论文还做了副本放置策略（最大化能从内存恢复的概率）和流量调度（checkpoint 流量插在训练流量的空隙里）。
5. 效果（作者自测，基于 DeepSpeed ZeRO-3，A100 / V100 实例）：取回 checkpoint 快至多 250×，checkpoint 频率高至多 8×，故障恢复快 13× 以上。

**2. 热备节点**：预留几台空闲节点，坏节点直接换热备，不用等调度器重新分配机器或等维修。

**3. 快速重启**：

1. 检测快：用心跳和 watchdog 尽快发现，而不是干等 NCCL 超时（第 6 节）。
2. 加载快：从内存 checkpoint 或对等节点加载；从文件系统读时各 rank 只读自己的分片。
3. 数据状态：数据加载器的位置要进 checkpoint，否则要么重复训练一部分数据，要么跳过一部分。PyTorch 的 `torchdata.stateful_dataloader.StatefulDataLoader` 提供 `state_dict()` / `load_state_dict()`。

另一条路是不重启：Gemini 2.5 报告训练系统在局部故障时自动减少 TPU slice 继续训练，每次中断只损失几十秒，故障 slice 恢复期间吞吐约 97%（[arXiv 2507.06261](https://arxiv.org/abs/2507.06261)，作者自测）。

### 6. 检测

**1. NCCL timeout 和 watchdog**：

1. `init_process_group` 的 `timeout` 对 NCCL 默认 10 分钟（其他后端 30 分钟）。超时后 collective 被异步中止、进程崩溃，因为 CUDA 是异步执行的，继续跑可能用到损坏的数据（[torch.distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)）。
2. 超时只告诉你「卡住了」，不告诉你是谁。Llama 3 用 PyTorch 的 NCCL flight recorder 记录每个 rank 最近的 collective 元数据和调用栈：对比各 rank 停在哪个 collective，没进去的那个 rank 就是嫌疑。

**2. 心跳**：每个节点上的 agent 监控本机的训练进程，进程退出或心跳停了就上报，调度器据此重启作业。比等 NCCL 超时快得多。

**3. 慢节点（straggler）**：节点没坏但变慢（降频、网卡降速），同步训练里其他所有卡都在 collective 里等它。Llama 3 原话是一个 straggler 就能拖慢几千张卡，他们做了工具优先排查可疑的通信、找出慢节点。方法：每个 rank 记录两次 collective 之间的计算时间，明显偏大的就是慢节点。

**4. Silent data corruption（SDC）**：硬件算错了但不报错，没有异常、没有超时，只在 loss 或梯度范数上留下痕迹。

1. Llama 3：54 天 419 次非计划中断里有 6 次是 SDC（作者自测）。
2. Gemini 2.5：遇到可疑指标就立刻确定性重放（deterministic replay）这一步，对比每个设备的中间结果校验和来定位；约 0.25% 的步被重放，其中 6% 确实是硬件损坏；出现间歇性 SDC 的芯片几分钟内被找出并踢掉（作者自测）。
3. 原理：重放同一步，结果和原来不一样，说明不是数据问题而是硬件问题；每个设备的校验和能指出是哪一个。

### 7. Resharding：换了并行配置后怎么加载

场景：用 FSDP 64 卡存的 checkpoint，想在 128 卡上接着训；或者 TP 从 8 改成 4。

DCP 的做法（文档：支持 load-time resharding，在一种集群拓扑下保存、在另一种下加载）：

1. 保存时 `.metadata` 记下每个张量的全局形状，以及每个分片在全局张量里的偏移和所在文件。
2. 加载时每个 rank 按**新**的并行配置算出自己需要哪一段，再到 metadata 里查这一段落在哪些旧分片、哪些文件，只读那几段字节。
3. 例：FSDP 64 → 128。每个参数原来沿第 0 维切 64 段，现在切 128 段，每个新 rank 读一个旧文件的一半。TP 8 → 4，每个新 rank 读两个旧分片拼起来。
4. 模型和优化器的状态字典用 `get_state_dict` / `set_state_dict` 取和设，它们屏蔽了 FSDP / TP 各自的 API，得到的状态可以换卡数加载。

前提：参数名（FQN）和全局形状不变，变的只是切法。如果改了模型代码里的张量布局（比如 QKV 从分开变成拼在一起），要另写转换脚本。

## 面试追问

::: details Q：为什么作业 MTBF 是单卡 MTBF 除以卡数？
同步训练中任何一张卡出事，整个作业都要停。假设各卡故障独立、单位时间的故障概率恒定（指数分布），$n$ 张卡的故障率相加是 $n/M_{\text{gpu}}$，倒数就是 $M_{\text{gpu}}/n$。所以卡数翻倍，作业被打断的频率翻倍。
:::

::: details Q：Young 公式 $\tau^* = \sqrt{2\delta M}$ 怎么来的？
单位时间浪费 $= \delta/\tau$（写 checkpoint）$+ \tau/(2M)$（每次故障平均丢半个间隔，单位时间故障 $1/M$ 次）$+ R/M$（重启）。前两项一个随 $\tau$ 减、一个随 $\tau$ 增，求导令和为 0 得 $\tau^* = \sqrt{2\delta M}$，此时两项相等。$R$ 和间隔无关，单独压。
:::

::: details Q：异步 checkpoint 之后还有什么限制？
训练只停 GPU → CPU 的拷贝，但后台写盘没完成前不能开始下一次，所以间隔不能短于写盘时间；CPU 内存要多放一份分片；checkpoint 线程的 collective 要走单独的 process group，否则和训练 collective 互相等会死锁。
:::

::: details Q：checkpoint 为什么是每参数 12 字节而不是 16？
显存里的 16 字节是 bf16 参数 2 + 梯度 2 + fp32 主权重 4 + Adam 8。bf16 参数可以从主权重 cast 出来，梯度一步用完就扔，都不用存；剩下主权重和 Adam 的 $m$、$v$ 共 12 字节。
:::

::: details Q：作业卡住了不报错，怎么定位？
先看是不是 NCCL hang：flight recorder 记录了各 rank 最近的 collective，找出没有进入某个 collective 的 rank。如果没卡住但变慢，是 straggler，比较各 rank 两次 collective 之间的计算时间。如果 loss 或梯度范数突然异常但没有报错，怀疑 SDC：确定性重放这一步，对比各设备的校验和。
:::

::: details Q：checkpoint 存在 CPU 内存里，机器坏了内存不就没了？
Gemini 的做法是每台机器的 checkpoint 在其他几台机器的内存里也存一份。坏一台时从对等机器拉；同一组副本所在的机器全坏了，才退回远端持久存储里较旧的 checkpoint。所以内存 checkpoint 用来提高频率、缩短恢复，持久存储照样要定期写。
:::

## 手撕

给定 $n$、单卡 MTBF、$\delta$、$R$，算作业 MTBF、最优间隔和 ETTR：

```python
import math

def plan(n_gpu, mtbf_gpu_h, delta_s, restart_s):
    M = mtbf_gpu_h * 3600 / n_gpu                       # 作业 MTBF（秒）
    tau = math.sqrt(2 * delta_s * M)                    # Young：两次 checkpoint 之间的训练时间
    waste = delta_s / tau + (tau / 2 + restart_s) / M   # 写 checkpoint + 丢半个间隔 + 重启
    return M / 3600, tau / 60, 1 - waste

# 从 Llama 3 反推单卡 MTBF：54 天 419 次非计划中断，约 16384 卡
mtbf_gpu = 54 * 24 / 419 * 16384
print(f"单卡 MTBF ≈ {mtbf_gpu:,.0f} h ≈ {mtbf_gpu / 8760:.1f} 年")

for n, delta, R in [(16384, 100, 600), (16384, 5, 600), (16384, 5, 120), (1024, 100, 600)]:
    M_h, tau_min, ettr = plan(n, mtbf_gpu, delta, R)
    print(f"n={n:>5} δ={delta:>3}s R={R:>3}s: M={M_h:5.2f} h  τ*={tau_min:5.1f} min  ETTR={ettr:.3f}")

# 单卡 MTBF ≈ 50,677 h ≈ 5.8 年
# n=16384 δ=100s R=600s: M= 3.09 h  τ*= 24.9 min  ETTR=0.812
# n=16384 δ=  5s R=600s: M= 3.09 h  τ*=  5.6 min  ETTR=0.916
# n=16384 δ=  5s R=120s: M= 3.09 h  τ*=  5.6 min  ETTR=0.959
# n= 1024 δ=100s R=600s: M=49.49 h  τ*= 99.5 min  ETTR=0.963
```

常见变体：给定目标 ETTR（比如 95%）反解 $\delta$ 和 $R$ 的上限；加入「写盘时间 $T_{\text{persist}}$」约束，$\tau = \max(\tau^*, T_{\text{persist}})$；checkpoint 大小用 $12\Psi / B_{\text{fs}}$ 算出 $\delta$ 再代入。

## 参考

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：§3.3.4 可靠性、Table 5 中断原因
- [Young, A first order approximation to the optimum checkpoint interval (CACM 1974)](https://doi.org/10.1145/361147.361115)
- [Daly, A higher order estimate of the optimum checkpoint interval for restart dumps (FGCS 2006)](https://doi.org/10.1016/j.future.2004.11.016)
- [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680)：ETTR
- [Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints (SOSP 2023)](https://doi.org/10.1145/3600006.3613145)
- [PyTorch Distributed Checkpoint 文档](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing](https://pytorch.org/blog/reducing-checkpointing-times/)
- [torch.distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html)：`init_process_group` 的 timeout
- [Gemini 2.5 技术报告](https://arxiv.org/abs/2507.06261)：SDC 重放、slice 级弹性

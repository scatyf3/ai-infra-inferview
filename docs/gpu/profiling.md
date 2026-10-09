---
title: Profiling：nsys / ncu / torch profiler
status: draft
tags: [profiling, nsys, ncu]
difficulty: 2
order: 5
related: [/inference/prefill-decode-roofline]
stack: [k-lang]
---

# Profiling：nsys / ncu / torch profiler

## 一句话结论

三层工具各看一层：
1. torch profiler 看 Python 算子和 CPU / GPU 时间线（有没有 GPU 空转）
2. nsys 看 kernel 序列、launch 间隙和 NCCL / 拷贝重叠情况
3. ncu 看单个 kernel 的带宽利用率、占用率和 stall 原因。先找「GPU 在等谁」，再看「单个 kernel 慢在哪」。

## torch profiler

先给一个能直接改的模板，下面三点逐条解释（API 见 [PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html) 和 [Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)）：

```python
import torch
from torch.profiler import profile, schedule, record_function, ProfilerActivity

sched = schedule(wait=1, warmup=2, active=3, repeat=1)   # 1. 跳过前几步，只录稳态

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    record_shapes=True,       # 记录每个算子的输入 shape
    with_stack=True,          # 2. 记录 Python 调用栈
    on_trace_ready=lambda p: p.export_chrome_trace("trace.json"),
) as prof:
    for _ in range(1 + 2 + 3):                   # wait + warmup + active
        with record_function("decode_step"):     # 在时间线上标一段，类似 NVTX
            out = model(x)
        prof.step()                              # 告诉 schedule 进入下一步

# 3. 汇总表 + 时间线（trace.json 拖进 https://ui.perfetto.dev）
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

### 1. warmup 和 repeat：只录稳态

前几次迭代不能代表稳态，它们会混进一次性开销：

- CUDA context 和 cuBLAS handle 初始化；
- caching allocator 第一次向驱动 `cudaMalloc` 显存，之后都是复用；
- `torch.compile` 编译、Triton autotune、cuDNN benchmark 选算法。

这些开销可能比一步 forward 本身还长。`schedule` 把迭代分成几段：

| 参数 | 这几步里 profiler 的状态 | 作用 |
|---|---|---|
| `wait` | 关着 | 跳过上面那些一次性开销 |
| `warmup` | 开着，但结果丢掉 | profiler 自己启动（CUPTI 初始化）也有开销，这几步把它吸收掉 |
| `active` | 开着，记录 | 真正要看的稳态 |
| `repeat` | 上面一轮重复几次 | 0 表示一直循环到退出；训练时可以隔一段录一轮 |

每次迭代结束都要调用 `prof.step()`，否则 schedule 不会往前走，什么都录不到。

### 2. with_stack：把 kernel 对应回 Python 行

时间线上只能看到 `aten::mm`、`ampere_bf16_gemm...` 这类名字，分不清是哪一层、哪行代码发起的。`with_stack=True` 会给每个算子记录 Python 调用栈，在时间线上点一个算子就能看到 `model.py:123` 这样的源码位置。汇总表也可以按调用栈分组：

```python
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=10))
```

`record_shapes=True` 同理，可以按 shape 分组（`group_by_input_shape=True`），用来区分同一个算子在 prefill 和 decode 下的耗时。代价是：两者都会让 profiler 自身开销变大、trace 文件变大，所以 `active` 别设太多步。

### 3. 看时间线：GPU 在等谁

Perfetto 里分上下两部分：上面是 CPU 线程（Python 算子 `aten::*` 和 `cudaLaunchKernel`），下面是 GPU stream（实际执行的 kernel），中间的箭头把每次 launch 连到它启动的 kernel。看三种情况：

1. **GPU 行有空隙，CPU 行很满**：GPU 在等 CPU 发下一个 kernel，瓶颈是 Python 和 launch 开销（overhead）。小 batch 的 decode 最常见：一个 kernel 只跑十几 µs，CPU 发一个 kernel 也要几 µs，再加上 Python 的分发，GPU 一半时间空着。解法是 CUDA Graph、减少算子数（融合），或者用 `torch.compile`。
2. **CPU 的 launch 远远跑在 GPU 前面**：GPU 一直有活，是 GPU-bound，这是好情况。接下来看汇总表里哪个 kernel 最耗时，再用 ncu 分析它。
3. **CPU 行出现很长的 `cudaStreamSynchronize` 或 `cudaMemcpy`（DtoH）**：有隐式同步。`.item()`、`.cpu()`、`print(tensor)`、`if tensor:`、`torch.nonzero` 都要等 GPU 算完、把结果拷回 CPU。CPU 在这里卡住，launch 队列就会排空，后面 GPU 跟着空转。

汇总表里两列别看混：**Self** 只算这个算子自己，不含它调用的子算子；**Total** 包含子算子。按 `self_cpu_time_total` 排序能找出 CPU overhead 大的算子，按 `cuda_time_total` 排序能找出最耗 GPU 时间的 kernel。新版 PyTorch 把 GPU 时间这一列叫 device time，`cuda_time_total` 报错的话换成 `device_time_total`。

还要注意，profiler 本身会给每个算子多加几 µs 的 CPU 时间，所以表里的 CPU 时间是偏高的。用它比较相对大小没问题，要算绝对的 launch 开销，用 nsys 更准。

## nsys

先给模板，下面三点逐条解释（参数见 [Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)）。Python 侧只做两件事：标出录制范围，给每一步打 NVTX 标记。

```python
import torch

WARMUP, ACTIVE = 5, 3
for i in range(WARMUP + ACTIVE):
    if i == WARMUP:
        torch.cuda.profiler.start()               # 1. 从这里开始录
    torch.cuda.nvtx.range_push(f"decode_step_{i}")  # 2. 在时间线上标一段
    out = model(x)
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.profiler.stop()                         # 1. 录到这里为止
```

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node \
  -o decode python run.py

# 3. 汇总表（GUI 里打开 decode.nsys-rep 看时间线）
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum decode.nsys-rep
```

### 1. capture range：只录稳态

道理和 torch profiler 的 warmup 一样：前几步混着初始化、allocator 第一次 malloc、compile 和 autotune，要跳过。

- `--capture-range=cudaProfilerApi`：nsys 从程序开头就挂上，但只在 `torch.cuda.profiler.start()` 和 `stop()` 之间记录。
- `--capture-range-end=stop`：遇到 `stop()` 就结束采集。
- 不想改代码的话，可以用 `--delay=<秒>` 和 `--duration=<秒>` 按时间截一段，只是不如按步数准。

nsys 的开销比 torch profiler 小得多，录几十步也没问题，主要限制是 trace 文件会变大。

### 2. NVTX：把 kernel 对应回代码

nsys 默认只看到 CUDA API 和 kernel 名字，不知道 Python 在干什么，所以要靠 NVTX 标记（`-t` 里要带 `nvtx`）：

- **手动标记**：`torch.cuda.nvtx.range_push` / `range_pop`，或者 `with torch.cuda.nvtx.range("prefill"):`。按 step、prefill / decode、每层去标，时间线上会多一行 NVTX，kernel 落在哪个范围里一目了然。
- **自动标记每个算子**：`with torch.autograd.profiler.emit_nvtx():` 会给每个 aten 算子打一个 NVTX range，粒度和 torch profiler 差不多。开销也会跟着变大，只在需要时开。

`--cuda-graph-trace=node` 是给 CUDA Graph 用的。默认整个 graph 在时间线上只显示成一块，看不到里面每个 kernel 的耗时；加上这个参数会按 kernel 展开。

### 3. 看时间线：GPU 在等谁

GUI 里从上到下大致是：

1. **CUDA HW**：每个 stream 一行，画的是实际执行的 kernel 和 memcpy。
2. **进程下的各个线程**：每个线程下面有一行 CUDA API（`cudaLaunchKernel`、`cudaStreamSynchronize` 等）和一行 NVTX。

点 API 调用会高亮它对应的 kernel。判断方法和 torch profiler 的三种情况一样：GPU 行有空隙是 launch 开销，CPU 远远领先是 GPU-bound，长的同步条是隐式同步。nsys 多出来的两样：

- **多 stream、多进程都能看**：NCCL 的 all-reduce kernel 和计算 kernel 有没有重叠，HtoD 拷贝是不是藏在计算后面，TP 多卡时每张卡是不是同步推进。这些 torch profiler 看不清。
- **GPU metrics 采样**：加 `--gpu-metrics-devices=all`（旧版参数名是 `--gpu-metrics-device`），时间线上会多出 SM Active、Tensor Active、DRAM Bandwidth 等曲线。decode 阶段 DRAM Bandwidth 高、Tensor Active 低；prefill 阶段 Tensor Active 高。它只能粗略看出 bound 的类型，要确认还得用 ncu。

汇总表里，`cuda_gpu_kern_sum` 按 kernel 名统计总耗时、次数和平均值，用来挑出最值得用 ncu 分析的 kernel；`cuda_api_sum` 看 CPU 侧 API 的耗时，`cudaStreamSynchronize` 排得很靠前就说明有同步问题。旧版 nsys 的 report 名字不一样（比如 `gpukernsum`），用 `nsys stats --help-reports` 查。

## ncu

先给模板，下面三点逐条解释（见 [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)）：

```bash
ncu --set full \
    -k regex:"gemm|gemv" \
    --launch-skip 20 --launch-count 5 \
    --target-processes all \
    -o decode_kernels \
    python run.py

# 命令行看结果（或者用 ncu-ui 打开 decode_kernels.ncu-rep）
ncu --import decode_kernels.ncu-rep --page details
```

### 1. 只抓要看的 kernel：-k / -s / -c

ncu 和前两个工具不一样，它不录时间线，而是把选中的 kernel 拿出来**重放**（replay）：每个 kernel 跑很多遍，每遍采一组硬件计数器，每遍之前把显存恢复原样。`--set full` 要重放几十遍，整个程序会慢几十到上百倍，所以一定要筛选：

- `-k regex:...`：按 kernel 名字过滤，名字从 nsys 的 `cuda_gpu_kern_sum` 里抄。
- `--launch-skip`（`-s`）：跳过前面若干次匹配的 launch，作用相当于 warmup。
- `--launch-count`（`-c`）：只抓几次。
- `--target-processes all`：vLLM 这类多进程程序，kernel 在子进程里跑，不加这个参数抓不到。

重放带来的两点差异，读数时要记住：

- **锁频**：ncu 默认把 GPU 时钟锁在 base clock（`--clock-control base`），这样结果稳定可复现，但 kernel 时长会比 nsys 里的长。要对比绝对时长，就加 `--clock-control none`。
- **冷 cache**：每遍重放前默认清空 cache（`--cache-control all`），量到的是冷 cache 下的数据。实际运行中，权重小到能放进 L2 的话，这和真实情况会有出入。

### 2. 实测判断 compute-bound 还是 memory-bound

[roofline](/inference/prefill-decode-roofline) 用公式比 $\text{FLOPs}/\text{peak}$ 和 $\text{Bytes}/\text{BW}$ 哪个大。ncu 直接给出这两项各自占峰值的百分比，在 GPU Speed Of Light Throughput（SOL）这一节：

| ncu 里的名字 | metric | 对应 roofline 的 |
|---|---|---|
| Compute (SM) Throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | FLOPs ÷ peak |
| DRAM Throughput | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Bytes ÷ BW（H100 的 HBM 在 ncu 里也叫 dram） |

按经验阈值判断（不是硬标准）：

1. **DRAM 高（≳ 60–80%），SM 低**：memory-bound。decode 的 GEMV、逐元素 kernel 都是这样。
2. **SM 高，DRAM 不高**：compute-bound。prefill 的大 GEMM 是这样。
3. **两个都低**：受延迟限制，见下面的面试追问。kernel 本身太短的话，就是 launch 开销。

有个坑：SOL 里还有一项 **Memory Throughput**（`gpu__compute_memory_throughput`），它取的是 L1、L2、DRAM 里最高的那个。所以它高不代表 HBM 被打满，可能只是 shared memory 或 L1 很忙。对照 roofline 要看 **DRAM Throughput**。

ncu 也能直接画 roofline（`--set full` 里包含，单独要就加 `--section SpeedOfLight_RooflineChart`），kernel 的点画在实测 AI 的位置。实测 AI 和按公式算的对不上，通常是 L2 命中，或者有额外的激活读写。

### 3. 定了 bound 之后看哪几节

SOL 告诉你瓶颈在哪一边，下面几节告诉你为什么：

| 情况 | 看哪一节 | 看什么 |
|---|---|---|
| memory-bound | Memory Workload Analysis | DRAM 实际读写的字节数和公式比，多出来的就是浪费；L2 命中率；表里每个请求用到多少 sector，看访存是不是合并的 |
| compute-bound | Compute Workload Analysis | 各个 pipe 的利用率。GEMM 应该是 Tensor 那一行高；如果 FMA（CUDA core）那一行高，说明没走上 tensor core |
| 两个都低 | Occupancy、Warp State Statistics | 理论 occupancy 被寄存器还是 shared memory 卡住；stall 原因里 long scoreboard 是等访存，barrier 是等同步 |

要定位到具体的源码行，看 Source 页：SASS 指令旁边会标出各自的 stall 次数。CUDA C++ 要用 `nvcc -lineinfo` 编译，才能对应回源码行。

## 三个工具怎么配合

1. **torch profiler**：先看 GPU 有没有空转，找出耗时最多的算子，以及它对应哪行 Python。
2. **nsys**：看多 stream、多卡之间的重叠和同步，量出准确的 launch 间隙，再从 `cuda_gpu_kern_sum` 挑出热点 kernel。
3. **ncu**：只抓这几个热点 kernel，用 SOL 判断 bound 的类型，再按上表往下钻。

## 面试追问

::: details Q：一个 kernel 在 ncu 里 DRAM 带宽只打到 30%，SM 也不满，可能是什么问题？
两边都不满说明受延迟限制而不是吞吐限制：occupancy 太低藏不住访存延迟、访存模式不合并导致事务数膨胀、或者有大量 `__syncthreads` 让 warp 互相等。看 `Warp State Statistics` 里的 stall 原因（long scoreboard 是等访存，barrier 是等同步）。
:::

## 手撕

常见题：给一段 decode 的 nsys 时间线截图，指出瓶颈并说下一步优化。

## 参考

- [Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute：Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

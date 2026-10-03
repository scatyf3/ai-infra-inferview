---
title: Profiling：nsys / ncu / torch profiler
status: todo
tags: [profiling, nsys, ncu]
difficulty: 2
order: 5
related: []
stack: [k-lang]
---

# Profiling：nsys / ncu / torch profiler

> 看哪几个指标

## 一句话结论

三层工具各看一层：torch profiler 看 Python 算子和 CPU / GPU 时间线（有没有 GPU 空转），nsys 看 kernel 序列、launch 间隙和 NCCL / 拷贝重叠情况，ncu 看单个 kernel 的带宽利用率、占用率和 stall 原因。先找「GPU 在等谁」，再看「单个 kernel 慢在哪」。

## 推导

- **先看 GPU 利用率**：时间线上 kernel 之间有空隙 → CPU 侧瓶颈（Python 调度、launch 开销），上 CUDA Graph 或减少算子数；kernel 挨得很紧 → 看单 kernel。
- **nsys**：`nsys profile -t cuda,nvtx python ...`，看 kernel 时长排序、拷贝和计算是否在不同 stream 上重叠；用 NVTX range 标出 prefill / decode / 每层。
- **ncu 关键指标**：`DRAM Throughput`、`SM Throughput`、`Achieved Occupancy`、`Memory Workload Analysis` 里的 L2 命中率；把实测带宽和算力放到 [roofline](/inference/prefill-decode-roofline) 上判断是 memory-bound 还是 compute-bound。
- **torch profiler**：`with_stack=True` 能把 kernel 对应回 Python 行；看 `cudaMemcpy` / `item()` 这类隐式同步点。

## 面试追问

::: details Q：一个 kernel 在 ncu 里 DRAM 带宽只打到 30%，SM 也不满，可能是什么问题？
两边都不满说明受延迟限制而不是吞吐限制：occupancy 太低藏不住访存延迟、访存模式不合并导致事务数膨胀、或者有大量 `__syncthreads` 让 warp 互相等。看 `Warp State Statistics` 里的 stall 原因（long scoreboard 是等访存，barrier 是等同步）。
:::

## 手撕

常见题：给一段 decode 的 nsys 时间线截图，指出瓶颈并说下一步优化。

## 参考

- [Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute：Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

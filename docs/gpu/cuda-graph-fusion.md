---
title: CUDA Graph 与 Kernel Fusion
status: todo
tags: [cuda-graph, fusion]
difficulty: 3
order: 4
related: []
stack: [f-graph, k-fused]
---

# CUDA Graph 与 Kernel Fusion

> 在框架侧的落地

## 一句话结论

decode 一步只有几毫秒，几百个小 kernel 的 launch 开销和 Python 调度开销能占一半以上；CUDA Graph 把整个 forward 录下来一次提交，kernel fusion 把多个小算子合成一个，两者都是在「让 GPU 别等 CPU」。

## 推导

- **launch 开销**：每个 kernel launch 约 5–10 μs 的 CPU 侧成本，decode 时每层十几个 kernel，80 层就是上千次 launch；GPU 执行每个 kernel 可能只要几微秒，CPU 反而成了瓶颈。
- **CUDA Graph**：capture 一次 forward 的全部 kernel 和参数，之后 `cudaGraphLaunch` 一次提交整张图；代价是 shape 必须固定，所以 vLLM 按 batch size 分桶预 capture 若干张图，并把输入 tensor 地址固定住。
- **kernel fusion**：RMSNorm + 残差加、SiLU + 乘法、RoPE + KV 写入这类 memory-bound 小算子，融合后省掉中间结果的 HBM 读写，收益直接按 [roofline](/inference/prefill-decode-roofline) 算。
- **框架落地**：vLLM 用 `torch.compile` 做 fusion、用 CUDA Graph 做 decode；两者可以叠加，compile 产出的 kernel 被 graph 捕获。

## 面试追问

::: details Q：CUDA Graph 为什么只用在 decode，不用在 prefill？
prefill 的 seq_len 每个请求都不一样，shape 不固定就要为每种 shape 单独 capture，图太多且命中率低；而 decode 每步每个序列只有 1 个 token，只有 batch size 一个维度变化，分几个桶就能覆盖。padding 到桶大小浪费的算力在 decode（memory-bound）场景几乎免费。
:::

## 手撕

常见题：用 `torch.cuda.CUDAGraph` 手动 capture 一个小模型的 forward 并重放，解释为什么输入要用 `copy_` 写进固定 buffer。

## 参考

- [CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch：CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

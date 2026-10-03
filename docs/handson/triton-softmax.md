---
title: Triton 版 Softmax
status: todo
tags: [triton, softmax, handson]
difficulty: 3
order: 7
related: []
stack: [k-lang, k-fused]
leetgpu: [5]
---

# Triton 版 Softmax

> 一行一个 program 的写法

## 一句话结论

row softmax 是「减最大值、exp、除以和」三步，naive 的 PyTorch 实现要读写 x 好几遍；Triton 版一行一个 program，把整行加载进寄存器，`tl.max` 和 `tl.sum` 两次 reduce 后直接写回，HBM 流量降到读一次写一次。这是 Triton 教程的第二课，也是理解 FlashAttention 里 online softmax 的前置。

## 推导

- **数值稳定**：先减行最大值再 exp，防止溢出；推导见 [数值稳定 softmax](/handson/stable-softmax)。
- **融合收益**：naive 版本 5 次读写（max、减、exp、sum、除），fused 版本 2 次；对 memory-bound 的算子就是约 2.5 倍。
- **mask**：行长不是 2 的幂时 `BLOCK = next_power_of_2(N)`，越界位置填 `-inf`，exp 后自然为 0。
- **大行**：单 block 装不下时要分块 + online softmax（维护 running max 和 sum），这就是 FlashAttention 的核心。

## 面试追问

::: details Q：num_warps 怎么选？
一行 N 个元素，每个线程处理 N / (32 × num_warps) 个；行短时 4 个 warp 够，行长到几万时增加 warp 数让每线程的寄存器压力降下来。Triton 教程里按 BLOCK_SIZE 用启发式（2048 以上 8 warps，4096 以上 16 warps），生产上交给 autotune。
:::

## 手撕

对应 [LeetGPU #5](https://leetgpu.com/challenges)。框架：

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * N + cols, y, mask=mask)
```

---
title: PyTorch 内部机制
status: todo
tags: [pytorch, autograd, allocator]
difficulty: 4
order: 1
related: []
stack: [3]
---

# PyTorch 内部机制

> dispatcher、autograd、custom op 注册、caching allocator 与碎片

## 一句话结论

PyTorch 的 Python 层只是壳：一次 `torch.add` 经过 dispatcher 按 tensor 的 device / dtype / autograd 等 key 逐层分发到最终 kernel，autograd 在分发链里插入一层记录反向图，caching allocator 在 `cudaMalloc` 上面做池化避免同步。写 custom op 就是往这张分发表里注册新条目。

## 推导

- **dispatcher**：每个算子有一张按 DispatchKey 索引的表（Autograd、CUDA、CPU、Python…），调用时取 tensor 的 key set 里优先级最高的那个；`torch.library` / `TORCH_LIBRARY` 就是往表里填函数。
- **autograd**：Autograd key 的 kernel 先创建 `grad_fn` 节点把输入存起来，再 redispatch 到真正的计算 kernel；`torch.no_grad` 只是把 Autograd key 从集合里排除。
- **custom op**：注册 schema + 各 backend 实现 + autograd formula；给 `torch.compile` 用还要注册 `FakeTensor` 实现以便推 shape。
- **caching allocator**：按 stream 维护大小分桶的空闲块池，释放时不还给 CUDA；碎片表现为 `reserved` 远大于 `allocated`，可用 `expandable_segments` 或固定 shape 缓解；推理框架常用它预分配 KV 池。

## 面试追问

::: details Q：`torch.cuda.empty_cache()` 为什么通常不该在训练循环里调？
它把空闲块真正 `cudaFree` 还给驱动，这是同步操作且后续再分配要重新 `cudaMalloc`，两者都慢；而且解决不了碎片（正在用的块位置没变）。它只在真要给别的进程腾显存时有用。
:::

## 手撕

常见题：用 `torch.library.custom_op` 注册一个带 autograd 的算子；解释 `x.view()` 和 `x.reshape()` 在 dispatcher 和 storage 层面的区别。

## 参考

- [Let's talk about the PyTorch dispatcher（ezyang）](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [PyTorch 文档：CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

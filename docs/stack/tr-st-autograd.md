---
title: "autograd"
---

# autograd

PyTorch 在 forward 时记下计算图和 backward 要用的张量（saved tensors），backward 时按反向拓扑序调用每个算子的梯度函数。训练激活显存的来源就是这些 saved tensors。

基于naive的autograd，我们有两个优化方向
- **hook**：FSDP、DDP 的梯度同步都挂在 autograd 的 hook 上，即酸楚梯度指挥立刻同步，更细粒度的梯度和多机同步的overlap
- **recompute**：checkpoint 让 forward 不存中间结果，backward 时重算，节约显存，但提升延迟



---
title: kernel mindset
status: draft
tags: [triton, cuda, guide, handson]
difficulty: 2
order: 0.3
related: [/handson/torch-primitives, /handson/triton_primitives]
stack: []
---

# kernel mindset

这些东西就是本质上沿不同的维度做不同的操作。

再细一点：**每个维度在 kernel 里只有两种角色。**

- **并行维**：输出沿这个维度互不依赖，直接切开分给不同 program → 进 grid
- **归约维**：多个输入合成一个输出 → 在 program 里循环 / 累加

给每个维度贴好标签，kernel 的骨架就定了。

## 单个维度：按输入 → 输出的映射分

### 1 → 1：逐元素

add、relu、gelu、cast。最简单的 linear kernel，代表题是 [Vector Add](./vector-add)。

- 唯一的维度是并行维，每个输入只读一次 → 必然 memory-bound
- 能做的优化只有两个：合并访存，和前后 kernel 融合（省掉中间结果写回显存）

1 → 1 还有几个变体，正好是难度台阶：

| 变体 | 例子 | 难点 |
|---|---|---|
| 带广播 | bias add | 一个输入被多个输出用 |
| 间接寻址 | embedding lookup | 下标从另一个 tensor 读出来，访存不连续 |
| 重排 | transpose | 读和写只有一侧能连续，要先进 shared memory 中转 |

### n → 1：reduce

sum、max。继续上难度，问题是怎么划分数据、怎么保证写入顺序，代表题是 [Reduce](/leetgpu/reduction)。分两种情况：

- **一个 program 装得下**：一行就是一个块，比如 [softmax](./stable-softmax)、[rmsnorm](./rmsnorm)。直接 `tl.sum` / `tl.max`，没有写入顺序的问题
- **装不下，要跨 program**：
  - **atomic**：简单；但浮点加法不满足结合律，每次结果可能不同
  - **两遍**：先写部分和，再起一个 kernel 合并
  - **split-K**：matmul 里 K 太长时就这么切
  - 本质都是需要一个全局同步点（kernel 结束 / atomic / 计数器），见 [Reduce](/leetgpu/reduction) 里的追问

还有一类**在线算法**：[online softmax](./online-softmax)、Welford 均值方差。它们把"先算 max 再算 sum"这种两遍归约合成一遍，flash attention 就靠这个。

### n → n 带前后依赖：scan

cumsum、prefix sum，[top-p sampling](./top-p-sampling) 里会用到。第 i 个输出依赖前 i 个输入，比 reduce 更难并行。

## 多维度：给每个维度贴标签

| kernel | 并行维 | 归约维 | 新增的难点 |
|---|---|---|---|
| softmax / rmsnorm | 行 | 列 | 归约 + 逐元素融合 |
| matmul | M、N（2D grid） | K（循环） | **数据复用**：一个输入被多个输出读，要分块；开始变成 compute-bound |
| attention | B、H、Q | KV 长度 | 两次 matmul 中间夹一个 softmax 归约，靠 online softmax 才能融合成一个 kernel（flash attention） |

matmul 比 reduce 难，不在于多了一维，而在于**数据复用**：逐元素和 reduce 的每个输入只读一次，matmul 的每个输入要被 N 或 M 个输出用。分块、shared memory、算术强度这些概念都从这里来。

## checklist

拿到任何 kernel，先回答：

1. 每个维度是并行维还是归约维？并行维进 grid，归约维进循环
2. 一个 program 负责多大的块？
3. 有没有一个输入被多个输出用？没有 → memory-bound；有 → 分块复用
4. 同一个输出位置会不会被多个 program 写？会 → atomic 或分两遍
5. memory-bound 还是 compute-bound？看算术强度

## 顺序

逐元素 → [rmsnorm](./rmsnorm) / [softmax](./triton-softmax) → [reduce](./cuda-reduce) → [matmul](./cuda-tiled-matmul) → flash attention；[top-p](./top-p-sampling) 是 scan 那条支线。

---
title: LeetGPU 题解
---

# LeetGPU 题解

[LeetGPU](https://leetgpu.com/challenges) 上做过的题，一题一页：题解代码 + 这道题特有的坑。概念和白板版写法在 [手撕高频](/handson/)，这里只管把 OJ 刷过。attention 类的题先看 [通用模板](./attention)。

## 题单

按 [手撕高频](/handson/) 的路线分组。实现等级：<code>torch</code> 只在乎 correctness，<code>triton</code> 在乎效率。题单在 `src/data/leetgpu-roadmap.ts`，直接改。

<LeetGPURoadmap />

## 全部题目

<LeetGPUBoard />

## 通用语法坑

PyTorch 题都是 `def solve(..., output, ...)`，结果要写进传入的 `output`，不看返回值。下面这些每道题都可能踩。

**会报错的**

1. **`dk = d_model / h`**：`/` 永远返回 float，`view` / `reshape` 报 "must be tuple of ints, but got float"。用 `//`。
2. **`view` / `transpose` / `reshape` 的结果没接住**：它们都返回新 tensor，不改原 tensor。单独一行 `Q.view(...).transpose(0, 1)` 等于没写，后面再按 3 维去 `transpose(1, 2)` 会报 "Dimension out of range"。
3. **transpose 之后直接 `view`**：transpose 只改 stride，tensor 不再连续，`view` 报 "view size is not compatible with input tensor's size and stride"。用 `reshape`，或先 `.contiguous()`。
4. **写 output 的方式不对**：
   - `return out` 或 `output = out` 都不会写到传入的显存，后者只是把局部变量指向新 tensor；
   - `out._copy(output)` 方法名和方向都错；
   - 正确写法是 `output.copy_(out)`，要求 out 能广播成 output 的 shape，维度数不对会报 "size of tensor a must match"。
5. **`matmul(..., out=output)` 而 shape 不一致**：PyTorch 会把 output resize 掉并报 UserWarning，写进去的布局也是错的。`out=` 只在 shape 完全对上时用，否则局部算完再 `copy_`。
6. **`torch.nn.Softmax(attn, dim=-1)`**：`nn.Softmax` 是 module 类，这是在构造对象。用函数版 `torch.softmax(attn, dim=-1)`。
7. **`torch.sqrt(d)`**：`torch.sqrt` 只吃 tensor，标量用 `math.sqrt(d)`。

**不报错、结果静默错的**

8. **3 维 tensor 用 `.T`**：`.T` 反转全部维度，不是只换最后两维。转置 K 用 `transpose(-1, -2)`。
9. **广播对齐错位**：matmul 的 batch 维从右往左对齐，维度数不同时，短的那个对上的是长的靠右的维度，不一定是你想的那一维。测试用例里两维长度恰好相等时不报错。缺的维度用 `unsqueeze` 显式补上。
10. **大小写变量拿错**：`v = V.unsqueeze(1)` 之后又写了 `matmul(attn, V)`。改过 shape 的变量起个不同的名字，或直接覆盖原名。

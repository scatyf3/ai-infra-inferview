---
title: GPTQ 推导细节
status: draft
tags: [quantization, gptq]
difficulty: 4
order: 8.5
related: [/inference/quantization]
---

# GPTQ 推导细节

> [量化](/inference/quantization) 页 GPTQ 一节的数学推导。

GPTQ 做 weight-only 量化：按 $K$ 维一行一行地量化 $W$，每量化完一行，就用 Hessian 的逆把误差分摊到还没量化的行上，使 $\|XW - X\hat{W}\|^2$ 最小。记号和量化页一致：$Y = XW$，$X \in \mathbb{R}^{M \times K}$ 是激活，$W \in \mathbb{R}^{K \times N}$ 是权重。

## 1. 目标按列拆开

记 $w_j \in \mathbb{R}^K$ 是 $W$ 的第 $j$ 列（一个输出通道），$\delta_j = \hat{w}_j - w_j$ 是它的量化误差：

$$
\|XW - X\hat{W}\|^2 = \sum_j \|X \delta_j\|^2 = \sum_j \delta_j^\top H \delta_j, \quad H = X^\top X \in \mathbb{R}^{K \times K}
$$

$X$ 是 calibration 数据上的激活。$H$ 只由激活决定，**所有 $N$ 列共用同一个 $H$**。论文写 $H = 2X^\top X$，是平方误差对 $w$ 的二阶导，系数 2 不影响结果。下面只看一列，省掉下标 $j$。

## 2. 量化一个元素后怎么补偿

把第 $k$ 个元素量化，它的误差 $e = \text{quant}(w_k) - w_k$ 就定死了。还没量化的元素可以随便调，问题变成：

$$
\min_\delta \ \delta^\top H \delta \quad \text{s.t.} \quad \delta_k = e
$$

拉格朗日乘子法：$H\delta = \lambda e_k$（$e_k$ 是第 $k$ 个单位向量），得 $\delta = \lambda H^{-1} e_k$，代入约束 $\lambda [H^{-1}]_{kk} = e$：

$$
\delta = \frac{e}{[H^{-1}]_{kk}} H^{-1}_{:,k}, \qquad \text{增加的误差} = \frac{e^2}{[H^{-1}]_{kk}}
$$

$\delta$ 的第 $k$ 个分量正好是 $e$，其余分量就是给还没量化的元素的补偿量。这里的 $H^{-1}$ 只算还没量化的那些元素。

### 直觉

取 $K = 2$，$H = \begin{bmatrix} h_{11} & h_{12} \\ h_{12} & h_{22} \end{bmatrix}$，量化 $w_1$ 后，$w_2$ 的补偿量是 $\delta_2 = -\frac{h_{12}}{h_{22}} e$。$h_{12} = \sum_m X_{m1} X_{m2}$ 衡量两个输入通道的相关性。如果两个通道的激活总是相等，$h_{12} = h_{22}$，$\delta_2 = -e$：$x_1 w_1 + x_2 w_2 = x_1(w_1 + w_2)$，$w_1$ 多出来的正好由 $w_2$ 减掉。如果两个通道不相关，$h_{12} = 0$，没法补偿。

## 3. 逐行做完

量化完第 $k$ 行就把它从「还没量化」里移除，$H^{-1}$ 做一步高斯消元：

$$
H^{-1} \leftarrow H^{-1} - \frac{H^{-1}_{:,k} \, H^{-1}_{k,:}}{[H^{-1}]_{kk}}
$$

因为所有列共用 $H$，第 $k$ 行的 $N$ 个元素可以一起量化、一起补偿：

```python
# W [K, N]，X 是 calibration 激活 [M, K]
H = X.T @ X                                          # [K, K]，所有列共用
Hinv = torch.linalg.inv(H)
for k in range(K):                                   # 按 K 维逐行
    q = quant(W[k])                                  # 第 k 行 N 个元素一起量化
    e = (W[k] - q) / Hinv[k, k]                      # [N]
    W[k + 1:] -= Hinv[k + 1:, k, None] * e           # 误差分摊到还没量化的行
    W[k] = q
    Hinv -= Hinv[:, k, None] * Hinv[k, :] / Hinv[k, k]   # 把第 k 行移出
```

## 4. 工程上的三点

让它能跑 175B 的模型：

1. **固定顺序**：前身 OBQ 每列贪心地挑增加误差最小的元素先量化，每列顺序不同，$H^{-1}$ 要各更新各的。GPTQ 让所有列按同一个顺序，$H^{-1}$ 的更新只做一遍，所有列共用。OPT-175B 用一张 A100 约 4 小时就能量化完。
2. **Cholesky**：上面逐步更新 $H^{-1}$ 会累积数值误差。GPTQ 预先对 $H^{-1}$ 做 Cholesky 分解，上三角因子的第 $k$ 行正好就是第 $k$ 步要用的那一行，不用再逐步更新。$H$ 先加上对角线均值的 1% 做 damping，防止不可逆。
3. **lazy batch**：每 128 行一块，块内逐行更新，块外的行等整块做完后用一次矩阵乘统一更新，把访存密集的逐行操作变成计算密集的 GEMM。

## 5. act-order

这是 GPTQ 和 outlier 的联系：$H$ 的对角线 $H_{kk} = \sum_m X_{mk}^2$ 是第 $k$ 个输入通道激活的平方和，outlier 通道的 $H_{kk}$ 特别大。按 $H_{kk}$ 从大到小的顺序量化，先处理 outlier 通道，它们的误差就有更多还没量化的行来补偿。代价是和 per-group 一起用时，量化顺序和组的划分对不上，kernel 要按额外的索引（`g_idx`）去找每行的 scale，会慢一些。

## 参考

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [Optimal Brain Compression（OBQ）](https://arxiv.org/abs/2208.11580)

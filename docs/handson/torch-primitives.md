---
title: torch 原语边界
status: draft
tags: [pytorch, guide, handson]
difficulty: 1
order: 0
related: [/handson/mha-gqa-forward, /handson/stable-softmax, /handson/rmsnorm, /handson/top-p-sampling]
stack: []
---

# torch 原语


### tensor itself

tensor
1. 一堆数据：**storage**，一维连续的一块内存，不带形状信息
2. shape+stride：怎么把这串数看成多维数组的**元数据**
   - `shape`：每一维多长
   - `stride`：每一维下标 +1，在 storage 里跳几个元素，
   - 当tensor连续时，stride = 大于当前index的shape数值乘
   - `storage_offset`：从 storage 第几个元素开始；另外还有 `dtype`、`device`
3. shape和stride的0 index从最高维度开始

`x[i, j]` 在 storage 里的位置是 `offset + i*stride[0] + j*stride[1]`。

```python
x = torch.arange(6).view(2, 3)   # storage: [0 1 2 3 4 5]
x.shape, x.stride()              # (2, 3), (3, 1)   行内相邻差 1，换一行跳 3

y = x.t()
y.shape, y.stride()              # (3, 2), (1, 3)   storage 没动，只对调了 stride
y.data_ptr() == x.data_ptr()     # True，共用同一块内存
y.is_contiguous()                # False
```

1. **contiguous**：stride 正好是行优先排列该有的值，shape `(a, b, c)` 对应 stride `(b*c, c, 1)`。transpose / permute 之后就不连续了。
2. **为什么 transpose 之后 `view` 报错**：`view` 不动数据，只换一套 shape + stride。现有内存排列表达不成目标 shape 时（比如 transpose 后把 `(N, h, dk)` 合成 `(N, d_model)`，`h` 和 `dk` 在内存里已经不相邻），只能报 "view size is not compatible with input tensor's size and stride"。
- `reshape`：能 view 就 view，不能就先拷一份连续的再 view
- `.contiguous()`：手动做这次拷贝
- **默认用 `reshape`**。只有要通过结果写回原 tensor 时才用 `view`：`view` 保证不拷贝，做不到就报错；`reshape` 可能悄悄拷一份，写进去的是副本，原 tensor 没变也不报错

#### 只改元数据 vs 会拷贝

| 只改元数据，和原 tensor 共享 storage | 分配新内存 |
|---|---|
| `view`、`transpose` / `permute` / `.T`、`unsqueeze` / `squeeze`、切片 `x[:, 1:3]`、`expand` | `clone`、不连续时的 `contiguous()`、不能 view 时的 `reshape`、`repeat`、`repeat_interleave`、`cat`、所有算术结果（`a + b`、`matmul` …） |

左边一列：
- 几乎零开销，不碰数据
- 通过它们写数据会改到原 tensor（同一块内存）

`expand` 把被扩展维的 stride 设成 0，下标怎么变都读同一个元素，不拷贝就实现了广播。GQA 里 `expand` 比 `repeat_interleave` 省显存就是这个原因。

排查用：`x.stride()`、`x.is_contiguous()`、`x.data_ptr()`（看两个 tensor 是否共用内存）。

### broadcasting

**规则**：两个 shape 从右往左对齐，每一维要么相等，要么有一边是 1；维度数不够的，在左边补 1。结果每一维取两者中较大的那个。

```
A:      (H_kv, q_div_kv, S, D)
B:      (H_kv,        1, S, D)   →  (H_kv, q_div_kv, S, D)   ✓ 1 被拉长
C:            (H_kv, S, D)       →  左边补 1 成 (1, H_kv, S, D)，H_kv 对上的是 q_div_kv
```

**哪些操作会广播**
- 逐元素运算：`+ - * /`、比较、`torch.where`
- `matmul`：只广播 batch 维（最后两维以外），最后两维做矩阵乘
- `copy_`：src 广播成 dst 的 shape，dst 本身不变形
- `masked_fill`：mask 广播成 x 的 shape；causal mask `(S, S)` 加到 `(H, S, S)` 的 score 上也是靠这个

**实现**：被拉长的维度 stride 设成 0，和 `expand` 一样，不拷贝、不占额外显存。

#### 容易踩的坑
- **维度数不同时对错位**：短的那个对上的是长的靠右几维，不一定是你想的那一维。长度恰好相等时不报错，结果静默错（GQA：`(H_kv, S, D)` 和 `(H_kv, q_div_kv, S, D)` 相乘，样例里 `H_kv == q_div_kv == 2`）
- **意外广播**：`(N,)` 加 `(N, 1)` 得到 `(N, N)`，不报错
- **归约漏了 `keepdim=True`**：`logits (B, V) - logits.max(dim=-1).values` 里减数是 `(B,)`，对上的是 V 那一维。`B != V` 时报错，`B == V` 时静默错。归约后还要和原 tensor 运算的，一律 `keepdim=True`
- **习惯**：参与运算的 tensor 先用 `unsqueeze` / `x[:, None]` 把维度数补齐，让每一维的对应关系写在代码里，不靠右对齐规则去猜

### modify tensor

原则：**默认 out-of-place；只有语义上就是"更新一块已有状态"时才 in-place**（参数、KV cache、预分配的 buffer）。

```python
h = torch.softmax(x @ w, dim=-1)        # 普通计算：产生新值，out-of-place
k_cache[:, pos] = k_new                 # 更新已有状态：KV cache 本来就该原地写
with torch.no_grad():
    param.add_(grad, alpha=-lr)         # 更新已有状态：参数
```

**训练（有 autograd）：默认 out-of-place**
- 手动 in-place 省不了多少显存：缓存分配器会复用释放的块，`torch.compile` 也会融合算子、复用 buffer
- 风险是实打实的：原地改了反向要用的中间结果，轻则报 "modified by an inplace operation"，重则梯度静默错
- 约定俗成的 in-place 只有几处：optimizer 更新参数（`param.add_`）、`zero_grad`、权重初始化（都在 `no_grad` 下）；`nn.ReLU(inplace=True)`（反向只要输出，覆盖输入安全）

**推理 / kernel 风格：原地写预分配 buffer 很常见**
- KV cache：显存预先分好，每步 `index_copy_` 或切片赋值写进去
- CUDA Graph：每次运行必须是同一块显存，只能原地更新
- LeetGPU 的 `solve(..., output)`：同一类，给你一块 buffer 往里写

#### 容易踩的坑
- **返回值没接住**：`view` / `transpose` / `reshape` 都是 out-of-place，单独一行 `Q.view(...)` 等于没写
- **`output = y`**：只是让局部名字指向别的 tensor，调用方那块显存没被写；要 `output.copy_(y)`
- **`x += y` ≠ `x = x + y`**：前者对 tensor 是原地的，`x` 是传进来的参数时会改到调用方
- **`out=` shape 不一致**：output 会被 resize，写进去的布局也错；只在 shape 完全对上时用
- **`copy_` 方向和广播**：`dst.copy_(src)`，src 先广播成 dst 的 shape，不兼容报 "size of tensor a must match"
- **通过 view 写会改原 tensor**：共用 storage。反过来也能利用：`output.view(N, h, dk).copy_(out.transpose(0, 1))` 直接写进 output，省一次 reshape

### operator

LeetGPU attention 系列（[#6](/leetgpu/softmax-attention)、[#12](/leetgpu/multi-head-attention)、[#80](/leetgpu/grouped-query-attention)、[#53](/leetgpu/causal-self-attention)）用到的全部算子。改 shape 的（`view` / `reshape` / `transpose` / `unsqueeze`）见上面 [tensor itself](#tensor-itself)，写回的（`copy_` / `out=`）见 [modify tensor](#modify-tensor)。

#### 矩阵乘

| 算子 | 用法 | 注意 |
|---|---|---|
| `torch.matmul(a, b)` / `a @ b` | `q @ k.transpose(-1, -2)` | 只对最后两维做矩阵乘，前面全当 batch 并[广播](#broadcasting)；1 维输入会被临时补成矩阵 |
| `x.transpose(-1, -2)` | 转置 K | 用负数索引，前面有几个 batch 维都不用改；别用 `.T`（反转全部维度） |

#### 归一化

| 算子 | 用法 | 注意 |
|---|---|---|
| `torch.softmax(x, dim=-1)` | 沿 key 维归一化 | 函数版。`torch.nn.Softmax` 是 module 类，`nn.Softmax(x)` 是在构造对象；内部已减 max，数值稳定 |
| `math.sqrt(d)` | 缩放 `1/√d_k` | `d` 是 Python int；`torch.sqrt` 只吃 tensor。也可以写 `d ** 0.5` |

#### 构造 tensor

| 算子 | 用法 | 注意 |
|---|---|---|
| `torch.full(shape, value, device=...)` | `torch.full((S, S), float('-inf'), device=Q.device)` | 默认在 CPU，和 GPU tensor 运算会报 device 不一致，一律传 `device=x.device` |
| `torch.ones(shape, dtype=torch.bool, device=...)` | 布尔 mask 的底 | 同上；`dtype=torch.bool` 后可以用 `~` 取反 |

构造类算子还有 `torch.zeros` / `torch.empty` / `torch.arange`，规则一样。`torch.empty` 不初始化，读之前必须写满。`x.new_zeros(shape)` 会继承 x 的 dtype 和 device，省得手写。

#### mask

| 算子 | 用法 | 注意 |
|---|---|---|
| `x.triu(k)` / `torch.triu(x, k)` | `full(-inf).triu(1)` 得加法 causal mask | 保留第 k 条对角线及右上方，其余置 0。`k=1` 不含主对角线；写成 `triu(0)` 会把对角线也盖掉，第一行全 -inf，softmax 出 NaN |
| `x.masked_fill(mask, value)` | `attn.masked_fill(ones(bool).triu(1), float('-inf'))` | mask 为 True 的位置填 value，mask 广播成 x 的 shape。out-of-place，原地版是 `masked_fill_` |

两种 causal mask 写法等价，都用 `triu(1)` 标出「看不到」的位置：加法 mask（`attn + full(-inf).triu(1)`）和布尔 mask（`attn.masked_fill(ones(bool).triu(1), -inf)`）。都放在 softmax 之前，`exp(-inf) = 0`。

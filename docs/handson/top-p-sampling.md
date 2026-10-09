---
title: Top-p 采样
status: draft
tags: [sampling, handson]
difficulty: 2
order: 5
related: [/handson/beam-search, /handson/stable-softmax, /handson/kernel-mindset, /stack/o-sampling, /stack/o-logits, /inference/speculative-sampling-math]
stack: [o-sampling]
leetgpu: [60]
---

# Top-p 采样

> 与 top-k / temperature 组合 · [LeetGPU #60 Top-p Sampling](https://leetgpu.com/challenges)（[题面](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/60_top_p_sampling)）

## 一句话结论

采样的三个旋钮按顺序作用：temperature 把 logits 除以 T 改变分布的尖锐度，top-k 只保留概率最高的 k 个 token，top-p（nucleus）只保留"累计概率刚达到 p"的最小前缀，然后在剩下的 token 里按重新归一化的概率采样。top-p 比 top-k 好在候选集大小随分布自适应：分布尖时只留几个，分布平时多留一些。实现是 `sort` + 前缀和 + mask，边界条件是"**前面的 token 累计概率 < p 就保留**"。

## 定义与约定

- 一个请求这一步的 logits $z \in \mathbb{R}^V$，V 是词表大小（LeetGPU 测 50,000；Llama 3 是 128K，[Llama 3 Herd](https://arxiv.org/abs/2407.21783) §3.2）。batch 时是 `[B, V]`，每行一个请求。
- **temperature** $T > 0$：$\pi = \text{softmax}(z / T)$。T < 1 分布变尖，T > 1 变平，T → 0 退化成 argmax（greedy），实现上 T = 0 单独走 argmax，不去除以 0。
- **top-k**：把 π 降序排成 $\pi_{(1)} \ge \pi_{(2)} \ge \dots$，只保留前 k 个。
- **top-p**（[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)）：保留满足 $\sum_{i \le n} \pi_{(i)} \ge p$ 的最小 n 个。等价地，第 i 个保留当且仅当它**前面**的累计概率 $S_{i-1} = \sum_{j<i} \pi_{(j)} < p$。
- 截断后重新归一化再采样。截断顺序：HF transformers 默认是 temperature → top-k → top-p（`_get_logits_processor` 里 warper 的 append 顺序）。

### 边界：为什么是"前面的累计 < p"

例子：降序概率 `[0.5, 0.3, 0.15, 0.05]`，前缀和 `[0.5, 0.8, 0.95, 1.0]`，排他前缀和（不含自己）`[0, 0.5, 0.8, 0.95]`。

| p | 保留 | 理由 |
|---|---|---|
| 0.3 | 第 1 个 | 第 1 个的排他前缀 0 < 0.3；它本身已经把累计推过 0.3 |
| 0.8 | 前 2 个 | 0.5 + 0.3 = 0.8 恰好达到 p；第 3 个前面已经是 0.8，不 < 0.8 |
| 0.9 | 前 3 个 | 第 3 个前面是 0.8 < 0.9，要靠它把累计推过 0.9 |

如果误写成"自己的前缀和 ≤ p 才保留"，p = 0.3 时一个都不剩。

LeetGPU 的参考实现用的是 `searchsorted(cumsum, p, right=False) + 1`：找到第一个前缀和 ≥ p 的位置，保留到它为止，和上面的定义完全一样。HF 的 `TopPLogitsWarper` 是升序排、删掉 `cumsum <= 1 - p` 的；升序前缀和 = 1 − 降序排他前缀和，所以也是同一个条件。

## 手撕

### LeetGPU #60：单个请求，要和参考逐 token 一致

题目只给一个 seed，判题比较的是采出来的 token id（整数，容差等于要求完全相等），所以**随机数的消耗方式必须和参考一模一样**：同样先 `torch.manual_seed(seed)`，同样对"截出来的 nucleus 向量"调 `torch.multinomial(·, 1)`。如果改成对整个长度 V、尾部填 0 的向量调 `multinomial`，分布一样，但同一个 seed 采出来的 token 可能不同，判题过不了。

```python
import torch


def exclusive_cumsum(p):
    # 排他前缀和：第 i 个位置是前 i-1 个的和。用"右移一位"而不是 cumsum - p，
    # 后者有浮点误差，可能和参考的 searchsorted 在边界上差一个 token
    c = p.cumsum(-1)
    return torch.cat([torch.zeros_like(c[..., :1]), c[..., :-1]], dim=-1)


# logits, p, seed, sampled_token are tensors on the GPU
def solve(logits: torch.Tensor, p: torch.Tensor, seed: torch.Tensor,
          sampled_token: torch.Tensor, vocab_size: int):
    probs = torch.softmax(logits, dim=-1)                  # 内部减 max，数值稳定
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    keep = exclusive_cumsum(sorted_p) < p                  # p 是 shape [1] 的 tensor，广播
    n_keep = int(keep.sum())                               # 保留的一定是前缀，数个数就行
    nucleus = sorted_p[:n_keep] / sorted_p[:n_keep].sum()  # 重新归一化
    torch.manual_seed(int(seed.item()))                    # 和参考一样的 RNG 状态
    j = torch.multinomial(nucleus, 1)                      # 在 nucleus 里的下标
    sampled_token[0] = sorted_idx[j]                       # 还原成词表里的 token id
```

对拍：把题目的 `reference_impl` 原样抄过来，随机 V ∈ [3, 3000)、logits 尺度 0–10、p ∈ (0, 1]，500 组全部逐 token 一致（CPU 上跑的）。

```python
def lg_ref(logits, p, seed, sampled_token, vocab_size):   # 题目参考，原样抄
    probs = torch.exp(logits - logits.max()); probs = probs / probs.sum()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    cutoff = min(torch.searchsorted(cumsum, p.item(), right=False).item() + 1, vocab_size)
    nucleus = sorted_probs[:cutoff] / sorted_probs[:cutoff].sum()
    torch.manual_seed(seed.item())
    sampled_token[0] = sorted_indices[:cutoff][torch.multinomial(nucleus, 1).item()]

torch.manual_seed(1)
for trial in range(500):
    V = int(torch.randint(3, 3000, (1,)))
    logits = (torch.randn(V) * float(torch.rand(1) * 10)).clamp(-100, 100)
    p = torch.rand(1).clamp(min=1e-3) if trial % 7 else torch.tensor([1.0])
    seed = torch.randint(0, 10000, (1,), dtype=torch.int32)
    a, b = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    lg_ref(logits, p, seed, a, V); solve(logits, p, seed, b, V)
    assert a.item() == b.item()
```

### 推理框架里：batch，每个请求的 T / k / p 不同

serving 时一个 batch 里每个请求的采样参数都可能不同，要打包成 `[B]` 的 tensor 一次处理，不能按请求写 for 循环。mask 照 [mask 的写法](/leetgpu/mask)，用下标广播出 bool 的 `allowed`：

```python
def sample(logits, temperature, top_k, top_p, generator=None):
    # logits [B, V]；temperature、top_p: float [B]；top_k: long [B]，0 表示不截断
    B, V = logits.shape
    greedy_tok = logits.argmax(dim=-1)                                  # T = 0 的请求用它
    probs = torch.softmax(logits.float() / temperature.clamp(min=1e-5)[:, None], dim=-1)
    sorted_p, idx = probs.sort(dim=-1, descending=True)                 # [B, V]

    rank = torch.arange(V, device=logits.device)[None, :]               # [1, V]，排序后的名次
    k = torch.where(top_k > 0, top_k, V)[:, None]                       # [B, 1]
    allowed_k = rank < k                                                # [B, V]
    sorted_p = sorted_p.masked_fill(~allowed_k, 0.0)
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)            # 截完 top-k 先重新归一化

    allowed_p = exclusive_cumsum(sorted_p) < top_p[:, None]             # 在归一化后的分布上做 top-p
    # 名次 0 一定保留（0 < k，排他前缀 0 < p），每行至少剩一个
    sorted_p = sorted_p.masked_fill(~allowed_p, 0.0)
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    j = torch.multinomial(sorted_p, 1, generator=generator)             # [B, 1]，排序后的下标
    tok = idx.gather(-1, j).squeeze(-1)                                 # 还原 token id
    return torch.where(temperature == 0, greedy_tok, tok)
```

top-k 和 top-p 都开时有两种语义：(a) 先截 top-k、**重新归一化**、再在新分布上做 top-p；(b) 两个条件都按原分布算再取交集。截完 top-k 后概率变大，前缀和更早到 p，所以 (a) 保留的 token 不多于 (b)。HF 的 `TopPLogitsWarper` 收到的 logits 已经被 top-k 置成 −inf，它内部再 softmax，就是 (a)；vLLM 的 PyTorch 路径也是先 top-k mask 再 softmax 做 top-p。上面的代码按 (a) 写。

检查：分布是否正确（20 万次采样的经验频率）：

```python
n = 200_000
# 只开 top-k = 3：应该是前 3 个的 softmax
L = torch.tensor([[2., 1., 0.5, 0., -1.]]).repeat(n, 1)
t = sample(L, torch.ones(n), torch.full((n,), 3), torch.ones(n))
emp = torch.bincount(t, minlength=5).float() / n
ref = torch.zeros(5); ref[:3] = torch.softmax(torch.tensor([2., 1., 0.5]), -1)
assert (emp - ref).abs().max() < 0.01

# 只开 top-p = 0.8：[0.5, 0.3, 0.15, 0.05] 只剩前两个，归一化成 [0.625, 0.375]
probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
t = sample(probs.log().repeat(n, 1), torch.ones(n), torch.zeros(n, dtype=torch.long), torch.full((n,), 0.8))
emp = torch.bincount(t, minlength=4).float() / n
assert (emp - torch.tensor([0.625, 0.375, 0., 0.])).abs().max() < 0.01

# 两个都开：[0.4, 0.3, 0.2, 0.1]，k = 3 → [4/9, 3/9, 2/9]，排他前缀 [0, 0.444, 0.778]
# p = 0.7 → 留前两个 → [4/7, 3/7]（若按原分布取交集，排他前缀 [0, 0.4, 0.7]，也是前两个；
# 换成 p = 0.75 两种语义就分开了：(a) 留 2 个，(b) 留 3 个）
probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
t = sample(probs.log().repeat(n, 1), torch.ones(n), torch.full((n,), 3), torch.full((n,), 0.7))
emp = torch.bincount(t, minlength=4).float() / n
assert (emp - torch.tensor([4 / 7, 3 / 7, 0., 0.])).abs().max() < 0.01
```

## 代价与更快的做法

### 排序是瓶颈

`sort` 是 O(V log V)，而 softmax、前缀和、mask 都是 O(V) 的单遍操作。B = 64、V = 128K 时一次要排 800 多万个元素。FlashInfer 的博客指出基于排序的 top-k / top-p 在大词表下开销很大（[Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html)）。

前缀和本身也是一个经典的并行原语（scan），第 i 个输出依赖前 i 个输入，见 [kernel mindset](/handson/kernel-mindset) 里的 scan。

### 用指数噪声代替 multinomial

vLLM 的采样不调 `multinomial`，而是：

```python
q = torch.empty_like(probs).exponential_()     # q_i ~ Exp(1)，独立
token = probs.div(q).argmax(dim=-1)
```

（[vLLM `topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)）。为什么对：$q_i / \pi_i$ 服从速率为 $\pi_i$ 的指数分布，独立指数变量里最小的是第 i 个的概率是 $\pi_i / \sum_j \pi_j$。所以 $\arg\max_i \pi_i / q_i = \arg\min_i q_i / \pi_i$ 恰好按 π 采样。好处是只要逐元素运算 + argmax，没有前缀和，也不需要 π 归一化（被 mask 成 0 的 token 永远不会被选中）。经验验证：

```python
pr = torch.tensor([0.5, 0.3, 0.2]).repeat(200_000, 1)
q = torch.empty_like(pr).exponential_()
emp = torch.bincount(pr.div(q).argmax(-1), minlength=3).float() / 200_000
assert (emp - torch.tensor([0.5, 0.3, 0.2])).abs().max() < 0.01
```

### 不排序的 top-p：拒绝采样

FlashInfer 的思路：不先把 nucleus 求出来，而是直接从原分布采一个候选（逆变换采样，累计到超过均匀随机数就停，大多数情况下不用扫完整个词表），再检查它是否在 nucleus 里；不在就用它的概率作为 pivot，把概率不超过 pivot 的 token 排除掉，在剩下的里重采。单 pivot 版本轮数没有上界；他们的 Dual Pivot Rejection Sampling 每轮至少把 pivot 的搜索区间减半，轮数是 $O(\log(1/\epsilon))$，ε 是浮点能表示的最小值（作者给出的证明，见上面的博客）。

## 面试追问

::: details Q：top-p 的 mask 为什么要用"前面的累计"而不是"自己的累计"？
如果 mask 掉 cumsum > p 的位置，当第一个 token 的概率就超过 p 时（比如 0.95，p = 0.9），所有 token 都被 mask，multinomial 会报错或采出垃圾。正确做法是用排他前缀和（不含自己）和 p 比较，保证"把累计推过 p 的那个 token"被保留，候选集至少有一个元素。写成 `cumsum - p_self` 数学上等价，但浮点上不是逐位相同，要和参考严格对齐时用右移一位的排他前缀和。
:::

::: details Q：temperature、top-k、top-p 的顺序有关系吗？
有。temperature 改变分布形状，在它之后做 top-p，nucleus 的大小会变：T 小时分布尖，同样的 p 保留更少 token。HF 默认顺序是 temperature → top-k → top-p。top-k 和 top-p 之间，先截 top-k 再重新归一化做 top-p，和"都按原分布算再取交集"结果也不同，前者保留的更少，见上面的说明。
:::

::: details Q：为什么 serving 里采样要 batch 化？
decode 阶段每步每个请求都要采样一次，按请求循环会变成 B 次小 kernel launch，和一次 forward 的时间比不可忽略。把每个请求的 T、k、p 打成 `[B]` 的 tensor，整个 batch 一次 sort、一次 mask、一次采样。greedy 请求（T = 0）走 argmax，用 `torch.where` 合并结果。
:::

## 参考

- Ari Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), ICLR 2020（nucleus sampling）
- Shanli Xing, Zihao Ye et al., [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html), FlashInfer blog, 2025
- [vLLM `vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)（排序 + mask 的 top-k/top-p、指数噪声采样）
- [HF transformers `TopPLogitsWarper`](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/logits_process.py)

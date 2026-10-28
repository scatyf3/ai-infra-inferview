---
title: MoE Grouped GEMM
status: draft
tags: [moe, gemm, kernel]
difficulty: 4
order: 4
related: [/gpu/tensor-core-gemm, /parallel/moe-ep, /gpu/quant-gemm, /gpu/memory-bound-kernels, /inference/prefill-decode-roofline, /posttrain/rl-train-infer-mismatch]
stack: [k-gemm]
train: [tr-k-gemm]
---

# MoE Grouped GEMM

> 几十上百个大小不一、运行时才知道形状的小 GEMM，怎么合成一个 kernel

## 一句话结论

MoE 层里每个 token 选 $k$ 个 expert，每个 expert 对分给它的 token 做一个 FFN。所以一层是 $E$ 个 GEMM：形状都是 $(m_e \times K) \cdot (K \times N)$，$K$、$N$ 相同，只有行数 $m_e$ 不同，而且 $m_e$ 运行时才知道、差别大、经常很小。

grouped GEMM 的想法是：**把所有 expert 的 tile 摊平成一个列表，一个 kernel 里的 block 按编号去领 tile，每个 tile 只属于一个 expert**。tile 内部就是普通的 [GEMM tiling](/gpu/tensor-core-gemm)，难点都在 tile 外面：

1. **排队**：token 按 expert 分好段，每段补齐到 $BM$ 的整数倍，保证一个 tile 不跨 expert。
2. **派活**：tile 列表的长度取决于运行时的 $m_e$，kernel 要能在 CPU 不知道 $m_e$ 的情况下启动，才能进 CUDA graph。
3. **代价**：补齐浪费的计算随 $BM$ 增长；decode 时每个 expert 只分到几个 token，又退化成 [GEMV](/gpu/tensor-core-gemm#面试追问)，而且要读的 expert 远比想象的多。

一层 MoE 的完整流程：

```
router：每个 token 选 top-k 个 expert 和权重
  → align：token 按 expert 排序、补齐到 BM            （排队）
  → grouped GEMM：gate 和 up 合在一起，N = 2I          （派活）
  → SiLU(gate) · up
  → grouped GEMM：down
  → 乘路由权重，同一个 token 的 k 个结果加起来（combine）
```

跨卡的 dispatch / combine 通信见 [MoE 与 Expert Parallel](/parallel/moe-ep)，本页只讲一张卡上的 kernel。

## 约定

$T$ 是这一步的 token 数，$E$ 是 expert 数，每个 token 选 $k$ 个。$T$ 个 token 一共产生 $Tk$ 个 **slot**，slot $s = t \cdot k + j$ 表示第 $t$ 个 token 的第 $j$ 个选择。expert $e$ 分到 $m_e$ 个 slot，$\sum_e m_e = Tk$。

例子用 DeepSeek-V3：hidden $d = 7168$，每个 expert 的中间维 $I = 2048$，$E = 256$ 个 routed expert，$k = 8$（[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)；另有 1 个 shared expert，是普通的稠密 GEMM，这里不算）。一个 expert 的权重是 gate+up $7168 \times 4096$ 加 down $2048 \times 7168$，共 $4.4 \times 10^7$ 个参数，FP8 下 44 MB。

## 朴素做法为什么不行

| 做法 | 问题 |
|---|---|
| `for e in experts: Y[e] = X[e] @ W[e]` | 每层 $2E$ = 512 次 launch；每个 GEMM 很小，$m_e$ 只有一个 tile，SM 喂不满；CPU 要先知道每个 $m_e$，得等 GPU 算完 router 再同步回来 |
| batched GEMM（`bmm`），所有 expert 一次算 | 要求每个 expert 行数一样，只能全部补到 $\max_e m_e$。路由不均时最热的 expert 可能是平均的好几倍，其余全是浪费 |

grouped GEMM 只把每个 expert 补到 $BM$ 的倍数，浪费小得多，而且一次 launch。

## 1. 排队：token 按 expert 分段

router 输出 `topk_ids`，形状 $[T, k]$。排队要产出三样东西（vLLM 里由 `moe_align_block_size` 生成）：

- `sorted_token_ids`：所有 slot 按 expert 排好，每个 expert 的那段补齐到 $BM$ 的整数倍，补的位置填一个无效值；
- `expert_ids`：每个长度为 $BM$ 的块属于哪个 expert；
- `num_tokens_post_padded`：补齐后的总长度。

小例子：$T = 4$，$k = 2$，$E = 4$，$BM = 2$。

```
topk_ids = [[0, 2],      # token 0 → slot 0（expert 0）、slot 1（expert 2）
            [1, 2],      # token 1 → slot 2（expert 1）、slot 3（expert 2）
            [0, 3],      # token 2 → slot 4（expert 0）、slot 5（expert 3）
            [2, 1]]      # token 3 → slot 6（expert 2）、slot 7（expert 1）

按 expert 分段、补齐到 BM = 2（P 是补的空位）：
expert 0: [0, 4]
expert 1: [2, 7]
expert 2: [1, 3, 6, P]
expert 3: [5, P]

sorted_token_ids       = [0, 4, 2, 7, 1, 3, 6, P, 5, P]
expert_ids（每 2 个一块）= [0,    1,    2,    2,    3   ]
num_tokens_post_padded = 10
```

slot $s$ 对应的激活行是 $X[s \,//\, k]$，比如 slot 7 是 token 3。

排好之后有两种用法：

| 用法 | 做法 | 代表 |
|---|---|---|
| **在 GEMM 里 gather** | 不复制 $X$。GEMM 读 A 时按 `sorted_token_ids[i] // k` 去取第几行 | vLLM 的 Triton `fused_moe` |
| **先 permute 成连续的** | 先把 $X$ 按排好的顺序复制成一个新张量，GEMM 读的就是规则的矩形 | DeepGEMM 的 contiguous 布局；EP 的 all-to-all 收到的 token 本来就按 expert 分好了 |

gather 省掉一次复制（$Tk \times d$ 个元素的读写），代价是 A 的 tile 不再是规则矩形。每一行内部沿 $K$ 还是连续的（7168 个 bf16 = 14 KB），访存照样能合并，但 Hopper 的 TMA 只能搬规则的矩形 tile，按索引挑行得用普通的 load 或 `cp.async`。

## 2. 派活：tile 列表由谁生成

tile 的总数是

$$
\sum_e \left\lceil \frac{m_e}{BM} \right\rceil \cdot \frac{N}{BN}
$$

它取决于运行时的 $m_e$。问题是 grid 大小要在 launch 时给定，而 launch 是 CPU 发的：

| 做法 | grid | CPU 要知道 $m_e$ 吗 | 能进 CUDA graph | 代价 |
|---|---|---|---|---|
| CPU 算好精确的 tile 数再 launch | 精确 | 要，得等 GPU 同步回来 | 不能 | 一次 GPU → CPU 同步 |
| **按上界 launch，多余的 block 直接退出** | 上界 | 不要 | 能 | 空 block 的调度开销 |
| **persistent kernel + GPU 上的 tile 调度** | SM 数 | 不要 | 能 | 每个 block 领 tile 时要查自己属于哪个 expert |
| **masked 布局**：每个 expert 预留固定容量 | 固定：$E \times$ 容量 $/ BM$ | 不要 | 能 | 容量按最坏情况留，显存和空 block 都浪费 |

上界很好算：每个 expert 最多补 $BM - 1$ 行，所以补齐后的长度不超过 $Tk + E(BM - 1)$。vLLM 的 `fused_moe` 就按这个长度开 grid，每个 block 先看自己的起始行是否超过 `num_tokens_post_padded`，超过就直接返回；没超过就读 `expert_ids` 知道该用哪个 expert 的权重：

```python
pid_m, pid_n = grouped_pid(program_id(0))          # 和普通 GEMM 一样的 L2 分组
if pid_m * BM >= num_tokens_post_padded:            # 补齐长度之外：空 block
    return
slots = sorted_token_ids[pid_m * BM : (pid_m + 1) * BM]
e = expert_ids[pid_m]                               # 这个 tile 属于哪个 expert
a = X[slots // top_k, k0:k0+BK]                     # gather：按 slot 找原始 token 行
b = W[e, k0:k0+BK, pid_n * BN : (pid_n + 1) * BN]
...
if MUL_ROUTED_WEIGHT:                               # down 投影：epilogue 里乘路由权重
    acc *= topk_weights[slots][:, None]
C[slots, ...] = acc                                 # 按 slot 写回，之后再按 token 加
```

CUTLASS 的 grouped GEMM 走 persistent 那条路：只开和 SM 数相当的 CTA，每个 CTA 循环领下一个 tile 编号，再根据每个 problem（expert）的 tile 数算出这个编号落在哪个 problem 里。DeepGEMM 在 decode 时用 masked 布局，因为 decode 要进 CUDA graph，形状必须是静态的（见 [MoE 与 EP](/parallel/moe-ep)）。

## 3. 补齐浪费多少：$BM$ 又是一次折中

每个 expert 的行数补到 $BM$ 的倍数，平均每个 expert 补约 $BM / 2$ 行（$m_e$ 对 $BM$ 取余大致均匀时）。被分到 token 的 expert 有 $E'$ 个，多算的比例约为

$$
\frac{E' \cdot BM / 2}{Tk}
$$

prefill 时 $T = 4096$，DeepSeek-V3 的 $Tk = 32768$，256 个 expert 都会分到 token，平均每个 128 行（假设路由均匀，粗算）：

| $BM$ | 补的行数 ≈ $256 \times BM / 2$ | 多算的比例 | tile 的算术强度 $\frac{BM \cdot BN}{BM + BN}$（$BN = 128$） |
|---|---|---|---|
| 128 | 16384 | 50% | 64 |
| 64 | 8192 | 25% | 43 |
| 16 | 2048 | 6% | 14 |

$BM$ 越大，复用越多，但补齐越浪费。这和 [内积与外积](/gpu/tensor-core-gemm#内积和外积-tiling-在两者之间折中) 那节是同一个折中，只是多了一项浪费。所以 MoE kernel 的 $BM$ 一般比普通 GEMM 小，而且按 $Tk / E$（每个 expert 平均分到多少行）来选：vLLM 为每种 $M$ 存了调好的 `BLOCK_SIZE_M`。

## 4. decode：每个 expert 只有几个 token

decode 一步 $T$ 不大，每个 expert 平均只分到 $Tk / E$ 个 slot。以 $T = 64$ 为例，$Tk / E = 2$，每个 expert 的 GEMM 只有 2 行，就是 GEMV。

更要命的是要读多少个 expert 的权重。假设路由均匀且互相独立，一个 expert 不被某个 token 选中的概率是 $1 - k/E$，$T$ 个 token 都没选它的概率是 $(1 - k/E)^T$，所以被至少一个 token 选中的 expert 数是

$$
E' = E \left[ 1 - \left( 1 - \frac{k}{E} \right)^T \right]
$$

| $T$ | 被选中的 expert $E'$ | 占全部 expert | 每个被选中的 expert 平均几行 = 算术强度 |
|---|---|---|---|
| 1 | 8 | 3% | 1 |
| 8 | 57 | 22% | 1.1 |
| 64 | 222 | 87% | 2.3 |
| 256 | 256 | 100% | 8 |
| 1024 | 256 | 100% | 32 |

（算术强度按「每读一个权重做几次乘加」算，就是每个 expert 的行数。）

两个结论：

1. **激活参数少，不等于读的字节少**。单个 token 只用 3% 的 expert，但 batch 64 时一步要读 87% 的 expert 权重。decode 的时间由读了多少个 expert 决定，不是由每个 token 激活了多少参数决定。
2. **算术强度要等所有 expert 都被选中之后，才随 batch 线性涨**，约为 $Tk / E$。要到 bf16 的 ridge 295，单卡上一步得有约 $295 \times 256 / 8 \approx 9400$ 个 token。这是大规模 EP 的出发点：把 expert 分到很多卡上，每张卡汇集整个集群发给它那几个 expert 的 token，每个 expert 分到的行数就上去了（见 [MoE 与 EP](/parallel/moe-ep)）。

kernel 层面，decode 的 grouped GEMM 还有一个形状问题：Hopper 的 `wgmma` 的 $M$ 固定是 64，一个 expert 只有 2 行时 97% 的 $M$ 是空的。常见做法是 **swap AB**：算 $Y^\top = W^\top X^\top$，让权重的 $N$ 维（几千）当 $M$，token 数当 $N$。`wgmma` 的 $N$ 可以小到 8，浪费就少得多。vLLM 的 `fused_moe` 里有对应的 `SWAP_AB` 开关。

## 5. 两个 GEMM 之间和之后

- **gate 和 up 合成一个 GEMM**：两个权重拼成 $d \times 2I$，一次 grouped GEMM 出来，和 dense 模型的 `gate_up_proj` 一样（见 [Kernel Fusion](/gpu/cuda-graph-fusion)）。
- **SiLU(gate) · up**：可以单独一个 elementwise kernel，也可以融合进第一个 GEMM 的 epilogue。融合的前提是同一个 tile 里同时有第 $j$ 列的 gate 和第 $j$ 列的 up，所以要把两者的列交错存放。
- **乘路由权重**：放在 down GEMM 的 epilogue 里（上面代码的 `MUL_ROUTED_WEIGHT`），这时结果还在寄存器里，几乎免费。
- **combine**：down 的输出按 slot 写回，每个 token 有 $k$ 行，要加成一行。可以单独一个归约 kernel；也可以在 epilogue 里直接 atomic add 到 token 的那一行，省一个 kernel，但浮点加法的顺序不再固定，结果会随调度变（见 [训推不一致](/posttrain/rl-train-infer-mismatch)）。

## 面试追问

::: details Q：为什么不用 batched GEMM（bmm）一次算所有 expert？
batched GEMM 要求每个 problem 形状一样，只能把所有 expert 都补到最热那个 expert 的行数。路由一不均匀，大部分算的都是 0。grouped GEMM 只把每个 expert 补到 $BM$ 的倍数，浪费最多是每个 expert $BM - 1$ 行。
:::

::: details Q：DeepSeek-V3 每个 token 只激活 37B 参数，为什么 decode 不能按 37B 估算读多少字节？
「每个 token 激活 8 个 expert」是对单个 token 说的。一个 batch 的 token 各选各的，并起来覆盖的 expert 多得多：均匀路由下 batch 64 就会选中 87% 的 routed expert。decode 是 memory-bound，时间由要读的 expert 权重决定，所以 batch 稍大一点，就接近要读整层的 expert。
:::

::: details Q：BLOCK_SIZE_M 怎么选？
越大，每个 tile 里权重的复用越多（算术强度 $\frac{BM \cdot BN}{BM + BN}$），但每个 expert 平均补 $BM / 2$ 行，浪费越多。每个 expert 平均分到的行数 $Tk / E$ 越少，$BM$ 应该越小。实际由 autotune 按 $M$ 选，vLLM 为常见的 GPU 和形状存了调好的配置。
:::

::: details Q：swap AB 是什么，为什么 decode 需要？
Hopper 的 `wgmma` 一条指令的 $M$ 固定是 64。decode 时每个 expert 只有几行，放在 $M$ 维上几乎全是空的。把乘法转置成 $Y^\top = W^\top X^\top$，权重的输出维（几千）当 $M$，token 数当 $N$，而 $N$ 可以小到 8，浪费就少了。
:::

::: details Q：为什么 vLLM 的 fused_moe grid 要按上界开，而不是按实际 tile 数？
实际 tile 数取决于 router 的输出，只有 GPU 知道。按实际数 launch 要先同步回 CPU，既慢，也没法录进 CUDA graph。按上界 $Tk + E(BM - 1)$ 开 grid，CPU 不需要任何运行时信息；多出来的 block 一看起始行超过了 `num_tokens_post_padded`，就直接退出。
:::

## 参考

- [vLLM `fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)：Triton 版 fused MoE，`moe_align_block_size`、gather、`MUL_ROUTED_WEIGHT`、`SWAP_AB`
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)：contiguous / masked 两种布局的 FP8 grouped GEMM
- [CUTLASS examples/24_gemm_grouped](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)：persistent + GPU 上的 tile 调度
- [MegaBlocks (arXiv:2211.15841)](https://arxiv.org/abs/2211.15841)：把 MoE 写成块稀疏矩阵乘，不丢 token、不按容量补齐
- [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)

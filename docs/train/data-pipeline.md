---
title: 预训练数据管线：从 token 到 GPU，不让数据拖后腿
status: draft
tags: [data-pipeline, megatron, dataloader, packing]
difficulty: 3
order: 9
related: [/train/compute-mfu, /train/stability, /train/fault-tolerance, /train/long-context, /posttrain/sft, /parallel/megatron-tp, /gpu/profiling]
stack: []
train: [tr-d-format, tr-d-mix, tr-d-pack, tr-d-loader]
---

# 预训练数据管线：从 token 到 GPU，不让数据拖后腿

> 离线 tokenize 成 .bin + .idx，mmap 按样本号随机读；按权重混合多个源；每个 DP rank 拿不同样本、TP 组内广播；记下 consumed_samples 才能精确恢复

## 一句话结论

GPU 每秒要的 token 很少（8B、MFU 40% 时每卡约 8200 token/s，int32 存储才 33 KB/s），所以数据管线的难点不是带宽，而是**确定性和随机访问**：

1. 离线把语料 tokenize 成扁平的二进制 token 流（Megatron 的 `.bin`）加每篇文档的偏移索引（`.idx`），训练时 `numpy.memmap` 按需读取。
2. 文档首尾相接、中间插 EOS，切成定长 `seq_len + 1` 的样本；样本号 → (文档, 偏移) 的映射表和打乱顺序都由种子决定、预先算好存盘。
3. 多个源按权重混合，用贪心的「谁落后最多就选谁」排出每个位置来自哪个源；某源要的 token 超过它的大小就重复多个 epoch。
4. 每个 DP rank 取全局 batch 里自己那一段；TP 组里只有 rank 0 读数据再广播，PP 只有首末 stage 需要数据。
5. checkpoint 里存 `consumed_samples`，重启后从同一个全局样本号继续，和 DP 大小无关。这是复现 loss spike、跳过坏 batch 的前提。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $N$ | 模型参数量 | Llama 3 8B：$8.03\times10^9$；70B：$70.6\times10^9$ |
| peak | 单卡 bf16 dense 峰值算力 | H100 SXM：989 TFLOP/s |
| $u$ | MFU，实际 FLOP/s ÷ peak | 0.4 |
| $L$ | 序列长度（seq_len） | 8192 |
| $V$ | 词表大小 | Llama 2：32000；Llama 3：128256 |
| $b_{\text{tok}}$ | 每个 token id 的存储字节 | uint16：2；int32：4 |
| $D$ | 总训练 token 数 | $10^{12}$（1T） |
| $w_k$ | 第 $k$ 个数据源的混合权重，$\sum_k w_k = 1$ | 0.1 |
| $S_k$ | 第 $k$ 个源的 token 数 | $5\times10^{10}$ |
| $e_k$ | 第 $k$ 个源被过几遍（epoch 数） | 2 |
| $B_{\mu}$ | micro-batch size（每个 DP rank 一次 forward 的样本数） | 2 |
| $d$ | 数据并行度（DP size） | 4 |

单位：GB = $10^9$ 字节，1 B（billion）= $10^9$，1 T = $10^{12}$。

### 1. 算需求：每卡每秒要喂多少 token

训练每 token 约 $6N$ FLOPs：forward $2N$（每个参数一次乘加），backward $4N$（对输入、对权重各一次矩阵乘）。推导见 [训练算力账](/train/compute-mfu)。所以单卡每秒消耗的 token 数：

$$
\text{tokens/s/GPU} = \frac{u \cdot \text{peak}}{6N}
$$

| 模型 | $6N$（FLOPs/token） | $u = 0.4$ 时 token/s/GPU | 每节点 8 卡 | int32 时每卡字节/s |
|---|---|---|---|---|
| 8B | $4.82\times10^{10}$ | **8200** | 6.6 万 | 33 KB/s |
| 70B | $4.24\times10^{11}$ | **930** | 7500 | 3.7 KB/s |

$u = 0.4$ 不是拍脑袋：Llama 3 405B 在 8192–16384 张 H100 上报告 bf16 MFU 38–43%（[Llama 3 Herd](https://arxiv.org/abs/2407.21783) Table 4，作者自测）。

两个结论：

1. **带宽不是问题。** 每卡几十 KB/s，任何磁盘、网络文件系统都喂得动。模型越大，每 token 越贵，需要的 token/s 越少。
2. **tokenize 必须离线做，但理由不是速度。** 理由是后面每一节都要用到的两件事：(a) 要把文档切成定长样本并打乱，必须事先知道每篇文档有多少 token；(b) 要能按样本号随机读、重启后精确接上，样本内容必须是确定的。在线 tokenize 时这两件事都做不到，而且同一份语料每次训练都要重新 tokenize 一遍。

### 2. 离线格式：.bin + .idx，mmap 读取

Megatron 的 `preprocess_data.py` 把 jsonl 语料 tokenize 后写成两个文件（[indexed_dataset.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py)）：

1. `.bin`：所有文档的 token id 首尾相接，一个扁平数组，没有任何分隔结构。
2. `.idx`：头部（魔数 `MMIDIDX`、版本、dtype 代码、序列数、文档数），然后三个数组：每篇的长度（int32）、每篇在 `.bin` 里的字节偏移（int64）、文档边界下标（int64）。

读第 $i$ 篇文档：从 `.idx` 取 `pointer[i]`、`length[i]`，在 `.bin` 的 memmap 上切 `length[i]` 个元素，零拷贝。

**为什么用 mmap**：

1. 语料远大于内存（15T token × 4 字节 = 60 TB），mmap 只把实际访问的页读进来。
2. 同一节点上 8 个 rank、每个 rank 的多个 dataloader worker 打开同一个文件，OS page cache 里只有一份，不会各自复制。
3. 打开文件是 $O(1)$，不需要反序列化。

**token 用几个字节存**：由词表大小决定，id 最大是 $V - 1$。Megatron 的 `DType.optimal_dtype`：$V < 65500$ 用 uint16，否则用 int32。

1. Llama 2，$V = 32000 < 65500$：uint16，每 token 2 字节。
2. Llama 3，$V = 128256 > 65535$：uint16 装不下，int32，每 token 4 字节。

例：1T token 的语料，uint16 是 2 TB，int32 是 4 TB。换大词表的 tokenizer，存储和 page cache 压力直接翻倍。

### 3. 预训练 packing：拼接、切块、跨文档 mask

预训练不像 SFT 那样装箱（[SFT 的 packing](/posttrain/sft)），而是把整个源当成一条无限长的 token 流：

1. **拼接**：每篇文档末尾加 EOS（`preprocess_data.py --append-eod`），全部首尾相接。
2. **切块**：第 $i$ 条样本取 token 流的 $[iL,\ iL + L]$，共 $L + 1$ 个 token。前 $L$ 个是 input，后 $L$ 个是 label（错一位）。相邻样本共享边界上那一个 token，这样每个 token 都恰好被预测一次。
3. **没有 padding**：除了每个 epoch 末尾不足一条的尾巴，所有位置都是真实 token。

Megatron 不真的物化这条流，而是建三张表，存成 `.npy` 缓存（[gpt_dataset.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py)）：

1. `document_index`：每个 epoch 把文档顺序打乱一次，多个 epoch 拼起来。
2. `sample_index`：第 $i$ 条样本从哪篇文档的哪个偏移开始，形状 `[num_samples + 1, 2]`。
3. `shuffle_index`：再把样本顺序打乱一次。

三张表都只由（数据路径、$L$、样本数、随机种子）决定，所以任何时候重建都一样。

**跨文档 attention 屏不屏蔽**：一条样本里有多篇文档，默认的因果 mask 会让后一篇看到前一篇。标准做法是屏蔽：从 EOS 位置推出文档边界，用 varlen attention 的 `cu_seqlens` 只在文档内算（机制和 [SFT 的 cu_seqlens](/posttrain/sft) 完全一样；Megatron 的开关是 `--dataloader-inter-document-masking`）。Llama 3 报告这种 mask 在 8K 的标准预训练里影响有限，但在超长序列的继续预训练里很重要（作者自测）。长上下文阶段的细节见 [长上下文训练](/train/long-context)。

数字例子：$L = 8192$，文档平均 1000 token，一条样本约 8 篇文档。屏蔽后 attention 的 FLOPs 从正比于 $L^2$ 变成 $\sum l_j^2 \approx 8 \times 1000^2$，约为原来的 $8\times10^6 / 8192^2 \approx 12\%$。

### 4. 数据混合：按权重从多个源采样

设有 $K$ 个源，权重 $w_k$，总训练 token $D$。第 $k$ 个源要贡献 $w_k D$ 个 token，它要被过的遍数：

$$
e_k = \frac{w_k D}{S_k}
$$

例子，$D = 1$T：

| 源 | $S_k$ | $w_k$ | $w_k D$ | $e_k$ |
|---|---|---|---|---|
| 网页 | 3T | 0.75 | 750B | 0.25 |
| 代码 | 100B | 0.15 | 150B | 1.5 |
| 数学 | 20B | 0.06 | 60B | 3 |
| 百科 | 8B | 0.04 | 40B | 5 |

$e_k > 1$ 就意味着重复。Muennighoff et al. 发现重复最多 4 遍时 loss 和全新数据相比变化可以忽略，继续重复则新增算力的价值逐渐降到 0（[Scaling Data-Constrained LMs](https://arxiv.org/abs/2305.16264)，作者自测）。所以上表里百科的 5 遍值得警惕：要么降权重，要么接受。

**Megatron 的 BlendedDataset 怎么实现**（[blended_megatron_dataset_builder.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/blended_megatron_dataset_builder.py)、[helpers.cpp](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/helpers.cpp)）：

1. 总共要 $n$ 条样本。给第 $k$ 个源请求 $\lceil w_k n \rceil \times (1 + 0.005)$ 条，多要 0.5% 当余量。
2. 每个源各自是一个 GPTDataset。它发现请求的样本数超过一遍的 token 数时，就把 epoch 数加到够为止，每个 epoch 用不同的随机顺序。这就是 $e_k$。
3. 混合顺序用贪心：排第 $j$ 个位置时，对每个源算 $\text{err}_k = w_k \cdot \max(j, 1) - c_k$（$c_k$ 是已经取了几条），选 err 最大的源。结果是任意前缀里各源的比例都紧贴 $w_k$，不会出现「前一半全是网页」。

例：$w = (0.7, 0.3)$，前 10 个位置排出来是 `0 1 0 0 1 0 0 1 0 0`，7 比 3。

### 5. 分布式采样：谁读、谁拿哪份

记全局 micro-batch 有 $B_\mu \cdot d$ 条样本。Megatron 的 `MegatronPretrainingSampler`（[data_samplers.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/datasets/data_samplers.py)）：

1. 从 `consumed_samples` 开始，按样本号顺序取 $B_\mu \cdot d$ 条为一组（样本号已经被 `shuffle_index` 打乱过，所以顺序取就是随机的）。
2. DP rank $r$ 取组内第 $[r B_\mu,\ (r+1) B_\mu)$ 条。

例：$B_\mu = 2$，$d = 4$，consumed = 0。rank 1 第一步拿样本 2、3，第二步拿 10、11。

同一个 DP rank 里还有 TP、PP、CP 的多张卡，它们要同一份数据（[Megatron TP](/parallel/megatron-tp)、[并行总览](/parallel/parallelism-overview)）：

| 维度 | 谁读 | 怎么分发 |
|---|---|---|
| TP | 只有 TP rank 0 跑 dataloader | `torch.distributed.broadcast` 给 TP 组内其他卡（`get_batch_on_this_tp_rank`） |
| PP | 只有首末 stage 需要数据 | 第一个 stage 要 `tokens`、`position_ids`；最后一个 stage 要 `labels`、`loss_mask`；中间 stage 什么都不要 |
| CP | CP 组的卡属于同一个 DP rank，拿到同一条样本 | 各自沿序列维切自己那一段：切成 $2c$ 块（$c$ 为 CP 度），rank $i$ 取第 $i$ 块和第 $2c-1-i$ 块，因果 mask 下负载均衡 |

TP 用广播而不是每卡各读一份：省掉重复的 IO 和 CPU 开销；更重要的是保证 TP 组内数据**逐位相同**。TP 的每张卡算同一批 token 的不同部分，数据一旦不同，all-reduce 出来的就是错的。

### 6. 可确定性恢复：consumed_samples

checkpoint 里除了模型和优化器，还要存一个整数 `consumed_samples`（全局已消耗的样本数，Megatron 里是 `args.consumed_train_samples`）。重启时：

1. 用同样的种子重建（或直接加载缓存的）三张索引表，得到同一个全局样本序列。
2. sampler 从 `consumed_samples` 开始取。

这样恢复后第 $t$ 步看到的数据和不中断时完全一样。而且因为记的是**全局**样本号，换一个 DP 大小重启（比如掉了一个节点），只要全局 batch size 不变，每一步的全局 batch 内容不变，只是在 rank 间分法不同。

不是 Megatron 的训练脚本，用 torchdata 的 `StatefulDataLoader`：它是 `DataLoader` 的替代品，多了 `state_dict()` / `load_state_dict()`，会汇总各个 worker 的状态；sampler 或 dataset 自己实现了 `state_dict` 就用它们的，否则按已出的 batch 数快进（[torchdata 文档](https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html)）。它不跨 rank 协调，所以每个 rank 存自己的那份。

**为什么对 loss spike 很重要**：

1. **复现**：spike 出现在第 $t$ 步，加载 $t - 100$ 步的 checkpoint、喂完全相同的数据重跑，才能判断是数据（某个 batch 有问题）还是优化器状态的问题。数据不确定，就无法区分。
2. **跳过坏 batch**：PaLM 的做法是从 spike 前约 100 步的 checkpoint 重启，并跳过约 200–500 个 batch（[PaLM](https://arxiv.org/abs/2204.02311) Sec. 5.1，作者自测）。「跳过」只有在数据位置可精确定位时才有意义：把 `consumed_samples` 往前加 $500 \times$ global batch size 即可。

排查流程见 [训练稳定性](/train/stability)；checkpoint 里还要存哪些状态见 [容错](/train/fault-tolerance)。

### 7. DataLoader 不成为瓶颈

第 1 节算过字节数很小，所以数据成为瓶颈几乎总是 **CPU 侧每个 batch 的处理太慢**（在线 tokenize、在 Python 里构造 mask、多模态的图片解码），或者 H2D 拷贝没和计算重叠。四个旋钮：

1. `num_workers`：worker 进程数，并行准备 batch。Megatron 默认 2（`--num-workers`）。纯 token 读取 2 个够用；有图片解码等重活时加大。
2. `prefetch_factor`：每个 worker 提前准备几个 batch，PyTorch 默认 2，即总共预取 $2 \times$ `num_workers` 个。
3. `pin_memory=True`：batch 放在页锁定内存里，配合 `.cuda(non_blocking=True)`，H2D 拷贝走 DMA、和 GPU 计算重叠。
4. `persistent_workers=True`：epoch 之间不销毁 worker，免得每个 epoch 开头重新 fork、重建 memmap。

**怎么判断数据是瓶颈**：

1. **计时**：每步单独记 `next(data_iter)` 的耗时。例：step 时间 1.0 s，取数据 50 ms，就是 5% 的吞吐损失。预取正常时这个数应该接近 0。
2. **profile**：torch profiler 时间线上，每步开头 GPU 空闲一段，同一时刻 CPU 卡在 dataloader 的 `__next__` 里（看法见 [Profiling](/gpu/profiling)）。
3. **替换实验**：把数据换成固定的假 batch（同形状）跑一遍。吞吐上去了，就是数据的问题；没变，就不是。

## 面试追问

::: details Q：8B 模型在 H100 上训练，每卡每秒要多少 token？数据带宽够吗？
$\text{tokens/s} = u \cdot \text{peak} / 6N = 0.4 \times 989\times10^{12} / (6 \times 8.03\times10^9) \approx 8200$。int32 存储每 token 4 字节，每卡约 33 KB/s，任何存储都喂得动。数据管线的瓶颈在 CPU 侧处理和确定性，不在带宽。
:::

::: details Q：为什么 Llama 3 的 token 要用 4 字节存，Llama 2 用 2 字节就够？
token id 最大是 $V - 1$。Llama 2 $V = 32000$，uint16（最大 65535）装得下；Llama 3 $V = 128256$ 装不下，Megatron 的 `optimal_dtype` 在 $V \ge 65500$ 时选 int32。同样 token 数的语料，存储翻倍。
:::

::: details Q：预训练的样本为什么取 seq_len + 1 个 token？
input 是前 $L$ 个、label 是后 $L$ 个，错一位做 next-token 预测，所以需要 $L + 1$ 个。第 $i$ 条样本取 $[iL, iL + L]$，相邻样本共享一个边界 token，每个 token 恰好作为 label 出现一次。
:::

::: details Q：某个源权重 0.04，总训练 1T token，这个源只有 8B token，会怎样？
它要贡献 $0.04 \times 10^{12} = 40$B token，$e = 40 / 8 = 5$ 遍。Megatron 会给这个源建 5 个 epoch 的文档顺序（每个 epoch 重新打乱）。Muennighoff et al. 发现 4 遍以内重复几乎不损失（作者自测），5 遍已超出，建议降权重。
:::

::: details Q：TP 组里的卡为什么不各自读数据，而要 rank 0 读完广播？
TP 的每张卡对同一批 token 算不同的部分，再 all-reduce 合并，数据必须逐位相同。各自读要求 dataloader 在每张卡上严格确定（worker 调度、随机增强都一致），广播直接消除这个风险，还省掉重复的 IO 和 CPU。
:::

::: details Q：训练中途掉了一个节点，DP 从 64 降到 56 重启，数据会乱吗？
Megatron 记的是全局 `consumed_samples`，样本序列由种子决定、和 DP 大小无关。只要全局 batch size 不变，每一步的全局 batch 内容不变，只是分到 rank 的方式变了（$B_\mu$ 或梯度累积步数相应调整）。如果用的是按 rank 存状态的 dataloader，换 DP 大小就没法直接恢复。
:::

::: details Q：怎么确认训练慢是因为数据？
每步单独计时 `next(data_iter)`；profiler 时间线看每步开头 GPU 是否空闲、CPU 是否卡在 dataloader；最直接的是把数据换成同形状的固定 batch，吞吐上升就是数据的问题。修法是加 `num_workers`、开 `pin_memory` + `non_blocking`、把在线处理挪到离线。
:::

## 手撕

写一个最小的 packed dataset：从扁平的 token 文件 mmap、按 `seq_len + 1` 切片返回 `(input, label)`；再写按权重混合两个源的 dataset（Megatron 的贪心排法）和从 `consumed_samples` 恢复的 DP 切分。

```python
import os, tempfile
import numpy as np
import torch

class PackedTokenDataset:
    """一个源 = 一个扁平的 token 文件（文档已用 EOS 拼好）。第 i 条样本取 seq_len+1 个 token。"""
    def __init__(self, path, seq_len, dtype=np.int32):
        self.tok = np.memmap(path, dtype=dtype, mode="r")    # 不读进内存，按需 page in
        self.seq_len = seq_len
        self.n = (len(self.tok) - 1) // seq_len               # 末尾不足一条的丢掉

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        x = torch.from_numpy(self.tok[s : s + self.seq_len + 1].astype(np.int64))
        return x[:-1], x[1:]                                  # input, label 错一位


def blend_index(weights, n):
    """Megatron 的贪心：每一步选「落后目标最多」的源。返回 (源编号, 该源内第几条)。"""
    w = np.asarray(weights, dtype=np.float64) / sum(weights)
    count = np.zeros(len(w), dtype=np.int64)
    src, local = np.empty(n, np.int64), np.empty(n, np.int64)
    for k in range(n):
        err = w * max(k, 1) - count                           # 目标条数 - 已取条数
        d = int(np.argmax(err))
        src[k], local[k] = d, count[d]
        count[d] += 1
    return src, local


class BlendedDataset:
    def __init__(self, datasets, weights, n, seed=1234):
        self.ds = datasets
        self.src, self.local = blend_index(weights, n)
        rng = np.random.default_rng(seed)
        self.order = []                                      # 每个源：多个 epoch 的打乱顺序拼起来
        for d, ds in enumerate(datasets):
            need = int((self.src == d).sum())
            epochs = -(-need // len(ds))                      # 向上取整：这个源要过几遍
            self.order.append(np.concatenate([rng.permutation(len(ds)) for _ in range(epochs)]))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, k):
        d = self.src[k]
        return self.ds[d][self.order[d][self.local[k]]]


def rank_indices(consumed, mbs, dp_rank, dp_size, total):
    """从 consumed_samples 继续；每个全局 micro-batch 有 mbs*dp_size 条，rank 取自己那一段。"""
    for start in range(consumed, total - mbs * dp_size + 1, mbs * dp_size):
        yield list(range(start + dp_rank * mbs, start + (dp_rank + 1) * mbs))


tmp = tempfile.mkdtemp()
for name, base in [("web", 0), ("code", 1000)]:
    np.arange(base, base + 100, dtype=np.int32).tofile(os.path.join(tmp, name + ".bin"))
web = PackedTokenDataset(os.path.join(tmp, "web.bin"), seq_len=8)
code = PackedTokenDataset(os.path.join(tmp, "code.bin"), seq_len=8)
print(len(web), web[1][0].tolist(), web[1][1].tolist())
# 12 [8, 9, 10, 11, 12, 13, 14, 15] [9, 10, 11, 12, 13, 14, 15, 16]

src, _ = blend_index([0.7, 0.3], 10)
print(src.tolist())                    # [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
mix = BlendedDataset([web, code], [0.7, 0.3], n=40)
print([int((mix.src == d).sum()) for d in range(2)], [len(o) for o in mix.order])
# [28, 12] [36, 12]   web 只有 12 条却要 28 条 → 过 3 遍
print(list(rank_indices(consumed=8, mbs=2, dp_rank=1, dp_size=2, total=16)))
# [[10, 11], [14, 15]]   从第 8 条恢复，rank 1 拿每组的后半
```

常见变体：让 `__getitem__` 从 EOS 位置推出 `cu_seqlens`（文档内 attention）；问 worker 进程多了以后 memmap 会不会重复占内存（不会，共享 page cache）。

## 参考

- [Megatron-LM indexed_dataset.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py)：`.bin` / `.idx` 格式、`optimal_dtype`
- [Megatron-LM gpt_dataset.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py)、[helpers.cpp](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/helpers.cpp)：document / sample / shuffle index，混合的贪心排法
- [Megatron-LM data_samplers.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/datasets/data_samplers.py)：`consumed_samples` 与 DP 切分
- [torchdata StatefulDataLoader](https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：MFU、文档内 attention mask
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)：重复数据的 epoch 数
- [PaLM](https://arxiv.org/abs/2204.02311)：loss spike 时回滚并跳过 batch
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)：`num_workers`、`pin_memory`、`prefetch_factor`

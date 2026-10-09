---
title: SFT：Packing、Loss Mask、长上下文
status: draft
tags: [sft, packing]
difficulty: 2
order: 1
related: [/posttrain/training-memory, /posttrain/lora-qlora, /posttrain/rlhf-ppo-dpo-grpo, /leetgpu/variable-length-causal-attention, /leetgpu/mask, /parallel/parallelism-overview]
stack: []
---

# SFT：Packing、Loss Mask、长上下文

> 数据怎么变成 `input_ids / labels / position_ids`，以及为什么要这样变

## 一句话结论

SFT（supervised fine-tuning）就是在「对话 → 期望回答」数据上做下一个 token 预测，loss 和预训练一样是交叉熵。工程上三件事：(1) 用和推理**完全相同的 chat template** 把对话拼成 token；(2) **loss mask**：只在 assistant 回答（含结束符）上算 loss，其他位置 label 设成 −100；(3) **packing**：把多条短样本首尾相接拼成定长序列，消灭 padding（常见数据里 padding 可占一半 token，[Krell et al. 2021](https://arxiv.org/abs/2107.02027) 作者自测），但必须配**块对角因果 mask**（或 varlen attention 的 `cu_seqlens`）和**按样本重置的 position_ids**，否则样本之间互相 attend。算力按每 token $6\Psi$ FLOPs 估，7B 全参在 8 张 H100 上（MFU 40%）每秒约 5.7–7.5 万 token；显存账和预训练完全一样，见 [训练显存账](/posttrain/training-memory)。

## 推导

### 0. SFT 在优化什么

记一条样本为 prompt $x$ 和回答 $y = (y_1, \dots, y_T)$，模型参数 $\theta$。SFT 最大化回答的条件似然：

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_n T_n}\sum_{n}\sum_{t=1}^{T_n} \log p_\theta\big(y^{(n)}_t \,\big|\, x^{(n)}, y^{(n)}_{<t}\big)
$$

和预训练的唯一区别是**求和只覆盖回答 token**，prompt 只当条件，不当目标。分母是这个 batch 里所有被计入 loss 的 token 数（后面会看到分母怎么取是个坑）。

它在后训练流水线里的位置：预训练 → SFT（学会格式、跟随指令）→ 偏好优化或 RL（见 [RLHF](/posttrain/rlhf-ppo-dpo-grpo)）。数据量通常不大：InstructGPT 的 SFT 集约 1.3 万条 prompt（[Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)，作者自测），LIMA 只用 1000 条精选样本（[Zhou et al. 2023](https://arxiv.org/abs/2305.11206)，作者自测，并据此提出「对齐主要是学格式」的 Superficial Alignment Hypothesis）。

### 1. 数据格式与 chat template

原始数据一般是 OpenAI 风格的消息列表：

```python
messages = [
    {"role": "system",    "content": "You are helpful."},
    {"role": "user",      "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
    {"role": "user",      "content": "再加 1？"},
    {"role": "assistant", "content": "3"},
]
```

chat template 把它渲染成一条字符串，用特殊 token 标出角色边界。以 ChatML（Qwen 系列用的格式）为例：

```text
<|im_start|>system\nYou are helpful.<|im_end|>\n
<|im_start|>user\n1+1=?<|im_end|>\n
<|im_start|>assistant\n2<|im_end|>\n
<|im_start|>user\n再加 1？<|im_end|>\n
<|im_start|>assistant\n3<|im_end|>\n
```

HF 里是 `tokenizer.apply_chat_template(messages, tokenize=False)`；推理时加 `add_generation_prompt=True`，末尾补上 `<|im_start|>assistant\n` 让模型接着写（[Transformers: Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)）。

**训练和推理必须用同一个模板。** 模型学到的是「看到 `<|im_start|>assistant\n` 就开始回答，写完输出 `<|im_end|>`」。训练时少一个换行、换一个角色名，线上 prompt 就落在训练分布外。

### 2. Loss mask：只在回答上算 loss

PyTorch 的 `F.cross_entropy(..., ignore_index=-100)` 会跳过 label 为 −100 的位置，HF 模型的 loss 默认就是这样。做法：`labels = input_ids.clone()`，把**不是 assistant 回答**的位置设成 −100。

约定：HF 的 causal LM 在模型内部做移位，用 `logits[:, :-1]` 预测 `labels[:, 1:]`，所以 `labels` 和 `input_ids` 对齐、不用自己错位。

上面的例子（为了看清楚，一个片段记一格）：

| 片段 | `<im_start>system\n…<im_end>\n` | `<im_start>user\n1+1=?<im_end>\n` | `<im_start>assistant\n` | `2` | `<im_end>` | `\n<im_start>user…` | `<im_start>assistant\n` | `3` | `<im_end>` |
|---|---|---|---|---|---|---|---|---|---|
| 算 loss？ | 否 | 否 | 否 | **是** | **是** | 否 | 否 | **是** | **是** |

要点：
1. **结束符 `<|im_end|>`（或 EOS）必须算 loss**。否则模型学不会停，推理时一直生成到 `max_tokens`。
2. **多轮对话一次 forward 算完所有 assistant 轮**，不要拆成「前 1 轮」「前 2 轮」两条样本：拆开后公共前缀被重复计算，因果 mask 本来就保证第一轮回答看不到后面内容，两种做法的梯度相同。
3. assistant 开头的 `<|im_start|>assistant\n` 是推理时由模板给出的，不需要模型生成，不算 loss。

怎么找到 assistant 的 token 区间？最稳的方法是用模板渲染前缀、按长度切，而不是分别 tokenize 再拼（BPE 在拼接边界上可能合并出不同的 token）：

```python
def build_example(tok, messages):
    ids = tok.apply_chat_template(messages, tokenize=True)
    labels = [-100] * len(ids)
    for n, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # 回答开始位置 = 「前 n 条 + assistant 头」的长度
        start = len(tok.apply_chat_template(messages[:n], tokenize=True,
                                            add_generation_prompt=True))
        # 回答结束位置 = 「前 n+1 条」的长度（含 <|im_end|>，可能再带一个换行）
        end = len(tok.apply_chat_template(messages[:n + 1], tokenize=True))
        labels[start:end] = ids[start:end]
    return ids, labels
```

这依赖模板「前缀稳定」：渲染前 $n$ 条的结果是渲染全部的前缀。有的模板会改写历史轮（例如删掉之前轮的思考内容），这时要用模板自带的 assistant mask（HF 的 `return_assistant_tokens_mask=True`，需要模板里写了 `{% generation %}` 标记）。TRL 的 `SFTTrainer` 有对应的开关（`assistant_only_loss`、`completion_only_loss`），以所装版本的 [文档](https://huggingface.co/docs/trl/sft_trainer) 为准。

**一定要 mask prompt 吗？** 默认是。但 [Shi et al. 2024](https://arxiv.org/abs/2405.14394) 发现在「指令长、回答短」或「样本很少」时，对指令也算 loss 反而更好，原因是减轻过拟合（作者自测）。这是可调的超参，不是铁律。

### 3. Loss 怎么平均：一个真实的坑

$\mathcal{L}$ 的分母应该是**整个优化步**里所有计入 loss 的 token 数。gradient accumulation 下常见错误是每个 micro-batch 先各自求平均，再对 micro-batch 求平均：

- micro-batch 1：10 个有效 token，平均 loss 1.0
- micro-batch 2：1000 个有效 token，平均 loss 2.0
- 正确：$(10 \times 1.0 + 1000 \times 2.0) / 1010 = 1.99$
- 错误：$(1.0 + 2.0) / 2 = 1.5$，短样本的 token 被放大了 50 倍权重

SFT 的样本长度差异大、loss mask 又让有效 token 数更不均，所以影响比预训练明显。HF 在 2024 年修过 Trainer 里的这个问题（[Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)）。正确写法：先统计这一步所有 micro-batch 的有效 token 总数 $N$，每个 micro-batch 用 `loss_sum / N` 反传。数据并行时 $N$ 还要跨卡 all-reduce。

另一个选择是「每条样本先平均、再对样本平均」，让每条样本权重相同、不偏向长回答。两种都可以，但要**有意识地选**，不能让 micro-batch 的切法决定权重。

### 4. Packing：为什么要做

不 packing 时，一个 batch 的样本 pad 到同一长度。设 max_len = 4096，样本平均 800 token：

- 每条样本计算 4096 个位置，有效 800 个，利用率 $800 / 4096 = 20\%$
- padding 位置的线性层 FLOPs 一点不少（attention 的 mask 只是把分数置 −∞，矩阵乘照算）

按长度分桶（每个 batch 里长度相近）能缓解，packing 则直接消灭 padding：按顺序把样本塞进长度 4096 的「箱子」里，只在最后一条放不下时留一点空。这是装箱问题，常用贪心：

```python
def pack(lengths, cap):
    """First-fit decreasing：长样本先放，每条放进第一个装得下的箱子"""
    bins = []                                   # 每个箱子：[剩余容量, [样本下标]]
    for idx in sorted(range(len(lengths)), key=lambda n: -lengths[n]):
        L = min(lengths[idx], cap)              # 超长样本截断（或单独处理）
        for b in bins:
            if b[0] >= L:
                b[0] -= L; b[1].append(idx); break
        else:
            bins.append([cap - L, [idx]])
    return [b[1] for b in bins]
```

利用率通常能到 95% 以上。Krell et al. 在 BERT phase-2 预训练上用 packing 得到 2 倍加速（作者自测）；HF 的 [packing + FlashAttention 2 博客](https://huggingface.co/blog/packing-with-FA2) 报告 SFT 吞吐最多 2 倍（作者自测）。

### 5. Packing 的三个配套：mask、position_ids、cu_seqlens

把 3 条样本 `a a a | b b | c c c` 拼成长度 8 的序列。用 `seg[t]` 表示第 $t$ 个 token 属于第几条样本。

**(1) attention mask：同一样本内的因果 mask。** query $i$ 只能看 key $j$，当且仅当两者属于同一条样本且 $j \le i$：

```python
seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])     # 每个 token 属于第几条样本
T = seg.numel()
i = torch.arange(T)[:, None]                      # (T, 1) query 下标
j = torch.arange(T)[None, :]                      # (1, T) key 下标
allowed = (seg[:, None] == seg[None, :]) & (j <= i)   # 同一样本 且 因果
scores = scores.masked_fill(~allowed, float("-inf"))
```

得到沿对角线排开的三个小三角，也就是块对角因果 mask。mask 的通用写法见 [Attention mask 怎么拼](/leetgpu/mask)。

不改 mask 会怎样：第二条样本的 token 能 attend 到第一条样本，训练时看到了推理时不存在的上下文。[Zhao et al. 2024](https://arxiv.org/abs/2402.13991) 在预训练上发现文档内因果 mask 能提升效果（作者自测）；Llama 3 报告这类 mask 在标准预训练中影响有限、但在超长序列的继续预训练中很重要（[Llama 3 Herd](https://arxiv.org/abs/2407.21783)，作者自测）。SFT 样本彼此无关，默认应该隔离。

**(2) position_ids：每条样本从 0 重新开始。**

```python
starts = torch.cat([torch.tensor([True]), seg[1:] != seg[:-1]])  # 每段第一个 token
first = torch.cummax(torch.where(starts, torch.arange(T), 0), 0).values
position_ids = torch.arange(T) - first           # tensor([0,1,2,0,1,0,1,2])
```

RoPE 只看相对位置，单从 attention 分数看似乎无所谓；但模型在预训练里见过的是「位置 0 是开头」，不重置会让第三条样本以为自己从位置 5 开始，和推理时不一致。重置后还有一个好处：HF 的 `DataCollatorWithFlattening` 就是靠 position_ids 里的「回到 0」推断样本边界。

**(3) cu_seqlens：给 varlen kernel 的边界。** 上面的 $T \times T$ 稠密 mask 只适合讲原理，长度 8k 时 mask 本身就有 $8192^2$ 个元素。实际用 FlashAttention 的 `flash_attn_varlen_func`，传累计长度：

```python
lengths = torch.tensor([3, 2, 3])
cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).int()   # [0, 3, 5, 8]
# out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
#                              max_seqlen_q=3, max_seqlen_k=3, causal=True)
```

kernel 按段算，跨段的块**直接不算**。这比「稠密 mask 再置 −∞」更省：attention FLOPs 从正比于 $(\sum_n l_n)^2$ 变成 $\sum_n l_n^2$。4 条 1024 的样本拼成 4096：$4 \times 1024^2$ 对比 $4096^2$，少 4 倍。逐行写法见 [Variable-Length Causal Attention](/leetgpu/variable-length-causal-attention)。

**(4) labels 的边界。** 每条样本最后一个 token 的「下一个 token」是下一条样本的第一个 token，这对预测没意义。因为样本开头总是 system/user 片段，label 本来就是 −100，所以 loss mask 已经顺带处理了；如果样本开头是要算 loss 的，要显式把每段第一个位置的 label 设 −100。

### 6. 吞吐与时间估算

训练每 token 约 $6\Psi$ FLOPs（推导见 [训练显存账](/posttrain/training-memory)），开 full gradient checkpointing 是 $8\Psi$。设 MFU（实际 FLOP/s 除以峰值）为 $u$，单卡吞吐：

$$
\text{tokens/s/GPU} = \frac{u \cdot \text{peak}}{6\Psi}
$$

7B 全参，H100 bf16 dense peak 989 TFLOP/s，$u = 0.4$：$0.4 \times 989\times10^{12} / (6 \times 7\times10^9) \approx 9400$ token/s；开 checkpointing 约 7100。8 卡约 5.7–7.5 万 token/s。这里忽略了 attention 的 FLOPs：每层每 token attention 约 $4 \cdot s_{\text{ctx}} \cdot h$ FLOPs（$QK^\top$ 和 $PV$ 各 $2 s_{\text{ctx}} h$），线性层约 $24h^2$（$2 \times (4h^2 + 3hi)$，$i \approx 2.69h$），比值 $s_{\text{ctx}} / (6h)$。$h = 4096$、varlen 下平均上下文 $s_{\text{ctx}} \approx 400$ 时只有 1.6%，可以忽略；seq 32k 的长样本就不能忽略了。

数据集 5 万条、平均 800 token、训 3 个 epoch：$5\times10^4 \times 800 \times 3 = 1.2\times10^8$ token。8 卡、开 checkpointing：$1.2\times10^8 / 5.7\times10^4 \approx 2100$ s，约 35 分钟。不 packing、全部 pad 到 4096：实际计算量是 $4096 / 800 = 5.1$ 倍，约 3 小时。

LoRA 时把 $6\Psi$ 换成约 $4\Psi$（不算底座的权重梯度），见 [LoRA / QLoRA](/posttrain/lora-qlora)。

### 7. 显存

SFT 的显存账和预训练完全一样，只看 $\Psi$、$s$、$b$：7B 全参模型状态 112 GB，需要 ZeRO-3 / FSDP 切到多卡（见 [ZeRO / FSDP](/parallel/zero-fsdp)）；LoRA 底座 14 GB + adapter 约 0.6 GB，单卡可行。激活：seq 4096、micro-batch 1、FlashAttention 下约 18 GB，full checkpointing 约 1 GB。

packing 的一个副作用是好的：每个 micro-batch 的 token 数固定，激活大小稳定，PyTorch allocator 的碎片也少；不 packing 时 batch 形状每步变化，容易出现「总空闲够、但没有连续大块」的 OOM。

### 8. 长上下文 SFT

激活随 $s$ 线性涨（FlashAttention 已去掉 $s^2$ 项）：7B、seq 32k、micro-batch 1 时 $34sbhL = 34 \times 32768 \times 4096 \times 32 \approx 146$ GB，开 full checkpointing 也还有 $2sbhL \approx 8.6$ GB 加上一层的 4.6 GB 峰值。超过单卡就要沿序列维切到多卡：

- **DeepSpeed Ulysses**（[Jacobs et al. 2023](https://arxiv.org/abs/2309.14509)）：每卡持有一段序列；进 attention 前做 all-to-all，把「按序列切」换成「按 head 切」，每卡对完整序列算一部分 head，算完再 all-to-all 换回来。并行度受 head 数限制。
- **Ring Attention**（[Liu et al. 2023](https://arxiv.org/abs/2310.01889)）：每卡持有一段 Q 和一段 KV，KV 块沿环传递，每卡依次和所有 KV 块算 attention 并用 online softmax 合并，通信和计算重叠。

和 packing 组合时注意：varlen 的段边界要在切分后仍然正确，因果 mask 下各卡负载不均（后面的段要算的 KV 更多），需要重排。这些在 [并行总览](/parallel/parallelism-overview) 里展开。

### 9. 常见坑清单

1. **模板不一致**：训练用 A 模板、推理用 B 模板，或训练时没加 system prompt 而线上有。
2. **EOS 没学**：label 里把 `<|im_end|>` 设成了 −100；或者 `pad_token = eos_token`，再「把所有 pad 位置设成 −100」时顺手把真正的 EOS 也 mask 掉。结果是模型不停。
3. **截断截掉了回答**：按 max_len 从右截断时，长 prompt 会把回答整个截掉，这条样本 loss 为 0（有效 token 数 0 时还可能出 NaN）。应该丢弃或从左截 prompt。
4. **分别 tokenize 再拼接**：`tok(prompt) + tok(answer)` 和 `tok(prompt + answer)` 在边界上可能不一样，训练看到的 token 序列线上不会出现。
5. **loss 分母**：见第 3 节。
6. **packing 没改 mask / position_ids**：用了 packing 但 attention 实现走的是普通 causal kernel。
7. **过拟合**：SFT 数据少，3 个 epoch 以上常见 loss 继续降、评测变差。质量和去重比数量重要（LIMA 的结论，作者自测）。
8. **学习率**：全参 7B 常用 1e-5 量级、LoRA 常用 1e-4 量级，配 warmup + cosine；从预训练的 lr 直接沿用会把模型训崩。

## 面试追问

::: details Q：packing 之后如果不改 attention mask 会怎样？
拼在一起的样本会互相 attend：后面样本的 token 看到了前面无关样本的内容，训练时的条件分布和推理时（每条样本单独）不一致。影响大小看数据和模型规模：Llama 3 报告标准预训练中影响有限、长序列继续预训练中重要（作者自测）。正确做法是 varlen attention（传 `cu_seqlens`）或块对角因果 mask，同时把 position_ids 按样本重置；varlen 还能把 attention FLOPs 从 $(\sum l)^2$ 降到 $\sum l^2$。
:::

::: details Q：为什么 EOS 必须算 loss？
模型怎么停是学出来的：只有在回答末尾把「下一个 token 是 EOS」作为监督信号，模型才会在合适的时候输出它。label 里把 EOS mask 掉，模型永远不会主动结束，推理时一直生成到长度上限。最常见的原因是 `pad_token` 和 `eos_token` 是同一个 id，按 id 把 pad 设成 −100 时连带 mask 了 EOS，应该按 attention_mask 或样本长度来 mask。
:::

::: details Q：多轮对话为什么一次 forward 算所有轮，而不是拆成多条？
因果 mask 保证第 $k$ 轮的回答只看得到它之前的内容，和单独把前 $k$ 轮拿出来做一条样本的 logits 完全一样，所以梯度相同。拆开会把公共前缀重复算很多遍：$K$ 轮对话的计算量从 $O(\text{总长})$ 变成 $O(K \cdot \text{总长})$。前提是模板前缀稳定（历史轮渲染方式不随后续轮变化）。
:::

::: details Q：gradient accumulation 下 loss 应该怎么归一化？
除以这个优化步里所有 micro-batch（以及所有数据并行 rank）的**有效 token 总数**，而不是每个 micro-batch 各自平均再平均。否则有效 token 少的 micro-batch 权重被放大：10 个 token 的 micro-batch 和 1000 个 token 的 micro-batch 被平均对待，真实 loss 1.99 被算成 1.5。实现上先统计总 token 数 $N$（跨 rank all-reduce），每个 micro-batch 反传 `loss_sum / N`。
:::

::: details Q：7B SFT，5 万条平均 800 token 的数据，8 张 H100 大概多久一个 epoch？
总 token $4\times10^7$。每 token $6\Psi = 4.2\times10^{10}$ FLOPs，开 checkpointing 是 $8\Psi$。MFU 40% 时单卡 $0.4 \times 989\text{T} / 5.6\times10^{10} \approx 7100$ token/s，8 卡约 5.7 万 token/s，一个 epoch 约 700 s，12 分钟左右。不 packing、pad 到 4096 的话乘以 5 倍。
:::

::: details Q：packing 会不会改变 loss 的权重？
会，取决于怎么平均。按 token 平均时，packing 不改变每个 token 的权重，长回答的样本贡献更多 token；按「每条样本先平均」时，需要用 `seg` 把每段的 loss 分开平均，不能简单对整条 packed 序列平均，否则一条 packed 序列里的几条样本被当成一条。
:::

## 手撕

给定几条已经 tokenize 好的样本（每条是 `ids` 和对应的 `labels`，非回答位置已是 −100），输出 packing 后的四个张量：

```python
import torch

def collate_packed(samples, cap):
    """samples: list of (ids: list[int], labels: list[int])，假设总长 ≤ cap"""
    input_ids, labels, position_ids, seg, lengths = [], [], [], [], []
    for n, (ids, lab) in enumerate(samples):
        L = len(ids)
        input_ids += ids
        lab = list(lab); lab[0] = -100          # 段首 token 不作为上一段的预测目标
        labels += lab
        position_ids += list(range(L))           # 每条样本从 0 开始
        seg += [n] * L
        lengths.append(L)
    pad = cap - len(input_ids)                   # 箱子剩余空间（first-fit 后通常很小）
    input_ids += [0] * pad; labels += [-100] * pad
    position_ids += list(range(pad)); seg += [len(samples)] * pad
    if pad:
        lengths.append(pad)
    cu_seqlens = torch.tensor([0] + lengths).cumsum(0).int()
    return (torch.tensor(input_ids), torch.tensor(labels),
            torch.tensor(position_ids), cu_seqlens, torch.tensor(seg))

ids, labels, pos, cu, seg = collate_packed(
    [([1, 11, 12], [-100, -100, 12]),            # 样本 a：prompt 1,11；回答 12
     ([1, 21],     [-100, 21]),                  # 样本 b
     ([1, 31, 32], [-100, 31, 32])], cap=8)      # 样本 c
# ids    = [1, 11, 12, 1, 21, 1, 31, 32]
# labels = [-100, -100, 12, -100, 21, -100, 31, 32]
# pos    = [0, 1, 2, 0, 1, 0, 1, 2]
# cu     = [0, 3, 5, 8]
T = ids.numel()
i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
allowed = (seg[:, None] == seg[None, :]) & (j <= i)  # 与 cu_seqlens 等价的稠密 mask
```

常见变体：多轮对话里只有 assistant 轮算 loss 时 labels 长什么样；问 `cu_seqlens` 和 `position_ids` 能否互相推出（能：position_ids 回到 0 的位置就是段首）。

## 参考

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394)
- [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
- [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：文档内 attention mask
- [HF blog: Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2)、[HF blog: Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)
- [HF Transformers: Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)、[HF TRL: SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)、[Ring Attention](https://arxiv.org/abs/2310.01889)

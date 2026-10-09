---
title: 给 vLLM / SGLang 加一个新模型
status: draft
tags: [vllm, sglang]
difficulty: 4
order: 3
related: [/parallel/megatron-tp, /framework/vllm-v1-architecture, /framework/torch-compile, /inference/attention-variants, /inference/kv-cache-paged-attention]
stack: [ld-load, f-model, f-runner, k-attn, sv-tokenize]
---

# 给 vLLM / SGLang 加一个新模型

> config 映射、weight loading、model runner、attention backend、tokenizer / chat template

## 一句话结论

一个 HF 模型 = `config.json`（结构超参）+ `*.safetensors`（参数名 → tensor）+ tokenizer 文件。给 vLLM / SGLang 加模型，就是把它翻译成框架自己的三件东西：

1. **模型定义**：用框架的张量并行 Linear 和 Attention 层重写 forward。输入是拍平的 `[num_tokens]`，attention 调框架的 paged KV 接口，不自己写；
2. **权重加载**：把 HF 的参数名映射到框架的参数上，合并 QKV / gate-up，按 TP rank 切片；
3. **注册**：让框架根据 `config.json` 里的 `architectures` 字段找到这个类。

动手前先试一下 vLLM 的 Transformers 后端（`--model-impl transformers`），能跑就不一定要重写（[vLLM 文档：Supported Models › Transformers](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)）。

## 推导

### 0. 起点：框架怎么找到你的类

`config.json` 里有一行 `"architectures": ["LlamaForCausalLM"]`。框架拿这个字符串查注册表：

- **vLLM**：[`registry.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) 里的字典，一行 `"LlamaForCausalLM": ("llama", "LlamaForCausalLM")` 表示「去 `models/llama.py` 里找 `LlamaForCausalLM` 类」，最后汇总进 `_VLLM_MODELS`。不改源码时用插件注册：`ModelRegistry.register_model("YourModelForCausalLM", "your_code:YourModelForCausalLM")`（[vLLM 文档：Registering a Model](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/registration.md)）；
- **SGLang**：在 `python/sglang/srt/models/` 下加一个文件，文件末尾写 `EntryClass = YourModelForCausalLM`，注册器导入文件时读这个变量。不改源码时设环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE`（[SGLang 文档：How to Support New Models](https://docs.sglang.io/docs/supported-models/support_new_models)）。

两家文档都建议找一个最像的已支持模型复制改写，下面以 Llama 为例。

### 1. config 映射：每个字段决定什么

以 Llama-3-8B 为例，括号里是它的值：

| HF 字段 | 决定什么 | 推出的 shape |
|---|---|---|
| `hidden_size`（4096） | 残差流宽度 $d$ | embedding `[V, d]` |
| `num_attention_heads`（32） | Q head 数 $H$ | $d_h = d / H = 128$（有 `head_dim` 字段时以它为准） |
| `num_key_value_heads`（8） | KV head 数 $H_{kv}$，GQA 组大小 $H / H_{kv} = 4$ | QKV 合并后输出维 $(H + 2H_{kv}) d_h = 48 \times 128 = 6144$ |
| `intermediate_size`（14336） | FFN 宽度 $f$ | gate-up 合并后输出维 $2f = 28672$ |
| `num_hidden_layers`（32） | 层数 $L$ | KV cache 每 token $2 L H_{kv} d_h \cdot 2\text{B} = 128$ KiB |
| `vocab_size`（128256） | 词表 $V$ | lm_head `[V, d]` |
| `rope_theta` / `rope_scaling`（新版 transformers 合成 `rope_parameters`） | RoPE 的频率和长上下文缩放方式 | 传给 `get_rope` |
| `rms_norm_eps`（1e-5） | RMSNorm 的 ε | |
| `tie_word_embeddings`（false） | lm_head 是否和 embedding 共享权重 | true 时 checkpoint 里没有 `lm_head.weight` |
| `sliding_window`、`layer_types` | 哪些层是滑窗 attention | 逐层传给 Attention |

新架构最先要确认的是 **attention 能不能用现成的 backend**：MLA、滑窗、attention sink、线性 attention 这类变体，backend 不支持就不只是写模型文件的事了（各种变体见 [Attention 变体](/inference/attention-variants)）。

### 2. 模型定义：用框架的层重写

vLLM 的 Llama attention（[`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)，删减后）：

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, hidden_size, num_heads, num_kv_heads, cache_config, quant_config, prefix):
        tp = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp                    # 本卡的 Q head 数
        self.num_kv_heads = max(1, num_kv_heads // tp)      # KV head 不够分时每卡至少 1 个（复制）
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads,
                                          quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size,
                                        quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(self.head_dim, max_position=..., rope_parameters=...)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5,
                              num_kv_heads=self.num_kv_heads, cache_config=cache_config,
                              quant_config=quant_config, prefix=f"{prefix}.attn")

    def forward(self, positions, hidden_states):          # hidden_states: [num_tokens, d]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)                           # 写 KV cache + 算 attention，全在里面
        return self.o_proj(out)[0]
```

和 HF 版本比，四处不同：

1. **没有 batch 维**。输入是所有请求的 token 拼成的 `[num_tokens, d]`，`positions` 给每个 token 在自己序列里的位置。原因见下方面试追问；
2. **没有 mask、没有 past_key_values 参数**。`self.attn` 从 forward context 里取本步的 attention metadata（每个序列的起止、block table、slot mapping），自己决定写 KV 到哪、读哪些 KV。模型代码完全看不到 KV cache；
3. **每层都有 `prefix`**。它是这一层在 state dict 里的完整名字（如 `model.layers.3.self_attn.attn`）。Attention 层靠它注册到全局表、找到自己那一层的 KV cache；量化配置也靠它判断这一层要不要量化（[vLLM 文档：Basic Model](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/basic.md)）；
4. **Linear 换成并行版本**，切法见 [Megatron TP](/parallel/megatron-tp)：

| 层 | 用哪个类 | 切哪一维 | 通信 |
|---|---|---|---|
| q/k/v | `QKVParallelLinear`（三个合成一个） | 输出维，按 head 切 | 无 |
| o_proj | `RowParallelLinear` | 输入维 | 一次 all-reduce |
| gate/up | `MergedColumnParallelLinear`（两个合成一个） | 输出维 | 无 |
| down | `RowParallelLinear` | 输入维 | 一次 all-reduce |
| embedding / lm_head | `VocabParallelEmbedding` / `ParallelLMHead` | 词表维 | gather |

**为什么 QKV 要合成一个 Linear**：三个小 GEMM 变一个大 GEMM，少 2 次 kernel launch，也少读 2 次输入 `hidden_states`。

**GQA 在 TP 下怎么切**：$H_{kv}$ 能被 TP 整除时各卡分 $H_{kv}/\text{TP}$ 个 KV head；$H_{kv} < \text{TP}$ 时每个 KV head 复制到 $\text{TP}/H_{kv}$ 张卡上。Llama-3-8B（$H = 32$，$H_{kv} = 8$）：

- TP = 4：每卡 8 个 Q head、2 个 KV head，本卡 QKV 输出维 $(8 + 2 \times 2) \times 128 = 1536$；
- TP = 16：每卡 2 个 Q head、1 个 KV head，每个 KV head 在 2 张卡上各存一份，KV cache 总量也跟着翻倍。

最后，模型类上加 `@support_torch_compile`，标出 `num_tokens` 那一维是动态的，就能享受 vLLM 的编译和 CUDA Graph（见 [torch.compile](/framework/torch-compile#_6-vllm-怎么用-torch-compile)）。

### 3. 权重加载：最容易出错的一步

HF checkpoint 里 q、k、v 是三个独立的 tensor，框架里是一个合并的 `qkv_proj.weight`。加载时要做两件事：**改名 + 定位**（放进合并参数的哪一段），**切片**（只取本 TP rank 的那一份）。

**改名表**。vLLM 的 Llama 把它写成 `WeightsMapper`，交给通用的 `AutoWeightsLoader` 执行（[`models/utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py)）：

```python
hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={
    # HF 名字里的片段: (框架参数名里的片段, shard_id)
    ".q_proj":    (".qkv_proj", "q"),
    ".k_proj":    (".qkv_proj", "k"),
    ".v_proj":    (".qkv_proj", "v"),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj":   (".gate_up_proj", 1),
})
```

很多模型（以及 SGLang 的 [`llama.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)）还是手写的老写法，逻辑一样：

```python
def load_weights(self, weights):                  # weights: 逐个产出 (HF 参数名, tensor)
    stacked = [(".qkv_proj", ".q_proj", "q"), (".qkv_proj", ".k_proj", "k"), (".qkv_proj", ".v_proj", "v"),
               (".gate_up_proj", ".gate_proj", 0), (".gate_up_proj", ".up_proj", 1)]
    params = dict(self.named_parameters())
    loaded = set()
    for name, w in weights:
        if "rotary_emb.inv_freq" in name:         # RoPE 频率是现算的，checkpoint 里有也跳过
            continue
        if self.config.tie_word_embeddings and "lm_head.weight" in name:
            continue                              # 共享 embedding，不单独加载
        for param_name, hf_name, shard_id in stacked:
            if hf_name in name:
                name = name.replace(hf_name, param_name)
                params[name].weight_loader(params[name], w, shard_id)
                break
        else:
            p = params[name]
            getattr(p, "weight_loader", default_weight_loader)(p, w)
        loaded.add(name)
    return loaded
```

**切片在 `weight_loader` 里**。每个并行 Linear 的参数上挂着自己的 `weight_loader`，知道本卡该取哪一段。`QKVParallelLinear.weight_loader`（[`layers/linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)）对 `shard_id="k"` 的处理：

```python
# 目标：本卡 qkv 参数里 k 那一段
offset = num_heads_per_rank * head_dim            # 跳过本卡的 q 段
size   = num_kv_heads_per_rank * head_dim
dst = param.narrow(0, offset, size)
# 来源：HF 的完整 k_proj.weight [H_kv * d_h, d]，取本卡负责的那几个 KV head
shard_rank = tp_rank // num_kv_head_replicas      # KV head 被复制时，相邻几张卡读同一段
src = w.narrow(0, shard_rank * size, size)
dst.copy_(src)
```

`RowParallelLinear` 则沿输入维切：`w.narrow(1, tp_rank * shard, shard)`。

**常见错误**：

1. 名字对不上：HF 是 `model.layers.0.self_attn.q_proj.weight`，有的 checkpoint 多一层前缀或叫 `attention.wq`。vLLM 加载完会比对「模型的参数」和「实际加载的参数」，有漏的直接报错 `Following weights were not initialized from checkpoint`；
2. 合并顺序错：q、k、v 在合并参数里的顺序必须和 forward 里 `split` 的顺序一致；
3. 有的 checkpoint 本身就是合并好的 qkv（如 Phi-3），这时 `shard_id=None`，loader 自己按 q/k/v 的大小拆；
4. TP 切分维度错：column 切输出维（dim 0），row 切输入维（dim 1）。PyTorch 的 Linear 权重是 `[out, in]`。

### 4. attention backend：不要自己写

模型里只写 `Attention(...)`，具体用哪个 kernel（FlashAttention、FlashInfer、Triton、MLA 专用……）由框架按硬件和配置选。好处是 paged KV、chunked prefill、CUDA Graph、FP8 KV 都自动生效。

需要动 backend 的情况：

- **逐层滑窗**：`config.json` 里要有 `layer_types`，模型代码逐层把窗口大小传给 Attention 的 `per_layer_sliding_window` 参数（Llama 的实现就这么做，见上面的文档 FAQ）；
- **全新的 attention 机制**（线性 attention、Mamba 混合）：要实现新的 layer 和 metadata，并把它注册成 custom op、加进 `splitting_ops`，否则 torch.compile 和 PIECEWISE CUDA Graph 会出问题（同上文档 FAQ；custom op 的原理见 [PyTorch 内部机制](/framework/pytorch-internals#_3-custom-op-往表里注册新条目)）。

### 5. tokenizer 与 chat template

大多数情况直接复用 HF tokenizer，要检查的是：

1. **chat template**：在 `tokenizer_config.json` 的 `chat_template` 字段里（Jinja2），决定 messages 怎么拼成 prompt。模板错了模型照样能跑，只是输出质量莫名变差，最难查。可以用 `--chat-template` 覆盖；
2. **结束符**：`generation_config.json` 里的 `eos_token_id` 可能是一个列表（如 Llama-3 同时有 `<|eot_id|>` 和 `<|end_of_text|>`），漏一个就停不下来；
3. **特殊 token**：工具调用、思考过程的标记，要配对应的 tool-call parser / reasoning parser；
4. **多模态**：图片 / 音频的预处理器和占位 token 的展开另有一套接口。

### 6. SGLang 的差别

SGLang 的模型文件长得和 vLLM 几乎一样（SGLang 早期直接复用 vLLM 的层），差在接口上：

| | vLLM | SGLang |
|---|---|---|
| attention 层 | `Attention(..., prefix=...)` | `RadixAttention(..., layer_id=...)` |
| forward 参数 | `input_ids, positions, intermediate_tensors, inputs_embeds` | `input_ids, positions, forward_batch`（`ForwardBatch` 里装着本步的 metadata） |
| logits | `compute_logits(hidden_states)` 单独调 | forward 里直接调 `self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)` |
| 注册 | `registry.py` 或 `ModelRegistry.register_model` | 文件里的 `EntryClass` 或 `SGLANG_EXTERNAL_MODEL_PACKAGE` |

`load_weights` 的 `stacked_params_mapping` 写法两家相同。

### 7. 怎么验证

1. **能加载**：vLLM 要求在 `tests/models/registry.py` 里登记一个 HF 仓库例子，CI 会用随机权重初始化一遍（[vLLM 文档：Unit Testing](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/tests.md)）；
2. **和 HF 对齐**：greedy 解码比较输出。vLLM 的测试工具有两档：`check_outputs_equal`（文本完全一致）和 `check_logprobs_close`（各自的 top-k logprob 互相包含，容忍 bf16 下 kernel 不同带来的小误差）。SGLang 用 `scripts/playground/reference_hf.py` 和 `python3 -m sglang.benchmark.one_batch --correct` 比 prefill logits；
3. **先 TP = 1 再 TP > 1**：TP = 1 对了、TP = 2 错了，问题几乎一定在 weight_loader 的切片；
4. **对不上时逐层比**：HF 和框架各跑同一个 prompt，打印每层输出的 hidden state，找第一个发散的层。

## 面试追问

::: details Q：为什么 vLLM 的模型 forward 不接收 [B, S, H] 的输入？
continuous batching 下一个 batch 里既有 prefill 的长序列又有 decode 的单 token。例：A 在 prefill 1000 个 token，B、C 各 decode 1 个，补成矩形要 3 × 1000 = 3000 行，有效的只有 1002 行，三分之二是 padding。所以把所有请求的 token 拼成一维 `[num_tokens, H]`，再用 metadata（每个序列的起止 `cu_seqlens`、slot mapping、block table）告诉 attention kernel 怎么分组。Linear、RMSNorm 这些逐 token 的层本来就不关心 token 属于哪个请求，直接算。见 [请求生命周期](/framework/request-lifecycle)。
:::

::: details Q：模型 $H_{kv} = 8$，想开 TP = 16，会怎样？
每个 KV head 被复制到 2 张卡上（`num_kv_head_replicas = 2`），每卡 1 个 KV head、2 个 Q head。能跑，但 KV cache 总显存翻倍，KV 投影的计算也重复了一份。要求 TP 能被 $H_{kv}$ 整除（或反过来），否则直接报错。
:::

::: details Q：`tie_word_embeddings = true` 的模型加载时要注意什么？
lm_head 和 embedding 是同一个矩阵，checkpoint 里通常没有 `lm_head.weight`（有的也会多存一份）。模型里让 `lm_head.weight` 直接指向 `embed_tokens.weight`，加载时跳过 `lm_head.weight`。否则要么报「权重未初始化」，要么加载了两份、一份没被用上。
:::

::: details Q：新模型输出全是乱码，怎么排查？
按概率从高到低：(1) chat template 或 BOS / EOS 不对，先用 completion 接口喂原始文本排除；(2) RoPE 参数不对（`rope_theta`、scaling 类型、`is_neox_style` 决定 q/k 的维度是前后对半配对还是相邻两两配对）；(3) 权重合并或切片错：TP = 1 和 TP = 2 对比；(4) 逐层和 HF 比 hidden state，定位到第一层发散的算子。
:::

## 手撕

给一个 HF 模型的 `state_dict` 键名列表，写出到 vLLM 合并 QKV 后的参数映射和 TP 切分规则：

```python
def plan(hf_shapes: dict, H: int, H_kv: int, d_h: int, tp: int, rank: int):
    """hf_shapes: {"model.layers.0.self_attn.q_proj.weight": (H*d_h, d), ...}
    返回: {框架参数名: [(HF 名, 目标 [行起, 行止), 源 [行起, 行止) 或 列切片)]}"""
    hq = H // tp
    hkv = max(1, H_kv // tp)
    rep = max(1, tp // H_kv)                       # 每个 KV head 被几张卡共享
    kv_rank = rank // rep
    out = {}
    for name in hf_shapes:
        if ".q_proj." in name:
            dst = name.replace(".q_proj.", ".qkv_proj.")
            out.setdefault(dst, []).append((name, (0, hq * d_h), (rank * hq * d_h, (rank + 1) * hq * d_h)))
        elif ".k_proj." in name or ".v_proj." in name:
            dst = name.replace(".k_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
            base = hq * d_h + (0 if ".k_proj." in name else hkv * d_h)
            src = (kv_rank * hkv * d_h, (kv_rank + 1) * hkv * d_h)
            out.setdefault(dst, []).append((name, (base, base + hkv * d_h), src))
        elif ".o_proj." in name or ".down_proj." in name:
            d_in = hf_shapes[name][1]
            out[name] = [(name, "all rows", f"cols [{rank * d_in // tp}, {(rank + 1) * d_in // tp})")]
    return out
```

用 Llama-3-8B（$H = 32$，$H_{kv} = 8$，$d_h = 128$）、TP = 4、rank = 1 检查：q 取源的第 1024–2048 行放到目标 0–1024 行；k 取源 256–512 行放到目标 1024–1280 行；v 放到 1280–1536 行。目标总共 1536 行，和第 2 节算的本卡 QKV 输出维一致。gate / up 的映射同理，留作练习。

## 参考

- [vLLM 文档：Basic Model](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/basic.md)
- [vLLM 文档：Registering a Model](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/registration.md)
- [vLLM 文档：Unit Testing](https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/tests.md)
- vLLM 源码：[`models/llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)、[`layers/linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)、[`models/registry.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
- [SGLang 文档：How to Support New Models](https://docs.sglang.io/docs/supported-models/support_new_models)
- SGLang 源码：[`models/llama.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)、[`layers/radix_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/radix_attention.py)
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053)（列切 / 行切的出处）

---
title: 给 vLLM / SGLang 加一个新模型
status: todo
tags: [vllm, sglang]
difficulty: 4
order: 3
related: []
stack: [ld-load, f-model, f-runner, k-attn, sv-tokenize]
---

# 给 vLLM / SGLang 加一个新模型

> config 映射、weight loading、model runner、attention backend、tokenizer / chat template

## 一句话结论

给 vLLM / SGLang 加模型，本质是把 HF 的 `config.json` + `safetensors` 翻译成框架自己的「模型定义 + 权重加载 + attention backend」三件套：模型定义用框架的并行 Linear 和 Attention 层重写 forward，权重加载把 HF 的参数名映射到框架的分片参数上，attention 直接调框架的 paged attention 接口而不是自己写。

## 推导

- **config 映射**：HF config 里的 `hidden_size`、`num_key_value_heads`、`rope_theta` 等字段对应到框架的 model config；新架构（如 MLA、滑窗）要先确认 attention backend 支持。
- **模型定义**：用 `ColumnParallelLinear` / `RowParallelLinear`（见 [Megatron TP](/parallel/megatron-tp)）替换 `nn.Linear`，QKV 合并成一个 Linear 减少 kernel 数；forward 的输入是扁平的 token 序列 + metadata，不是 `[B, S, H]`。
- **weight loading**：`load_weights` 遍历 safetensors 的 `(name, tensor)`，按映射表找到目标参数，合并 QKV / gate-up、按 TP rank 切分后拷贝；这是最容易出错的一步。
- **tokenizer / chat template**：多数情况直接复用 HF tokenizer，只有自定义的 chat template、特殊 token 和多模态输入需要额外处理。

## 面试追问

::: details Q：为什么 vLLM 的模型 forward 不接收 [B, S, H] 的输入？
continuous batching 下一个 batch 里既有 prefill 的长序列又有 decode 的单 token，padding 成矩形会浪费大量算力。所以输入是把所有请求的 token 拼成一维 `[num_tokens, H]`，再用 metadata（每个序列的起止、slot mapping、block table）告诉 attention kernel 怎么分组，见 [请求生命周期](/framework/request-lifecycle)。
:::

## 手撕

常见题：给一个 HF 模型的 `state_dict` 键名列表，写出到 vLLM 合并 QKV 后的参数映射和 TP 切分规则。

## 参考

- [vLLM 文档：Adding a New Model](https://docs.vllm.ai/en/latest/contributing/model/)
- [SGLang 文档：How to Support a New Model](https://docs.sglang.ai/supported_models/support_new_models.html)

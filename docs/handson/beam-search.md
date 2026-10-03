---
title: Simple Beam Search
status: todo
tags: [beam-search, handson]
difficulty: 2
order: 6
related: []
stack: [o-sampling]
leetgpu: [98]
---

# Simple Beam Search

> 长度惩罚与 early stopping

## 一句话结论

beam search 每步保留 k 条累计 log-prob 最高的序列：对每条 beam 展开 vocab 个候选，在 k × V 个候选里取 top-k 作为下一步的 beam，遇到 EOS 的 beam 移到完成集并补一条新的。比贪心好在不会被局部最优的 token 卡死，代价是 k 倍的计算和 KV cache，推理服务里很少用。

## 推导

- **打分**：序列分数是 log-prob 之和，长序列天然吃亏，常用长度归一化 score / len^α。
- **展开与裁剪**：`[k, V]` 的 log-prob 加上每条 beam 的累计分数，flatten 后 `topk(k)`，再用整除和取余还原出「来自哪条 beam、选了哪个 token」。
- **KV cache**：被选中的 beam 要把父 beam 的 KV 复制过来（或用 block table 共享前缀，PagedAttention 的 copy-on-write 就是为此）；见 [KV cache 与 PagedAttention](/inference/kv-cache-paged-attention)。
- **结束条件**：完成集里有 k 条且最好的未完成 beam 分数已不可能超过它们。

## 面试追问

::: details Q：为什么 LLM 服务基本不用 beam search？
一是 k 倍的 KV cache 和算力让吞吐掉 k 倍；二是对开放式生成 beam search 倾向于产出重复、保守的高频句子，采样（temperature + top-p）反而质量更好。beam search 主要留在翻译、摘要这类有标准答案的任务上。
:::

## 手撕

对应 [LeetGPU #98](https://leetgpu.com/challenges)。框架：

```python
def beam_search(model, prompt_ids, k, max_len, eos_id):
    beams = [(prompt_ids, 0.0)]          # (tokens, cum_logprob)
    finished = []
    for _ in range(max_len):
        cands = []
        for toks, score in beams:
            logp = model.next_token_logprobs(toks)      # [V]
            top_lp, top_id = logp.topk(k)
            # 展开 k 个候选，加到 cands
        # cands 按 score 排序取前 k；EOS 的移入 finished
        # 没有未完成 beam 时 break
    return max(finished, key=lambda x: x[1] / len(x[0]))
```

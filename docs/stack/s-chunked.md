---
title: Chunked Prefill
---

# Chunked Prefill

一个长 prompt 的 prefill 会独占好几步，同一批里正在 decode 的请求只能干等，ITL 出现尖刺。chunked prefill 把长 prefill 切成固定 token 数的块，每一步拿一块和 decode 请求拼在一起跑。

- **效果**：decode 延迟更平稳；compute-bound 的 prefill 块和 memory-bound 的 decode 混在一起跑，GPU 利用率更高。
- **token 预算**：每一步的总 token 数有上限，块大小是在 TTFT 和 ITL 之间做取舍。
- **代价**：同一个 prompt 的 attention 要分几次算，每一块都要重新读前面各块的 KV。

**延伸阅读**：[Sarathi-Serve (OSDI '24)](https://arxiv.org/abs/2403.02310)

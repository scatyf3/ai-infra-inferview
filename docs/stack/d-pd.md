---
title: PD 分离
---

# PD 分离

prefill 是 compute-bound，decode 是 memory-bound，放在同一批 GPU 上会互相干扰：prefill 拉高 decode 的 ITL，decode 占着显存让 prefill 排队。PD 分离把两个阶段放到不同的实例池，prefill 算完后把 KV 传给 decode 实例。

- **好处**：两边可以分别选择并行方式和卡数，分别针对 TTFT 和 TPOT 优化。
- **代价**：KV 要通过 RDMA / NVLink 传输，模型越大、prompt 越长，传输量越大。
- **配比**：P 实例和 D 实例的数量比例要随负载调整。

**延伸阅读**：[DistServe (OSDI '24)](https://arxiv.org/abs/2401.09670) · [Splitwise](https://arxiv.org/abs/2311.18677) · [Mooncake](https://arxiv.org/abs/2407.00079)

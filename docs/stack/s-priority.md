---
title: 优先级调度
---

# 优先级调度

默认的 FCFS 对所有请求一视同仁，但实际业务有在线和离线之分，有付费等级，有不同的延迟 SLA。优先级调度决定谁先进 batch，以及显存不够时先抢占谁。

- **难点**：生成长度事先不知道，经典的短作业优先（SJF）没法直接用，只能预测长度，或者用多级反馈队列来近似。
- **公平性**：按 token 计量每个用户得到的服务量（如 Virtual Token Counter），防止大用户挤占小用户。
- **和抢占联动**：高优先级请求到达时，可以抢占低优先级请求的 decode。

**延伸阅读**：[FastServe](https://arxiv.org/abs/2305.05920) · [Fairness in Serving LLMs](https://arxiv.org/abs/2401.00588)

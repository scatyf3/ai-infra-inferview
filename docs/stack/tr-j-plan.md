---
title: "算力规划 / MFU"
---

# 算力规划 / MFU

训练总算力 $C \approx 6ND$，时长 $= C / (n \cdot \text{peak} \cdot \text{MFU} \cdot \text{ETTR})$。MFU 只算模型需要的 FLOPs，recompute 不算；大规模 dense 训练在 30–55% 之间。Chinchilla 给出算力最优约每参数 20 个 token，但小模型普遍 over-train。

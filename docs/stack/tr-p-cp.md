---
title: "CP"
---

# CP

context parallel 把一条序列切到 $c$ 张卡上，激活按 $c$ 降。attention 要看到全部 K、V：Ring Attention 环形传 KV，Llama 3 直接 all-gather KV，Ulysses 用 all-to-all 换成按 head 切。

- **causal 负载均衡**：切成 $2c$ 块，卡 $i$ 拿第 $i$ 和第 $2c-1-i$ 块。

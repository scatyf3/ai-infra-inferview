---
title: "TP / SP"
---

# TP / SP

Megatron 的 column + row parallel 让每层 forward 只要两次 all-reduce。sequence parallel 把 LayerNorm、dropout 这段也按序列切，all-reduce 拆成 reduce-scatter + all-gather，通信量不变、激活除以 TP 度。

- **限制**：通信在关键路径上，TP 度通常不超过一台机器的 8 张卡。

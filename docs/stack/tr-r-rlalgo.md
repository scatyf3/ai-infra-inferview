---
title: "RL 算法"
---

# RL 算法

PPO 要 actor、critic、reference、reward 四个模型；DPO 把偏好直接写成分类 loss；GRPO 用组内均值当基线，去掉 critic。DAPO、GSPO 等变体各改一个零件。训练时盯熵、回答长度、KL 和梯度范数。

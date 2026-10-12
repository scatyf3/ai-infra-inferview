---
title: "容错 / checkpoint"
---

# 容错 / checkpoint

同步训练任何一张卡出事整个作业都停，作业 MTBF 是单卡 MTBF 除以卡数，万卡规模约几小时一次。最优 checkpoint 间隔 $\tau^* = \sqrt{2\delta M}$，要提高有效训练时间就压写入阻塞 $\delta$ 和重启时间。

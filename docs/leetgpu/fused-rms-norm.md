---
title: Fused Residual Add and RMS Norm
status: draft
tags: [rmsnorm, triton, leetgpu]
difficulty: 3
order: 6
related: [/handson/rmsnorm, /leetgpu/rms-normalization]
stack: [k-fused]
leetgpu: [83]
---

# Fused Residual Add and RMS Norm


1. 这道题是更贴近真llm里的rms norm，input是(N,C)，per token normalize
2. 这里题目给的限制是N和C都小于65k

这里的问题是如何映射计算，一个naive的想法就是每个program（block）负责一个token，但是
1. N太大，block不够咋办
	1. **grid 有上限，但很大：** CUDA 的 grid x 维最多是 $2^{31}-1$。Triton 的 `program_id(0)` 就对应这一维，N=65536 离上限还差得远。要注意的是 y 维和 z 维，它们的上限只有 65535。所以一维的 grid 要放在 axis 0 上，不要放到 axis 1。
	2. **同时能跑的 program 有限，但不需要你管：** 每个 SM 能同时驻留的 block 数受寄存器、shared memory 和线程数限制。比如 T4 有 40 个 SM，每个 SM 驻留几个 block，同一时刻也就一两百个 program 在跑。剩下的由硬件调度器排队，前一批跑完再发下一批，一批叫一个 wave。你只需要按 N 发 grid，硬件会分 wave 调度。
2. 如果N小C大，为啥不在token 内部也split？
	1. 理论上可以更大并行，但是通常来讲得不偿失
	2. 每个 program 只有一段的部分和，要合成整行的 rms 就得跨 program 通信。要么拆成两个 kernel，多一次 launch，也多一遍 HBM 读写；要么用 atomic，再加上某种全局同步。代价比较大。


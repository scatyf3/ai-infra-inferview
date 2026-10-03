---
title: OS 与网络：进程线程、虚拟内存、锁与原子操作、TCP / RPC / gRPC
status: todo
tags: [os, network, grpc]
difficulty: 2
order: 3
related: []
stack: [sv-api]
---

# OS 与网络：进程线程、虚拟内存、锁与原子操作、TCP / RPC / gRPC

> 基础系统知识

## 一句话结论

推理服务离不开 OS 和网络基础：进程 / 线程决定并发模型（vLLM 一个进程一张卡），虚拟内存和 pinned memory 决定 H2D 拷贝快慢，锁与原子操作决定调度器和 KV 分配器怎么写，TCP / gRPC 决定多节点之间怎么传请求和 KV。

## 推导

- **进程 vs 线程**：进程隔离地址空间、崩了不连坐，线程共享内存、切换便宜；Python 下受 GIL 限制，CPU 密集只能多进程，所以 TP worker 是多进程 + NCCL 通信。
- **虚拟内存**：page fault、mmap 和 page cache 决定权重加载速度；pinned（page-locked）memory 才能走 DMA 异步拷贝，CUDA 的 `cudaMallocHost` 就是为此。
- **锁与原子操作**：无锁队列、CAS 用在高频小临界区；GPU 上的 `atomicAdd` 是同一思路，竞争多时退化严重，所以 reduce 要先 block 内归约再原子加。
- **TCP / RPC / gRPC**：gRPC 基于 HTTP/2 多路复用，适合控制面；大块数据（KV 传输）走 RDMA / NCCL，不走 gRPC。

## 面试追问

::: details Q：为什么 H2D 拷贝要用 pinned memory 才能和计算 overlap？
pageable 内存可能被 OS 换页，DMA 引擎不能直接读，CUDA 会先拷到一个内部 pinned 缓冲再 DMA，这一步是同步的。pinned 内存地址固定，DMA 可以直接读，`cudaMemcpyAsync` 才真正异步，才能和 kernel 在不同 stream 上重叠。
:::

## 手撕

常见题：用条件变量实现一个有界阻塞队列；解释一次 gRPC 调用从序列化到 socket 发送经过哪些层。

## 参考

- [OSTEP: Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [CUDA C++ Best Practices Guide：Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)

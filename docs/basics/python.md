---
title: Python：GIL、asyncio、多进程 vs 多线程
status: todo
tags: [python, gil, asyncio]
difficulty: 2
order: 1
related: []
stack: [sv-api]
---

# Python：GIL、asyncio、多进程 vs 多线程

> CPython 对象模型

## 一句话结论

GIL 让一个 CPython 进程同一时刻只有一个线程执行字节码，所以 CPU 密集用多进程、IO 密集用多线程或 asyncio；推理框架的 API server 用 asyncio 处理并发请求，engine 用独立进程避开 GIL，两者之间用队列 / ZMQ 通信。

## 推导

- **GIL**：保护引用计数等解释器内部状态；C 扩展（torch 算子、NCCL 调用）执行时会释放 GIL，所以 GPU 计算本身不被 GIL 卡住，卡住的是 Python 侧的调度和 tokenize 逻辑。
- **asyncio**：单线程事件循环，`await` 处让出控制权；适合大量等待网络的请求处理，但一个同步的慢函数会阻塞整个 loop，要用 `run_in_executor` 甩出去。
- **多进程 vs 多线程**：多进程绕开 GIL 但有序列化和启动开销；vLLM V1 把 EngineCore 放在单独进程里，API server 进程只做前后处理，就是这个取舍。
- **对象模型**：一切皆 `PyObject`，有引用计数和类型指针；列表存的是指针数组，所以 Python 列表做数值计算慢，必须下沉到 tensor。

## 面试追问

::: details Q：GIL 存在的话，为什么 PyTorch 多线程 DataLoader 还能加速？
数据加载的瓶颈多在磁盘 IO、解码（PIL / OpenCV 是 C 扩展，会释放 GIL）上，这些阶段不持有 GIL，多线程可以并行。纯 Python 的变换逻辑仍受 GIL 限制，所以 DataLoader 默认用多进程 worker。
:::

## 手撕

常见题：用 asyncio 写一个带并发上限的请求发送器（`Semaphore` + `gather`）；解释 `async for` 流式返回 token 的实现。

## 参考

- [Python 文档：asyncio](https://docs.python.org/3/library/asyncio.html)
- [What is the Python GIL](https://realpython.com/python-gil/)

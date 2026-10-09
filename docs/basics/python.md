---
title: Python：GIL、asyncio、多进程 vs 多线程
status: draft
tags: [python, gil, asyncio, multiprocessing]
difficulty: 2
order: 1
related: [/basics/rust, /basics/cpp, /basics/os-network, /framework/vllm-v1-architecture, /framework/serving-layer]
stack: [sv-api]
---

# Python：GIL、asyncio、多进程 vs 多线程

> 推理框架的 API server、调度器、tokenizer 胶水层都是 Python；面试考的是「CPU 侧为什么会拖慢 GPU」

## 一句话结论

CPython 的 GIL 让一个进程里同一时刻只有一个线程在执行 Python 字节码。所以：
1. IO 密集（等网络、等 GPU）用 asyncio 或线程；
2. CPU 密集的纯 Python 逻辑要么下沉到会释放 GIL 的 C / C++ / Rust 扩展，要么拆到独立进程；
3. 拆进程的代价是跨进程序列化（pickle / msgpack）和启动开销。

vLLM V1 把调度循环（EngineCore）放进单独进程，API server 进程只管 HTTP、tokenize 和 detokenize，两边用 ZMQ 通信，就是这个取舍的典型例子。

## 推导

### 0. 先说清楚：CPython 的对象长什么样

Python 里一切都是 `PyObject`：对象头里有**引用计数**和**类型指针**，后面才是数据。实测（CPython 3.13，64 位）：

```python
import sys
sys.getsizeof(1)          # 28 字节：一个小整数也是完整对象
sys.getsizeof(1.0)        # 24 字节
sys.getsizeof([])         # 56 字节：空 list 的对象头
sys.getsizeof(list(range(1000)))   # 8056 = 56 + 1000 × 8：list 里存的是 8 字节指针
```

所以 1000 个整数的 list 实际占约 8 KB 指针 + 1000 × 28 B 的整数对象（小整数 -5~256 是缓存的单例），而 `np.int64` 数组只要 8 KB 连续内存。这就是「Python 列表做数值计算慢、必须下沉到 tensor」的原因：每个元素都要解引用、查类型、改引用计数，没法向量化。

**引用计数与 GC**：引用计数降到 0 立即释放（所以 `del x` 之后 tensor 的显存马上还给 caching allocator）；循环引用靠分代 GC 周期性扫描回收（[gc 文档](https://docs.python.org/3/library/gc.html)）。两个和 infra 相关的点：

- **引用计数会写内存**。`fork` 之后子进程哪怕只是「读」父进程的对象，也会改它的引用计数，触发 copy-on-write，把共享页逐页复制一遍。官方文档建议在 fork 前调 `gc.freeze()`，避免子进程的 GC 去碰这些长期对象（[gc.freeze](https://docs.python.org/3/library/gc.html#gc.freeze)）。
- **循环引用里的 tensor 不会立即释放**，要等 GC 扫到。显存「看起来泄漏」时可以先 `gc.collect()` 再看 `torch.cuda.memory_allocated()` 是否下降。

### 1. GIL 是什么，何时释放

**定义**：GIL（Global Interpreter Lock）是解释器级别的一把锁，线程必须持有它才能执行字节码（[Glossary：GIL](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)）。它存在的主要原因是引用计数的增减不是原子的，用一把大锁比给每个对象加锁便宜。

**什么时候切换**：持锁线程每隔一个「switch interval」被请求让出，默认 5 ms（`sys.getswitchinterval()` 返回 `0.005`，见 [sys.setswitchinterval](https://docs.python.org/3/library/sys.html#sys.setswitchinterval)）。阻塞 IO 和大部分 C 扩展的长计算会主动释放 GIL。

实测纯 Python 循环，两线程并没有更快（M 系列 Mac，3.13）：

```python
import threading, time

def work(n=10_000_000):
    s = 0
    for i in range(n):
        s += i

t = time.perf_counter(); work(); work()
print("serial   ", time.perf_counter() - t)      # ≈ 0.50 s

t = time.perf_counter()
ths = [threading.Thread(target=work) for _ in range(2)]
[x.start() for x in ths]; [x.join() for x in ths]
print("2 threads", time.perf_counter() - t)      # ≈ 0.48 s，没有加速
```

**对推理服务意味着什么**：

| 代码 | 持有 GIL 吗 | 影响 |
| --- | --- | --- |
| `torch.matmul`、NCCL 调用、`cudaMemcpy` | C++ 侧执行时释放 | GPU 计算本身不受 GIL 影响 |
| HF tokenizers `encode_batch` | Rust 侧释放（见 [Rust：PyO3](/basics/rust#_10-pyo3-给-python-写扩展并释放-gil)） | 放线程池能真并行 |
| 调度器循环、构造 batch metadata、采样后处理、detokenize 拼字符串 | 持有 | 和同进程的 HTTP 处理抢同一把锁 |

decode 一步在 H100 上跑小模型只要几毫秒，vLLM 团队观察到 Llama-8B 在 H100 上一步 GPU 时间低到约 5 ms，此时 CPU 侧的调度和前后处理开销就显得很突出（[vLLM V1 博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)）。GIL 让这些 CPU 工作只能串行，GPU 就在等。

**free-threaded CPython**：PEP 703 给出了去掉 GIL 的构建（`python3.13t`），3.14 起官方支持但仍是可选构建（[PEP 703](https://peps.python.org/pep-0703/)、[What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html)）。单线程有额外开销，官方给的 pyperformance 平均值约 1%（macOS aarch64）到 8%（x86-64 Linux）（[Free-threading HOWTO](https://docs.python.org/3/howto/free-threading-python.html)）。目前主流推理框架仍按「有 GIL」来设计。

### 2. 三种并发模型

| | 线程 `threading` | 协程 `asyncio` | 进程 `multiprocessing` |
| --- | --- | --- | --- |
| 调度者 | OS 抢占式 | 事件循环，协作式（只在 `await` 让出） | OS |
| 能否并行跑 Python 字节码 | 不能（GIL） | 不能（单线程） | 能 |
| 单位开销 | 每线程一个 OS 栈（默认数 MB 虚拟地址） | 一个协程是一个 Python 对象，KB 级 | 一个完整解释器 + 导入 torch 的内存和秒级启动 |
| 共享数据 | 直接共享，要加锁 | 直接共享，`await` 之间天然无竞争 | 要序列化或用共享内存 |
| 典型用途 | 调用会释放 GIL 的阻塞函数 | 成千上万个 HTTP / SSE 连接 | 每张卡一个 worker、EngineCore |

推理服务的标准组合：**API server 用 asyncio 管连接，CPU 重活放线程池或单独进程，GPU worker 每卡一个进程**。

### 3. asyncio 事件循环

**基本模型**：一个线程里跑一个事件循环。`async def` 定义协程，`await` 表示「这里要等，先去跑别人」。事件循环底层用 `epoll` / `kqueue` 监听 socket，哪个就绪就恢复哪个协程（[asyncio 事件循环](https://docs.python.org/3/library/asyncio-eventloop.html)）。

**关键推论**：两个 `await` 之间的代码是独占执行的。一个同步慢函数会让**所有连接**一起停住。下面的 heartbeat 每 10 ms 醒一次，记录实际间隔：

```python
import asyncio, time

def tokenize(n=3_000_000):          # 纯 Python CPU 计算，模拟慢的同步函数
    return sum(i * i for i in range(n))

async def heartbeat(log):
    last = time.perf_counter()
    for _ in range(30):
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        log.append((now - last) * 1e3)
        last = now

async def main(mode):
    log = []
    hb = asyncio.create_task(heartbeat(log))
    await asyncio.sleep(0.05)
    if mode == "block":
        tokenize()                                  # 直接调用：阻塞整个 loop
    else:
        await asyncio.to_thread(tokenize)           # 甩到线程：loop 继续跑
    await hb
    print(f"{mode:6s} max gap = {max(log):6.1f} ms")

asyncio.run(main("block"))    # 实测 max gap ≈ 122 ms
asyncio.run(main("thread"))   # 实测 max gap ≈ 28 ms
```

两点观察：

1. 直接调用时 heartbeat 停了 100 多毫秒，对应到服务里就是所有流式请求的 ITL 同时抖一下。asyncio 的 debug 模式会把超过 100 ms 的回调记成 slow callback（[Developing with asyncio](https://docs.python.org/3/library/asyncio-dev.html)）。
2. 甩到线程后仍有约 28 ms 的抖动：这个 `tokenize` 是纯 Python，在工作线程里照样持有 GIL，事件循环线程每次要等 switch interval 才能抢回来。官方文档也写明 `to_thread` 一般只能让 **IO 密集**的函数不阻塞（[asyncio.to_thread](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)）。真正的 tokenizer 是 Rust 实现、计算时释放 GIL，所以放线程池有效；纯 Python 的 CPU 活就只能拆进程。

**另一个坑**：`asyncio.create_task()` 返回的 task 要自己保存引用，事件循环只持有弱引用，否则 task 可能在执行中途被 GC 回收（[create_task 文档](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)）。

### 4. 生成器与迭代器：流式输出的底层

**迭代器**是实现了 `__next__` 的对象；**生成器**是带 `yield` 的函数，调用后返回一个迭代器，每次 `next()` 执行到下一个 `yield` 就暂停并保存栈帧。**异步生成器**（`async def` + `yield`）用 `async for` 消费，暂停点里可以 `await`。

流式返回 token 就是一个异步生成器：引擎每产出一个 token，生成器 `yield` 一帧 SSE，HTTP 框架把它写进 socket。好处是内存里只有当前这一帧，而不是整段输出。

```python
async def stream_tokens(engine, req_id):
    async for out in engine.generate(req_id):   # 引擎每步产出增量
        yield out.delta_text

async def sse(engine, req_id):
    async for tok in stream_tokens(engine, req_id):
        yield f"data: {tok}\n\n"                # SSE 帧：data 行 + 空行
    yield "data: [DONE]\n\n"
```

客户端断开时，框架会在 `yield` 处抛 `CancelledError` / `GeneratorExit`，这里要做 abort：通知引擎释放这个请求的 KV block，否则显存被一个已经没人看的请求占着。SSE 和 HTTP 层面的细节见 [OS 与网络](/basics/os-network#_7-http-与-sse-llm-api-怎么流式返回)。

### 5. 装饰器与上下文管理器

**装饰器**就是「接收函数、返回新函数」的高阶函数，`@deco` 等价于 `f = deco(f)`。**上下文管理器**实现 `__enter__` / `__exit__`，`with` 块保证 `__exit__` 在异常时也执行，相当于 C++ 的 RAII（见 [C++：RAII](/basics/cpp#_1-raii-资源跟着对象走)）。`torch.no_grad()`、`torch.cuda.stream(s)`、`torch.profiler.profile()` 都是上下文管理器，`no_grad` 同时也能当装饰器用。

```python
import functools, time
from contextlib import contextmanager

@contextmanager
def cuda_timer(name, sync=None):
    # sync 传 torch.cuda.synchronize：GPU 是异步的，不同步只量到了 launch 时间
    if sync: sync()
    t = time.perf_counter()
    try:
        yield
    finally:                       # 异常路径也会执行
        if sync: sync()
        print(f"{name}: {(time.perf_counter() - t) * 1e3:.2f} ms")

def timed(fn):
    @functools.wraps(fn)           # 保留 __name__ / __doc__，否则日志和 profiler 里全叫 wrapper
    def wrapper(*args, **kwargs):
        with cuda_timer(fn.__name__):
            return fn(*args, **kwargs)
    return wrapper

@timed
def step(n):
    return sum(range(n))

step(1_000_000)
```

### 6. 多进程：启动方式与序列化代价

**三种 start method**（[multiprocessing 文档](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)）：

1. `fork`：直接 `fork()` 当前进程，子进程继承父进程全部内存（copy-on-write），启动快。但「安全地 fork 一个多线程进程是有问题的」，父进程里别的线程持有的锁会以锁住的状态被复制过去。
2. `spawn`：启动全新解释器，重新 import 主模块，参数通过 pickle 传过去。慢（要重新 `import torch`），但干净。
3. `forkserver`：先起一个干净的 server 进程，以后从它 fork。

**CUDA 不能跨 fork 使用**：父进程初始化 CUDA 之后再 fork，子进程里 CUDA 会出错，PyTorch 文档要求用 `spawn` 或 `forkserver`（[PyTorch multiprocessing notes](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)）。vLLM 的 `VLLM_WORKER_MULTIPROC_METHOD` 默认是 `fork`，但检测到 CUDA 已初始化时会打 warning 并强制改成 `spawn`（[vllm/utils/system_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/utils/system_utils.py)）。Python 3.14 起 POSIX 上（macOS 除外）默认 start method 从 `fork` 改为 `forkserver`（[What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html)），老代码依赖 fork 继承全局变量的会出问题。

**序列化代价**：跨进程传对象要 pickle → 写管道 / socket → unpickle。实测一个 64 MB 的 float32 numpy 数组（M 系列 Mac）：

```python
import pickle, numpy as np
x = np.zeros(16 * 1024 * 1024, dtype=np.float32)              # 64 MB
b = pickle.dumps(x, protocol=5)                                # ≈ 11 ms，产出 64 MB bytes
y = pickle.loads(b)                                            # ≈ 5 ms

bufs = []
b = pickle.dumps(x, protocol=5, buffer_callback=bufs.append)   # ≈ 0.5 ms，b 只有 121 字节
# 数据本身作为 out-of-band buffer 留在 bufs[0]（零拷贝的 memoryview），由调用方决定怎么发
```

protocol 5 的 out-of-band buffer 就是为了避免大块数据的额外拷贝（[PEP 574](https://peps.python.org/pep-0574/)、[pickle 文档](https://docs.python.org/3/library/pickle.html)）。估算公式：一次跨进程传输 ≈ 序列化 + 拷贝到内核 + 拷贝出内核 + 反序列化，每一项都是 $\text{size} / \text{memcpy 带宽}$ 量级（单核 memcpy 通常 10–20 GB/s），所以 64 MB 走一圈就是几到十几毫秒，和一次 decode step 同量级。

vLLM 的做法：前端和 EngineCore 之间用 ZMQ socket + msgspec msgpack，而不是 pickle（[core_client.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core_client.py)）；tensor / ndarray 小于 256 字节时内联编码，更大的作为单独的帧发送，避免拷进消息体（`VLLM_MSGPACK_ZERO_COPY_THRESHOLD`，见 [serial_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/serial_utils.py)）。传的是 token id 和请求元数据，不传 KV，KV 一直留在 GPU 上。

真正要共享大块 CPU 数据时用共享内存：`multiprocessing.shared_memory`（[文档](https://docs.python.org/3/library/multiprocessing.shared_memory.html)），或 `torch.multiprocessing` 传 tensor 时只传共享内存句柄。原理见 [OS 与网络：共享内存](/basics/os-network#_4-共享内存-ipc-与零拷贝)。

### 7. 为什么 vLLM V1 把 EngineCore 拆成单独进程

把前面几节串起来：

1. API server 要在一个 asyncio 循环里服务上千个连接，还要 tokenize / detokenize，这些都占 GIL。
2. 调度器每一步要构造 batch、分配 KV block、准备输入，这也是 Python 代码，也占 GIL。
3. 放在一个进程里，两者抢同一把锁，结果是 GPU 跑完一步要等 CPU 把下一步准备好。vLLM 0.6.0 先把 API server 拆成单独进程、用 ZeroMQ 做 IPC；V1 进一步做成独立的 EngineCore 循环，专门负责调度和执行模型，前后处理和它重叠（[vLLM V1 博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)）。
4. 代价：每步多一次 IPC 序列化，所以消息里只放必要字段，用 msgpack 而不是 pickle。

完整的进程结构见 [vLLM V1 架构](/framework/vllm-v1-architecture)。

### 8. 常见坑

1. **在 `async def` 里调同步慢函数**：`requests.get`、`time.sleep`、大 tokenizer 调用、同步的 `torch.cuda.synchronize()`，都会冻结事件循环。
2. **隐式 GPU 同步**：`tensor.item()`、`.tolist()`、`print(tensor)`、用 GPU tensor 做 `if` 判断，都会等 GPU 跑完，打断 CPU 和 GPU 的流水。
3. **在 Python 里逐元素循环 tensor**：每次索引都是一次 kernel launch 或一次 Python 对象创建。
4. **fork 之后用 CUDA**：见上一节；DataLoader 的 worker 也一样，worker 里不要碰 CUDA。
5. **可变默认参数** `def f(x, cache={})`：默认值只在定义时求值一次，所有调用共享同一个 dict。
6. **闭包的晚绑定**：`[lambda: i for i in range(3)]` 三个函数都返回 2，要写 `lambda i=i: i`。
7. **fire-and-forget 的 task 被回收**：`create_task` 的返回值不存，见第 3 节。

## 面试追问

::: details Q：GIL 存在的话，为什么 PyTorch 多线程 DataLoader 还能加速？
数据加载的瓶颈多在磁盘 IO 和解码上，PIL / OpenCV 是 C 扩展，执行时会释放 GIL，这些阶段多线程可以并行。纯 Python 的变换逻辑仍受 GIL 限制，所以 DataLoader 默认用多进程 worker（`num_workers > 0` 时每个 worker 是一个进程），代价是每个 batch 要跨进程传回主进程（通过共享内存传 tensor）。
:::

::: details Q：asyncio 和多线程都是「单核」的，什么时候选哪个？
都受 GIL 限制，区别在调度方式和规模。asyncio 只在 `await` 处切换，两个 `await` 之间不会被打断，共享状态不用加锁；一个协程只是一个 Python 对象，上万个连接也撑得住。线程是 OS 抢占，任何字节码之间都可能切换，要加锁；每个线程有自己的栈，几千个就很重。所以连接管理用 asyncio；要调用一个没有 async 版本、但会释放 GIL 的阻塞函数（文件 IO、Rust tokenizer），用 `run_in_executor` / `to_thread` 甩到线程池。
:::

::: details Q：为什么 vLLM 的 TP worker 是多进程而不是多线程？
三个原因。① GIL：每个 worker 都要跑 Python 的 model runner 逻辑（准备输入、launch kernel），多线程就串行了。② CUDA 上下文：一卡一进程最简单，`CUDA_VISIBLE_DEVICES` / device 绑定清晰，一个 worker 崩了不会带着整个地址空间一起坏。③ NCCL 的常见用法就是一个进程（rank）一张卡。代价是进程间要靠 NCCL（GPU 数据）和共享内存 / ZMQ（控制消息）通信。
:::

::: details Q：跨进程传一个大 tensor，怎么避免拷贝？
1. 如果是 GPU tensor，根本不要经过 CPU：同机用 CUDA IPC handle（`torch.multiprocessing` 传 CUDA tensor 时就是这么做的），跨卡用 NCCL。
2. CPU tensor 用共享内存：`torch.multiprocessing` 会把 tensor 的 storage 移到共享内存，只 pickle 一个句柄；或者自己用 `shared_memory.SharedMemory` + `np.ndarray(buffer=shm.buf)`。
3. 必须走 socket 时，用 pickle protocol 5 的 out-of-band buffer 或 msgpack 的多帧发送，避免把数据先拷进一个大 bytes 再发。
:::

::: details Q：free-threaded Python 出来之后，还需要拆进程吗？
短期内仍然需要。① 很多 C 扩展还没适配 free-threaded 构建，导入时可能重新启用 GIL。② 单线程有约 1%–8% 的额外开销。③ 拆进程还有 GIL 之外的好处：故障隔离、一卡一进程的 CUDA 管理、可以独立扩缩。free-threading 的价值更多在 API server 内部，比如 tokenize 和 detokenize 能在线程里真并行。
:::

## 手撕

### 带并发上限的请求发送器 + SSE 流式生成器

压测客户端和网关都会写这段：最多 `limit` 个请求同时在飞，结果按提交顺序返回。

```python
import asyncio, random, time

async def fake_request(i: int) -> str:
    await asyncio.sleep(random.uniform(0.05, 0.15))   # 模拟网络 + 推理
    return f"resp-{i}"

async def bounded_gather(n: int, limit: int) -> list[str]:
    sem = asyncio.Semaphore(limit)
    inflight = peak = 0

    async def one(i: int) -> str:
        nonlocal inflight, peak
        async with sem:                     # 最多 limit 个协程同时进入
            inflight += 1
            peak = max(peak, inflight)
            try:
                return await fake_request(i)
            finally:
                inflight -= 1

    out = await asyncio.gather(*(one(i) for i in range(n)))   # 结果顺序 = 提交顺序
    print(f"peak in-flight = {peak}")
    return out

async def stream_tokens(prompt: str):
    for tok in prompt.split():
        await asyncio.sleep(0.01)           # 模拟等下一步 decode
        yield tok

async def sse(prompt: str):
    async for tok in stream_tokens(prompt):
        yield f"data: {tok}\n\n"
    yield "data: [DONE]\n\n"

async def main():
    t = time.perf_counter()
    res = await bounded_gather(n=20, limit=5)
    print(res[:3], f"{time.perf_counter() - t:.2f}s")   # 20 个请求、并发 5：约 4 轮 × 0.1 s
    async for frame in sse("hello paged attention"):
        print(repr(frame))

asyncio.run(main())
```

要点：`Semaphore` 控并发；`gather` 保序；`finally` 保证计数在异常时也回退。追问「某个请求失败怎么办」：`gather(..., return_exceptions=True)` 把异常当结果返回，或者改用 3.11 的 `asyncio.TaskGroup`，一个失败就取消其余。

## 参考

- [Python 文档：asyncio](https://docs.python.org/3/library/asyncio.html)
- [Python 文档：Developing with asyncio](https://docs.python.org/3/library/asyncio-dev.html)
- [Python 文档：multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [PEP 703：Making the GIL Optional](https://peps.python.org/pep-0703/)
- [PEP 574：Pickle protocol 5 with out-of-band data](https://peps.python.org/pep-0574/)
- [PyTorch：Multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [vLLM V1 alpha release 博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

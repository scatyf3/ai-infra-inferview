---
title: OS 与网络：进程线程、虚拟内存、pinned memory、IPC、TCP / SSE、RDMA
status: draft
tags: [os, network, rdma, pinned-memory, sse]
difficulty: 2
order: 3
related: [/basics/python, /basics/cpp, /inference/kv-cache-paged-attention, /parallel/collective-comm, /stack/hw-link, /stack/sv-stream, /stack/kv-offload]
stack: [sv-api]
---

# OS 与网络：进程线程、虚拟内存、pinned memory、IPC、TCP / SSE、RDMA

> 推理服务的很多「玄学慢」最后都落在 OS 和网络上：H2D 拷贝没 overlap、`/dev/shm` 太小、SSE 被代理缓冲、跨机 KV 传输走了 TCP

## 一句话结论

1. **进程 / 线程**决定并发模型：Python 下 CPU 密集只能多进程，所以推理引擎是一卡一进程。
2. **虚拟内存**用页表把连续的虚拟地址映射到不连续的物理页，PagedAttention 用 block table 对 KV 做了同一件事。
3. **pinned memory** 地址固定、不会被换出，DMA 引擎才能直接读，`cudaMemcpyAsync` 才真正异步。
4. **共享内存 / CUDA IPC / RDMA** 都是在做零拷贝：让数据不经过多余的 CPU 拷贝和内核协议栈。
5. 网络上要分清**延迟和带宽**：控制面（请求、token 流）走 TCP 上的 HTTP / SSE / gRPC，数据面（KV、梯度、权重）走 NVLink / RDMA。

## 推导

### 0. 进程与线程

- **进程**：OS 分配资源的单位，有独立的虚拟地址空间、文件描述符表。一个进程崩了不影响别的进程。
- **线程**：调度的单位，同一进程的线程共享地址空间，切换时不用换页表，比进程切换便宜。
- **上下文切换**：保存寄存器、切内核栈；进程间切换还要换页表、TLB 失效。量级在微秒（经验值，随 CPU 和工作集变化），对每秒几千次的调度循环可以忽略，对每秒百万次的细粒度同步就不行。

推理服务的选择（Python 侧原因见 [Python：GIL](/basics/python#_1-gil-是什么-何时释放)）：

| 组件 | 形态 | 原因 |
| --- | --- | --- |
| API server | 一个进程，asyncio 单线程 + 线程池 | 上千连接，IO 密集 |
| 调度器（vLLM EngineCore） | 独立进程 | 不和 HTTP 处理抢 GIL |
| GPU worker | 一卡一进程 | GIL；CUDA context 和 NCCL rank 一一对应；故障隔离 |

### 1. 虚拟内存与分页

**定义**：每个进程看到的是一段连续的**虚拟地址**；物理内存被切成固定大小的**页**（x86-64 默认 4 KiB），**页表**记录「虚拟页号 → 物理页号」。CPU 的 MMU 每次访存都做这个翻译，TLB 缓存最近的翻译结果（[OSTEP：Paging 章节](https://pages.cs.wisc.edu/~remzi/OSTEP/)）。

```python
PAGE = 4096

def translate(vaddr: int, page_table: dict[int, int]) -> int:
    vpn, offset = divmod(vaddr, PAGE)          # 虚拟页号 + 页内偏移
    if vpn not in page_table:
        raise LookupError("page fault")        # 真实系统：陷入内核，分配 / 读入物理页后重试
    return page_table[vpn] * PAGE + offset

translate(0x1234, {1: 0x9})                    # vpn=1 → ppn=9，结果 0x9234
```

几个推理里会碰到的推论：

1. **按需分配**：`malloc` 一大块只是保留了虚拟地址，第一次写某页时才触发 page fault 分配物理页。所以「分配很快、第一次写很慢」。
2. **`mmap` 文件**：把文件映射进地址空间，访问时由 page fault 从磁盘读进 page cache（[mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html)）。safetensors 加载权重就是 mmap，每个 TP rank 只碰自己那一片，不需要先把整个文件读进内存（见 [权重加载](/stack/ld-load)）。第二次启动快，是因为文件还在 page cache 里。
3. **换出**：内存紧张时 OS 可以把页换到磁盘，物理地址随之改变，这正是下一节 DMA 要避开的情况。

**和 PagedAttention 的类比**：vLLM 论文明确说 PagedAttention 的灵感来自 OS 的虚拟内存和分页（[Kwon et al., 2023](https://arxiv.org/abs/2309.06180)）。对照：

| OS 虚拟内存 | PagedAttention |
| --- | --- |
| 页（4 KiB） | KV block（默认 16 token） |
| 进程的页表 | 每个序列的 block table |
| 虚拟页号 → 物理页号 | 逻辑 block 号 → 物理 block 号 |
| 按需分配，消除外部碎片 | 生成到新 block 才分配，消除预留浪费 |
| fork 后 copy-on-write | parallel sampling / beam search 分叉后 CoW |
| 换出到磁盘 | 抢占时 swap 到 CPU 内存 |
| 由 MMU 硬件做翻译 | 由 attention kernel 自己按 block table 查表 |

翻译公式完全同构：

```python
BLOCK = 16
def kv_slot(token_idx: int, block_table: list[int]) -> int:
    logical_block, offset = divmod(token_idx, BLOCK)
    return block_table[logical_block] * BLOCK + offset    # 和上面的 translate 一样

kv_slot(37, [7, 2, 11])   # token 37 在逻辑 block 2 → 物理 #11，slot = 11*16 + 5 = 181
```

区别在最后一行：GPU 上没有 MMU 帮你翻译 KV 的索引，kernel 里多一次 gather，这就是 block 不能太小的原因。完整设计见 [KV Cache 与 PagedAttention](/inference/kv-cache-paged-attention)。

### 2. Pinned memory 与 DMA

**DMA**（Direct Memory Access）：GPU 上的拷贝引擎（copy engine）直接通过 PCIe 读写主机内存，不占 CPU。前提是它要知道一个**不会变**的物理地址。

- **pageable 内存**（普通 `malloc` / 普通 CPU tensor）：OS 随时可能把页换出或迁移，DMA 不能直接读。CUDA 的做法是先由 CPU 把数据拷进一个内部的 pinned 中转缓冲，再 DMA。这一步 CPU 参与、和调用方同步。
- **pinned（page-locked）内存**：用 `cudaHostAlloc` / `cudaMallocHost` 分配，或用 `cudaHostRegister` 把已有内存锁住。OS 保证不换出，DMA 直接读，`cudaMemcpyAsync` 才真正异步，可以和其他 stream 上的 kernel 重叠（[CUDA Best Practices：Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)）。

PyTorch 里对应的写法：

```python
import torch, time

def h2d_gbps(x: torch.Tensor, iters=20) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        x.to("cuda", non_blocking=True)        # pinned 时才真正异步
    torch.cuda.synchronize()
    return x.numel() * x.element_size() * iters / (time.perf_counter() - t) / 1e9

n = 256 * 1024 * 1024 // 4                      # 256 MB fp32
pageable = torch.empty(n)
pinned = torch.empty(n, pin_memory=True)        # 等价于 cudaHostAlloc
print(f"pageable {h2d_gbps(pageable):.1f} GB/s, pinned {h2d_gbps(pinned):.1f} GB/s")
```

（本页未在 GPU 上实测，数字请在自己机器上跑；`DataLoader(pin_memory=True)` 做的就是把 batch 放进 pinned 内存。）

**能跑多快**：PCIe 的理论带宽 = 每 lane 速率 × lane 数 × 编码效率。Gen5 每 lane 32 GT/s、128b/130b 编码，x16 单方向 $32 \times 16 \times \frac{128}{130} / 8 \approx 63$ GB/s，双向约 128 GB/s，和 NVIDIA 给的 H100 PCIe Gen5「128 GB/s」一致（[H100 规格](https://www.nvidia.com/en-us/data-center/h100/)）。Gen4 是它的一半，约 32 GB/s 单方向。实际到不了理论值：Best Practices Guide 举的例子是 Gen3 x16（理论约 15.75 GB/s）上 pinned 能到约 12 GB/s。

**算一笔**：KV offload 把一个请求 1 GB 的 KV 从 CPU 拉回 GPU，按 Gen5 有效 50 GB/s 估算约 20 ms；同样的 KV 重新 prefill 要多久，取决于前缀长度和算力（见 [KV Offload](/stack/kv-offload)）。这就是「搬回来是否比重算快」的判断依据。

**代价**：pinned 内存不能被换出，是稀缺资源，锁太多会拖慢整机；分配本身也比 `malloc` 重得多（官方文档原话：「Pinned memory should not be overused」）。所以框架会预分配一个 pinned 缓冲池反复使用，而不是每次临时分配。

### 3. 锁、条件变量与原子操作

- **mutex**：同一时刻只有一个线程进入临界区。持锁时间越长、竞争越多，吞吐越差。
- **条件变量**：配合 mutex 使用，「等待某个条件成立」时释放锁并睡眠，被 `notify` 唤醒后重新持锁。必须用 `while` 检查条件，防止虚假唤醒（手撕见下文）。
- **原子操作 / CAS**：硬件保证的不可分割的读改写（`fetch_add`、`compare_exchange`），适合计数器、无锁队列这种很小的临界区。代价是多核争同一 cache line 时会退化，C++ `shared_ptr` 的计数就是例子（见 [C++：智能指针](/basics/cpp#_3-智能指针)）。
- **GPU 上同一思路**：`atomicAdd` 在很多线程争同一地址时串行化，所以 reduce 先在 block 内用 shared memory 归约，每个 block 只做一次全局原子加（见 [Reduce](/handson/reduce)）。

### 4. 共享内存 IPC 与零拷贝

**零拷贝**的意思是：数据从生产者到消费者，不经过多余的中间拷贝（尤其是用户态 ↔ 内核态的拷贝）。常见手段：

1. **POSIX 共享内存**：`shm_open` + `mmap`，两个进程把同一块物理页映射进各自的地址空间（[shm_overview(7)](https://man7.org/linux/man-pages/man7/shm_overview.7.html)）。Linux 上它就是 `/dev/shm`（tmpfs）下的一个文件。
2. **CUDA IPC**：同机两个进程共享一块显存，传的是一个 handle 而不是数据。`torch.multiprocessing` 传 CUDA tensor 就是这样（[PyTorch multiprocessing notes](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)）。
3. **RDMA / GPUDirect**：跨机器、跨设备的零拷贝，见第 8、9 节。

Python 里用共享内存传 numpy，只在写入时拷一次（已实测）：

```python
import numpy as np
from multiprocessing import Process, shared_memory

def worker(name, shape, dtype):
    shm = shared_memory.SharedMemory(name=name)           # 按名字 attach 同一块物理内存
    x = np.ndarray(shape, dtype=dtype, buffer=shm.buf)    # ndarray 直接指向共享页，零拷贝
    x *= 2                                                 # 父进程立即可见
    del x; shm.close()

if __name__ == "__main__":
    a = np.arange(8, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    x = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    x[:] = a
    p = Process(target=worker, args=(shm.name, a.shape, a.dtype)); p.start(); p.join()
    print(x)                                               # [ 0.  2.  4. ... 14.]
    del x; shm.close(); shm.unlink()                       # 不 unlink 就会一直留在 /dev/shm
```

**`/dev/shm` 在容器里是个坑**：Docker 默认 `/dev/shm` 只有 64 MB（[Docker run 文档 `--shm-size`](https://docs.docker.com/engine/containers/run/)）。NCCL 在 GPU 之间没法 P2P 时会退回 SHM transport 走主机内存（[NCCL_SHM_DISABLE](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-shm-disable)），PyTorch DataLoader 的 worker 也用共享内存回传 batch，64 MB 不够就会报 bus error 或 NCCL 错误。vLLM 的 Docker 示例里加 `--ipc=host` 就是为此（[vLLM docker.md](https://github.com/vllm-project/vllm/blob/main/docs/deployment/docker.md)），或者用 `--shm-size` 调大。

### 5. TCP 基础、延迟与带宽

**延迟 vs 带宽**：发送 $n$ 字节的时间可以写成 $T = \alpha + n / B$。$\alpha$ 是固定开销（协议栈处理、网络往返），$B$ 是带宽。小消息看 $\alpha$，大消息看 $B$。这个模型和集合通信用的 α-β 模型是同一个（见 [集合通信：α-β 模型](/parallel/collective-comm#α-β-模型-带宽项与延迟项)）。

**TCP 要点**：

1. **建连**：三次握手要 1 个 RTT，再加 TLS 握手。所以客户端和网关都用连接池 / keep-alive，不为每个请求新建连接。
2. **可靠有序**：丢包重传，接收端按序交付。一个包丢了，后面的数据都要等它（队头阻塞）。
3. **Nagle 算法**：默认会攒小包再发，降低包数但增加延迟。流式推 token 这种「很多小写入、对延迟敏感」的场景要设 `TCP_NODELAY` 关掉它（[tcp(7)](https://man7.org/linux/man-pages/man7/tcp.7.html)）。主流 HTTP 服务器通常默认已设置。
4. **拷贝路径**：普通 socket 发送 = 用户缓冲 → 内核 socket 缓冲 → 网卡 DMA，接收方反过来，每一跳都要 CPU 参与。这在 100 Gb/s 以上就成了瓶颈，也是 RDMA 存在的原因。

### 6. RPC 与 gRPC

**RPC**：让调用远程函数看起来像调用本地函数，框架负责序列化参数、发送、等待、反序列化结果。

**gRPC**：默认用 Protocol Buffers 定义接口和编码消息（[Introduction to gRPC](https://grpc.io/docs/what-is-grpc/introduction/)），跑在 HTTP/2 上（[gRPC over HTTP/2](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md)），一条连接上可以多路复用很多并发调用，并支持服务端流式、客户端流式和双向流式 RPC（[Core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/)）。

在推理系统里的位置：TGI 的 Rust router 通过 gRPC 调 Python model server，SGLang Model Gateway 也有 gRPC 模式（见 [Rust](/basics/rust#_0-为什么-ai-infra-里会碰到-rust)）。它们传的是请求、token id、控制消息这类**小而频繁**的数据。KV cache、权重这类 GB 级数据不走 gRPC：protobuf 编解码和 TCP 拷贝都会成为瓶颈，走 NCCL / RDMA。

### 7. HTTP 与 SSE：LLM API 怎么流式返回

OpenAI 兼容 API 的 `stream=true` 用的是 **Server-Sent Events**：一个普通的 HTTP 响应，`Content-Type: text/event-stream`，body 不关闭，服务器每有新内容就写一段（[HTML 标准：Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)）。格式是若干 `data: ...` 行加一个空行组成一条事件：

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

`data: [DONE]` 是 OpenAI 的约定，不是 SSE 标准的一部分。服务端怎么用异步生成器产出这些帧见 [Python：生成器](/basics/python#_4-生成器与迭代器-流式输出的底层)，用 `curl -N` 可以直接看到逐条到达。

几个工程坑：

1. **代理缓冲**：nginx 默认会缓冲上游响应，用户就看到「等很久然后一次性全出来」。要关 `proxy_buffering`，或让上游返回 `X-Accel-Buffering: no` 头（[nginx proxy 模块](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering)）。
2. **状态码已发出**：流开始后不能再改 HTTP 状态码，中途出错只能在消息体里发一条错误事件（见 [流式输出](/stack/sv-stream)）。
3. **客户端断开**：服务端要能感知并 abort 请求，释放 KV block，否则显存被没人看的请求占着。
4. **为什么不用 WebSocket**：SSE 是单向的普通 HTTP，能直接穿过现有的负载均衡和代理，对「一个请求、一串回复」足够了。

### 8. RDMA：InfiniBand 与 RoCE

**RDMA**（Remote Direct Memory Access）：网卡直接读写远端机器的已注册内存，数据不经过对端 CPU，也不经过内核协议栈（kernel bypass）。和 TCP 对比：

| | TCP socket | RDMA |
| --- | --- | --- |
| 数据路径 | 用户缓冲 → 内核 → 网卡，对端反过来 | 网卡直接 DMA 用户（或 GPU）内存 |
| CPU 参与 | 每个包都要协议栈处理 | 只负责提交请求，传输由网卡完成 |
| 前提 | 无 | 内存要先注册（pin 住，让网卡知道物理地址），和第 2 节同一个原因 |
| 编程接口 | `send` / `recv` | verbs：queue pair、work request、completion queue |

**两种承载网络**：

- **InfiniBand**：从链路层起就是专门为 RDMA 设计的网络，需要 IB 交换机和子网管理器。当前主流是 NDR 400 Gb/s，例如 ConnectX-7 网卡（[NVIDIA InfiniBand Adapters](https://www.nvidia.com/en-us/networking/infiniband-adapters/)），400 Gb/s ÷ 8 = 50 GB/s 单方向。
- **RoCE v2**：把 IB 传输层封装在 UDP/IP 里跑在以太网上，可以走普通 IP 路由，目的端口 4791。IB 传输假设链路几乎不丢包，所以以太网要配成 lossless：用 PFC（按优先级暂停）加 ECN（拥塞标记）（[NVIDIA Cumulus：RoCE](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-511/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/)、[NVIDIA DOCA：RoCE](https://docs.nvidia.com/doca/sdk/RDMA+over+Converged+Ethernet)）。好处是复用以太网设施，代价是拥塞控制和调参更麻烦。

带宽的层级（数字见 [互联](/stack/hw-link) 和 [集合通信：拓扑](/parallel/collective-comm#拓扑-节点内-nvlink-跨节点-ib)）：NVLink 4 单方向 450 GB/s > PCIe Gen5 x16 约 63 GB/s > IB NDR 每网卡 50 GB/s。跨机带宽比机内低约一个数量级，所以 TP 放机内，PP / DP / PD 分离的 KV 传输才跨机。

### 9. GPUDirect：让 GPU 直接和网卡、存储、别的 GPU 对话

没有 GPUDirect 时，GPU 数据发到网络要先拷到主机内存（bounce buffer），再由网卡读走，多一次 PCIe 往返和一份主机内存占用。GPUDirect 是一组让这次中转消失的技术：

1. **GPUDirect P2P**：同机两张 GPU 直接互相读写显存（经 NVLink 或 PCIe），不经过主机内存。
2. **GPUDirect RDMA**：网卡等第三方 PCIe 设备直接 DMA 读写 GPU 显存（[GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)）。NCCL 跨机通信、PD 分离里 prefill 节点把 KV 直接写进 decode 节点的显存，都依赖它。GPU 和网卡最好挂在同一个 PCIe switch 下，否则数据要绕 CPU 根复合体，带宽打折。
3. **GPUDirect Storage**：NVMe 存储和显存之间直接 DMA，避开 CPU 的 bounce buffer（[GDS 概览](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)），用于加速权重加载和 KV 落盘。

推理里的实际组件：NVIDIA 的 NIXL 为推理框架提供点对点传输抽象，屏蔽 CPU / GPU 内存和各种存储后端（[NIXL](https://github.com/ai-dynamo/nixl)）；Mooncake 的 Transfer Engine 支持 RDMA、TCP、NVLink 等多种传输，可以聚合多张 RDMA 网卡的带宽（[Mooncake](https://github.com/kvcache-ai/Mooncake)、[论文](https://arxiv.org/abs/2407.00079)）。

## 面试追问

::: details Q：为什么 H2D 拷贝要用 pinned memory 才能和计算 overlap？
pageable 内存可能被 OS 换出或迁移，DMA 引擎不能直接读，CUDA 会先由 CPU 拷到一个内部 pinned 缓冲再 DMA，这一步对调用方是同步的。pinned 内存物理地址固定，copy engine 可以直接读，`cudaMemcpyAsync` 立即返回，拷贝在 copy engine 上执行，可以和另一个 stream 上的 kernel 同时进行。PyTorch 里要同时满足三个条件：源 tensor 是 pinned、`non_blocking=True`、拷贝和计算在不同 stream 上。
:::

::: details Q：PagedAttention 和 OS 分页有什么不一样？
思路相同（逻辑块 → 物理块的映射表、按需分配、共享 + CoW、换出），差别在实现层面。① 翻译者：OS 由 MMU 硬件 + TLB 翻译，对程序透明；PagedAttention 由 attention kernel 自己读 block table，是显式的 gather。② 粒度：OS 页 4 KiB，KV block 是 16 个 token × 所有层 × K/V，按模型可能是几百 KB 到几 MB。③ 换出：OS 换到磁盘，vLLM 抢占时可以 swap 到 CPU，也可以直接丢掉以后重算（recompute），因为 KV 可以从 token 重新算出来，而普通内存页不行。
:::

::: details Q：容器里跑多卡推理，启动时报 NCCL 错误或 bus error，可能是什么？
先看 `/dev/shm`：Docker 默认只有 64 MB。NCCL 在 GPU 间不能 P2P 时会走 SHM transport，DataLoader / 多进程框架也用共享内存，空间不够就会失败。解法是 `--ipc=host` 或 `--shm-size=16g` 这类设置。其他常见原因：容器里看不到 IB 设备（没挂 `/dev/infiniband`）、`NCCL_SOCKET_IFNAME` 选错了网卡。用 `NCCL_DEBUG=INFO` 看它实际选了哪个 transport。
:::

::: details Q：PD 分离的 KV 传输为什么不用 gRPC / HTTP？
算一笔：Llama-70B 一个 4k token 请求的 KV 是 GB 量级（按 [显存账本](/inference/memory-accounting) 的公式算），要在 decode 开始前传完。走 TCP 需要 GPU → 主机内存 → 内核 → 网卡，对端反过来，每一跳都要 CPU 拷贝，单连接很难跑满 400 Gb/s，CPU 也被占满。RDMA + GPUDirect 让网卡直接从 prefill GPU 的显存读、写进 decode GPU 的显存，不经过 CPU，也不需要序列化，接近线速。gRPC 留给控制面：告诉 decode 节点「KV 在哪、什么时候好」。
:::

::: details Q：用户反馈流式输出「卡一下然后一次性出来一大段」，怎么排查？
从服务端往外逐层看。① 引擎本身的 ITL 是否有尖峰（比如 prefill 插队、事件循环被同步函数阻塞，见 [Python：asyncio](/basics/python#_3-asyncio-事件循环)）。② 服务端是否在攒批推送（有些实现每 N 个 token 才 flush 一次）。③ 中间代理是否在缓冲：nginx 的 `proxy_buffering`、某些 CDN / API 网关会缓冲整个响应。④ TCP 层的 Nagle（`TCP_NODELAY`）。用 `curl -N` 直连引擎和经过代理各测一次，就能定位是哪一层。
:::

## 手撕

### 用条件变量实现有界阻塞队列

推理引擎的请求队列、tokenizer 线程和调度线程之间的交接都是这个结构（已实测）：

```python
import threading
from collections import deque

class BoundedQueue:
    def __init__(self, cap: int):
        self.cap, self.q = cap, deque()
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)     # 两个条件变量共用一把锁
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while len(self.q) >= self.cap:                 # while 而不是 if：防虚假唤醒
                self.not_full.wait()                       # 原子地「释放锁 + 睡眠」，醒来时重新持锁
            self.q.append(item)
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while not self.q:
                self.not_empty.wait()
            item = self.q.popleft()
            self.not_full.notify()
            return item

if __name__ == "__main__":
    bq, out = BoundedQueue(cap=4), []

    def producer():
        for i in range(100):
            bq.put(i)
        bq.put(None)                                       # 哨兵：通知消费者结束

    def consumer():
        while (x := bq.get()) is not None:
            out.append(x)

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert out == list(range(100)); print("ok")
```

追问：为什么要两个条件变量？只用一个的话，`notify` 可能唤醒同类（生产者唤醒生产者），对方检查条件不满足又睡回去，真正该醒的那个没醒；用 `notify_all` 能避免死等但会惊群。有界的意义：生产者比消费者快时提供**背压**，不会把内存撑爆，推理服务里对应的就是请求队列满了返回 429。

## 参考

- [OSTEP: Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [CUDA C++ Best Practices Guide：Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [HTML 标准：Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [Linux tcp(7)](https://man7.org/linux/man-pages/man7/tcp.7.html)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

---
title: PyTorch 内部机制
status: draft
tags: [pytorch, autograd, allocator]
difficulty: 4
order: 1
related: [/handson/torch-primitives, /framework/torch-compile, /framework/cuda-graph, /basics/python, /inference/memory-accounting]
stack: [3]
---

# PyTorch 内部机制

> dispatcher、autograd、custom op 注册、caching allocator 与碎片

## 一句话结论

PyTorch 的 Python 层只是壳。一次 `torch.add(a, b)` 先进 C++，再由 **dispatcher** 按输入 tensor 带的 key（Autograd、CUDA、CPU……）逐层分发：Autograd 那一层记下反向图，然后把调用转给 CUDA 那一层真正发 kernel。显存由 **caching allocator** 管：它在 `cudaMalloc` 上面做池化，tensor 释放后块回到池子而不还给驱动，所以 `nvidia-smi` 看到的占用（reserved）总是大于 tensor 实际用的（allocated）。写 custom op 就是往 dispatcher 的表里注册新条目。

tensor 本身的 shape / stride / storage、`view` 和 `reshape` 的区别见 [torch 原语](/handson/torch-primitives)，这里不重复。

## 推导

### 0. 一次算子调用经过哪几层

以 `c = a * b`（`a`、`b` 都在 CUDA 上，`a.requires_grad=True`）为例，按调用顺序：

1. **Python 绑定**：`Tensor.__mul__` 是自动生成的 C++ 绑定，解析参数后调用 `at::mul`。
2. **dispatcher**：算出这次调用的 dispatch key，查表找到对应 kernel。
3. **Autograd kernel**：创建反向节点 `MulBackward0`，把反向要用的 `a`、`b` 存进去，再 **redispatch**。
4. **CUDA kernel**：分配输出（走 caching allocator），把 elementwise kernel 发到当前 stream 上，立刻返回。GPU 什么时候算完，CPU 不等。

第 1–3 步全在 CPU 上，每个算子要花几微秒。这就是 decode 小 batch 会被 CPU 卡住、需要 [CUDA Graph](/framework/cuda-graph) 的根源。

### 1. dispatcher：一张按 key 索引的表

**定义**。每个算子（`aten::mul`、`aten::add`……）有一张表，下标是 **DispatchKey**，值是 kernel 函数指针。DispatchKey 是一个枚举，常见的有 `CPU`、`CUDA`、`AutogradCUDA`、`AutocastCUDA`、`Python`、`Functionalize`、`BackendSelect`（完整列表见 [c10/core/DispatchKey.h](https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h)）。key 之间有固定的优先级，Autograd 一类排在后端 key（CPU / CUDA）前面。

**怎么选 key**（Yang, 2020，[Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)）：

```python
def dispatch(op, *args):
    ks = KeySet()
    for t in tensor_args(args):
        ks |= t.key_set                 # 每个 tensor 自带 key：CUDA 上且 requires_grad → {CUDA, AutogradCUDA, ...}
    ks |= tls.included                  # 线程局部「打开」的 key，比如 tracing
    ks -= tls.excluded                  # 线程局部「屏蔽」的 key，比如 no_grad 屏蔽 Autograd
    key = ks.highest_priority()
    kernel = op.table[key]
    if kernel is FALLTHROUGH:           # 这个 key 对该算子没登记实现：跳过，换下一个
        return dispatch_with(op, ks - {key}, *args)
    return kernel(*args)
```

Autograd kernel 做完自己的事后，把 Autograd 加进 `tls.excluded` 再调一次 `dispatch`，于是第二次选到的是 CUDA。这一步叫 **redispatch**。

这张表就是 PyTorch 的扩展点：autocast、vmap、functionalization、`torch.compile` 用的 FakeTensor，都是往某个 key 上挂一层 kernel，不用改算子本身。

### 2. autograd：前向时就把反向图记好

PyTorch 的 autograd 是**动态图**：前向每执行一个算子，就在图里加一个节点（[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)）。

```python
x = torch.randn(4, requires_grad=True)
y = x * 2
z = y.sin().sum()
z.grad_fn                    # <SumBackward0>
z.grad_fn.next_functions     # ((<SinBackward0>, 0),)
```

每个节点记两样东西：指向上游节点的边（`next_functions`），以及反向公式要用的 **saved tensors**。`sin` 的导数是 `cos(y)`，所以 `SinBackward0` 要存 `y`。`backward()` 从 `z` 出发，按拓扑序调用每个节点的反向公式。

**为什么推理一定要关 autograd**：saved tensors 会把激活一直留在显存里，直到图被释放。算一下：一层 Linear 的输入是 `[N, d]`，N = 4096 个 token，d = 4096，bf16 下是 4096 × 4096 × 2 B = 32 MiB。一层有好几个这样的激活，32 层就是几个 GB，全是白占。

两种关法：

| | 做了什么 | 限制 |
|---|---|---|
| `torch.no_grad()` | 把 Autograd key 加进 `tls.excluded`，不建图、不存激活 | 无 |
| `torch.inference_mode()` | 在 no_grad 基础上，再关掉 view tracking 和 version counter 的维护（[文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)） | 这里面创建的 tensor 以后不能参与需要 autograd 的计算 |

**version counter** 是每个 tensor 上的一个整数，原地修改（`add_`、`copy_`）时加 1。反向时发现 saved tensor 的版本号变了就报错，防止反向用到被改过的值。推理里没有反向，维护它是纯开销。

### 3. custom op：往表里注册新条目

自己写的 CUDA / Triton kernel 想让 PyTorch 当成普通算子用，需要注册三样东西：

1. **schema**：名字、输入输出类型、哪些参数会被原地修改（`mutates_args`）；
2. **各后端的实现**：CUDA 上调哪个 kernel；
3. **可选**：fake 实现（只算输出的 shape / dtype，不碰数据），以及 autograd 公式。

`torch.library.custom_op` 把这三步包成装饰器（[Custom ops 教程](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)，[torch.library 文档](https://docs.pytorch.org/docs/stable/library.html)）：

```python
import torch

@torch.library.custom_op("mylib::fused_silu_mul", mutates_args=())
def fused_silu_mul(x: torch.Tensor) -> torch.Tensor:
    # 真实场景这里调自己的 CUDA / Triton kernel；输入 [N, 2h]，输出 [N, h]
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up

@fused_silu_mul.register_fake
def _(x):
    # 只描述输出长什么样。torch.compile 抓图时用 FakeTensor 跑这个函数推 shape
    return x.new_empty(x.shape[:-1] + (x.shape[-1] // 2,))
```

**为什么要 fake 实现**：Dynamo 抓图时输入是 FakeTensor（有 shape、dtype、device，没有数据），没法真跑 kernel；没有 fake 实现，compile 就只能在这里 graph break。详见 [torch.compile](/framework/torch-compile)。

**推理框架怎么用**：vLLM 把整个 attention（读写 paged KV cache、选 backend）注册成一个 custom op `vllm::unified_attention_with_output`（注册工具是 [`direct_register_custom_op`](https://github.com/vllm-project/vllm/blob/main/vllm/utils/torch_utils.py)），让 Dynamo 把它当黑盒，不去追踪里面的动态逻辑；PIECEWISE CUDA Graph 也正是在这个 op 处切图（[vLLM 文档：torch.compile integration](https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md)）。

### 4. caching allocator：为什么不直接 `cudaMalloc`

**问题**。`cudaMalloc` 要进驱动、改页表，比一次 kernel launch 慢得多；`cudaFree` 还会隐式同步整个 device。训练和推理每步都要分配释放成百上千个中间 tensor，直接调会慢到不可用。

**做法**：向驱动要大段显存（segment），切成块（block）发给 tensor；tensor 释放后块回到池子，下次分配优先复用。规则写在 [CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp) 开头的注释和常量里：

| 规则 | 数值 |
|---|---|
| 请求大小向上取整 | 512 B 的倍数 |
| 小块池 | ≤ 1 MiB 的请求从 2 MiB 的 segment 里切 |
| 中等请求 | 1–10 MiB 的请求，没有合适空闲块时申请一个 20 MiB 的 segment 再切 |
| 大请求 | > 10 MiB 的请求按 2 MiB 取整后直接 `cudaMalloc` |
| 选块 | 找**能放下的最小**空闲块（best fit），多出来的部分切开留在池里 |
| stream | 块按 stream 分开；在 stream A 上释放的块只能再分给 stream A |
| OOM 时 | 先释放没被切开的缓存块重试，还不行就释放全部这类块再试，最后才报 OOM |

**两个读数**：

- `torch.cuda.memory_allocated()`：活着的 tensor 占了多少；
- `torch.cuda.memory_reserved()`：allocator 从驱动拿了多少，约等于 `nvidia-smi` 里这个进程的占用（再加上 CUDA context 本身的几百 MB）。

两者的差就是池里的空闲块。

**碎片**：PyTorch 不能像 Java GC 那样搬动 tensor 把空隙压实，因为 tensor 的地址已经交给 kernel 了。空闲块散在各个 segment 中间就拼不成大块：

```text
segment 20 MiB: [used 4][free 6][used 4][free 6]
malloc(8 MiB): 没有 >= 8 MiB 的空闲块 -> 再 cudaMalloc 一段，或者 OOM
reserved = 20, allocated = 8, 空闲 12 但一块都放不下
```

表现就是 **reserved 远大于 allocated，却 OOM**。诊断用 `torch.cuda.memory_summary()`，或者 `torch.cuda.memory._record_memory_history()` 录下每次分配再导出 snapshot 看（[Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)）。

缓解手段：

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：segment 先映射一段虚拟地址，不够时在尾部原地扩展，不用另开新段（[CUDA semantics 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)，标注为 experimental）；
2. 固定 shape：每步分配同样大小的块，复用率接近 100%；
3. 启动时一次性预分配。这是推理框架的做法，见下一节。

**`empty_cache()` 的真实作用**：把池里**完全空闲**的 segment 还给驱动，「so that those can be used by other GPU applications」（同上文档）。正被 tensor 占着的块不动，所以它解决不了碎片。

### 5. 推理框架怎么利用这些

vLLM 启动时的显存规划（[`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) 里的 `profile_run` 和 `initialize_kv_cache`）：

1. 加载权重；
2. 用最大的 batch（`max_num_batched_tokens` 个 token）跑一次假的 forward，记下激活的峰值；
3. KV 池大小 = 总显存 × `gpu_memory_utilization`（当前默认 0.92，见 [`config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)）− 权重 − 激活峰值 − CUDA Graph 等其他开销；
4. 按这个大小**一次性**分配 KV cache 的 tensor，之后整个服务生命周期不再释放。

KV 是推理里最大、最动态的一块显存。把它从 caching allocator 的日常分配里拿出来，由 block table 在框架层面分页管理（见 [PagedAttention](/inference/kv-cache-paged-attention)），碎片问题就只剩激活这一小块。具体每部分占多少见 [显存账本](/inference/memory-accounting)。

## 面试追问

::: details Q：`torch.cuda.empty_cache()` 为什么通常不该在训练循环里调？
它把池里整段空闲的 segment `cudaFree` 还给驱动。`cudaFree` 会同步 device，下一步再分配又要重新 `cudaMalloc`，两头都慢。它也解决不了碎片，因为被占用的块位置没变。只有在真要给同卡上的别的进程腾显存时才有用。
:::

::: details Q：reserved 40 GB、allocated 25 GB，还 OOM 了，怎么查？
15 GB 的差是池里的空闲块，OOM 说明其中没有一块大到能放下这次请求，也就是碎片。步骤：(1) 看 OOM 报错里请求的大小；(2) `memory_summary()` 看大块池的空闲块分布；(3) 用 `_record_memory_history` 录 snapshot，找出是哪类变长 tensor 把 segment 切碎的。处理：开 `expandable_segments`、把变长的 shape 对齐到几个桶、或者预分配。
:::

::: details Q：`no_grad` 和 `inference_mode` 差在哪？推理该用哪个？
都不建反向图。`inference_mode` 还关掉了 view tracking 和 version counter 的维护，每个算子少一点 CPU 开销；代价是里面产生的 tensor 以后不能拿去参与要求梯度的计算。纯推理服务用 `inference_mode`；RL 训练里 rollout 出来的 tensor 之后要进训练图的，用 `no_grad` 更安全。
:::

::: details Q：多个 stream 共用一个 tensor 时，allocator 会出什么问题？
块是按「分配时的 stream」回收的。tensor 在 stream A 上分配、在 stream B 上被 kernel 使用，A 侧释放后块可能立刻被 A 上的新 tensor 复用，而 B 上的 kernel 还没读完，数据就被覆盖了。要用 `tensor.record_stream(B)` 告诉 allocator：等 B 上已发出的工作完成后才能回收这个块。
:::

::: details Q：custom op 不写 fake 实现会怎样？
eager 模式下完全正常。`torch.compile` 抓图时用 FakeTensor 推 shape，遇到没有 fake 实现的 op 推不出输出形状，只能 graph break 或报错。所以给推理框架写 kernel，fake 实现基本是必需的。
:::

## 手撕

**题 1**：用 `torch.library.custom_op` 注册一个带 autograd 的 `y = x * scale`（`scale` 是 Python float），并用 `torch.library.opcheck` 检查注册是否正确。

```python
import torch

@torch.library.custom_op("mylib::scale", mutates_args=())
def scale(x: torch.Tensor, s: float) -> torch.Tensor:
    return x * s

@scale.register_fake
def _(x, s):
    return torch.empty_like(x)

def setup_context(ctx, inputs, output):
    _, s = inputs
    ctx.s = s                         # 反向只需要标量 s，不存 x，省显存

def backward(ctx, grad_out):
    return grad_out * ctx.s, None     # 对 x 的梯度；s 不是 tensor，返回 None

scale.register_autograd(backward, setup_context=setup_context)

x = torch.randn(8, requires_grad=True, dtype=torch.float64)
torch.library.opcheck(scale, (x, 3.0))          # 检查 schema、fake、autograd 注册是否一致
torch.autograd.gradcheck(lambda t: scale(t, 3.0), (x,))
```

**题 2**：解释 `x.view()` 和 `x.reshape()` 在 dispatcher 和 storage 层面的区别。两者都是 aten 算子、都走 dispatcher；`view` 只改 shape / stride，不能满足时直接报错；`reshape` 能 view 就 view，不能就调 `contiguous()` 拷一份新 storage。stride 的推导见 [torch 原语：只改元数据 vs 会拷贝](/handson/torch-primitives#只改元数据-vs-会拷贝)。

## 参考

- Yang, [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)（2020）
- Yang, [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/)（2019，tensor / storage / dispatcher 的总览）
- [c10/core/DispatchKey.h](https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h)
- [c10/cuda/CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
- [PyTorch 文档：CUDA semantics / memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch 文档：Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch 文档：Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch 教程：Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

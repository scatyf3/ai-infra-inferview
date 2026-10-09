---
title: CUDA Graph：从一次 capture 到 vLLM 的三种图
status: draft
tags: [cuda-graph, vllm, torch-compile]
difficulty: 3
order: 2.5
related: [/framework/torch-compile, /gpu/cuda-graph-fusion, /framework/request-lifecycle, /inference/prefill-decode-roofline]
stack: [f-graph]
---

# CUDA Graph：从一次 capture 到 vLLM 的三种图

## 一句话结论

CUDA Graph 把一串 kernel 连同参数**录下来**，以后一次 launch 重放整串，省掉每个 kernel 的 CPU 侧开销。代价是录下来的东西全部固定：shape、指针地址、控制流。

vLLM 里「三种图」说的是 `cudagraph_mode` 的三种主要取值（[vLLM 文档：CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)）：

| 模式 | 录什么 | 一句话 |
|---|---|---|
| `PIECEWISE` | attention **之外**的部分，按 attention 切成很多段，每段一张图 | 什么 batch 都能用，attention 照常 eager 跑 |
| `FULL` | 整个 forward 一张图，attention 也在图里 | 最快，但 attention backend 必须支持被 capture |
| `FULL_AND_PIECEWISE`（当前默认） | 纯 decode batch 用 FULL，其余用 PIECEWISE | 两边的好处都要，代价是显存和 capture 时间最多 |

另外还有 `NONE`（关掉，调试用）和 `FULL_DECODE_ONLY`（只给纯 decode 录 FULL，其余 eager，PD 分离的 decode 实例用）。

## 1. 问题：decode 为什么会被 CPU 卡住

先说定义。PyTorch eager 模式下，每个算子都要走一遍 CPU 侧的流程：Python 解释器 → dispatcher → 选 kernel → `cudaLaunchKernel`。这段 CPU 时间记作 $c$（每个算子），GPU 真正执行的时间记作 $g$。CPU 是异步往 GPU 的队列里塞 kernel 的，所以一步 forward 的时间大致是：

$$t_\text{step} \approx \max\Big(\underbrace{N_\text{op} \cdot c}_{\text{CPU 发射}},\ \underbrace{\textstyle\sum g}_{\text{GPU 执行}}\Big)$$

CPU 发得比 GPU 跑得慢，GPU 就在两个 kernel 之间空等。

**代入数字**（量级估算，$c$ 取 10 μs 是假设值，实际随框架和 CPU 变化，用 profiler 看 kernel 之间的空隙能测出来）：decode 是 memory-bound，GPU 时间 ≈ 权重字节数 ÷ 显存带宽（推导见 [Prefill vs Decode Roofline](/inference/prefill-decode-roofline)）。H100 SXM 带宽 3.35 TB/s。

| 模型 | 权重（bf16） | GPU 时间 | 层数 $L$ × 每层算子数 | CPU 时间（$c$ = 10 μs） | 谁是瓶颈 |
|---|---|---|---|---|---|
| Llama-3-8B | 16 GB | 16 / 3350 ≈ 4.8 ms | 32 × 15 ≈ 480 | 4.8 ms | 打平 |
| 1B 模型 | 2.5 GB | ≈ 0.75 ms | 16 × 15 ≈ 240 | 2.4 ms | CPU，GPU 闲 70% |
| 8B，TP = 4 | 每卡 4 GB | ≈ 1.2 ms | 480 + 通信算子 | ≥ 4.8 ms | CPU |

tldr
- **模型越小、TP 越大，CPU 越是瓶颈**。TP 把每卡的 GPU 时间除以 TP，但每张卡的 CPU 还是要发全部算子。tldr是基本上0.x～1B是严重cpu bound，正是indep spec decoding面临的最大工程问题
- **prefill 一般不怕**：GPU 时间随 token 数线性涨（compute-bound），几千个 token 的 prefill 要几十上百 ms，CPU 那几 ms 被盖住了
- 还取决于cpu的能力，垃圾server低频cpu一般这个问题很严重

CUDA Graph 把 $N_\text{op} \cdot c$ 换成「一次 graph launch」的开销，公式里第一项几乎消失。

## 2. CUDA Graph 是什么

### capture → instantiate → replay

CUDA 提供 **stream capture**：在一条 stream 上开始录制，之后往这条 stream 发的 kernel 不执行，只被记成图里的节点（kernel 名、grid/block 大小、**参数值**，包括指针）。录完实例化成可执行图，之后每次 launch 一下就把整串 kernel 交给 GPU。

```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (layer : layers) launch_kernels(layer, stream);   // 不执行，只录
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, ...);              // 一次性的准备开销

for (step : steps) {
    copy_new_input_into_static_buffer();               // 输入必须写进录制时的那块地址
    cudaGraphLaunch(exec, stream);                     // 一次提交整串 kernel
}
```



底层只有上面这**一种**录法：stream capture。runtime API 叫 `cudaStreamBeginCapture`，driver API 叫 `cuStreamBeginCapture`，是同一个机制。`torch.compile`、`torch.cuda.graph` 都是包在它外面的层，最后都落到这里：

| 层 | 谁调用 stream capture | 你要自己管什么 |
|---|---|---|
| CUDA | 你自己调 `cudaStreamBeginCapture` / `EndCapture` / `Instantiate` | 全部：静态地址、shape、实例化、显存 |
| PyTorch 手动 | `torch.cuda.graph(g)`，内部就是 `cudaStreamBeginCapture`（[ATen/cuda/CUDAGraph.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGraph.cpp)） | 静态 buffer、`copy_`、warmup、显存池，见第 3 节 |
| `torch.compile(mode="reduce-overhead")` | Inductor 编译完，自动给结果套一层 CUDA Graph（CUDAGraph Trees） | 基本不用管：第一次运行只 warmup 不录，第二次才录；输入自动拷进静态地址；graph break 切出的每段各录一张，按执行路径组成一棵树；输入地址变了就重录（[CUDAGraph Trees 文档](https://github.com/pytorch/pytorch/blob/main/docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md)） |
| vLLM | `torch.compile` 只负责融合，不用 `reduce-overhead`；录图是自己的 `CUDAGraphWrapper` 调 `torch.cuda.graph`（[vllm/compilation/cuda_graph.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py)） | 框架管：分桶、piecewise / full，见第 4、5 节 |

`torch.compile` 里那些中间层（Dynamo、AOTAutograd、Inductor）做的是**生成更好的 kernel**（抓图、融合、生成 Triton），和录不录 CUDA Graph 是两件事。默认的 `mode="default"` 根本不录图；`reduce-overhead` 只是在编译结果外面多包一层 CUDA Graph（`torch.compile` 的 docstring：「reduces the overhead of python with CUDA graphs, useful for small batches」）。所以「用 `torch.compile` 录」和「用 CUDA 录」不是两种并列的方法，前者最终还是调用后者。

另外，CUDA 层面除了 stream capture，还能用 `cudaGraphAddKernelNode` 等 API 手动一个个加节点（上面括号里提到的那种），这才是真正的「另一种建图方式」。框架基本不用它，因为 kernel 是 cuBLAS、Triton 等库内部发的，拿不到每个节点的参数，只能靠 capture 把它们录下来。

### 录下来的东西全部固定

图里存的是**参数的值**，不是「去哪找参数」。由此推出所有限制：

| 录制时固定的 | 后果 | 推理框架怎么应对 |
|---|---|---|
| 指针 | 每次 replay 读写同一块地址 | 输入 `copy_` 进静态 buffer；输出也是静态 buffer，要保留就 `clone` |
| shape、grid 大小 | 换 shape 必须换一张图 | 按 batch 大小分桶，各录一张，实际 batch padding 到最近的桶 |
| CPU 上的代码 | 录的时候跑一次，replay 时**不会**再跑 | CPU 侧的逻辑（算 metadata、选分支）放到图外面 |
| 控制流 | `if x.sum() > 0` 这类分支只录了当时走的那条 | 分支外提，或每个分支录一张图 |

PyTorch 文档把违规分成两类（[PyTorch：CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)）：
- **直接报错**：capture 期间做 CPU-GPU 同步，比如 `.item()`、`print(tensor)`、`torch.nonzero`（输出 shape 依赖数据，要同步）
- **静默错**：动态 shape、换了地址、依赖 CPU 的工作、动态控制流。replay 照样跑，结果是错的

## 3. PyTorch 录图

```python
model = model.cuda().eval()
static_x = torch.zeros(B, D, device='cuda')          # 静态输入 buffer，地址此后不变

# 1. warmup：必须在 side stream 上跑几次，让 cuBLAS 选好算法、caching allocator 稳定下来
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3):
        model(static_x)
torch.cuda.current_stream().wait_stream(s)

# 2. capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad():
    static_y = model(static_x)                         # static_y 也是固定地址

# 3. replay
def run(x):
    static_x.copy_(x)        # 写进录制时的那块地址；写成 static_x = x 只是换了名字指向，图看不到
    g.replay()
    return static_y.clone()  # 不 clone 的话，下一次 replay 会覆盖它
```

`static_x = x` 和 `static_x.copy_(x)` 的区别就是 [modify tensor](/handson/torch-primitives#modify-tensor) 里讲的 `output = y` 那个坑：前者只改了 Python 名字，图里记的指针还指向旧 buffer。

### 显存


**1. 录图后中间结果的显存不能还回去**

不录图时，一次 forward 里每层的中间结果（比如 MLP 的 gate、up 输出）用完就还给 caching allocator，下次分配可能拿到别的地址，无所谓。

录图以后不行：图里记死了这些中间 tensor 的**地址**，每次 replay 都往同样的地址读写。这块显存要是还回去、又分给了别的 tensor，replay 就会把别人的数据覆盖掉。所以 PyTorch 在 capture 期间把所有分配都放进一个**这张图专用的池**（private pool），图活着，池就一直留着（[PyTorch：Graph memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#graph-memory-management)）。

tldr：每个图内都需要存中间结果，就像之前写的推理框架一样，中间的tensor也要preallocate

**2. 51 张图各占一个池太浪费，可以共用**

vLLM 默认录 51 个 batch 大小（见第 4 节）。每张图一个池的话，总量 ≈ 80 KiB × (1 + 2 + 4 + 8 + 16 + … + 512) = 80 KiB × 10503 ≈ **820 MiB**。

但同一时刻只会 replay 一张图：这一步 batch 是 37，就只跑 B = 40 那张。各张图的中间结果从来不会同时需要，完全可以放在**同一块**显存里：`torch.cuda.graph(g, pool=shared_pool)`。

为什么安全：PyTorch 文档说，几张图只要**互不读取对方的输出、且不同时 replay**，就可以共享一个池，并直接拿 vLLM 当例子（[Sharing memory across captures](https://docs.pytorch.org/docs/stable/notes/cuda.html#sharing-memory-across-captures)）。推理正好满足。唯一要注意的是，图的**输出**也在共享池里，下一次 replay（哪怕是另一张图）会把它覆盖，所以下一次 replay 前要把输出用掉或 `clone` 出来。文档里还有另一个条件「按 capture 的顺序 replay」，那是给训练里固定执行顺序的多张图用的；推理的 replay 顺序随 batch 大小变，靠的是前一个条件。

vllm和sglang都用torch这套graph来管理显存吗？
对，录图的显存两家都交给 PyTorch 的 graph 池，并且都用全局共享池

**3. 为什么从大到小录**

共享池里，后录的图先找池里现成的空闲块，够大就切一块用，不够才向驱动要新的。
1. **先录 B = 512**：池长到约 40 MiB。再录 496、480……，需要的块都比现成的小，切着用就行，池基本不再长。
2. **反过来先录 B = 1**：池里只有很小的块。录更大的图时现成的块不够大，只能再要新的；越往后越大，每次都可能要新的，前面那些小块又用不上。

所以 vLLM 所有图共用一个全局池，并且从大到小录（`gpu_model_runner.py` 的注释：「Capture the large shapes first so that the smaller shapes can reuse the memory pool allocated for the large shapes」）。总量约等于最大那张图的峰值，几十 MiB，而不是几百 MiB。代码见下面「手撕」里的 `reversed(self.sizes)`。

**小结：两层复用**

| 情况 | 总显存（Llama-3-8B，51 张图） |
|---|---|
| 两层都不复用 | 32 层 × 80 KiB × 10503 ≈ 26 GiB |
| 只有图内复用（层和层之间；PyTorch 默认每张图一个池） | 80 KiB × 10503 ≈ 820 MiB |
| 图内 + 图间都复用（vLLM：共享池 + 从大到小录） | 80 KiB × 512 ≈ 40 MiB |

1. 图内复用是 allocator 自动做的；图间复用要自己传 `pool=`，再配合从大到小录。
2. 上面每个 token 80 KiB 是按主要 buffer 粗估的量级，实际以 `torch.cuda.memory_stats()` 测出来的为准。

## 4. 推理里怎么套：分桶 + padding

shape 固定的约束，靠「录一组 batch 大小，运行时向上取整」解决。我们录制51个batchsize的51个图，然后运行的时候根据batch向上pad，大小分别是
1. bs=1,2,4
2. 8~256 8个一个graph
3. 256～512 16个一个graph

`vllm/config/compilation.py`：

```python
cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))
```

这里的b并非超参数的batching，而是当前step的`batch * num_step_tokens`

**padding 为什么几乎免费**：batch 13 pad 到 16，多算 3 行。decode 的时间主要花在把权重读一遍（与 batch 无关），多 3 行只多一点点激活计算。超过最大桶的 batch 退回 eager，这时 batch 已经够大，GPU 时间盖得住 CPU 开销。

**运行时拿什么查表**：录好的图很多，每一步要按一个 key 找出该 replay 哪一张。这个 key 不只是 batch 大小。vLLM V1 把所有请求的 token 拍平成一维，batch 的大小是 token 总数 `num_tokens`，而 token 数相同的 batch，attention 的形状可能完全不同：

| 这一步的 batch                                 | num_tokens    | num_reqs | uniform          | 能用的图             |
| ------------------------------------------ | ------------- | -------- | ---------------- | ---------------- |
| 37 个请求各 decode 1 个 token                   | 37 → pad 到 40 | 37       | True（每个都是 1）     | FULL             |
| 10 个请求做 spec decode，各验证 1 + 3 个 token      | 40            | 10       | True（每个都是 4）     | FULL             |
| 1 个 20 token 的 prefill chunk + 20 个 decode | 40            | 21       | False（20 和 1 混着） | PIECEWISE（默认模式下） |

Linear、MLP 这些逐 token 的算子只看 40 这个数，三行一样。attention 不一样，它还要读每个请求缓存里的全部上下文（decode 的 1 个 token 要和自己请求的几千个历史 K、V 做点积），工作量取决于每个请求的 query 和上下文各多长。

这些长度本身（`seq_lens`、`cu_seqlens`、block table）可以放进固定地址的 GPU buffer，每步 `copy_` 新值，kernel 运行时去读，录图没问题。录不住的是 **launch 之前 CPU 根据这些长度做的决定**：走 decode 还是 prefill 的 kernel、grid 开多大、split-KV 切几段。它们作为 launch 参数被记死在图里，三行各不相同（细节见第 5 节）。

todo
1.  decode 还是 prefill 的 kernel，这不是混合的吗
2. grid 开多大、split-KV 切几段 这些超参数的影响啥回事？按照之前我的经验，关掉autotune直接用一个经验参数即可...

为什么不干脆用一个大 mask 把各请求隔开、shape 只看 `num_tokens`？
1. K 那一边是所有请求的上下文拼起来。37 个请求各 2000 上下文，score 矩阵是 40 × 74000，有用的只有 37 × 2000，白算约 40 倍；KV 还按 block 散在各处。所以实际的 kernel 按 `cu_seqlens` 只算每个请求自己的那块，不用 mask 隔请求。
2. 我们有个gpu算子，来模拟这个batching，然而那个模拟是通过bug实现的


所以查表的 key 是 `BatchDescriptor`（[vllm/forward_context.py](https://github.com/vllm-project/vllm/blob/main/vllm/forward_context.py)）：

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int               # pad 之后的 token 总数，决定逐 token 算子的 shape
    num_reqs: int | None = None   # 请求数；FULL 图的 attention metadata 和它有关，PIECEWISE 图填 None（任意请求数都能用）
    uniform: bool = False         # 所有请求的 query 长度是否相同（纯 decode，或 spec decode 的 1 + k）
    has_lora: bool = False        # 开了 LoRA 会多出 LoRA 的 kernel，要另录一张图
    num_active_loras: int = 0     # 不同 adapter 的个数；可选地按它再细分，因为有的 LoRA kernel 的 grid 和它有关
```

其中 `uniform` 决定**能不能用 FULL 图**。原因在 attention kernel：大多数 backend 在 launch 前由 CPU 根据 `max_query_len` 选 decode 还是 prefill 的 kernel 路线，再按每个请求的 query 长度定 grid，录图会把这些全部冻结。
1. uniform 的 batch：请求数定了，请求 i 就占第 i·q 到 (i+1)·q 个 token，形状固定。每步变的只有各请求的 KV 长度，kernel 从 GPU buffer 里读，`copy_` 新值进去就行。
2. 非 uniform 的 batch：这一步是 20 + 20×1，下一步可能是 7 + 33×1，prefill 请求排在哪、多长一直在变，grid 和路线跟着变，冻结不住。

所以卡住的是 **kernel launch config**，不是 mask。长度、`cu_seqlens`、block table 这些是数据，放在固定地址的 buffer 里每步更新就行；被冻结的是 launch 前 CPU 定下的东西：调哪个 kernel、grid 多大、split-KV 切几段。uniform 的 batch 在请求数固定时这些每步都一样，非 uniform 的每步都变。

这不是绝对的：`ALWAYS` 级别的 backend（FlashAttention 3、Triton attention）用固定 grid、所有 metadata 都从 GPU buffer 读，混合 batch 也能录 FULL 图，`FULL` 模式下就这么用。默认的 `FULL_AND_PIECEWISE` 有意只给 uniform decode 录 FULL 图，混合 batch 走 attention 在图外 eager 跑的 PIECEWISE 图，因为里面有 prefill，GPU 时间长，剩下的 CPU 开销盖得住（[vLLM 设计文档：CUDA Graphs](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md)）。各模式和 backend 的支持等级见第 5 节。

## 5. 三种图

### attention 为什么是难点

除 attention 外，decoder 层里的算子（Linear、RMSNorm、RoPE、SiLU×乘、残差）都是**逐 token** 的：输入是 `(num_tokens, hidden)`，shape 只依赖 `num_tokens`，不关心这些 token 属于几个请求、每个请求多长。按 `num_tokens` 分桶就能全部录下来。

attention 不行。它要知道每个请求的 query 长度、KV 长度、`cu_seqlens`、block table。同样是 `num_tokens = 16`：
- 16 个请求各 decode 1 个 token
- 1 个请求 prefill 16 个 token
- 2 个 decode + 1 个 14 token 的 prefill chunk

三种情况 kernel 走的分支、grid 大小、split-KV 的切法都可能不同，而这些往往是 CPU 根据 metadata 算出来的，录不进图。能不能录取决于 attention backend 有没有把这些做成「固定 grid + 从 GPU buffer 读 metadata」。vLLM 用 `AttentionCGSupport` 标注每个 backend 的能力：

| 等级 | 含义 | 例子（vLLM 文档） |
|---|---|---|
| `ALWAYS` | 任意 batch（含 prefill/decode 混合）都能录 | FlashAttention 3、Triton Attention |
| `UNIFORM_BATCH` | 所有请求 query 长度相同时能录（含 spec decode） | FlashAttention 2、FlashMLA |
| `UNIFORM_SINGLE_TOKEN_DECODE` | 只有纯 decode（query 长度全是 1）能录 | FlashInfer、CUTLASS MLA |
| `NEVER` | 不能录 | 未标注的 backend |

### PIECEWISE：绕开 attention

做法：[torch.compile](./torch-compile) 抓出整个模型的 FX 图后，在 `splitting_ops`（默认是各种 attention op，比如 `vllm::unified_attention_with_output`）处把图切开。$L$ 层的模型切成 $L+1$ 段：

```
[embed → L0 的 QKV proj]  attn_0  [L0 的 o_proj → MLP → L1 的 QKV proj]  attn_1  ...  attn_{L-1}  [最后的 MLP → lm_head 之前]
   piece 0 (graph)        eager           piece 1 (graph)                eager                   piece L (graph)
```

每段按 `num_tokens` 各录一张图，attention 在段与段之间 eager 跑。

- 好处：**任何 batch 都能用**，prefill、decode、混合 batch 都行，因为图里只有逐 token 的算子；不挑 attention backend
- 代价：每步还剩 $L$ 次 eager attention + $L+1$ 次 graph launch，CPU 开销没消干净。图的数量是 $(L+1) \times$ 桶数，32 层 × 51 个桶 ≈ 1700 张，capture 慢（vLLM 在 capture 时把 `gc.collect` 也 patch 掉了，否则每段都 gc 一次太慢）

### FULL：整个 forward 一张图

做法：attention 也录进去，整个 forward 只剩一次 graph launch。前提是 attention backend 支持（见上表），并且 attention metadata 都写在固定地址的 GPU buffer 里，每步 `copy_` 新值进去。

- 好处：CPU 开销最低
- 代价：受 backend 能力限制。比如 FlashInfer 只能录纯 decode，混合 batch 录不了

### FULL_AND_PIECEWISE：按 batch 选（当前默认）

两套图都录。运行时看 batch：

```python
def dispatch(desc: BatchDescriptor):
    desc = pad_to_capture_size(desc)            # num_tokens 向上取整到最近的桶
    if desc in full_keys:                       # 优先级 FULL > PIECEWISE > NONE
        return FULL, desc
    pw = replace(desc, num_reqs=None, uniform=False)   # piecewise 图只看 num_tokens，其余字段抹掉再查
    if pw in piecewise_keys:
        return PIECEWISE, pw
    return NONE, desc                           # 超过最大桶，或无合适的图：eager
```

FULL 只为 `uniform=True`（纯 decode / spec decode）录，其余走 PIECEWISE。decode 是最怕 CPU 开销的阶段（见第 1 节），所以给它最快的 FULL；混合 batch 里有 prefill，GPU 时间长，PIECEWISE 剩下的那点 CPU 开销盖得住。

实现上是两层嵌套的 `CUDAGraphWrapper`：外层包整个模型（FULL），内层包每一段（PIECEWISE）。dispatcher 把选好的模式放进 forward context，每个 wrapper 只在模式和自己一致时才录或重放，否则直接调用里面的函数。所以一个 batch 只有一层生效。

### 五种取值对比

| `cudagraph_mode` | 纯 decode batch | 含 prefill 的 batch | 显存 / capture 时间 | 什么时候用 |
|---|---|---|---|---|
| `NONE` | eager | eager | 0 | 调试 |
| `PIECEWISE` | piecewise | piecewise | 中 | backend 一种都不支持；pooling 模型的默认 |
| `FULL` | full | full（backend 得是 `ALWAYS`） | 中 | 小模型、短 prompt；或要做跨 attention 的融合 |
| `FULL_DECODE_ONLY` | full | eager | 低 | PD 分离的 decode 实例，几乎没有 prefill |
| `FULL_AND_PIECEWISE` | full | piecewise | 最高 | 默认，大多数模型最快 |

**自动降级**：backend 支持不了用户选的模式时，vLLM 退到最近的能用的模式。比如选了 `FULL` 但 backend 只能录 uniform batch：开着 piecewise 编译就变成 `FULL_AND_PIECEWISE`，否则变成 `FULL_DECODE_ONLY`。

### 和 torch.compile 融合的冲突

PIECEWISE 在 attention 处切图，切开的两段之间 Inductor 看不到，**跨 attention 的融合做不了**。例子是 `AttnQuantFusionPass`：把 attention 输出的 FP8 量化融进 attention kernel。开了这类 pass 时 vLLM 设 `splitting_ops=[]` 不切图，模式只能是 `FULL` 或 `FULL_DECODE_ONLY`。torch ≥ 2.9 上可以开 `use_inductor_graph_partition=True`，让 Inductor 在自己的图里分区，编译时看到完整的图，capture 时仍然绕开 attention（实验特性）。

## 面试追问

::: details Q：CUDA Graph 只能用在 decode 吗？
老答案是「是」：prefill 的 shape 变化太多。vLLM V1 之后不对了。PIECEWISE 图只看 `num_tokens`，一个 300 token 的 prefill chunk 也能 pad 到 304 的桶走图，只有 attention 在图外。默认最大桶 512，chunked prefill 把 prefill 切成 chunk 后，很多 chunk 都落在桶里。FULL 图仍然主要给纯 decode。
:::

::: details Q：replay 前为什么必须 `copy_` 进静态 buffer？
图里记的是录制时的指针值。`static_x = x` 只是让 Python 名字指向新 tensor，图还是读旧地址，读到的是上一次的输入，不报错，结果静默错。
:::

::: details Q：图里有 `.item()` 会怎样？
capture 时直接报错：`.item()` 要等 GPU 算完把值拷回 CPU（同步），capture 期间不允许。即使绕过去，CPU 拿到的值也只是录制那一次的，replay 时不会再读。
:::

::: details Q：录 51 个桶，显存为什么没炸？
所有图共用一个显存池，从最大的桶开始录，小桶复用大桶分配过的块，中间 buffer 的总量约等于最大那张图的。代价是同一时刻只能 replay 一张图。还剩的开销是每张图本身的元数据，和 PIECEWISE 下图的数量成正比，所以 `FULL_AND_PIECEWISE` 显存最多。显存预算见 [显存账本](/inference/memory-accounting)。
:::

::: details Q：batch 13 pad 到 16，浪费了多少？
多算 3/16 的激活计算，但 decode 的时间主要是读一遍权重，和 batch 大小基本无关，所以实际慢不了多少。这也是小 batch 档位要密（1、2、4、8、16…）的原因：bucket 越小，相对浪费越大。
:::

::: details Q：为什么 TP 越大，CUDA Graph 收益越大？
每卡的 GPU 时间按 TP 缩小，每卡 CPU 要发的算子数不变（还多了 all-reduce）。TP = 4 时 8B 模型每卡 GPU 时间约 1.2 ms，CPU 发射约 5 ms，不用图 GPU 闲置 75% 左右。
:::

## 手撕

按 batch 分桶的 graph runner，就是 vLLM 的最小版本：

```python
import bisect, torch

class GraphRunner:
    def __init__(self, model, sizes, D, device='cuda'):
        self.model, self.sizes = model, sorted(sizes)
        self.max = self.sizes[-1]
        self.x = torch.zeros(self.max, D, device=device)       # 所有图共用一块输入 buffer
        self.graphs, self.outs = {}, {}
        self.pool = torch.cuda.graph_pool_handle()

        with torch.no_grad():
            s = torch.cuda.Stream()                             # warmup 在 side stream
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    model(self.x)
            torch.cuda.current_stream().wait_stream(s)

            for b in reversed(self.sizes):                      # 从大到小录，小图复用大图的池
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self.outs[b] = model(self.x[:b])            # x[:b] 是 view，地址就是 self.x 的开头
                self.graphs[b] = g

    @torch.no_grad()
    def __call__(self, x):
        n = x.shape[0]
        if n > self.max:
            return self.model(x)                                # 超过最大桶：eager
        b = self.sizes[bisect.bisect_left(self.sizes, n)]       # 向上取整到最近的桶
        self.x[:n].copy_(x)
        self.x[n:b].zero_()                                     # padding 行填 0，结果丢掉
        self.graphs[b].replay()
        return self.outs[b][:n].clone()
```

要能说出的点：
- `self.x[:b]` 是切片 view，和 `self.x` 共用 storage，所以所有图的输入地址都在同一块 buffer 里，一次 `copy_` 就能喂给任意一张图
- 从大到小录、共用 `pool`：输出 `self.outs[b]` 也在共享池里，只有当前这张图的输出可信，所以返回前要 `clone`
- padding 行的结果直接丢掉。真实框架里 attention metadata 也要 pad：padding 的请求 KV 长度设成 0、slot 指向保留的空 block（vLLM 留 block 0 给 padding），避免写坏别的请求的 KV

## 参考

- [vLLM 文档：CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)：五种 `CUDAGraphMode`、`BatchDescriptor`、dispatcher、`AttentionCGSupport` 表
- [vllm/config/compilation.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py)：`cudagraph_capture_sizes` 默认公式、`splitting_ops`
- [vllm/v1/worker/gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)：`capture_model`，从大到小录
- [vllm/compilation/cuda_graph.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py)：`CUDAGraphWrapper`、全局 graph pool
- [PyTorch：CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)：限制、warmup、共享显存池
- [CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

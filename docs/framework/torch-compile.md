---
title: torch.compile：Dynamo / AOTAutograd / Inductor
status: draft
tags: [torch-compile, dynamo, inductor]
difficulty: 4
order: 2
related: [/framework/pytorch-internals, /framework/cuda-graph, /gpu/cuda-graph-fusion, /gpu/triton, /inference/prefill-decode-roofline]
stack: [f-graph]
---

# torch.compile：Dynamo / AOTAutograd / Inductor

> dynamo 抓图、graph break、inductor 生成什么

## 一句话结论

`torch.compile` 分三段（Ansel et al., ASPLOS 2024，[PyTorch 2 论文](https://pytorch.org/assets/pytorch2-2.pdf)）：

1. **Dynamo** 在 CPython 执行字节码之前截住函数，符号执行字节码，把 tensor 操作记成一张 FX 图，并记下「这张图在什么条件下还能用」（guard）。遇到抓不了的东西就 **graph break**，把函数切成几段。
2. **AOTAutograd** 让这张图再走一遍 dispatcher，展开成 ATen 层的算子图；训练时同时 trace 出反向图。
3. **Inductor** 把 ATen 图里相邻的 pointwise / reduction 算子融合，生成 Triton（GPU）或 C++（CPU）kernel；matmul 默认还是调 cuBLAS。

推理框架用它主要拿两样：小算子融合（省 HBM 读写），以及一张干净的静态图，方便后面录 [CUDA Graph](/framework/cuda-graph)。

## 推导

### 0. 为什么要编译：eager 下每个算子都要往返一次 HBM

eager 模式一个算子一个 kernel：每个 kernel 从 HBM 读输入、算完写回 HBM，下一个 kernel 再读回来。以 RMSNorm 为例：

```python
def rms_norm(x, w, eps=1e-6):          # x: [N, d]
    var = x.pow(2).mean(-1, keepdim=True)   # kernel 1: 读 x 写 x²；kernel 2: 读 x² 写 [N, 1]
    x = x * torch.rsqrt(var + eps)          # kernel 3、4: 小 tensor；kernel 5: 读 x 写 x
    return x * w                            # kernel 6: 读 x 写 y
```

符号：$N$ 是 token 数，$d$ 是 hidden，$b$ 是每元素字节数。粗略地数，eager 有约 $k = 3$ 次「整张 $[N, d]$ 读 + 写」（pow、乘 rsqrt、乘 w），融合成一个 kernel 后只剩读一次 $x$、写一次 $y$：

$$
\text{Bytes}_\text{eager} \approx 2k \cdot N d b, \qquad \text{Bytes}_\text{fused} \approx 2 N d b
$$

代入 prefill 的数字：$N = 8192$，$d = 4096$，bf16 时 $Ndb = 64$ MiB ≈ 67 MB。H100 带宽 3.35 TB/s：

- eager：$6 \times 67 / 3.35 \times 10^{6}$ ≈ 0.12 ms
- 融合：$2 \times 67 / 3.35 \times 10^{6}$ ≈ 0.04 ms

每层两次 RMSNorm，32 层合计 7.7 ms 对 2.6 ms。这是下限估计：实际 eager 还会在中间转 fp32，字节数更多。RMSNorm 这类算子是 memory-bound 的（AI ≪ ridge，见 [roofline](/inference/prefill-decode-roofline)），省字节就是省时间。

decode 时 $N$ 只有几十，字节数可以忽略，收益变成 **kernel 数变少**：6 个 kernel 变 1 个，每层省 5 次 launch。这又回到 CUDA Graph 要解决的 CPU 开销问题。

### 1. Dynamo：在字节码层面抓图

**钩子**：[PEP 523](https://peps.python.org/pep-0523/) 允许替换 CPython 的 frame evaluation 函数。Dynamo 装上自己的函数，每次 Python 函数被调用、字节码执行之前先过它（PyTorch 2 论文 §3）。

**流程**（伪代码，按论文 §3 的描述整理）：

```python
def dynamo_eval_frame(frame):
    for guard_fn, compiled in cache.get(frame.f_code, []):
        if guard_fn(frame.f_locals):          # 命中：直接跑编译好的版本
            return compiled(frame)
    # 未命中：符号执行字节码
    graph, guards, resume_fns = symbolic_trace(frame.f_code, frame.f_locals)
    compiled = backend(graph)                  # backend 默认是 AOTAutograd + Inductor
    cache[frame.f_code].append((make_guard_fn(guards), compiled))
    return compiled(frame)
```

`symbolic_trace` 逐条模拟字节码：碰到 tensor 操作就往 FX 图里加节点；碰到普通 Python 逻辑（读 `self.config.num_heads`、`for` 循环）就当场求值、把循环展开，同时记一条 **guard**。

**guard** 是一组布尔检查，描述「这张图是在什么假设下抓的」：输入 tensor 的 dtype、device、shape（或 shape 的符号关系）、读到的 Python 常量、`nn.Module` 的属性等。论文说 guard 有 30 多种。下次调用时只要有一条 guard 不成立，就要重新抓一次（**recompile**）。同一段代码最多重编 `torch._dynamo.config.recompile_limit` 次（当前默认 8，见 [torch/_dynamo/config.py](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py)），超过就退回 eager。

### 2. graph break

**定义**：Dynamo 遇到它没法放进图里的东西时，把当前函数切开。前半段编成一张图，那条指令交回 CPython 正常执行，后半段生成一个「续接函数」，重新进 Dynamo 抓下一张图（论文 §3.8 "Graph Breaks and Continuation Functions"）。

常见原因：

1. **依赖 tensor 值的控制流**：`if x.sum() > 0:`。抓图时没有真实数据，不知道该走哪个分支；
2. `print`、日志、`.item()`、`.tolist()`：要把 GPU 上的值读回 Python；
3. 调用 Dynamo 不认识的 C 扩展或第三方库；
4. 没写 fake 实现的 custom op（见 [PyTorch 内部机制：custom op](/framework/pytorch-internals#_3-custom-op-往表里注册新条目)）。

**代价**：每多一个 break，就多一次「图 → Python → 图」的切换，而且 break 两侧的算子没法融合。break 太多，基本就退化成 eager。

**排查**：

```bash
TORCH_LOGS="graph_breaks,recompiles" python run.py   # 打印每个 break 的位置和原因、每次重编是哪条 guard 失败
```

```python
torch.compile(model, fullgraph=True)   # 出现任何 graph break 直接报错，推理框架常用
```

### 3. AOTAutograd：从 Python 图到 ATen 图

Dynamo 产出的 FX 图里是 `torch.nn.functional.silu` 这种 Python 层的调用。AOTAutograd 拿 FakeTensor 再跑一遍这张图，在 dispatcher 层把每个调用记成 ATen 算子（论文 §3.9）。顺便做三件事：

1. **decomposition**：把复杂算子拆成简单算子（如 `log2` 拆成 `log` 乘常数）。论文写作时 Inductor 用了 191 个 decomposition，这样后端只需支持一个较小的算子集；
2. **functionalization**：把原地修改（`add_`、`copy_`）改写成纯函数形式，方便后面做融合和重排；
3. **训练时 trace 反向**：把前向和反向 trace 成一张联合图，再用 min-cut 算法切开。哪些激活存下来、哪些在反向里重算，就是这一步决定的。

推理只有前向，第 3 步不发生。

### 4. Inductor：生成什么代码

Inductor 把 ATen 图降成循环级的 IR（论文写作时有 433 个算子的 lowering），然后：

1. **融合**：相邻的 pointwise 和 reduction，只要迭代空间对得上，就合进同一个 kernel；
2. **代码生成**：GPU 上生成 Triton kernel，CPU 上生成 C++/OpenMP；
3. **matmul / conv**：默认调 cuBLAS / cuDNN 这类外部库。`mode="max-autotune"` 会额外生成 Triton matmul 模板，跟 cuBLAS 现场比速度，选最快的。

上面的 RMSNorm 编译后大致是这样一个 Triton kernel（示意，真实输出用 `TORCH_LOGS=output_code` 查看）：

```python
@triton.jit
def fused_rms_norm(x_ptr, w_ptr, y_ptr, d, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                     # 一个 program 处理一行（一个 token）
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    x = tl.load(x_ptr + row * d + cols, mask=mask, other=0.).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / d + eps)    # reduction 留在寄存器里
    w = tl.load(w_ptr + cols, mask=mask)
    tl.store(y_ptr + row * d + cols, (x * r * w).to(tl.bfloat16), mask=mask)
```

`x` 从 HBM 读一次，`y` 写一次，中间结果都在寄存器里。Triton 怎么写见 [Triton](/gpu/triton)。

论文给的整体收益：A100 上 180 多个模型，推理几何平均加速 2.27×，训练 1.41×（作者自测）。

### 5. 动态 shape

**问题**：guard 默认把每个输入的 shape 当常数。推理里 batch 的 token 数每步都变，每个新 shape 都触发一次重编，几步就撞到 `recompile_limit`。

`torch.compile(dynamic=...)` 的三种取值（[torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)）：

- `None`（默认）：先按静态 shape 编；发现某个维度变了，重编时把它换成符号；
- `True`：一开始就尽量用符号 shape；
- `False`：永远按具体 shape 编。

也可以对单个维度标 `torch._dynamo.mark_dynamic(x, 0)`。

**符号 shape** 的意思是：图里的维度是一个符号 `s0`，guard 记的是 `s0` 满足的关系，不是具体值。有个例外叫 **0/1 特化**：大小为 0 或 1 的维度总会被当成常数，因为 PyTorch 很多逻辑（broadcast、是否连续）对 1 有特判（论文 §5 动态 shape 一节）。这就是为什么 batch = 1 常常会单独编一次。

### 6. vLLM 怎么用 torch.compile

vLLM V1 默认开启 compile（[vLLM 文档：torch.compile integration](https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md)）。几个和通用用法不一样的设计：

1. **只有一个动态维度**。模型类上加 [`@support_torch_compile`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/decorators.py) 装饰器，标出 `input_ids`、`positions` 的第 0 维是动态的（Llama 里写成 `dynamic_arg_dims={"input_ids": {0: "b"}, ...}`，见 [llama.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)）。输入已经被拍平成 `[num_tokens]`（原因见 [加新模型](/framework/add-model-vllm-sglang)），所以整张图唯一会变的大小就是 `num_tokens`，权重的 shape 全是常数。
2. **attention 是黑盒**。整个 attention 注册成 custom op `vllm::unified_attention_with_output`，Dynamo 不看里面。图在这里被切开（`splitting_ops`），每两次 attention 之间的部分单独交给 Inductor。$L$ 层模型切出来，去重后只有 3 种子图：第一层 attention 之前、两层 attention 之间（$L-1$ 份共用一份编译结果）、最后一层 attention 之后。切开的每一段同时也是 PIECEWISE CUDA Graph 的一段，见 [CUDA Graph：PIECEWISE](/framework/cuda-graph#piecewise-绕开-attention)。
3. **启动时编完，服务期间不再编**。所有编译在接请求之前完成，避免某个请求触发编译、卡住整个 engine。编译结果按「配置 + 模型代码 + PyTorch 版本」算 hash，存在 `~/.cache/vllm/torch_compile_cache/<hash>/`，下次启动直接加载；Dynamo 的 guard 被丢弃，不做运行时检查。
4. **只对指定 shape autotune**。默认按符号 shape 编一份通用 kernel；`compile_sizes: [1, 2, 4, 8]` 会为这几个 `num_tokens` 再各编一份全静态的版本并开 autotune。autotune 要几秒到几分钟，所以默认关闭。
5. **自定义融合 pass**。vLLM 在 Inductor 里挂了自己的 FX pass，做通用 Inductor 不会做的融合（[vLLM 文档：Fusion passes](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md)），例如：

| 融合 | 收益（作者自测，随模型和硬件变） | 默认 |
|---|---|---|
| AllReduce + RMSNorm（TP > 1） | 端到端 5–20% | Hopper / Blackwell 上 O2 开 |
| RMSNorm + FP8/FP4 量化 | 1–4% | O1 按条件开 |
| SiLU×Mul + 量化 | 1–4% | O1 按条件开 |
| Attention 输出 + 量化 | 3–7% | 默认关，且要求不切图 |

最后一行要求 attention 前后在同一张图里，和 PIECEWISE 切图冲突，取舍见 [CUDA Graph：和 torch.compile 融合的冲突](/framework/cuda-graph#和-torch-compile-融合的冲突)。

## 面试追问

::: details Q：vLLM 里 torch.compile 和 CUDA Graph 分别解决什么？能同时用吗？
compile 负责减少 kernel 数量和 HBM 往返（融合 RMSNorm、残差、激活、量化），CUDA Graph 负责消除剩下那些 kernel 的 launch 开销。两者正交：先 compile 出融合后的 kernel，再把这组 kernel capture 成 graph。vLLM V1 默认两者都开，而且切图的位置相同（都在 attention 处），见 [CUDA Graph](/framework/cuda-graph) 和 [CUDA Graph 与 fusion](/gpu/cuda-graph-fusion)。
:::

::: details Q：为什么 vLLM 敢把 Dynamo 的 guard 丢掉？
guard 的作用是保证「抓图时的假设现在仍然成立」。vLLM 控制了模型的全部输入：权重 shape 固定，唯一变的是 `num_tokens`，它已被标成符号维度；模型代码里不应有依赖 tensor 值的 Python 分支。在这个约束下 guard 几乎总是成立，每步检查一遍是纯 CPU 开销。代价是一旦模型代码违反约束（比如按 batch 大小走了不同的 Python 分支），会静默用错图。所以 vLLM 文档提供了 `unbacked` 等更保守的动态 shape 模式做取舍。
:::

::: details Q：编译时间太长怎么办？
(1) 复用编译缓存：把 `~/.cache/vllm/torch_compile_cache` 拷到部署机上，命中时跳过 Dynamo 和 Inductor；(2) 少编静态 shape，不开 `compile_sizes` 和 max-autotune；(3) 调试时先关 compile 确认问题与它无关。冷启动的大头通常还是权重加载和 CUDA Graph capture，不一定是 compile。
:::

::: details Q：Inductor 生成的 matmul 比 cuBLAS 快吗？
大多数大 shape 下 cuBLAS 更快或持平，所以默认直接调 cuBLAS。`max-autotune` 会把 Triton 模板和 cuBLAS 放在一起实测、选最快的；在 decode 这种 M 很小的瘦长 GEMM 上，Triton 模板有时能赢。vLLM 文档里有一例：`8×2048×3072` 的 mm，Triton 配置 0.0130 ms，cuBLAS 0.0160 ms（作者自测）。
:::

::: details Q：同一个模型，batch=1 和 batch=2 为什么各编一次？
0/1 特化：大小为 1 的维度会被当成常数并加一条 guard，所以 batch=1 编出来的图不能复用到 batch=2。第二次调用时 guard 失败触发重编；默认 `dynamic=None` 下，这次重编会把该维度换成符号，之后的 batch 大小就不再重编。
:::

## 手撕

给一段含数据依赖分支的代码，指出 graph break 的位置并改写：

```python
@torch.compile
def f(x):
    y = x * 2
    if y.sum() > 0:          # graph break：Python 的 if 需要具体的布尔值，要把 GPU 上的结果读回来
        return y.relu()
    return y.tanh()

# 改法 1：两个分支都算，用 torch.where 选（适合两边都很便宜的情况）
@torch.compile(fullgraph=True)
def f1(x):
    y = x * 2
    return torch.where(y.sum() > 0, y.relu(), y.tanh())

# 改法 2：torch.cond 把两个分支都编进图里，运行时只执行一边
@torch.compile(fullgraph=True)
def f2(x):
    y = x * 2
    return torch.cond(y.sum() > 0, lambda t: t.relu(), lambda t: t.tanh(), (y,))
```

验证：用 `TORCH_LOGS=graph_breaks` 跑原版能看到 break 的原因；改写后加 `fullgraph=True` 不报错，说明整段是一张图。`torch.cond` 的用法和限制见 [torch.cond 文档](https://docs.pytorch.org/docs/stable/cond.html)。

## 参考

- Ansel et al., [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://pytorch.org/assets/pytorch2-2.pdf)（ASPLOS 2024）
- [PEP 523：Adding a frame evaluation API to CPython](https://peps.python.org/pep-0523/)
- [PyTorch 文档：torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch 文档：Dynamo overview](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [PyTorch 文档：Dynamic shapes](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [PyTorch 文档：torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [vLLM 文档：torch.compile integration](https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md)
- [vLLM 博客：torch.compile in vLLM](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM 文档：Fusion torch.compile passes](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md)

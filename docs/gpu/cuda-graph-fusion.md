---
title: CUDA Graph 与 Kernel Fusion
status: draft
tags: [cuda-graph, fusion]
difficulty: 3
order: 4
related: [/framework/cuda-graph, /framework/torch-compile, /gpu/triton, /gpu/tensor-core-gemm, /inference/prefill-decode-roofline, /inference/flash-attention, /leetgpu/fused-rms-norm, /leetgpu/fused-qkv-projection-with-rope-and-kv-cache-update]
stack: [f-graph, k-fused]
---

# CUDA Graph 与 Kernel Fusion

> 本页讲 kernel fusion：融合什么、省多少、谁来融。CUDA Graph 的原理、PyTorch 用法和 vLLM 的三种图见单独一篇 [CUDA Graph：从一次 capture 到 vLLM 的三种图](/framework/cuda-graph)，这里只留一节摘要。

## 一句话结论

kernel fusion 把几个相邻的算子合成一个 kernel，中间结果留在寄存器 / shared memory 里，不写回 HBM 再读出来。它对 memory-bound 的逐元素、逐行算子（残差加、RMSNorm、SiLU × up、RoPE、量化）有效，省两样东西：中间结果的 HBM 字节，和 kernel 的个数。在 LLM 推理里算一下账会发现，逐元素融合省的字节相对权重读取很小，decode 下主要收益是**少 kernel**；真正大头的融合是 attention（FlashAttention 不把 $S \times S$ 的分数矩阵写回 HBM）。CUDA Graph 则处理剩下那些 kernel 的 CPU 发射开销，两者叠加使用。

## 推导

### 符号和一般公式

- $T$：这一步的 token 数（decode 时 = batch，prefill 时 = 这个 chunk 的 token 数）
- $d$：hidden 维；$I$：MLP 中间维
- $b$：每个元素的字节数，bf16 = 2
- BW：HBM 带宽，H100 SXM 3.35 TB/s

一个 memory-bound 算子的时间约等于它读写的字节数 ÷ BW（[roofline](/inference/prefill-decode-roofline)），再加一个和数据量无关的固定开销 $t_k$（CPU 发射、GPU 上 kernel 启动和收尾；不开 CUDA Graph 时主要是 CPU 那部分，用 nsys 量 kernel 之间的空隙，见 [Profiling](/gpu/profiling#nsys)）。$n$ 个算子融合成 1 个：

$$
\Delta t \approx \frac{\text{Bytes}_{\text{unfused}} - \text{Bytes}_{\text{fused}}}{\text{BW}} + (n - 1)\, t_k
$$

字节怎么数：不融合时，每个算子都要把输入从 HBM 读进来、把输出写回去；融合后只剩**外部输入读一次、最终输出写一次**，中间结果不落 HBM。

### 例 1：残差加 + RMSNorm

pre-norm 的 decoder 层每层两处「残差加，紧接着 norm」：$h = x + \text{attn\_out}$ 之后 $\text{Norm}(h)$ 喂给 MLP；$h' = h + \text{mlp\_out}$ 之后是下一层的 input norm。RMSNorm 的定义是 $y = h / \sqrt{\text{mean}(h^2) + \epsilon} \cdot \gamma$。

| | 读 | 写 | 合计（单位 $T d b$） |
|---|---|---|---|
| 不融合：`add` | $x$、$\text{attn\_out}$ | $h$ | 3 |
| 不融合：`rmsnorm` | $h$ | $y$ | 2 |
| **融合** | $x$、$\text{attn\_out}$ | $h$（下一次残差要用，必须写）、$y$ | **4** |

$\gamma$ 只有 $d$ 个元素，忽略。融合省 1 份 $Tdb$（20%）和 1 次 launch。

vLLM 的写法：norm 层多接一个 `residual` 参数，返回 `(normed, new_residual)`（`vllm/model_executor/models/llama.py` 的 `LlamaDecoderLayer.forward`）：

```python
if residual is None:              # 第一层：还没有残差
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:                             # 融合：residual += hidden_states; hidden_states = norm(residual)
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
```

底下的 CUDA kernel 是 `fused_add_rms_norm_kernel`（[csrc/libtorch_stable/layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu)）：一个 block 处理一行，第一遍算 $h = x + \text{residual}$、写回 residual、累加 $h^2$；block 内归约出 rms 后第二遍再读 $h$ 做归一化。一行只有 $d \times 2 = 16$ KB（$d = 8192$），第二遍读基本命中 L1 / L2，不算额外的 HBM 流量。

**代入数字**（Llama-3-70B，$d = 8192$，$L = 80$ 层，每层 2 处，共 160 处）：

| 场景 | 每处省 $Tdb$ | 160 处共省 | ÷ 3.35 TB/s | 同一步的主要开销 |
|---|---|---|---|---|
| decode，$T = 64$ | 1 MiB | 168 MB | 50 μs | 读一遍权重 140 GB ≈ 42 ms |
| prefill，$T = 8192$ | 128 MiB | 21 GB | 6.4 ms | $2PT / \text{peak} = 2 \times 70\text{e}9 \times 8192 / 989\text{e}12 \approx 1.16$ s |

两种场景下省的字节都只占 0.1%–0.5%。所以这个融合在 decode 下的意义主要是 **160 个 kernel 变成 80 个**：不开 CUDA Graph 时每个 kernel 都有微秒级的 CPU 发射开销，开了也还有 GPU 侧的启动和收尾。小模型、大 TP 时每卡 GPU 时间短，kernel 个数的影响更大（[CUDA Graph 第 1 节的表](/framework/cuda-graph#_1-问题-decode-为什么会被-cpu-卡住)）。

### 例 2：gate / up 合并 + SwiGLU

Llama 的 MLP 是 $\text{down}(\text{SiLU}(x W_g) \odot x W_u)$。两步融合：

1. **横向融合两个 GEMM**：$W_g$ 和 $W_u$ 拼成一个 $d \times 2I$ 的矩阵，一次 GEMM 出 `gate_up`（$T \times 2I$）。$x$ 只读一次、少一次 launch，而且 $N$ 翻倍，tile 更多，小 batch 时更容易把 SM 喂满（[GEMV 的并行度问题](/gpu/tensor-core-gemm#为什么-decode-的-gemv-打不满)）。vLLM 里是 `MergedColumnParallelLinear`，Q / K / V 三个投影同理合成 `QKVParallelLinear`（[vllm/model_executor/layers/linear.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)）。
2. **SiLU × up 融成一个 kernel**：

```python
# vLLM SiluAndMul.forward_native：参考实现，eager 下是两个 kernel
d = x.shape[-1] // 2
return F.silu(x[..., :d]) * x[..., d:]      # 切片是 view，不拷贝
```

| | 读 | 写 | 合计（单位 $T I b$） |
|---|---|---|---|
| 不融合：`silu` | gate | tmp | 2 |
| 不融合：`mul` | tmp、up | out | 3 |
| **融合**（`act_and_mul_kernel`） | gate、up | out | **3** |

省 40% 的字节和 1 次 launch。CUDA 实现在 [csrc/libtorch_stable/activation_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu)。

### 例 3：QKV + RoPE + 写 KV cache

attention 之前有一串小算子：QKV 投影 → 对 q、k 做 RoPE → 把 k、v 按 slot 写进 paged KV cache。vLLM 里分别是 `QKVParallelLinear`、`rotary_embedding` kernel（原地改 q、k）、`reshape_and_cache_flash` kernel；编译期的 `rope_kvcache_fusion` pass 会把 RoPE 和写 cache 合成一次调用（[vllm/compilation/passes/fusion/](https://github.com/vllm-project/vllm/tree/main/vllm/compilation/passes/fusion)）。读一次 k、旋转、直接写到 cache 的目标 slot，省掉「旋转后的 k 写回 HBM 再读出来写 cache」这一来一回。手写一遍这个融合见 [LeetGPU：Fused QKV Projection with RoPE and KV Cache Update](/leetgpu/fused-qkv-projection-with-rope-and-kv-cache-update)。

### 例 4：融进 GEMM 的 epilogue

GEMM 算完一个 $C$ tile 时，结果还在寄存器里（[三级 tiling](/gpu/tensor-core-gemm#三级-tiling)）。在写回之前顺手做 bias、激活、残差加、量化到 FP8，就省掉一个完整的逐元素 kernel 和 $C$ 的一次读写。代价是要能改 GEMM kernel：CUTLASS 有 epilogue 的扩展点；Inductor 在 `mode="max-autotune"` 时用 Triton 模板生成 matmul，把后面的 pointwise 算子融进 epilogue（Ansel et al., [PyTorch 2, ASPLOS 2024](https://docs.pytorch.org/assets/pytorch2-2.pdf)）。

量化场景特别划算：norm 之后紧接着一个 FP8 GEMM，不融合时 norm 写 bf16、quant kernel 读 bf16 写 fp8；融合后 norm 直接输出 fp8。vLLM 的 `rms_quant_fusion`、`act_quant_fusion`（SiLU × up + FP8 量化）就是这类 pass。

### 例 5：attention，收益最大的融合

不融合的 attention：$S = QK^\top$ 写回 HBM → softmax 读 $S$ 写 $P$ → $PV$ 读 $P$。一层、一个序列、$H$ 个 head，分数矩阵是 $H \cdot S_{len}^2$ 个元素。$S_{len} = 8192$、$H = 64$、bf16 时：

$$
64 \times 8192^2 \times 2\ \text{B} = 8\ \text{GiB}
$$

写一遍、读两遍（softmax、$PV$）、softmax 还要写一遍 $P$，每层几十 GB 的 HBM 流量，而 $Q$、$K$、$V$ 本身只有 $3 \times 8192 \times 8192 \times 2$ B ≈ 400 MB。FlashAttention 把三步融成一个 kernel，按 tile 用 online softmax 累加，分数矩阵从不落 HBM，HBM 流量从 $O(S_{len}^2)$ 降到 $O(S_{len} \cdot d)$ 量级（Dao et al., 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）。这里字节省的是数量级，不是 20%。详见 [FlashAttention](/inference/flash-attention)。

### 什么时候不融合，或融不了

1. **中间需要全局同步**：融合后的 kernel 里，后一步只能用本 block 算出来的东西。比如 softmax 的一行超过一个 block，要先归约出全局 max 和 sum，就得切成两个 kernel（[Triton softmax 的大 N 写法](/handson/triton-softmax#大-n-先-max-再-sum-vs-online-合并)）。
2. **把 compute-bound 的 kernel 拖慢**：往 GEMM 里塞太多 epilogue 逻辑会多占寄存器，逼着用更小的 tile，GEMM 本身的效率掉下来，得不偿失。
3. **跨通信**：TP 的 all-reduce 夹在 o_proj 和下一个 norm 之间，普通 kernel 融不过去，要专门的通信 + 计算融合 kernel（vLLM 的 `allreduce_rms_fusion` pass 调 FlashInfer 的实现）。
4. **被切图挡住**：vLLM 的 PIECEWISE CUDA Graph 在 attention 处切开，Inductor 看不到跨 attention 的算子，`attn_quant_fusion` 这类融合只能在不切图（FULL）时开，见 [CUDA Graph：和 torch.compile 融合的冲突](/framework/cuda-graph#和-torch-compile-融合的冲突)。

### 谁来做融合

| 方式 | 例子 | 特点 |
|---|---|---|
| 手写 CUDA / Triton custom op | vLLM `csrc` 下的 `fused_add_rms_norm`、`act_and_mul`、`rotary_embedding`、`reshape_and_cache_flash`、`concat_and_cache_mla_rope_fused` | 最快，但每种组合要单独写；模型代码要显式调用 |
| Inductor 自动融合 | `torch.compile` 把 pointwise、reduction、scatter 合并成少量 Triton kernel | 不用改模型；PyTorch 2 论文报告 180+ 个模型上推理几何平均 2.27× 加速（A100，作者自测）；消融实验里关掉融合和 inlining，fp16 HuggingFace 推理从 1.91× 掉到 0.80×，比 eager 还慢 |
| 框架自定义编译 pass | vLLM 在 Inductor 前做 pattern match，把识别出的子图换成上面的 custom op（`add_rms_fusion`、`rms_quant_fusion`、`act_quant_fusion`、`rope_kvcache_fusion`、`allreduce_rms_fusion` 等） | 模型代码写成朴素的 PyTorch，编译时自动换成手写 kernel |

关掉融合反而更慢的原因，论文里说得很直接：Inductor 先把大算子拆成很多原语算子（decomposition），要靠融合才能拼回原来的性能（Table 4）。

### CUDA Graph（摘要）

eager 模式下一步 forward 的时间约为 $\max(N_{op} \cdot c,\ \sum g)$：$N_{op}$ 是 kernel 个数，$c$ 是每个 kernel 的 CPU 侧开销，$g$ 是 GPU 执行时间。decode 时 $\sum g$ 只有几毫秒，CPU 那一项可能更大。

- **fusion** 减小 $N_{op}$，同时减小 $\sum g$（少读写字节）；
- **CUDA Graph** 把整串 kernel 录下来一次 launch，把 $N_{op} \cdot c$ 换成一次图启动的开销；代价是 shape、指针、控制流全部固定，所以要按 batch 分桶、输入 `copy_` 进静态 buffer。

两者正交、可以叠加：先 `torch.compile` 出融合后的 kernel，再把这组 kernel capture 成图，vLLM V1 默认两者都开。录什么、怎么分桶、FULL / PIECEWISE 怎么选，全部见 [CUDA Graph](/framework/cuda-graph)。

## 面试追问

::: details Q：残差加和 RMSNorm 融合，省多少？
不融合读写 $5Tdb$（add 读 2 写 1，norm 读 1 写 1），融合后 $4Tdb$（读 x 和 attn_out，写新的 residual 和 norm 输出），省 20% 和一次 launch。新的 residual 必须写回，因为下一处残差加还要用。70B decode、batch 64 时 160 处一共省 168 MB，约 50 μs，相对 42 ms 的权重读取可以忽略；decode 下真正的收益是 kernel 数减半。
:::

::: details Q：既然逐元素融合省的字节很少，为什么推理框架还要做？
三个原因：1. decode 时 kernel 个数直接决定 CPU 发射开销和 GPU 上的空隙，小模型、大 TP 时尤其明显；2. 融合量化（norm + FP8 quant、SiLU × up + quant）省掉的是一整个额外的 kernel 和一遍读写；3. prefill 长序列时激活大，绝对字节数可观。而真正决定性的融合是 attention，它省的是 $O(S^2)$ 的流量。
:::

::: details Q：CUDA Graph 为什么只用在 decode，不用在 prefill？
这是 vLLM V0 时代的答案：prefill 的 seq_len 每个请求都不一样，整图要为每种 shape 单独 capture。V1 的 PIECEWISE 图只录 attention 之外逐 token 的算子，只按 `num_tokens` 分桶，prefill chunk 和混合 batch 也能走图；整图（FULL）仍主要给纯 decode。详见 [CUDA Graph](/framework/cuda-graph#面试追问)。
:::

::: details Q：能不能把 RMSNorm 融进后面的 QKV GEMM？
数学上可以：$\text{Norm}(h) W = \frac{1}{\text{rms}(h)} \cdot h\,(\text{diag}(\gamma) W)$。$\gamma$ 可以离线乘进 $W$；$1/\text{rms}$ 是每行一个标量，可以挪到 GEMM 的 epilogue 里乘；$\text{rms}$ 要整行的平方和，而 GEMM 沿 K 循环时本来就会读完 $h$ 的整行，可以顺手累加。代价是要改 GEMM kernel，没法直接用 cuBLAS，而且 decode 下 GEMM 是读权重的 memory-bound，省掉的那一遍激活读写相对权重很小。
:::

::: details Q：torch.compile 和 CUDA Graph 分别解决什么？
compile 负责减少 kernel 数量、少读写中间结果（融合），CUDA Graph 负责消除剩下 kernel 的 CPU 发射开销。先 compile 再 capture。vLLM 的 PIECEWISE 模式在 attention 处切图，切点两边无法互相融合，这是两者唯一的冲突点。
:::

## 手撕

1. **写一个融合 kernel**：SiLU × up，一行一个 program，读 gate 和 up 各一次、写一次：

```python
@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0)                     # x: (T, 2I) 行主序，前 I 列是 gate，后 I 列是 up
    for c0 in range(0, I, BLOCK):              # I 很大（如 28672）时分块循环
        cols = c0 + tl.arange(0, BLOCK)
        mask = cols < I
        g = tl.load(x_ptr + row * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
        y = g * tl.sigmoid(g) * u              # SiLU(g) = g * sigmoid(g)
        tl.store(out_ptr + row * I + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

# grid = (T,)；silu_and_mul_kernel[grid](x, out, I, BLOCK=1024)
```

残差加 + RMSNorm 的融合版见 [Fused Residual Add and RMS Norm](/leetgpu/fused-rms-norm)。

2. **看 Inductor 融了什么**：

```python
def mlp_act(x):                                   # 和 SiluAndMul.forward_native 一样
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

f = torch.compile(mlp_act)
f(torch.randn(64, 2 * 14336, device="cuda", dtype=torch.bfloat16))
# TORCH_LOGS=output_code python run.py：生成的代码里 silu 和 mul 在同一个 Triton kernel 里
```

3. **CUDA Graph 手动 capture**：用 `torch.cuda.CUDAGraph` 录一个小模型的 forward 并重放，解释为什么输入要 `copy_` 进固定 buffer；按 batch 分桶的完整版见 [CUDA Graph 手撕](/framework/cuda-graph#手撕)。

## 参考

- [Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation (ASPLOS 2024)](https://docs.pytorch.org/assets/pytorch2-2.pdf)：TorchInductor 的融合、Table 4 消融
- [Ivanov et al. Data Movement Is All You Need: A Case Study on Optimizing Transformers (arXiv:2007.00072)](https://arxiv.org/abs/2007.00072)：训练 BERT 时数据搬运是主要瓶颈，靠融合等手段减少 22.91% 的数据搬运，编码器层 1.30× 加速（作者自测）
- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- vLLM 源码：[layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu)、[activation_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu)、[compilation/passes/fusion/](https://github.com/vllm-project/vllm/tree/main/vllm/compilation/passes/fusion)
- [CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [PyTorch：CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

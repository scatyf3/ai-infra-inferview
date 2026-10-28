---
title: Memory-bound Kernel 与 Kernel Fusion
status: draft
tags: [kernel, bandwidth, fusion]
difficulty: 3
order: 5
related: [/gpu/gpu-architecture, /gpu/tensor-core-gemm, /gpu/profiling, /gpu/triton, /inference/prefill-decode-roofline, /inference/flash-attention, /framework/torch-compile, /framework/cuda-graph, /handson/cuda-reduce, /handson/rmsnorm, /handson/online-softmax, /handson/top-p-sampling, /gpu/quant-gemm, /gpu/moe-grouped-gemm, /leetgpu/fused-rms-norm, /leetgpu/fused-qkv-projection-with-rope-and-kv-cache-update]
stack: [k-fused]
---

# Memory-bound Kernel 与 Kernel Fusion

> 算得再快也没用，时间等于字节除以带宽

## 一句话结论

推理里除了 GEMM 和 attention，剩下的 kernel 几乎都是 memory-bound：norm、激活、RoPE、残差加、量化、采样、写 KV cache、MoE 的排队……每个元素只做几次运算，算术强度在 1 FLOP/B 上下，离 H100 的 ridge（295 FLOP/B）差两个数量级。它们的时间下界就是

$$
t \ge \frac{\text{必须读写的字节}}{\text{HBM 带宽}}
$$

优化只有两个方向：

1. **少搬**：中间结果不落 HBM（kernel fusion），或者用更小的 dtype，把分子变小。
2. **搬满**：让每个字节都以接近峰值的带宽搬，靠四件事：访存合并、向量化、在途字节够多、block 够多。

fusion 除了省字节，还省 kernel 个数。算一下账会发现，逐元素融合省的字节相对权重读取很小，decode 下主要收益是**少 kernel**；真正大头的融合是 attention（FlashAttention 不把 $S \times S$ 的分数矩阵写回 HBM）。融合之后剩下的那些 kernel 的 CPU 发射开销，交给 [CUDA Graph](/framework/cuda-graph) 去掉，那是第 3 层（kernel → forward）的事。

衡量写得好不好，看的不是 FLOP 利用率，而是**有效带宽 = 必须读写的字节 ÷ 实际耗时**，拿它和同样字节数的 device-to-device memcpy 比。memcpy 是这块卡上搬数据的实际上限。

难度主要取决于访存模式：

| 类型 | 例子 | 一个输出要读什么 | 难点 |
|---|---|---|---|
| elementwise | 残差加、SiLU · up、RoPE、类型转换、per-tensor 量化 | 同一位置的几个输入 | 访存合并 + 向量化就够 |
| 行归约 | RMSNorm、LayerNorm、softmax、per-token 量化 | 整行 | 线程之间要归约；一行放不放得进片上，决定读一遍还是两遍 |
| 全局归约 / 选择 | 求和、argmax、top-k / top-p 采样 | 整个张量或整个词表 | 跨 block 汇总；确定性 |
| gather / scatter | embedding 查表、写 paged KV cache、MoE permute | 按索引找到的行 | 地址由索引决定，只能保证行内连续 |
| 重排 | 转置、权重 repack | 别处的元素 | 读和写总有一边不连续，要过 shared memory |

## 先算账

RMSNorm，$X$ 是 $[T, d] = [4096, 8192]$ 的 bf16：

- 字节：读 $X$ 64 MiB，写 $Y$ 64 MiB，$\gamma$ 只有 16 KiB，共 128 MiB。按 3.35 TB/s 至少要 **40 μs**。
- FLOP：每个元素平方、累加、乘 rsqrt、乘 $\gamma$，约 4 次，共 $1.3 \times 10^8$。就算全在 FP32 CUDA core 上（67 TFLOP/s）也只要 **2 μs**。

算术强度约 1 FLOP/B，时间完全由字节决定。把计算再优化一倍，耗时不变；少读写一份 $X$，耗时就少四分之一。

decode 时 $T$ 很小，问题又变了：$T = 64$ 时同一个 RMSNorm 只有 2 MiB，0.6 μs 就搬完了，比 kernel launch 本身的开销还小。这时瓶颈是 kernel 的个数：融合把 kernel 变少，剩下的发射开销靠 [CUDA Graph](/framework/cuda-graph#_1-问题-decode-为什么会被-cpu-卡住)。

## 少搬：融合

kernel fusion 把几个相邻的算子合成一个 kernel，中间结果留在寄存器 / shared memory 里，不写回 HBM 再读出来。

### 省多少：一般公式

- $T$：这一步的 token 数（decode 时 = batch，prefill 时 = 这个 chunk 的 token 数）
- $d$：hidden 维；$I$：MLP 中间维
- $b$：每个元素的字节数，bf16 = 2
- BW：HBM 带宽，H100 SXM 3.35 TB/s

一个 memory-bound 算子的时间约等于它读写的字节数 ÷ BW，再加一个和数据量无关的固定开销 $t_k$（CPU 发射、GPU 上 kernel 启动和收尾；不开 CUDA Graph 时主要是 CPU 那部分，用 nsys 量 kernel 之间的空隙，见 [Profiling](/gpu/profiling#nsys)）。$n$ 个算子融合成 1 个：

$$
\Delta t \approx \frac{\text{Bytes}_{\text{unfused}} - \text{Bytes}_{\text{fused}}}{\text{BW}} + (n - 1)\, t_k
$$

字节怎么数：不融合时，每个算子都要把输入从 HBM 读进来、把输出写回去；融合后只剩**外部输入读一次、最终输出写一次**，中间结果不落 HBM。

几个最常见的融合，每元素读写的字节（bf16 记 2 B，FP8 记 1 B），下面逐个推导：

| 融合 | 不融合 | 融合后 | 省多少 |
|---|---|---|---|
| 残差加 + RMSNorm | add 读 2 写 1，norm 读 1 写 1：10 B | 读 $x$、residual，写新 residual、norm 输出：8 B | 20% + 1 次 launch |
| SiLU(gate) · up | silu 读 1 写 1，mul 读 2 写 1：10 B | 读 gate、up，写一次：6 B | 40% + 1 次 launch |
| RMSNorm + per-token FP8 量化 | norm 读 2 写 2，量化读 2 写 1：7 B | 读 2，写 1：3 B | 57% + 1 次 launch |
| elementwise 融进 GEMM 的 epilogue | GEMM 写 $C$，elementwise 再读写 $C$ | 在寄存器里做完再写 | 省 $C$ 的一次完整读写 |
| attention（FlashAttention） | 分数矩阵 $S$、$P$ 写回再读 | 分数只在片上 | $O(S_{len}^2) \to O(S_{len} d)$ |

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

1. **横向融合两个 GEMM**：$W_g$ 和 $W_u$ 拼成一个 $d \times 2I$ 的矩阵，一次 GEMM 出 `gate_up`（$T \times 2I$）。$x$ 只读一次、少一次 launch，而且 $N$ 翻倍，tile 更多，小 batch 时更容易把 SM 喂满（[GEMV 的并行度问题](/gpu/tensor-core-gemm#面试追问)）。vLLM 里是 `MergedColumnParallelLinear`，Q / K / V 三个投影同理合成 `QKVParallelLinear`（[vllm/model_executor/layers/linear.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)）。
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

### 例 3：RMSNorm + per-token FP8 量化

norm 之后紧接着一个 FP8 GEMM 时，不融合是 norm 写 bf16、quant kernel 读 bf16 写 fp8；融合后 norm 直接输出 fp8，省掉一整个 kernel 和一遍 bf16 读写（上表 57%）。能融，是因为 per-token 量化要的 amax 也是按行算的，和 RMSNorm 的行归约在同一个 block 里，一行读进寄存器之后两件事一起做完。vLLM 的 `rms_quant_fusion`、`act_quant_fusion`（SiLU × up + FP8 量化）就是这类 pass。

### 例 4：QKV + RoPE + 写 KV cache

attention 之前有一串小算子：QKV 投影 → 对 q、k 做 RoPE → 把 k、v 按 slot 写进 paged KV cache。vLLM 里分别是 `QKVParallelLinear`、`rotary_embedding` kernel（原地改 q、k）、`reshape_and_cache_flash` kernel；编译期的 `rope_kvcache_fusion` pass 会把 RoPE 和写 cache 合成一次调用（[vllm/compilation/passes/fusion/](https://github.com/vllm-project/vllm/tree/main/vllm/compilation/passes/fusion)）。读一次 k、旋转、直接写到 cache 的目标 slot，省掉「旋转后的 k 写回 HBM 再读出来写 cache」这一来一回。手写一遍这个融合见 [LeetGPU：Fused QKV Projection with RoPE and KV Cache Update](/leetgpu/fused-qkv-projection-with-rope-and-kv-cache-update)。

### 例 5：融进 GEMM 的 epilogue

GEMM 算完一个 $C$ tile 时，结果还在寄存器里（[三级 tiling](/gpu/tensor-core-gemm#三级-tiling)）。在写回之前顺手做 bias、激活、残差加、量化到 FP8，就省掉一个完整的逐元素 kernel 和 $C$ 的一次读写。代价是要能改 GEMM kernel：CUTLASS 有 epilogue 的扩展点；Inductor 在 `mode="max-autotune"` 时用 Triton 模板生成 matmul，把后面的 pointwise 算子融进 epilogue（Ansel et al., [PyTorch 2, ASPLOS 2024](https://docs.pytorch.org/assets/pytorch2-2.pdf)）。

### 例 6：attention，收益最大的融合

不融合的 attention：$S = QK^\top$ 写回 HBM → softmax 读 $S$ 写 $P$ → $PV$ 读 $P$。一层、一个序列、$H$ 个 head，分数矩阵是 $H \cdot S_{len}^2$ 个元素。$S_{len} = 8192$、$H = 64$、bf16 时：

$$
64 \times 8192^2 \times 2\ \text{B} = 8\ \text{GiB}
$$

写一遍、读两遍（softmax、$PV$）、softmax 还要写一遍 $P$，每层几十 GB 的 HBM 流量，而 $Q$、$K$、$V$ 本身只有 $3 \times 8192 \times 8192 \times 2$ B ≈ 400 MB。FlashAttention 把三步融成一个 kernel，按 tile 用 online softmax 累加，分数矩阵从不落 HBM，HBM 流量从 $O(S_{len}^2)$ 降到 $O(S_{len} \cdot d)$ 量级（Dao et al., 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）。这里字节省的是数量级，不是 20%。融合之后 attention 变成 compute-bound，详见 [FlashAttention](/inference/flash-attention)。

### 融合的边界：什么时候融不了

1. **归约的范围**：融合后的 kernel 里，后一步需要的东西必须是同一个 block 能自己算出来的。RMSNorm 需要整行的平方和，而 GEMM 的一个 tile 只有一行里的 $BN$ 列，所以 RMSNorm 一般融不进前一个 GEMM 的 epilogue，只能和残差加、量化这些同样按行处理的步骤融在一起。softmax 的一行超过一个 block 时，要先归约出全局 max 和 sum，也得切成两个 kernel（[Triton softmax 的大 N 写法](/handson/triton-softmax#大-n-先-max-再-sum-vs-online-合并)）。
2. **把 compute-bound 的 kernel 拖慢**：往 GEMM 里塞太多 epilogue 逻辑会多占寄存器，逼着用更小的 tile，GEMM 本身的效率掉下来，得不偿失。
3. **跨通信**：TP 的 all-reduce 夹在 o_proj 和下一个 norm 之间，普通 kernel 融不过去，要专门的通信 + 计算融合 kernel（vLLM 的 `allreduce_rms_fusion` pass 调 FlashInfer 的实现）。
4. **被切图挡住**：vLLM 的 PIECEWISE CUDA Graph 在 attention 处切开，Inductor 看不到跨 attention 的算子，`attn_quant_fusion` 这类融合只能在不切图（FULL）时开，见 [CUDA Graph：和 torch.compile 融合的冲突](/framework/cuda-graph#和-torch-compile-融合的冲突)。

### 谁来做融合

| 方式 | 例子 | 特点 |
|---|---|---|
| 手写 CUDA / Triton custom op | vLLM `csrc` 下的 `fused_add_rms_norm`、`act_and_mul`、`rotary_embedding`、`reshape_and_cache_flash`、`concat_and_cache_mla_rope_fused` | 最快，但每种组合要单独写；模型代码要显式调用 |
| Inductor 自动融合 | `torch.compile` 把 pointwise、reduction、scatter 合并成少量 Triton kernel | 不用改模型；PyTorch 2 论文报告 180+ 个模型上推理几何平均 2.27× 加速（A100，作者自测）；消融实验里关掉融合和 inlining，fp16 HuggingFace 推理从 1.91× 掉到 0.80×，比 eager 还慢 |
| 框架自定义编译 pass | vLLM 在 Inductor 前做 pattern match，把识别出的子图换成上面的 custom op（`add_rms_fusion`、`rms_quant_fusion`、`act_quant_fusion`、`rope_kvcache_fusion`、`allreduce_rms_fusion` 等） | 模型代码写成朴素的 PyTorch，编译时自动换成手写 kernel |

关掉融合反而更慢的原因，论文里说得很直接：Inductor 先把大算子拆成很多原语算子（decomposition），要靠融合才能拼回原来的性能（Table 4）。Inductor 怎么抓图、生成什么代码见 [torch.compile](/framework/torch-compile)。

## 搬满：四件事

### 1. 访存合并

一个 warp 的 32 个线程同时发 load，地址挨在一起时合成几个完整的 32 B sector，否则每个线程各占一个 sector，带宽浪费几倍。规则是**让相邻线程读相邻地址**，原理见 [GPU 执行模型：coalescing](/gpu/gpu-architecture#基本功一-coalescing-读-hbm-时让-32-个地址挨在一起)。

### 2. 向量化：每个线程一次读 16 B

带宽是一样的，为什么一次读 16 B（`uint4`，8 个 bf16）比读 2 B 快？因为同样多的字节，load 指令少了 8 倍：

- 发射槽：memory-bound 的 kernel 每条 load 之后只有几条计算，指令少了，LSU 和发射槽都不再是瓶颈。
- 在途字节：每条 load 带 16 B，同样的 warp 数下路上的字节多了 8 倍，更容易满足下一条（[CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)）。

```cuda
// out = silu(gate) * up，bf16；每个线程每次处理 8 个元素（16 B）
// 假设 n 是 8 的倍数、指针 16 B 对齐，尾部另行处理
__global__ void silu_mul(const bf16* gate, const bf16* up, bf16* out, int n) {
    int stride = gridDim.x * blockDim.x * 8;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride) {
        uint4 g = *(const uint4*)(gate + i);   // 一条 16 B 的 load
        uint4 u = *(const uint4*)(up + i);
        uint4 o;
        // 拆成 8 个 bf16，转 fp32 算 silu(g) * u，再打包回 bf16
        ...
        *(uint4*)(out + i) = o;                // 一条 16 B 的 store
    }
}
```

前提是地址 16 B 对齐：每行的长度是 8 的倍数（bf16），起始地址对齐。不对齐的尾巴单独用标量处理。

### 3. 在途字节够多

按 Little 定律，要让 HBM 一直忙，每个 SM 同时要有约 7–13 KB 的请求在路上（推导和几种设计的对照见 [GPU 执行模型：Little 定律](/gpu/gpu-architecture#_4-需要多少-little-定律)）。在途字节 = 驻留 warp 数 × 每个 warp 同时在等的字节。两种凑法：

- 多挂 warp（occupancy 高）；
- 每个线程连着发几条互不依赖的 16 B load，全部发出去之后再用（ILP），occupancy 不高也够。

### 4. block 够多，别留尾巴

grid 至少要让每个 SM 都有活，最好是好几波。元素太少、block 不够的时候，前三件事都做对了也跑不满，这正是 decode 下小 kernel 的处境，只能靠融合减少 kernel 数。常见写法是 grid-stride loop：开「SM 数 × 每 SM 能驻留的 block 数」个 block，每个线程循环处理多段，block 数不随数据量变。

## 行归约：一行放不放得进片上

RMSNorm $y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$ 要先有整行的平方和，才能算每个输出。最省字节的做法是**一行只读一遍**：读进寄存器，归约出平方和，再从寄存器里算输出写回。能不能做到，看一行放不放得进一个 block 的寄存器：

| 一行多大 | 怎么分 | 例子 |
|---|---|---|
| 小（几百个元素） | 一个 warp 一行，一个 block 处理好几行 | 每个 head 的 QK-norm，$d = 128$ |
| 中（几千到一两万） | 一个 block 一行，整行留在寄存器 | RMSNorm，$d = 8192$ |
| 大（几十万） | 读两遍，或一行拆给多个 block | 对 128K 词表的 logits 做 softmax / 采样 |

$d = 8192$ 的 bf16 一行 16 KB。256 个线程，每个线程拿 32 个元素，就是 4 条 16 B 的 load。归约分两层：warp 内用 shuffle 做 5 步，8 个 warp 的结果经 shared memory 再归约一次，最后把 rsqrt 广播给所有线程。完整写法见 [手撕 RMSNorm](/handson/rmsnorm)。

一行太大放不下时只能多读几遍。softmax 朴素写法要读三遍（max、sum、归一化），[online softmax](/handson/online-softmax) 把 max 和 sum 合成一遍，总共两遍。

## 全局归约：跨 block 怎么汇总

整个张量求和、argmax 这类，每个 block 先算自己那段的部分结果，再汇总。三种汇总方式：

| 方式 | 做法 | 代价 |
|---|---|---|
| 两个 kernel | 第一个写每个 block 的部分和，第二个读回来加 | 多一次 launch；结果确定 |
| atomic | 每个 block 直接 `atomicAdd` 到结果 | 一个 kernel；但浮点加法顺序随调度变，结果每次可能不同 |
| 最后一个 block 汇总 | 每个 block 写完部分和后计数，最后到的那个负责加 | 一个 kernel，结果确定，写法复杂一点 |

推导和代码见 [手撕 CUDA Reduce](/handson/cuda-reduce)。结果不确定在 RL 里会变成训推不一致，见 [训推不一致](/posttrain/rl-train-infer-mismatch)。采样（top-k / top-p）是对每一行词表做选择，属于行归约里「一行很大」的那一档，见 [手撕 Top-p 采样](/handson/top-p-sampling)。

## gather / scatter：让不规则只发生在「哪一行」

embedding 查表 `out[t] = table[ids[t]]`、写 paged KV cache（按 `slot_mapping` 把每个 token 的 K、V 写进对应的 page）、MoE 按 expert 排队（[MoE Grouped GEMM](/gpu/moe-grouped-gemm#_1-排队-token-按-expert-分段)），地址都由一个索引数组决定。

做法是**让一个 warp（或一个 block）负责一整行**：索引只决定去哪一行，行内的 $d$ 个元素仍然连续，照样能合并、能向量化。反过来，如果让相邻线程处理不同的行，每个线程的地址都跳到别处，合并就没了。

## 重排：经过 shared memory 转一道

转置这类 kernel 按行读、按列写（或反过来），总有一边不连续。做法是一个 block 先按行合并地读一个 tile 进 shared memory，再按列从 shared memory 取、合并地写出去。shared memory 里列方向的访问会撞 bank，要加 padding 或 swizzle，见 [GPU 执行模型：手撕矩阵转置](/gpu/gpu-architecture#手撕-矩阵转置)。权重 repack（[量化 GEMM](/gpu/quant-gemm#权重离线重排)）也是重排，但在加载模型时只做一次，不在推理路径上。

## 怎么判断写得好不好

1. **先算下界**：必须读写的字节 ÷ 带宽，和实测时间比，得到有效带宽。
2. **和 memcpy 比**：同样字节数的 device-to-device 拷贝能跑到多快，就是这块卡上实际能达到的上限。离它很近，就没什么可优化的了。
3. **用 ncu 找原因**：SOL 里看 DRAM Throughput（不是 Memory Throughput）；Memory Workload Analysis 里看实际读写的字节是否比公式多（多出来的是浪费），以及每个请求用了几个 sector（访存是否合并）。见 [Profiling：ncu](/gpu/profiling#_2-实测判断-compute-bound-还是-memory-bound)。

## 面试追问

::: details Q：残差加和 RMSNorm 融合，省多少？
不融合读写 $5Tdb$（add 读 2 写 1，norm 读 1 写 1），融合后 $4Tdb$（读 x 和 attn_out，写新的 residual 和 norm 输出），省 20% 和一次 launch。新的 residual 必须写回，因为下一处残差加还要用。70B decode、batch 64 时 160 处一共省 168 MB，约 50 μs，相对 42 ms 的权重读取可以忽略；decode 下真正的收益是 kernel 数减半。
:::

::: details Q：既然逐元素融合省的字节很少，为什么推理框架还要做？
三个原因：1. decode 时 kernel 个数直接决定 CPU 发射开销和 GPU 上的空隙，小模型、大 TP 时尤其明显；2. 融合量化（norm + FP8 quant、SiLU × up + quant）省掉的是一整个额外的 kernel 和一遍读写；3. prefill 长序列时激活大，绝对字节数可观。而真正决定性的融合是 attention，它省的是 $O(S^2)$ 的流量。
:::

::: details Q：RMSNorm 能融进前一个 GEMM 的 epilogue 吗？
一般不能。RMSNorm 要整行的平方和，GEMM 的一个 tile 只有这一行的 $BN$ 列，算不出整行的结果，除非一个 block 负责整行的所有列。所以实践里 RMSNorm 和同样按行处理的残差加、per-token 量化融在一起，而不是和 GEMM 融。
:::

::: details Q：那能不能把 RMSNorm 融进后面的 QKV GEMM？
数学上可以：$\text{Norm}(h) W = \frac{1}{\text{rms}(h)} \cdot h\,(\text{diag}(\gamma) W)$。$\gamma$ 可以离线乘进 $W$；$1/\text{rms}$ 是每行一个标量，可以挪到 GEMM 的 epilogue 里乘；$\text{rms}$ 要整行的平方和，而 GEMM 沿 K 循环时本来就会读完 $h$ 的整行，可以顺手累加。代价是要改 GEMM kernel，没法直接用 cuBLAS，而且 decode 下 GEMM 是读权重的 memory-bound，省掉的那一遍激活读写相对权重很小。
:::

::: details Q：带宽是固定的，为什么向量化 load 能变快？
同样的字节，16 B 一条的 load 比 2 B 一条的少 8 倍指令。memory-bound 的 kernel 里 load 指令本身就占了大部分发射槽，指令少了不再卡在发射上；而且每条 load 带的字节多，同样多的 warp 能让更多字节在路上，更容易满足 Little 定律。
:::

::: details Q：decode 时这些小 kernel 的瓶颈是什么？
batch 小时每个 kernel 只搬几 MB，不到 1 μs 就搬完了，比 launch 的开销还小，GPU 大部分时间在等下一个 kernel 发过来。kernel 这一层能做的是融合，减少 kernel 数；剩下的 CPU 发射开销由 [CUDA Graph](/framework/cuda-graph) 在 forward 层面一次性去掉。
:::

::: details Q：求和用 atomicAdd 有什么问题？
浮点加法不满足结合律，atomic 的到达顺序由调度决定，每次运行的求和顺序不同，结果的低位可能不一样。要逐位可复现，就用两个 kernel 或「最后一个 block 汇总」，顺序固定。
:::

::: details Q：gather kernel 里，线程该怎么对应到数据？
一个 warp 或一个 block 负责一整行，线程在行内按列分。这样不规则的只是「去哪一行」，行内地址连续，访存能合并、能用 16 B 向量化。如果相邻线程负责不同的行，每个线程的地址都跳开，合并就没了。
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

残差加 + RMSNorm 的融合版见 [Fused Residual Add and RMS Norm](/leetgpu/fused-rms-norm)，一行只读一遍的 RMSNorm 见 [手撕 RMSNorm](/handson/rmsnorm)。

2. **看 Inductor 融了什么**：

```python
def mlp_act(x):                                   # 和 SiluAndMul.forward_native 一样
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

f = torch.compile(mlp_act)
f(torch.randn(64, 2 * 14336, device="cuda", dtype=torch.bfloat16))
# TORCH_LOGS=output_code python run.py：生成的代码里 silu 和 mul 在同一个 Triton kernel 里
```

## 参考

- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [CUDA C++ Best Practices Guide：Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Mark Harris：Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Ansel et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation (ASPLOS 2024)](https://docs.pytorch.org/assets/pytorch2-2.pdf)：TorchInductor 的融合、Table 4 消融
- [Ivanov et al. Data Movement Is All You Need: A Case Study on Optimizing Transformers (arXiv:2007.00072)](https://arxiv.org/abs/2007.00072)：训练 BERT 时数据搬运是主要瓶颈，靠融合等手段减少 22.91% 的数据搬运，编码器层 1.30× 加速（作者自测）
- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- vLLM 源码：[layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu)、[activation_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu)、[compilation/passes/fusion/](https://github.com/vllm-project/vllm/tree/main/vllm/compilation/passes/fusion)

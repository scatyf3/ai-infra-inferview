---
title: SM 里的计算单元：CUDA Core / Tensor Core / SFU
status: draft
tags: [cuda, sm, tensor-core, sfu]
difficulty: 3
order: 1.2
related: [/gpu/gpu-architecture, /gpu/special-instructions, /gpu/tensor-core-gemm, /gpu/profiling, /inference/flash-attention, /inference/quantization]
stack: [hw-gpu]
train: [tr-hw-gpu]
---

# SM 里的计算单元：CUDA Core / Tensor Core / SFU

> [GPU 执行模型](/gpu/gpu-architecture) 讲了 warp 怎么被调度，但没展开 `w.step()` 背后是哪块硬件在算。本页把一个 SM 拆到子分区，讲每种计算单元一个周期能做多少、一条 warp 指令要占它几个周期，以及为什么 AI 的算力几乎全在 Tensor Core 上。数字默认 H100 SXM（sm_90）。每 SM 每周期的吞吐来自 [CUDA C++ Best Practices Guide 的指令吞吐表](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)（下文简称「吞吐表」），单元个数来自 [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。

## 1. 一个子分区里有什么

H100 每个 SM 分成 4 个子分区（sub-partition，ncu 里叫 SMSP）。每个子分区有自己的 warp scheduler 和 L0 指令 cache，每周期独立发一条指令。warp 按 `warp_id % 4` 分到子分区上（[Jia et al., 2018, arXiv:1804.06826](https://arxiv.org/abs/1804.06826) 在 Volta 上实测）。每个子分区里的单元：

| 单元 | 每 SMSP | 每 SM | 执行什么 | ncu 里的 pipe |
|---|---|---|---|---|
| warp scheduler | 1 | 4 | 每周期挑一个 warp 发一条指令 | — |
| 寄存器文件 | 16384 × 32 bit = 64 KB | 256 KB | 本子分区所有 warp 的寄存器 | — |
| FP32 单元（即「CUDA core」） | 32 | 128 | FP32 add / mul / FMA，以及整数乘法 `IMAD` / `IMUL` | FMA |
| INT32 单元 | 16 | 64 | 位运算、移位、比较等其余整数指令 | ALU |
| FP64 单元 | 16 | 64 | 双精度 | FP64 |
| SFU | 4 | 16 | `ex2`、`lg2`、`rsqrt`、`rcp`、`sin`、`cos`，int ↔ float 转换 | XU |
| Tensor Core（第 4 代） | 1 | 4 | 小矩阵乘加 | Tensor |

NVIDIA 说的 CUDA core 就是表里的一条 FP32 lane，H100 SXM 的「16896 个 CUDA core」= 132 SM × 128。pipe 的分工以 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 的定义为准。SFU 的个数没有官方数字，16 / SM 是从吞吐表里 `ex2` 等指令的 16 results / 周期 / SM 推出来的。

FP64 只有数据中心卡才给这么多。GeForce 卡（sm_86 / 89）的 FP64 吞吐是 2 / 周期 / SM，只有 H100 的 1/32（吞吐表）。

## 2. 一条 warp 指令占一个单元几个周期

[GPU 执行模型](/gpu/gpu-architecture) 第 4 节里，`SM.cycle` 只要找到操作数就绪的 warp 就发射。真实情况多一个条件：**这条指令要用的单元得空着**。

```python
### conceptual
class SubPartition:              # SMSP，每个 SM 4 个
    lanes = {                    # 每周期能处理几个线程的这类指令 = 吞吐表数字 / 4
        'fma':  32,
        'alu':  16,
        'fp64': 16,
        'xu':    4,
    }
    tc_fma_per_clk = 512         # 1 个 Tensor Core，bf16 稠密（第 3 节推出来）
    busy_until = {}              # 每个单元在哪个周期之后空出来

    def cycle(self, now):
        w = 任选一个 warp：下一条指令的操作数已就绪，且它要用的单元 busy_until <= now
        if w is None:
            return               # stall：要么在等数据，要么单元被占着（ncu 叫 math pipe throttle）
        inst = w.next_inst
        self.busy_until[inst.pipe] = now + occupy(inst)
        w.issue()                # 结果要过 latency 个周期，依赖它的指令才能发

def occupy(inst):                # 一条 warp 指令占单元几个周期
    if inst.pipe == 'tensor':
        return inst.m * inst.n * inst.k // tc_fma_per_clk
    return 32 // lanes[inst.pipe]
```

代入几条常见指令：

| 指令 | 单元 | 占几个周期 |
|---|---|---|
| `FFMA`（FP32 FMA） | FMA | 32 / 32 = 1 |
| `LOP3`（三输入位运算） | ALU | 32 / 16 = 2 |
| `MUFU.EX2`（exp2） | XU | 32 / 4 = **8** |
| `mma.sync` m16n8k16 bf16 | Tensor Core | 16 × 8 × 16 / 512 = 4 |
| `wgmma` m64n256k16 bf16 | 4 个 Tensor Core（一个 warpgroup 的 4 个 warp 正好在 4 个子分区上） | 64 × 256 × 16 / (4 × 512) = **128** |

由此可以读出三件事。

1. **发射只占 scheduler 1 个周期，单元可能被占好几个周期。** 一条 `MUFU.EX2` 发出去以后，XU 要忙 8 个周期，但 scheduler 下个周期就能给 FMA 发别的指令。所以不同单元上的指令是重叠执行的，softmax 里的 exp 和 FMA 可以交错着跑。
2. **一条 `wgmma` 让 Tensor Core 忙 128 个周期。** 这 128 个周期里 scheduler 几乎没事做，所以 `wgmma` 设计成异步的：发出去之后接着发 softmax、发 TMA，最后再 `wgmma.wait_group`。FA3 的 pingpong 就是利用这段空闲（[Shah et al., 2024, arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）。
3. **吞吐之外还有延迟。** Volta 上 `FFMA` 的依赖延迟是 4 个周期（Jia et al. §4.1，作者自测）。按 Little 定律，要让 FMA 单元每周期都有活干，每个子分区要同时有 1 条/周期 × 4 周期 = **4 条互不依赖的 FFMA** 在路上。可以是 4 个 warp 各一条，也可以是 1 个 warp 自己有 4 个独立的累加器。CUDA core 上的 GEMM 每个线程算一个小 tile、同时维护很多累加器，就是在补这里的并行度（[cuda-tiled-matmul](/handson/cuda-tiled-matmul)）。

## 3. Tensor Core：算力几乎都在这里

### 一个 Tensor Core 每周期做多少

NVIDIA 不直接公布 H100 每个 Tensor Core 的数字，但可以从峰值反推。产品页的 bf16 稠密峰值是 989 TFLOPS（标的 1979 是 2:4 稀疏，除以 2）：

$$
\frac{989 \times 10^{12}}{132\ \text{SM} \times 2\ \text{FLOP/FMA} \times 1.83 \times 10^{9}\ \text{Hz}} \approx 2048\ \text{FMA / 周期 / SM}
$$

即每个 Tensor Core 每周期 512 次 bf16 FMA。这和白皮书「每 SM、同频下是 A100 的 2 倍」一致（A100 是每个 Tensor Core 256、每 SM 1024）。

这里用 1.83 GHz 而不是 H100 SXM 的最高 boost 1.98 GHz：FP32 的 67 TFLOPS 是按 1.98 GHz 算的（132 × 128 × 2 × 1.98 GHz），Tensor Core 的峰值按 1.83 GHz 算才得到整数。PCIe 版有同样的现象，NVIDIA 工程师在论坛里指出跑 Tensor Core 时的最高 boost 单独规定，比板卡最高 boost 低（[forum: How to calculate the Tensor Core FP16 performance of H100](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727/2)）。

和 FP32 单元比，每 SM 每周期 2048 对 128，**Tensor Core 是 16 倍**。

### 为什么能多出 16 倍：省的是搬操作数

乘法器本身不贵，贵的是每次乘加都要从寄存器文件读操作数。算一下每次 MAC 要读多少字节寄存器：

| 做法 | 一条指令的 MAC 数 | 从寄存器读的字节 | 每 MAC 读多少 |
|---|---|---|---|
| `FFMA` | 32（每线程 1 次） | 3 个 FP32 源操作数 × 4 B × 32 线程 = 384 B | **12 B** |
| `mma.sync` m16n8k16，bf16 输入、fp32 累加 | 2048 | A 16×16×2 B + B 16×8×2 B + C 16×8×4 B = 1280 B | **0.625 B** |
| `wgmma` m64n256k16，A、B 都在 shared memory | 262144 | 只有累加器 64×256×4 B = 64 KB | **0.25 B** |

`mma.sync` 读的 A、B 元素在 Tensor Core 内部被广播给多个乘法器反复用，每 MAC 的寄存器流量降到 `FFMA` 的约 1/19。`wgmma` 更进一步，把 A、B 移到 shared memory，寄存器只放累加器。Blackwell 的 `tcgen05.mma` 把累加器也搬进了 TMEM，寄存器基本不参与（见 [普通 SIMT 之外的指令](/gpu/special-instructions) 第 3 节）。这和 tiling 是同一个道理，只是发生在 Tensor Core 内部：一次读进来，多次使用。

### 每代翻倍，SFU 原地不动

每 SM 每周期（FP16 / BF16 稠密 FMA，SFU 数 results）：

| | V100（sm_70） | A100（sm_80） | H100（sm_90） | B200（sm_100） |
|---|---|---|---|---|
| Tensor Core | 8 × 64 = 512 | 4 × 256 = 1024 | 4 × 512 = 2048 | 4096 |
| FP32 FMA | 64 | 64 | 128 | 128 |
| SFU | 16 | 16 | 16 | 16 |
| Tensor Core / SFU | 32 | 64 | 128 | **256** |

V100、A100 的 Tensor Core 数字来自各自白皮书，B200 取自 FA4 给的 8192 FLOP / 周期 / SM（[Tri Dao, FlashAttention-4 blog](https://tridao.me/blog/2026/flash4/)），FP32 和 SFU 来自吞吐表。Tensor Core 每代翻倍，SFU 从 V100 到 B200 没变过，第 4 节的 softmax 问题就是这么来的。

### 精度越低，吞吐越高

H100 SXM 的稠密峰值，按 1.83 GHz 折算到每 SM 每周期：

| 输入类型 | 累加 | 稠密峰值 | FMA / 周期 / SM |
|---|---|---|---|
| TF32 | FP32 | 495 TFLOPS | 1024 |
| BF16 / FP16 | FP32 | 989 TFLOPS | 2048 |
| FP8（E4M3 / E5M2） | FP32（实际精度见下） | 1979 TFLOPS | 4096 |
| INT8 | INT32 | 1979 TOPS | 4096 |

位宽每减半，吞吐翻倍。所以量化不仅省显存带宽，compute-bound 的 prefill 也能直接变快（[量化](/inference/quantization)）。

### Tensor Core 里的加法不完全是 IEEE

NVIDIA 不公开 Tensor Core 内部的舍入细节。Fasi 等人在 V100、T4、A100 上用构造的输入去测，以 V100 为例：乘积求和用的是 round-towards-zero（截断），不是 IEEE 默认的就近舍入；中间的部分和不做规格化，只有最终结果规格化（[Fasi et al., 2021, PeerJ CS 7:e330](https://peerj.com/articles/cs-330)）。

这在 FP8 上变成了实际问题。DeepSeek-V3 报告 H800 上 FP8 GEMM 的累加只保留约 14 位尾数，K 很大时误差明显。他们的办法是每 128 个 K 就把 Tensor Core 的部分和取出来，到 CUDA core 上用 FP32 累加，顺便乘上分块的 scale（[DeepSeek-AI, 2024, arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §3.3.2）：

```python
acc = zeros_fp32(BM, BN)                          # 寄存器里，由 CUDA core 维护
for k0 in range(0, K, 128):
    part = wgmma_fp8(A[:, k0:k0+128], B[k0:k0+128, :])   # Tensor Core，内部约 14 位精度
    acc += part * scale_a[:, k0 // 128] * scale_b[k0 // 128]   # FFMA，完整 FP32
```

代价是每 128 个 K 多一轮 `FFMA`，FMA 单元和 Tensor Core 的活要重叠起来才不掉速。DeepGEMM 就是这么实现的。

## 4. SFU：老硬件，新瓶颈

SFU 是图形时代就有的单元。它用查表加二次插值来近似超越函数，同一套硬件在图形管线里还负责顶点属性插值（[Oberman & Siu, ARITH 2005](https://ieeexplore.ieee.org/document/1467649)）。所以它只给近似结果（PTX 里是 `ex2.approx` 这类），也一直只有每 SM 16 个。

**吞吐只在「每个元素都要算一次」时才成问题。** 对比两种用法：

| 用法 | 每次 SFU 调用对应多少 Tensor Core / FMA 的活 | 会不会成为瓶颈 |
|---|---|---|
| RMSNorm 的 `rsqrt` | 每行 1 次，对应这一行 $d$ = 4096 次 FMA | 不会 |
| attention softmax 的 `exp` | 每个 (q, k) 对 1 次，对应 $4d$ FLOP 的 matmul（$d$ = 128 时 512 FLOP） | 会 |

H100 上 exp 要花掉 matmul 一半时间的推算见 [普通 SIMT 之外的指令](/gpu/special-instructions) 第 4 节。上面的表里 Tensor Core / SFU 从 H100 的 128 涨到 B200 的 256，同样的推算在 B200 上 exp 就和 matmul 一样长了。FA4 的办法是让一部分 exp2 不走 SFU，改在 FMA 单元上用三次多项式算，比例用一个参数调（[FlashAttention-4 blog](https://tridao.me/blog/2026/flash4/)）：

```python
def exp2_soft(x):                      # 只用 FMA / ALU 单元
    n = floor(x); f = x - n            # f ∈ [0, 1)
    p = ((p3 * f + p2) * f + p1) * f + p0   # 2^f 的三次近似，3 条 FFMA（Horner）
    return as_float(as_int(p) + (n << 23))  # 把 n 加进指数位，等于乘 2^n，整数指令
```

FMA 单元每 SM 每周期 128 条 lane，就算一次 exp 要 5、6 条指令，也能分担掉一大块。但 softmax 自己的 max、减法、缩放也在 FMA 上，所以不能全挪过去。

### 类型转换也走 XU

ncu 的定义里，XU 还负责 int ↔ float 转换。吞吐表里 H100 几个相关的数（results / 周期 / SM）：float → half 和 float → bf16 都是 16，float → int32 是 16，int32 → float 是 64。

这对 W4A16 这类反量化 kernel 很关键。decode 时 batch = 1，HBM 3.35 TB/s 按 int4 算每秒送进来 6.7 × 10¹² 个权重，摊到每 SM 每周期：

$$
\frac{6.7 \times 10^{12}}{132 \times 1.83 \times 10^{9}} \approx 28\ \text{个权重 / 周期 / SM}
$$

朴素的 int → float → half 要经过 float → half 这一步，只有 16 / 周期 / SM，比 HBM 送数据还慢，kernel 会卡在 XU 上而不是带宽上。FasterTransformer 的做法是绕开转换指令：fp16 里 1024 到 2048 之间的数，指数固定、尾数就是整数偏移，所以把 4-bit 整数直接塞进 1024.0（`0x6400`）的尾数低位，再减掉 1024（[Kim et al., 2022, arXiv:2211.10017](https://arxiv.org/abs/2211.10017)）：

```python
# q 是 0..15 的 int4，一个 32-bit 寄存器里打包两个 fp16
h = as_half2(0x64006400 | (q_pair & 0x000F000F))   # = 1024 + q，1 条 LOP3，ALU 单元
w = h - half2(1024.0, 1024.0)                       # = q，1 条 HSUB2，FMA 单元，一次两个
```

两条指令都走整宽的单元，每条处理两个权重，按吞吐表是每 SM 每周期 128（LOP3：64 条 × 2）和 256（HSUB2：128 条 × 2），远高于 28。Marlin 等 int4 kernel 都沿用这个技巧（[Frantar et al., 2024, arXiv:2408.11743](https://arxiv.org/abs/2408.11743)）。

## 5. 在 ncu 里看是哪个单元满了

ncu 的 Compute Workload Analysis 会给每个 pipe 的利用率，warp state 里的 `math pipe throttle` 表示「warp 在等某个被占满的计算单元」（[Profiling](/gpu/profiling)）。常见的对应关系：

| 哪个 pipe 高 | 通常是什么 kernel | 下一步 |
|---|---|---|
| Tensor | GEMM、attention 的 matmul 部分 | 已经在峰值附近了，看能不能降精度 |
| FMA | 没用上 Tensor Core 的 matmul（fp32、形状不对齐）、elementwise 里的大量 FMA | 先确认 matmul 有没有走 Tensor Core（第 3 节） |
| XU | softmax 的 exp、大量类型转换 | exp 和 matmul 重叠（FA3）或部分软件模拟（FA4）；转换换成位运算技巧 |
| ALU | 下标计算、位运算多的反量化 | 把下标计算挪出循环，或交给 TMA |
| 都不高 | memory-bound 或延迟受限 | 回到 [GPU 执行模型](/gpu/gpu-architecture) 看 coalescing 和 occupancy |

想确认某段代码到底编译成了 `HMMA` / `HGMMA`（Tensor Core）还是 `FFMA`、`MUFU.EX2` 还是多项式，就看 SASS（`cuobjdump -sass` 或 ncu 的 Source 页）。

## 面试追问

::: details Q：H100 有 16896 个 CUDA core，是不是相当于 16896 个 CPU 核？
不是。CUDA core 只是一条 FP32 lane，没有自己的取指和 PC。能独立发指令的是 SM 子分区，H100 有 132 × 4 = 528 个，每个每周期发一条 warp 指令，相当于一个 32 宽 SIMD、16 路超线程的顺序核。单看 FP32，一个子分区和一个 AVX-512 服务器核同一量级（都是每周期 32 次 FMA）。
:::

::: details Q：Tensor Core 每 SM 的乘加吞吐是 FP32 单元的 16 倍，它省在哪？
省在搬操作数。`FFMA` 每做一次乘加要从寄存器读 3 个 4 B 操作数，共 12 B。`mma.sync` m16n8k16 一条指令 2048 次乘加只读 1280 B，每次 0.625 B，因为 A、B 的元素在 Tensor Core 内部被多个乘法器复用。`wgmma` 把 A、B 放进 shared memory，寄存器只放累加器，每次乘加只剩 0.25 B。
:::

::: details Q：FP8 GEMM 为什么还要 CUDA core 参与？
Hopper 的 Tensor Core 做 FP8 时内部累加只保留约 14 位尾数，K 一大误差就明显。DeepSeek-V3 每 128 个 K 把部分和取出来，在 CUDA core 上用 FP32 累加，同时乘上 per-block 的 scale。代价是多一轮 FFMA，要和下一轮 wgmma 重叠才不掉速。
:::

::: details Q：为什么 softmax 的 exp 在 Blackwell 上比 Hopper 更成问题？
SFU 从 V100 到 B200 一直是每 SM 每周期 16 次，Tensor Core 每代翻倍，两者之比从 H100 的 128 变成 B200 的 256。head dim 128 时，H100 上 exp 已经要花 matmul 一半的时间，B200 上就一样长了。FA4 把一部分 exp2 改成在 FMA 单元上用三次多项式算。
:::

## 参考

- [CUDA C++ Best Practices Guide：Throughput of Native Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)：各类指令每 SM 每周期的吞吐，含类型转换
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：每 SM 的 FP32 / INT32 / FP64 / Tensor Core 个数，Tensor Core 同频 2× A100
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：FMA / ALU / XU / Tensor 各 pipe 的分工，`math pipe throttle`
- Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*, [arXiv:1804.06826](https://arxiv.org/abs/1804.06826)：warp 到 scheduler 的映射，指令延迟
- Fasi, Higham, Mikaitis, Pranesh, *Numerical behavior of NVIDIA tensor cores*, [PeerJ CS 7:e330](https://peerj.com/articles/cs-330), 2021
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §3.3.2：FP8 累加精度与提升到 CUDA core
- Oberman & Siu, *A High-Performance Area-Efficient Multifunction Interpolator*, [ARITH 2005](https://ieeexplore.ieee.org/document/1467649)：SFU 的查表 + 二次插值
- Shah et al., *FlashAttention-3*, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)；Zadouri et al., *FlashAttention-4*, [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) 与 [blog](https://tridao.me/blog/2026/flash4/)
- Kim et al., *Who Says Elephants Can't Run*, [arXiv:2211.10017](https://arxiv.org/abs/2211.10017)：int → fp16 的 magic number 转换
- Frantar et al., *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*, [arXiv:2408.11743](https://arxiv.org/abs/2408.11743)

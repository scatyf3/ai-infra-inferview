---
title: FSDP2 与 DTensor：PyTorch 原生分布式训练是怎么实现的
status: draft
tags: [fsdp2, dtensor, device-mesh, pytorch, hsdp, tensor-parallel]
difficulty: 4
order: 4.5
related: [/parallel/zero-fsdp, /parallel/comm-overlap, /framework/pytorch-internals, /train/parallel-config, /parallel/megatron-tp, /posttrain/training-memory, /train/fault-tolerance, /parallel/collective-comm]
stack: []
train: [tr-p-dp, tr-c-config]
---

# FSDP2 与 DTensor：PyTorch 原生分布式训练是怎么实现的

> DeviceMesh 管「哪些卡一组」，DTensor 管「一个张量怎么分在这些卡上」，FSDP2 只是一组 hook：按 module 把 DTensor 参数 all-gather 回来、用完再切回去

## 一句话结论

PyTorch 原生分布式训练分三层：**DeviceMesh** 把卡号排成多维数组，每一维对应一个通信组；**DTensor** 是「一个逻辑上的全局张量 + 每个 mesh 维度上的 placement（`Shard(dim)` / `Replicate()` / `Partial()`）」，placement 之间的转换就是集合通信（Shard→Replicate 是 all-gather，Partial→Replicate 是 all-reduce，Partial→Shard 是 reduce-scatter）；**FSDP2**（`fully_shard`）把每个参数沿 dim 0 切成 DTensor，在 module 的 forward / backward 前后挂 hook 做 all-gather 和 reduce-scatter。和 FSDP1 的最大区别是不再把一组参数拼成 `FlatParameter`，换来更简单的冻结参数、state dict 和 TP 组合，以及不依赖 `recordStream` 的确定性显存；torchtitan 在 Llama-2-7B 上报告显存平均低 7%、吞吐高 1.5%（作者自测）。ZeRO-3 本身的显存和通信账见 [ZeRO / FSDP](/parallel/zero-fsdp)，这页只讲 PyTorch 是怎么实现的。

## 推导

### 0. 记号

| 符号 | 含义 | 例子取值 |
|---|---|---|
| $\Psi$ | 参数量 | $7\times10^9$ |
| $L$ | Transformer 层数 | 32 |
| $P_\ell$ | 一层（一个 `fully_shard` 单元）的参数量，取 $\Psi / L$ | $2.19\times10^8$ |
| $N$ | FSDP 的分片数（`dp_shard` 那一维的卡数） | 8 |
| $R$ | HSDP 的副本数（`dp_replicate` 那一维） | 4 |
| $t$ | TP 度 | 8 |
| $k$ | gradient accumulation 的 micro-batch 数 | 4 |
| $\text{BW}$ | 单方向链路带宽 | NVLink 450 GB/s，跨节点 IB 50 GB/s |
| $\alpha$ | 每次消息的固定延迟 | 5 µs |
| $T_{\text{tok}}$ | 每张卡（有 TP 时是每个 TP 组）每步的 token 数 | 32768 |
| $b$ | 每个元素的字节数 | bf16 为 2，fp32 为 4 |
| $\text{peak}\cdot\text{MFU}$ | 实际算力 | H100 bf16 dense 989 TFLOP/s × 50% = 494 TFLOP/s |

单位：GB = $10^9$ 字节，MB = $10^6$ 字节。带宽和 $\alpha$ 沿用 [计算通信 Overlap](/parallel/comm-overlap) 的取值；all-gather / reduce-scatter 的时间用 α-β 模型 $(N-1)\alpha + \frac{N-1}{N}\cdot\frac{M}{\text{BW}}$，$M$ 是完整结果的字节数（推导见 [集合通信](/parallel/collective-comm)）。API 以当前 PyTorch 稳定版文档为准（`torch.distributed.fsdp.fully_shard`、`torch.distributed.tensor`）。

### 1. DeviceMesh：多维的进程组

**定义**：`init_device_mesh(device_type, mesh_shape, mesh_dim_names=...)` 把 `world_size` 个 rank 按行优先排成形状为 `mesh_shape` 的数组，并为每一维建好通信组。**最后一维是最内层**，相邻 rank 落在这一维上。

例：8 张卡，`init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))`：

```text
mesh = [[0, 1, 2, 3],
        [4, 5, 6, 7]]
rank 5:  mesh["tp"] -> [4, 5, 6, 7]   (同一行)
         mesh["dp"] -> [1, 5]         (同一列)
```

（在本机用 gloo 后端起 8 个进程验证过。）

1. `mesh["tp"]` 取子 mesh，返回的是**当前 rank 所在**的那一组；`mesh["tp"].get_group()` 拿到对应的 ProcessGroup。子 mesh 复用父 mesh 建好的通信器，不重新建（[DeviceMesh recipe](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)）。
2. 多维可以一起取：`mesh["dp_replicate", "dp_shard"]` 是一个 2D 子 mesh，HSDP 用它（第 6 节）。
3. 哪个维度放最内层就是 rank 排布问题：通信最频繁的 TP 放最后一维，落在同一台机器的 NVLink 上。公式和 Llama 3 的 [TP, CP, PP, DP] 顺序见 [并行配置](/train/parallel-config)。

### 2. DTensor：全局张量 + placement

**定义**：DTensor 是 `torch.Tensor` 的子类，存三样东西：本地分片（`to_local()`）、DeviceMesh、每个 mesh 维度一个 placement（[DTensor 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html)）。

| placement | 含义 | 2 卡例子：逻辑张量 $X$ 是 `[[0,1],[2,3],[4,5],[6,7]]` |
|---|---|---|
| `Shard(0)` | 沿 dim 0 按 `torch.chunk` 切，每卡一段 | rank 0：`[[0,1],[2,3]]`；rank 1：`[[4,5],[6,7]]` |
| `Replicate()` | 每卡一份完整的 | 两卡都是完整的 $X$ |
| `Partial()` | 每卡一个加数，逻辑值是各卡**之和**（还没做的归约） | rank 0：$A_0$，rank 1：$A_1$，逻辑值 $A_0 + A_1$ |

`Partial` 从哪来：行切的矩阵乘。$A = \begin{bmatrix}0&1&2&3\\4&5&6&7\end{bmatrix}$ 按列（`Shard(1)`）切、$B = \begin{bmatrix}0&1\\2&3\\4&5\\6&7\end{bmatrix}$ 按行（`Shard(0)`）切，每卡算本地乘积：rank 0 得 $\begin{bmatrix}2&3\\10&19\end{bmatrix}$，rank 1 得 $\begin{bmatrix}26&31\\66&79\end{bmatrix}$，两者之和才是 $AB = \begin{bmatrix}28&34\\76&98\end{bmatrix}$。

不均匀时按 `torch.chunk` 的语义，最后几段可以短一些：长度 3 的向量切 2 卡是 `[0, 1]` 和 `[2]`。

**redistribute：placement 的转换就是集合通信**（同一文档）：

| 从 → 到 | 集合通信 | 上例的结果 |
|---|---|---|
| `Shard(d)` → `Replicate()` | all-gather | 两卡都拿到 4×2 的 $X$ |
| `Partial()` → `Replicate()` | all-reduce | 两卡都是 `[[28,34],[76,98]]` |
| `Partial()` → `Shard(0)` | reduce-scatter | rank 0：`[28, 34]`；rank 1：`[76, 98]` |
| `Replicate()` → `Shard(d)` | 无通信，本地切片 | |
| `Shard(i)` → `Shard(j)` | all-to-all | |

（上面三个例子用真实 DTensor 在本机 gloo 上跑过，结果一致。）`full_tensor()` 就是 `redistribute` 到全 `Replicate` 再 `to_local()`。

**sharding propagation**：每个算子有一条规则，由输入的 placement 推出输出的 placement，必要时先对输入 redistribute。例：`Replicate() @ Shard(1)` → `Shard(1)`，不通信（列切）；`Shard(1) @ Shard(0)` → `Partial()`，也不通信，归约推迟到有人要 `Replicate` 时才做（行切）。这两条正是 Megatron TP 的列切 + 行切、一层只要一次 all-reduce 的原因，见 [Megatron TP](/parallel/megatron-tp)。

### 3. FSDP1 → FSDP2 改了什么

**FSDP1**：一个 wrap 单元里的所有参数先展平、拼成一个一维的 `FlatParameter`，padding 后切成 $N$ 段。一段分片可能横跨好几个参数的边界。

**FSDP2**：每个参数**各自**沿 dim 0 切成 `Shard(0)` 的 DTensor（per-parameter sharding）；`model.named_parameters()` 返回原来的名字和 DTensor 参数，优化器和梯度裁剪看到的就是 DTensor（[torchtitan FSDP.md](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/FSDP.md)）。

| | FSDP1（`FlatParameter`） | FSDP2（per-parameter DTensor） |
|---|---|---|
| 冻结参数 | `use_orig_params=False` 时一个单元内要么全冻结、要么全可训练；`True` 时可以混，但梯度显存按没冻结算（FSDP1 docstring） | 每个参数单独标 `requires_grad`，LoRA 的冻结底座和 adapter 可以放进同一个单元 |
| dtype | 拼成一个张量，必须同一 dtype | 冻结参数可以是别的 dtype（all-gather 按字节拼）；同一单元里**可训练**参数的原始 dtype 仍要一致（源码 `_init_mp_dtypes` 的检查） |
| state dict | 分片不对齐参数边界，取每个参数的分片要额外通信 | `model.state_dict()` 直接是分片的 DTensor，不通信 |
| 和 TP 组合 | 参数被展平拼接，单个参数自己的切法（比如 TP 切的是哪一维）不好单独表达 | TP 之后参数已经是 DTensor，FSDP 在它上面再切一维（第 6 节） |
| 显存 | 靠 `recordStream` + CPU 同步（`limit_all_gathers`）限制在途的 all-gather | 不用 `recordStream`，用 event 在确定的时刻释放，不需要 CPU 同步 |
| 大模型初始化 | meta device 上建模型后要 `param_init_fn` / `reset_parameters` 在切之前物化 | 先在 meta 上 `fully_shard`，再 `to_empty(device="cuda")`，只分配分片 |

**为什么 `recordStream` 让显存不确定**：all-gather 在通信 stream 上分配输出、在计算 stream 上使用。`record_stream` 告诉 caching allocator「等计算 stream 用完再回收」（机制见 [PyTorch 内部机制](/framework/pytorch-internals)），但 allocator 是在**之后某次分配时**才去查 event，CPU 跑得越超前，块被扣留得越久，峰值显存取决于 CPU 和 GPU 的相对速度。FSDP1 只能用 `limit_all_gathers=True` 让 CPU 等待，把在途的 all-gather 限制在两个单元（FSDP1 docstring）。FSDP2 改为由下一个单元在拷贝完成后显式释放上一个单元的 all-gather 输出（源码 `_fsdp_param_group.py` 里的 Note），释放时刻是确定的。

**代价**：per-parameter 切法下，一个单元的 all-gather 前要把各参数的分片拷进一块连续 buffer（copy-in），gather 完再拷回各参数（copy-out），比 `FlatParameter` 多两次显存拷贝（`foreach_all_gather_copy_out`）。

**数字**：torchtitan 论文（[arXiv 2410.06511](https://arxiv.org/abs/2410.06511) 附录 B.1）报告 Llama-2-7B 上 FSDP2 比 FSDP1 显存平均低 7%、吞吐平均高 1.5%（作者自测）；torchtitan 的 FSDP.md 给的是 8×H100 上 Llama-7B 峰值显存低 7%、MFU 更高、loss 曲线一致（作者自测）。

### 4. `fully_shard` 的运行时

**分组**：`fully_shard(module)` 把 `module.parameters()` 里还没被子模块认领的参数编成一组，一组一次 all-gather、一次 reduce-scatter。所以要**自底向上**调：先每个 Transformer block，最后 root（[fully_shard 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)）：

```python
for block in model.layers:
    fully_shard(block)
fully_shard(model)        # 认领剩下的 embedding、最后的 norm、输出 head
```

`fully_shard` 不是包装器，它把 `module.__class__` 换成一个继承了 `FSDPModule` 的动态类，在上面挂四个 hook（文档）：

1. **pre-forward**：all-gather 本组参数，用完整参数替换 DTensor。
2. **post-forward**：`reshard_after_forward=True` 时释放完整参数，参数变回 DTensor 分片（ZeRO-3）；`False` 时留着到 backward，省掉 backward 的那次 all-gather（对应 FSDP1 的 `SHARD_GRAD_OP`，即 ZeRO-2 式）。默认值 `None`：非 root 为 `True`，root 为 `False`，因为 backward 一开始就要用 root 的参数。
3. **pre-backward**：如果 forward 后切回去了，再 all-gather 一次。
4. **post-backward**：本组梯度一起 reduce-scatter，每卡只留自己那段，释放完整参数和完整梯度。

**预取（prefetch）**：

1. **隐式（默认）**：all-gather 放在单独的 stream 上。CPU 一般跑在 GPU 前面，GPU 还在算第 $\ell$ 层时，CPU 已经走到第 $\ell+1$ 层的 pre-forward 发出了它的 all-gather，于是自然重叠。backward 按 forward 结束顺序的逆序，在第 $\ell$ 层的 pre-backward 里预取第 $\ell-1$ 层（源码 `_backward_prefetch`）。
2. **显式**：`block.set_modules_to_forward_prefetch([下一层, 下下层])` / `set_modules_to_backward_prefetch`。用在三种情况：CPU 跑不到 GPU 前面（CPU-bound）时隐式预取失效；想同时预取 2 层以上换更多重叠（多占显存）；想更早发出第一个 all-gather（[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)）。

**3 层的时序**（3 个 block 都 `reshard_after_forward=True`，每格一个时间片）：

```text
slot   1    2    3    4    5    6    7    8    9
ag    AG0  AG1  AG2       AG2  AG1  AG0
comp       F0   F1   F2        B2   B1   B0
rs                                  RS2  RS1  RS0
```

1. AG0 暴露：第一层之前没有计算可以重叠。
2. slot 5 的 AG2 暴露：F2 刚把 L2 切回去，B2 马上又要。所以 root 默认不 reshard；torchtitan 对**最后一个 block** 也设 `reshard_after_forward=False`（论文附录 A、B.1）。
3. RS0 暴露：和 DDP 最后一个 bucket 一样，backward 算完才开始，后面没有计算可藏。
4. 显存峰值在 slot 6 附近：L2 的完整参数和梯度、预取来的 L1 完整参数同时在卡上。

**算一笔**：7B、32 层、8 卡 NVLink，`param_dtype=bf16`、`reduce_dtype=fp32`。

1. 每卡常驻的分片：$P_\ell / N = 2.73\times10^7$ 个参数，fp32 存 109 MB；cast 成 bf16 作为 all-gather 输入 54.7 MB。
2. all-gather 一层的完整结果：$M = 2.19\times10^8 \times 2 = 437.5$ MB。每卡收 $\frac{7}{8} \times 437.5 = 383$ MB，时间 $7 \times 5\,\mu s + 383\text{ MB} / 450\text{ GB/s} \approx 0.89$ ms。
3. reduce-scatter 一层 fp32 梯度：$M = 875$ MB，约 1.74 ms。
4. 一个 micro-batch：forward、backward 各 32 次 all-gather，32 次 reduce-scatter，共 $64 \times 0.89 + 32 \times 1.74 \approx 112$ ms。
5. 对比计算：每卡 32768 token 时一层 forward 约 29 ms，一步约 2.8 s（$6\Psi T_{\text{tok}} / (\text{peak}\cdot\text{MFU})$）。通信是计算的 4%，能藏住；藏住的条件推导见 [计算通信 Overlap](/parallel/comm-overlap)。

### 5. MixedPrecisionPolicy

**定义**（文档）：

| 字段 | 管什么 | 不设时 |
|---|---|---|
| `param_dtype` | all-gather 之前把分片 cast 成这个 dtype，forward / backward 用它算 | 保持原 dtype |
| `reduce_dtype` | 梯度 reduce-scatter 用的 dtype | 跟 `param_dtype` 走 |
| `output_dtype` | forward 输出 cast 成什么 | 不 cast |
| `cast_forward_inputs` | 是否把输入 cast 成 `param_dtype` | `True` |

关键：**分片参数本身保持原 dtype（fp32），它就是主权重**；优化器在 fp32 分片上更新。bf16 的完整参数只在 all-gather 之后临时存在。reduce-scatter 的输出最后 cast 回分片参数的 dtype，所以分片梯度也是 fp32（源码 `_cast_and_view_sharded_grads`）。

**和 16 字节账对上**（[训练显存账](/posttrain/training-memory)），每卡、每参数：

| 东西 | dtype | 常驻？ | 字节 |
|---|---|---|---|
| 分片参数（= 主权重） | fp32 | 常驻 | $4/N$ |
| 分片梯度 | fp32 | 常驻到 optimizer step | $4/N$ |
| Adam $m$、$v$ | fp32 | 常驻 | $8/N$ |
| **合计** | | | $16/N$ |

7B、8 卡：$16 \times 7\times10^9 / 8 = 14$ GB，和 ZeRO-3 页的数一样。区别在于 ZeRO 论文记法里的「bf16 参数 2 + bf16 梯度 2」在 FSDP2 里不常驻，只在当前层临时出现：2 层完整 bf16 参数（当前 + 预取）0.875 GB，当前层的 bf16 梯度 0.44 GB，fp32 的 reduce-scatter 输入 buffer 0.875 GB，合计约 2 GB（自己算的）。

**`reduce_dtype` 选 fp32 还是 bf16**：

1. fp32：reduce-scatter 字节翻倍，每层 1.74 ms 对 bf16 的 0.89 ms，一步 56 ms 对 28 ms；reduce-scatter 的输入 buffer 也翻倍。
2. bf16：ring reduce-scatter 每一跳把收到的部分和与本地梯度相加后按 bf16 存，尾数只有 7 位，卡数越多，舍入次数越多。
3. torchtitan 默认 `mixed_precision_param="bfloat16"`、`mixed_precision_reduce="float32"`（[configs.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/configs.py)）。注意 PyTorch 的默认值：只设 `param_dtype=bf16` 不设 `reduce_dtype`，梯度会用 bf16 归约。

### 6. 2D：FSDP + TP，以及 HSDP

**FSDP + TP**。mesh 取 `("dp", "tp")`，TP 在最后一维（节点内）。顺序是**先 TP 再 FSDP**：

1. `parallelize_module(block, mesh["tp"], plan)`：`ColwiseParallel` 把权重变成 `Shard(0)` 的 DTensor（`nn.Linear` 的 weight 是 `[out, in]`，切输出维），`RowwiseParallel` 变成 `Shard(1)`；切法本身见 [Megatron TP](/parallel/megatron-tp)。
2. `fully_shard(block, mesh=mesh["dp"])`：在每个 TP 分片上再沿 dim 0 切 $N$ 份。参数成为 2D mesh 上的 DTensor。本机在 (dp=2, tp=2) 的 mesh 上验证：列切权重的 placement 是 `(_StridedShard(dim=0, sf=2), Shard(dim=0))`，`sf` 等于 TP 度。`_StridedShard` 的意思是：从全局张量看，一个 dp rank 拿到的行不连续，每个 TP 块里各有一段。
3. 所有参数最好都落在同一个 2D mesh 上（norm 用 `SequenceParallel`、embedding 和 head 也做 TP）。本机 torch 2.5 上，如果有的参数只在 dp mesh 上，`foreach` 版 AdamW 和 `clip_grad_norm_` 报「cross-mesh operation」错误。

**为什么要加 TP**：固定 global batch 下扩卡（strong scaling）。FSDP 跨节点 all-gather 藏得住的条件是 $T_{\text{tok}} \ge \frac{N-1}{N}\cdot\frac{b \cdot \text{peak}\cdot\text{MFU}}{2\,\text{BW}} = \frac{N-1}{N}\cdot\frac{2 \times 494\text{T}}{2 \times 50\text{G}} \approx 9.9$k（推导见 [计算通信 Overlap](/parallel/comm-overlap)，$N$ 大时 $\frac{N-1}{N} \approx 1$）。1024 卡、global batch 4M token：

1. 纯 FSDP：每卡 $4194304 / 1024 = 4096$ 个 token，不到 9.9k，all-gather 暴露。
2. TP 8 × FSDP 128：TP 组内 8 张卡吃同一份数据，每个 TP 组 32768 个 token。每卡的 all-gather 字节和计算量都除以 $t$，比值不变，条件里的 $T_{\text{tok}}$ 变成每个 TP 组的 token 数，32768 满足。
3. torchtitan 论文 3.3.2 节：「Combining FSDP and TP allows strong scaling with a fixed problem/batch size」。

**HSDP**：mesh 取 `(R, N)`、名字 `("dp_replicate", "dp_shard")`，传给 `fully_shard(mesh=mesh["dp_replicate", "dp_shard"])`，参数 placement 是 `(Replicate(), Shard(0))`。节点内做 FSDP，节点间做 DDP：

1. all-gather、reduce-scatter 只在 $N$ 张卡（一台机器）内，走 NVLink。
2. reduce-scatter 之后，每卡把自己那 $1/N$ 的梯度在 $R$ 个副本间 all-reduce，走 IB，和 backward 重叠。

例：7B、4 台 × 8 卡。纯 FSDP-32：模型状态每卡 3.5 GB，但每层 all-gather 都要跨 IB，约 8.6 ms（NVLink 下 0.89 ms），每个 micro-batch 64 次。HSDP（$R = 4$、$N = 8$）：每卡 14 GB，层内通信全在 NVLink；跨节点只有每步一次 fp32 梯度分片的 all-reduce，$\Psi/N \times 4 = 3.5$ GB，$2 \times \frac{3}{4} \times 3.5\text{ GB} / 50\text{ GB/s} \approx 105$ ms。显存够就选 HSDP。

### 7. 常见坑

1. **单元太大**：只对 root 调 `fully_shard`，整个模型一次 all-gather，完整的 bf16 参数（7B 是 14 GB）同时在卡上，而且没有下一层可以预取，通信完全暴露。文档明确说不要只包 root。
2. **单元太小**：每个 Linear 一个单元。q_proj（$4096 \times 4096$，bf16 33.5 MB）的 all-gather 约 100 µs，其中延迟项 35 µs 占 35%；一个 block 一个单元时延迟项只占 4%。每个单元还有自己的 hook 和 copy-in / copy-out。**标准做法：一个 Transformer block 一个单元，root 再包一次。**
3. **root 也要 `fully_shard`**：不在任何单元里的参数（embedding、最后的 norm、head）不归 FSDP 管，不切、梯度也不跨卡归约，各卡各自更新就漂开了。root 的 hook 还负责把输入搬到设备上、在 backward 结束时等所有 reduce-scatter 完成。
4. **optimizer 要在 `fully_shard` 之后建**：`fully_shard` 用新的 `nn.Parameter`（包着 DTensor）替换了模块参数（源码 `_fsdp_param.py`），之前建的 optimizer 拿的是旧参数。
5. **调 `model(x)`，不要调 `model.forward(x)`**：后者绕过 hook，参数没有 all-gather。
6. **gradient accumulation**：前 $k-1$ 个 micro-batch 调 `model.set_requires_gradient_sync(False)`，最后一个设回 `True`，每步只做 $L$ 次 reduce-scatter 而不是 $kL$ 次。代价：没归约的梯度以**完整形状**留在卡上（按 `reduce_dtype` 累加），7B 用 fp32 是 $4\Psi = 28$ GB，往往放不下。HSDP 下改用 `set_requires_all_reduce(False)`：reduce-scatter 照做（节点内、便宜），只把跨节点 all-reduce 推迟到最后一个 micro-batch，累加的是 $1/N$ 的分片，显存不涨。
7. **checkpoint 用 DCP**：`model.state_dict()` 已经是分片 DTensor，用 `torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer)` 取模型和优化器状态、`dcp.save` 每卡写自己的分片，换卡数加载时 DCP 按 metadata 重切。FSDP2 不直接给完整 state dict，需要时用 `DTensor.full_tensor()` 或 `StateDictOptions(full_state_dict=True)`。见 [容错](/train/fault-tolerance)。

## 面试追问

::: details Q：FSDP2 和 FSDP1 最本质的区别是什么？
FSDP1 把一个单元的参数拼成一个 `FlatParameter` 再切，一段分片横跨多个参数；FSDP2 每个参数单独沿 dim 0 切成 DTensor。后者让「对单个参数做不同的事」变简单（冻结、不同 dtype 的冻结参数、fp8 all-gather），state dict 不需要通信，和 TP 的 DTensor 能直接叠成 2D；显存管理不靠 `recordStream`，峰值更低、更确定。代价是 all-gather 前后要做 copy-in / copy-out。
:::

::: details Q：`Partial` 这个 placement 有什么用？为什么不算完就 all-reduce？
它表示「每卡有一个加数，还没归约」。行切矩阵乘的输出天然是 `Partial`。把归约推迟，下一步才知道该用哪种通信：要完整结果就 all-reduce，下一步要按行切（sequence parallel）就 reduce-scatter，通信量只有 all-reduce 的一半。FSDP 的梯度在 reduce-scatter 之前也是 `Partial`：每卡用自己的数据算出一份梯度，总梯度是它们的和。
:::

::: details Q：`reshard_after_forward=False` 省什么、花什么？
省 backward 前的 all-gather：每个 micro-batch 从 $2L$ 次降到 $L$ 次。花显存：forward 结束到 backward 用完之前，每层完整的 bf16 参数都留着，7B 是 $2\Psi = 14$ GB。和 DeepSpeed ZeRO-2 不完全一样，参数在 backward 后仍会切回去，下一步 forward 再 gather。root 默认就是 `False`，因为 backward 开始马上要用它；torchtitan 对最后一个 block 也这么设。有 PP 时 torchtitan 默认整个模型都不 reshard，避免每个 micro-batch 都 all-gather。
:::

::: details Q：FSDP + TP 时，为什么先 `parallelize_module` 再 `fully_shard`？两个 mesh 怎么取？
TP 决定每张卡算哪一片，这是计算的切法；FSDP 只决定这一片在 DP 组里怎么存，算之前 gather 回来。先 TP 后，参数已经是 tp mesh 上的 DTensor，`fully_shard(mesh=mesh["dp"])` 在本地分片上再切，得到 (dp, tp) 2D mesh 上的 DTensor。mesh 用 `init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))`，tp 在最后一维，保证 TP 组在同一台机器内。
:::

::: details Q：FSDP2 下 `clip_grad_norm_` 为什么能直接用？
梯度是 DTensor。每卡对本地分片算平方和，结果是一个 `Partial` 的 norm，`full_tensor()` 时做一次 all-reduce 求和再开根，所以 `torch.nn.utils.clip_grad_norm_` 不用改。前提是所有梯度在同一个 mesh 上；有 PP 时还要跨 stage 再加一次。细节见 [优化器与梯度裁剪](/train/optimizer-lr)。
:::


## 手撕

**题 1**：用 numpy 模拟 2 卡上的 DTensor：三种 placement、redistribute 对应的集合通信、两条 matmul 传播规则，最后用它做一步 FSDP（all-gather → 各卡算梯度 → reduce-scatter 求平均），和单卡吃整个 batch 的梯度对比。

```python
import numpy as np

N = 2  # 2 张卡

class DT:  # 1D mesh 上的 DTensor：locals[r] 是 rank r 的本地张量
    def __init__(s, locals_, pl):
        s.locals, s.pl = locals_, pl  # ("shard", dim) / ("rep",) / ("partial",)

    def full(s):
        if s.pl[0] == "shard":
            return np.concatenate(s.locals, axis=s.pl[1])
        return s.locals[0] if s.pl[0] == "rep" else sum(s.locals)  # partial：各 rank 之和

def distribute(x, pl):
    if pl[0] == "shard":
        return DT(np.array_split(x, N, axis=pl[1]), pl)  # torch.chunk 语义
    return DT([x.copy() for _ in range(N)], pl)

COMM = {("shard", "rep"): "all-gather", ("partial", "rep"): "all-reduce",
        ("partial", "shard"): "reduce-scatter", ("rep", "shard"): "本地切片",
        ("shard", "shard"): "all-to-all"}

def redistribute(t, pl):
    return distribute(t.full(), pl), COMM[(t.pl[0], pl[0])]

def matmul(a, b):  # 两条 sharding propagation 规则
    out = [a.locals[r] @ b.locals[r] for r in range(N)]
    if (a.pl, b.pl) == (("rep",), ("shard", 1)):
        return DT(out, ("shard", 1))       # 列切：不通信
    if (a.pl, b.pl) == (("shard", 1), ("shard", 0)):
        return DT(out, ("partial",))       # 行切：结果待归约
    raise NotImplementedError

A, B = np.arange(8.0).reshape(2, 4), np.arange(8.0).reshape(4, 2)
p = matmul(distribute(A, ("shard", 1)), distribute(B, ("shard", 0)))
print("Partial:", [l.tolist() for l in p.locals])
for dst in [("rep",), ("shard", 0)]:
    q, op = redistribute(p, dst)
    print(dst, "via", op, [l.tolist() for l in q.locals])
c = matmul(distribute(A, ("rep",)), distribute(B, ("shard", 1)))
print("Rep @ Shard(1) ->", c.pl, np.allclose(c.full(), A @ B))

# 一步 FSDP：W 沿 dim 0 切，各 rank 吃不同数据，loss = mean(y^2)
rng = np.random.default_rng(0)
W = rng.standard_normal((4, 3))                       # y = x @ W.T
xs = [rng.standard_normal((5, 3)) for _ in range(N)]
w = distribute(W, ("shard", 0))                       # 每卡只存 2 行
W_full, op1 = redistribute(w, ("rep",))               # pre-forward
grads = []
for r in range(N):
    y = xs[r] @ W_full.locals[r].T
    grads.append((2 * y).T @ xs[r] / y.size)
g, op2 = redistribute(DT(grads, ("partial",)), ("shard", 0))  # post-backward
g.locals = [l / N for l in g.locals]                  # 求平均
X = np.concatenate(xs)
ref = (2 * X @ W.T).T @ X / (X.shape[0] * 4)          # 单卡吃整个 batch
print(op1, "+", op2, "matches single-device:", np.allclose(g.full(), ref), g.locals[0].shape)

# Partial: [[[2.0, 3.0], [10.0, 19.0]], [[26.0, 31.0], [66.0, 79.0]]]
# ('rep',) via all-reduce [[[28.0, 34.0], [76.0, 98.0]], [[28.0, 34.0], [76.0, 98.0]]]
# ('shard', 0) via reduce-scatter [[[28.0, 34.0]], [[76.0, 98.0]]]
# Rep @ Shard(1) -> ('shard', 1) True
# all-gather + reduce-scatter matches single-device: True (2, 3)
```

**题 2**：写一个 FSDP2 + TP 的训练配置。**需要 8 张 GPU，`torchrun --nproc_per_node=8 train.py`，这里不跑。** 同样的 TP plan 和 `fully_shard` 调用在本机 CPU / gloo、torch 2.5 上验证过：forward 结果和不切分的模型一致，所有参数都在 (dp, tp) mesh 上。

```python
import os
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module)

class Block(nn.Module):                      # 只有 MLP 的 block，结构够演示
    def __init__(s, d):
        super().__init__()
        s.norm, s.w1, s.w2 = nn.LayerNorm(d), nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
    def forward(s, x):
        return x + s.w2(F.gelu(s.w1(s.norm(x))))

class Model(nn.Module):
    def __init__(s, V=32000, d=4096, L=32):
        super().__init__()
        s.emb, s.layers = nn.Embedding(V, d), nn.ModuleList(Block(d) for _ in range(L))
        s.norm, s.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
    def forward(s, tok):
        x = s.emb(tok)
        for blk in s.layers:
            x = blk(x)
        return s.head(s.norm(x))

torch.distributed.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))   # tp 在最内层
torch.manual_seed(0)
model = Model()                               # 大模型应在 meta device 上建，fully_shard 后 to_empty

# 1. TP：激活在 block 之间按序列切（Shard(1)），进 w1 前 all-gather，出 w2 后 reduce-scatter
parallelize_module(model, mesh["tp"], {
    "emb": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
    "norm": SequenceParallel(),
    "head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
})
for blk in model.layers:
    parallelize_module(blk, mesh["tp"], {
        "norm": SequenceParallel(),
        "w1": ColwiseParallel(input_layouts=Shard(1)),
        "w2": RowwiseParallel(output_layouts=Shard(1)),
    })

# 2. FSDP：在 dp 维上再切；每个 block 一个单元，root 最后包
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for i, blk in enumerate(model.layers):
    fully_shard(blk, mesh=mesh["dp"], mp_policy=mp,
                reshard_after_forward=(i < len(model.layers) - 1))   # 最后一层不切回
fully_shard(model, mesh=mesh["dp"], mp_policy=mp)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)   # 一定在 fully_shard 之后建
g = torch.Generator().manual_seed(mesh["dp"].get_local_rank())  # 同一 TP 组同一份数据
k = 4                                                  # gradient accumulation
for step in range(10):
    for micro in range(k):
        model.set_requires_gradient_sync(micro == k - 1)   # 前 k-1 次不 reduce-scatter
        tok = torch.randint(0, 32000, (2, 1025), generator=g).cuda()  # 输入 1024，能被 tp 整除
        logits = model(tok[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1).float(), tok[:, 1:].flatten())
        (loss / k).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

model_sd, optim_sd = get_state_dict(model, opt)        # 分片 DTensor，不通信
dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id="ckpt/step10")
```

## 参考

- [PyTorch 文档：`torch.distributed.fsdp.fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)：hook 时序、`reshard_after_forward`、`MixedPrecisionPolicy`、`FSDPModule` 的方法
- [PyTorch 文档：DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)：placement 与 redistribute 对应的集合通信
- [PyTorch recipe：DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [PyTorch 教程：Getting Started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：隐式 / 显式预取、混合精度、DCP
- [PyTorch 教程：Large Scale Transformer model training with Tensor Parallel](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)：`parallelize_module` 与 FSDP + TP
- [torchtitan FSDP.md](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/FSDP.md)：FSDP1 → FSDP2 的动机、API 对照、meta device 初始化
- [TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://arxiv.org/abs/2410.06511)（Liang et al., 2024）
- [FSDP2 RFC（pytorch/pytorch#114299）](https://github.com/pytorch/pytorch/issues/114299)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)（Zhao et al., 2023）：FSDP1 的 `FlatParameter` 设计
- 源码：[`torch/distributed/fsdp/_fully_shard/`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp/_fully_shard)（`_fsdp_param_group.py` 的预取与 hook，`_fsdp_collectives.py` 的 reduce-scatter 与 dtype 处理）

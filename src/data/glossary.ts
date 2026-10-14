import type { GlossaryEntry } from '../lib/glossary'

/**
 * 术语表。def 写一句话，能给数字就给数字；细节留给 link 指向的页面。
 * 目前只在训练栈页面生效（见 inGlossaryScope）。
 */
export const glossary: GlossaryEntry[] = [
  // ---------- 效率与可靠性指标 ----------
  {
    term: 'MFU', match: ['MFU'], full: 'Model FLOPs Utilization',
    def: '实测 token/s × 每 token 模型需要的 FLOPs（约 $6N$）÷（卡数 × dense 峰值）。recompute 多做的不算进分子。大规模 dense 训练常见 30–55%。',
    link: '/train/compute-mfu',
  },
  {
    term: 'HFU', match: ['HFU'], full: 'Hardware FLOPs Utilization',
    def: '分母同 MFU，分子算上 recompute 实际多做的 FLOPs，所以 HFU ≥ MFU。full recompute 下每 token 做 $8N$，HFU = 8/6 · MFU。',
    link: '/train/compute-mfu',
  },
  {
    term: 'ETTR', match: ['ETTR'], full: 'Effective Training Time Ratio',
    def: '墙钟时间里真正在推进训练的比例，扣掉故障、重启、写 checkpoint。Llama 3 报告 >90%。',
    link: '/train/compute-mfu',
  },
  {
    term: 'MTBF', match: ['MTBF'], full: 'Mean Time Between Failures',
    def: '平均多久出一次故障。同步训练任何一张卡出事整个作业都停，作业 MTBF ≈ 单卡 MTBF ÷ 卡数；Llama 3 405B 在约 16K 张卡上约 3 小时一次。',
    link: '/train/fault-tolerance',
  },
  {
    term: 'SDC', match: ['SDC'], full: 'Silent Data Corruption',
    def: '硬件算错了但不报错，没有异常也没有超时，只在 loss 或梯度范数上留下痕迹。',
    link: '/train/fault-tolerance',
  },

  // ---------- 并行 ----------
  {
    term: 'DP', match: ['DP'], full: 'Data Parallel',
    def: '每张卡一份完整模型，各吃不同的数据，backward 后 all-reduce 梯度。',
    link: '/parallel/parallelism-overview',
  },
  {
    term: 'DDP', match: ['DDP'], full: 'Distributed Data Parallel',
    def: 'PyTorch 的 DP 实现：梯度分桶，backward 还没算完就开始 all-reduce 已就绪的桶。',
    link: '/parallel/parallelism-overview',
  },
  {
    term: 'TP', match: ['TP'], full: 'Tensor Parallel',
    def: '把一层的权重矩阵按行 / 列切到多卡（Megatron 式），每个 Transformer 层 forward 要 2 次 all-reduce，所以一般只放在机内 NVLink 上。',
    link: '/parallel/megatron-tp',
  },
  {
    term: 'SP', match: ['SP'], full: 'Sequence Parallel',
    def: '配合 TP 用：把 LayerNorm、dropout 这些 TP 没切的部分按序列维切开，all-reduce 换成 reduce-scatter + all-gather，通信量不变，省激活显存。',
    link: '/parallel/megatron-tp',
  },
  {
    term: 'PP', match: ['PP'], full: 'Pipeline Parallel',
    def: '按层切成 $p$ 个 stage，micro-batch 流水着过。代价是 bubble：1F1B 下空闲时间约为计算时间的 $\\frac{p-1}{m}$（$m$ 是 micro-batch 数）。',
    link: '/train/pipeline-schedules',
  },
  {
    term: 'CP', match: ['CP'], full: 'Context Parallel',
    def: '按序列维把 Q、K、V 切到多卡，attention 时轮流交换 KV 块，各块结果用 online softmax 合并。用于长上下文训练。',
    link: '/train/long-context',
  },
  {
    term: 'EP', match: ['EP'], full: 'Expert Parallel',
    def: 'MoE 的专家分到不同卡上，token 按路由结果 all-to-all 发给对应专家，算完再 all-to-all 发回来。',
    link: '/parallel/moe-ep',
  },
  {
    term: 'ZeRO', match: ['ZeRO'], full: 'Zero Redundancy Optimizer',
    def: 'DP 各卡不再各存一份完整状态：ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 连参数也切。',
    link: '/parallel/zero-fsdp',
  },
  {
    term: 'FSDP', match: ['FSDP'], full: 'Fully Sharded Data Parallel',
    def: 'PyTorch 版的 ZeRO-3：参数分片存，每层算之前 all-gather 出完整参数，backward 后 reduce-scatter 梯度。',
    link: '/parallel/zero-fsdp',
  },
  {
    term: 'FSDP2', match: ['FSDP2'],
    def: '基于 DTensor 重写的 FSDP（`fully_shard`）：每个参数按第 0 维分片，不再把一组参数拼成一个 flat buffer。',
    link: '/train/fsdp2-dtensor',
  },
  {
    term: 'HSDP', match: ['HSDP'], full: 'Hybrid Sharded Data Parallel',
    def: '一组卡内用 FSDP 分片，组与组之间完整复制、只 all-reduce 梯度。对应 2D mesh：`dp_replicate × dp_shard`。',
    link: '/train/fsdp2-dtensor',
  },
  {
    term: 'DTensor', match: ['DTensor'],
    def: 'PyTorch 的分布式张量：一个全局张量 + 一个 DeviceMesh + 每个 mesh 维度上的 placement（Shard / Replicate / Partial）。',
    link: '/train/fsdp2-dtensor',
  },
  {
    term: 'DeviceMesh', match: ['DeviceMesh'],
    def: '把所有 rank 排成多维网格，每一维对应一种并行（如 dp × tp），每一维各自建通信组。',
    link: '/train/fsdp2-dtensor',
  },
  {
    term: '1F1B', match: ['1F1B'], full: 'One Forward One Backward',
    def: 'PP 调度：稳态下每个 stage 交替做一个 micro-batch 的 forward 和一个的 backward。bubble 和 GPipe 一样，但同时在途的激活最多 $p$ 份。',
    link: '/train/pipeline-schedules',
  },
  {
    term: 'GPipe', match: ['GPipe'],
    def: '最早的 PP 调度：先把 $m$ 个 micro-batch 的 forward 全做完再做 backward，要同时存 $m$ 份激活。',
    link: '/train/pipeline-schedules',
  },
  {
    term: 'DualPipe', match: ['DualPipe'],
    def: 'DeepSeek-V3 的双向 PP 调度：把一个 forward chunk 和一个 backward chunk 配对，让一个的计算盖住另一个的 all-to-all 和 PP 通信。',
    link: '/train/pipeline-schedules',
  },

  // ---------- 通信与硬件 ----------
  {
    term: 'NCCL', match: ['NCCL'], full: 'NVIDIA Collective Communications Library',
    def: 'NVIDIA 的集合通信库，提供 all-reduce / all-gather / reduce-scatter / all-to-all，是 PyTorch 分布式在 GPU 上的默认后端。',
    link: '/parallel/collective-comm',
  },
  {
    term: 'NVLink', match: ['NVLink'],
    def: '机内 GPU 之间的直连。H100 的 NVLink 4 每卡双向 900 GB/s（单向 450），约是跨机 IB 的 9 倍。',
    link: '/stack/tr-hw-link',
  },
  {
    term: 'IB', match: ['IB'], full: 'InfiniBand',
    def: '跨机网络。NDR 每卡 400 Gb/s ≈ 50 GB/s，约是 NVLink 单向的 1/9，所以 TP、EP 一般不跨机。',
    link: '/stack/tr-hw-link',
  },
  {
    term: 'DCP', match: ['DCP'], full: 'Distributed Checkpoint',
    def: 'PyTorch 的分片 checkpoint：每个 rank 只写自己持有的分片，外加一个记录全局形状的 `.metadata`，换卡数也能加载。',
    link: '/train/fault-tolerance',
  },

  // ---------- 算子与精度 ----------
  {
    term: 'GEMM', match: ['GEMM'], full: 'General Matrix Multiply',
    def: '矩阵乘。训练时一个 Linear 层有 3 个：forward $Y = XW$、$dX = dY\\,W^\\top$、$dW = X^\\top dY$，合起来每 token $6N$ FLOPs。',
    link: '/gpu/tensor-core-gemm',
  },
  {
    term: 'FP8', match: ['FP8', 'fp8'],
    def: '8 位浮点，有 E4M3（精度高）和 E5M2（范围大）两种。H100 的 fp8 dense 峰值 1979 TFLOP/s，是 bf16 的 2 倍；训练里只把 GEMM 的输入换成 fp8，配 scale。',
    link: '/train/fp8-training',
  },
  {
    term: 'MXFP8', match: ['MXFP8'], full: 'Microscaling FP8',
    def: '每 32 个元素共享一个 2 的幂次 scale（E8M0），由 Blackwell 的 Tensor Core 直接支持，不用软件按块缩放。',
    link: '/train/fp8-training',
  },
  {
    term: 'recompute', match: ['recompute'], full: 'activation checkpointing',
    def: 'forward 时不存中间激活，backward 用到时再算一遍。full recompute 每 token 多一次 forward，$6N \\to 8N$，换来激活显存大幅下降。',
    link: '/posttrain/training-memory',
  },

  // ---------- 优化器与训练阶段 ----------
  {
    term: 'AdamW', match: ['AdamW'],
    def: 'Adam + 解耦的 weight decay。每参数存 fp32 的 $m$、$v$；混合精度下 bf16 参数和梯度 4 B + fp32 主权重 4 B + $m, v$ 8 B，共 16 B / 参数。',
    link: '/train/optimizer-lr',
  },
  {
    term: 'WSD', match: ['WSD'], full: 'Warmup-Stable-Decay',
    def: '学习率调度：warmup 后长时间保持峰值，最后一小段快速衰减。stable 段的任何 checkpoint 都能接着训或单独退火。',
    link: '/train/optimizer-lr',
  },
  {
    term: 'LoRA', match: ['LoRA'], full: 'Low-Rank Adaptation',
    def: '冻结原权重 $W$，只训练低秩增量 $BA$（秩 $r \\ll d$）。$d \\times d$ 的层可训参数从 $d^2$ 降到 $2rd$，优化器状态跟着降。',
    link: '/posttrain/lora-qlora',
  },
  {
    term: 'GRPO', match: ['GRPO'], full: 'Group Relative Policy Optimization',
    def: '每个 prompt 采 $G$ 个回答，用组内 reward 的均值和标准差算 advantage，省掉 PPO 的 critic 模型。',
    link: '/posttrain/rlhf-ppo-dpo-grpo',
  },
]

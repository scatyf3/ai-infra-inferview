// LeetGPU 题单：按 docs/handson/index.md 的学习路线分组。直接改这个文件，页面自动更新。
//
// impl 实现等级（一道题可以标多个）：
//   torch  只在乎 correctness，PyTorch 写对就行
//   triton 在乎效率，要用 Triton（或 CUDA）写出像样的 kernel
// 同一道题可以出现在多个分组里（比如 RMSNorm 先 torch 写对，再 triton 写快），各条的 impl 独立。
// optional: true = 选做；note = 这题在这里练什么。

export type Impl = 'torch' | 'triton'

export interface RoadmapItem {
  id: number
  impl: Impl[]
  note?: string
  optional?: boolean
}

export interface RoadmapGroup {
  key: string
  title: string
  desc?: string
  items: RoadmapItem[]
}

export interface RoadmapStage {
  title: string
  groups: RoadmapGroup[]
}

export const roadmap: RoadmapStage[] = [
  {
    title: '1. 概念正确性',
    groups: [
      {
        key: 'attention',
        title: 'Attention',
        desc: 'Naive → MHA → GQA → MLA，mask / shape，with KV Cache',
        items: [
          { id: 6, impl: ['torch', 'triton'], note: '单头基础；triton 版 = 分块 + online softmax，就是 FlashAttention' },
          { id: 12, impl: ['torch'], note: '拆 head / 合并 head' },
          { id: 80, impl: ['torch'], note: '广播分组' },
          { id: 53, impl: ['torch', 'triton'], note: 'causal mask；triton 版 = causal FlashAttention' },
          { id: 26, impl: ['torch'], note: 'Q 与 K/V 长度不同' },
          { id: 59, impl: ['torch'], note: '带状 mask' },
          { id: 102, impl: ['torch'], note: '变长序列打包后的 mask' },
          { id: 112, impl: ['torch'], note: '额外的 sink 位置参与 softmax' },
          { id: 61, impl: ['torch', 'triton'], note: 'RoPE 单独练：成对旋转、cos/sin 表；triton 版是逐元素 kernel' },
          { id: 115, impl: ['torch', 'triton'], note: '投影 + RoPE + 写 KV cache：decode 时 RoPE 的位置要带 cache 偏移；triton 版练 cache 布局' },
          { id: 55, impl: ['torch'], note: 'ALiBi：score 上加位置偏置；主流模型基本都用 RoPE，只有 BLOOM / MPT 用', optional: true },
          { id: 96, impl: ['torch', 'triton'], note: '从 cache 读出、反量化后做 attention；triton 版练 cache 布局' },
          { id: 114, impl: ['torch'], note: 'MLA，路线终点' },
        ],
      },
      {
        key: 'decoder-block',
        title: 'Decoder Block 其余部分',
        desc: 'RMSNorm、RoPE、SwiGLU FFN',
        items: [
          { id: 50, impl: ['torch'] },
          { id: 61, impl: ['torch'] },
          { id: 54, impl: ['torch'] },
          { id: 84, impl: ['torch'] },
          { id: 106, impl: ['torch'] },
          { id: 93, impl: ['torch'], note: '综合题：拼成整个 Llama block' },
          { id: 74, impl: ['torch'], note: 'LayerNorm + GELU 版的 block', optional: true },
        ],
      },
    ],
  },
  {
    title: '2. Kernel',
    groups: [
      {
        key: 'linear-reduce-matmul',
        title: 'linear, reduce, matmul',
        items: [
          { id: 4, impl: ['triton'], note: '分层归约' },
          { id: 17, impl: ['triton'], note: '归约' },
          { id: 16, impl: ['triton'], note: 'scan' },
          { id: 3, impl: ['triton'], note: '合并访存' },
          { id: 2, impl: ['triton'], note: '朴素版' },
          { id: 22, impl: ['triton'], note: 'tiling' },
          { id: 30, impl: ['triton'] },
          { id: 57, impl: ['triton'], note: '低精度' },
        ],
      },
      {
        key: 'fused-norm',
        title: 'fused LayerNorm / RMSNorm',
        items: [
          { id: 5, impl: ['triton'], note: '按行归约，norm 系列同一个套路' },
          { id: 50, impl: ['triton'] },
          { id: 113, impl: ['triton'] },
          { id: 83, impl: ['triton'] },
        ],
      },
      {
        key: 'attention-opt',
        title: 'optimization trick',
        desc: 'Paged Attention、FlashAttention；LeetGPU 没有专门的题，拿普通 attention 题练',
        items: [
          { id: 6, impl: ['triton'], note: 'FlashAttention：分块 + online softmax' },
          { id: 53, impl: ['triton'], note: 'causal FlashAttention' },
          { id: 115, impl: ['triton'], note: 'Paged：自己加一层 block table' },
          { id: 96, impl: ['triton'], note: 'Paged：自己加一层 block table' },
          { id: 111, impl: ['triton'], note: 'FlashAttention 反向', optional: true },
          { id: 119, impl: ['triton'], note: '稀疏选块', optional: true },
        ],
      },
    ],
  },
  {
    title: '路线外 · 推理相关',
    groups: [
      {
        key: 'sampling',
        title: '采样',
        items: [
          { id: 60, impl: ['torch'], optional: true },
          { id: 104, impl: ['torch'], optional: true },
          { id: 29, impl: ['torch'], optional: true },
          { id: 98, impl: ['torch'], optional: true },
          { id: 87, impl: ['torch'], optional: true },
        ],
      },
      {
        key: 'quant',
        title: '量化',
        items: [
          { id: 32, impl: ['triton'], optional: true },
          { id: 64, impl: ['triton'], optional: true },
          { id: 81, impl: ['triton'], optional: true },
        ],
      },
      {
        key: 'misc',
        title: '其他',
        items: [
          { id: 67, impl: ['torch'], optional: true },
          { id: 85, impl: ['torch'], optional: true },
        ],
      },
    ],
  },
]

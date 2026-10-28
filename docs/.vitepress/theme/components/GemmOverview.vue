<script setup lang="ts">
// GEMM 优化总览：按存储层级排开，每一步搬运对应一级 tile 和几种手段；颜色 = 这个手段在省什么
type Kind = 'bytes' | 'instr' | 'latency' | 'compute' | 'fill' | 'fuse' | 'limit'
interface Chip { text: string; kind: Kind; href: string; tag?: string }
interface Step { moves: string; title: string; chips: Chip[] }

const kinds: Record<Kind, { label: string; color: string }> = {
  bytes: { label: '少读 HBM', color: '#3b82f6' },
  instr: { label: '少发 load 指令', color: '#f59e0b' },
  latency: { label: '藏延迟', color: '#8b5cf6' },
  compute: { label: '算得快', color: '#ec4899' },
  fill: { label: '喂满 SM', color: '#14b8a6' },
  fuse: { label: '省一次读写', color: '#22c55e' },
  limit: { label: '约束', color: '#94a3b8' },
}

const A = {
  l1: '#第一级-block-tile-放进-shared-memory',
  l2: '#l2-让同时跑的-block-读同一份数据',
  reg: '#第二级-每个线程算一小块-寄存器-tile',
  three: '#三级-tiling',
  instr: '#第三级-指令-tile',
  hopper: '#hopper-上的变化',
  limit: '#shared-memory-和寄存器卡住上限',
  wave: '#wave-quantization-tile-数和-sm-数对不齐',
  gemv: '#面试追问',
}

const levels = [
  { name: 'HBM', sub: '80 GB · 3.35 TB/s' },
  { name: 'L2', sub: '50 MB' },
  { name: 'shared memory', sub: '228 KB / SM' },
  { name: '寄存器', sub: '256 KB / SM，每线程 ≤ 255 个' },
  { name: 'CUDA core / Tensor Core', sub: 'bf16 989 TFLOP/s' },
]

const steps: Step[] = [
  {
    moves: 'A / B 条带',
    title: 'L2：同一波 block 共享条带',
    chips: [{ text: 'grouped ordering（GROUP_SIZE_M）：一波铺成方块，有效 AI 64 → 676', kind: 'bytes', href: A.l2 }],
  },
  {
    moves: 'As / Bs，一次 BK 段',
    title: '第 1 级 block tile：BM × BN × BK',
    chips: [
      { text: '块内复用：AI = BM·BN / (BM + BN)', kind: 'bytes', href: A.l1 },
      { text: '多级流水：算这一段时异步搬下一段（cp.async）', kind: 'latency', href: A.three },
      { text: 'TMA：一个线程搬整块，顺手 swizzle', kind: 'latency', href: A.hopper, tag: 'Hopper' },
    ],
  },
  {
    moves: 'a / b 片段',
    title: '第 2 级 warp tile：WM × WN',
    chips: [
      { text: '线程 tile TM × TN：一次 load 换 TM·TN 次 FMA', kind: 'instr', href: A.reg, tag: 'CUDA core' },
      { text: 'float4 向量 load（As 转置存）', kind: 'instr', href: A.reg },
      { text: 'ldmatrix：按 fragment layout 分给 32 个线程', kind: 'instr', href: A.instr },
      { text: 'wgmma 直接读 smem，省掉这一步', kind: 'instr', href: A.hopper, tag: 'Hopper' },
    ],
  },
  {
    moves: 'fragment → MAC',
    title: '第 3 级指令 tile：mma m16n8k16 → wgmma m64nNk16 → tcgen05',
    chips: [
      { text: '一条指令算一整块，形状由硬件定', kind: 'compute', href: A.instr },
      { text: 'warp specialization：producer 搬，consumer 算', kind: 'latency', href: A.hopper, tag: 'Hopper' },
    ],
  },
]

const epilogue: Step = {
  moves: 'C tile 写回 HBM',
  title: 'epilogue：C 还在寄存器里',
  chips: [{ text: '顺手做 bias / 激活 / 量化再写回（kernel fusion）', kind: 'fuse', href: A.three }],
}

const side = [
  {
    title: '横切：tile 选多大',
    chips: [
      { text: '上限：smem（BK × stage 数）、寄存器（BM·BN 个 fp32 累加器）', kind: 'limit', href: A.limit },
      { text: 'wave quantization：tile 数对齐 SM 数', kind: 'fill', href: A.wave },
      { text: 'Stream-K：按 K 迭代平均分给 SM', kind: 'fill', href: A.wave },
      { text: '离最优：搬运已贴着 I/O 下界，剩下差在 epilogue、流水填充和频率', kind: 'limit', href: '#离最优还有多远' },
    ] as Chip[],
  },
  {
    title: 'M 很小（decode GEMV，AI ≈ batch）：tiling 无从复用',
    chips: [
      { text: 'continuous batching', kind: 'bytes', href: A.gemv },
      { text: 'W4A16 weight-only 量化', kind: 'bytes', href: A.gemv },
      { text: 'speculative decoding', kind: 'bytes', href: A.gemv },
      { text: 'split-K 补并行度', kind: 'fill', href: A.gemv },
    ] as Chip[],
  },
]
</script>

<template>
  <figure class="widget gov">
    <div class="legend">
      <span v-for="(k, key) in kinds" :key="key"><i :style="{ background: k.color }" />{{ k.label }}</span>
    </div>

    <div class="ladder">
      <template v-for="(lv, i) in levels" :key="lv.name">
        <div class="mem" :class="{ compute: i === levels.length - 1 }">
          <b>{{ lv.name }}</b>
          <small>{{ lv.sub }}</small>
        </div>
        <div class="memgap" />
        <template v-if="i < steps.length">
          <div class="arrow"><span>↓</span><small>{{ steps[i].moves }}</small></div>
          <div class="step">
            <div class="stitle">{{ steps[i].title }}</div>
            <div class="chips">
              <a v-for="c in steps[i].chips" :key="c.text" class="chip" :href="c.href" :style="{ '--c': kinds[c.kind].color }">
                {{ c.text }}<em v-if="c.tag">{{ c.tag }}</em>
              </a>
            </div>
          </div>
        </template>
      </template>
      <div class="arrow up"><span>↑</span><small>{{ epilogue.moves }}</small></div>
      <div class="step">
        <div class="stitle">{{ epilogue.title }}</div>
        <div class="chips">
          <a v-for="c in epilogue.chips" :key="c.text" class="chip" :href="c.href" :style="{ '--c': kinds[c.kind].color }">{{ c.text }}</a>
        </div>
      </div>
    </div>

    <div class="side">
      <div v-for="s in side" :key="s.title" class="sbox">
        <div class="stitle">{{ s.title }}</div>
        <div class="chips">
          <a v-for="c in s.chips" :key="c.text" class="chip" :href="c.href" :style="{ '--c': kinds[c.kind].color }">{{ c.text }}</a>
        </div>
      </div>
    </div>

    <figcaption class="muted">
      从上往下是数据离计算单元越来越近；每往下一层用一级 tile，让搬上来的数据在更近的存储里被多次复用。点任一手段跳到对应小节。
    </figcaption>
  </figure>
</template>

<style scoped>
.gov { margin: 20px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--wg-muted); margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

.ladder { display: grid; grid-template-columns: 132px minmax(0, 1fr); column-gap: 14px; }
.mem {
  border: 1.5px solid var(--wg-text); border-radius: 8px; padding: 6px 8px; background: var(--vp-c-bg);
  display: flex; flex-direction: column; align-items: center; text-align: center; line-height: 1.3;
}
.mem b { font-size: 13px; }
.mem small { font-size: 11px; color: var(--wg-muted); }
.mem.compute { border-color: #ec4899; background: color-mix(in srgb, #ec4899 10%, var(--vp-c-bg)); }
.memgap { min-height: 0; }

.arrow { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0; color: var(--wg-muted); }
.arrow span { font-size: 18px; line-height: 1; }
.arrow small { font-size: 11px; text-align: center; line-height: 1.3; }
.arrow.up span { color: #22c55e; }

.step { padding: 8px 0 10px; border-left: 2px dashed var(--wg-border); padding-left: 12px; }
.stitle { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  font-size: 12px; line-height: 1.45; padding: 3px 8px; border-radius: 6px;
  border-left: 3px solid var(--c); background: color-mix(in srgb, var(--c) 12%, var(--vp-c-bg));
  color: var(--wg-text); text-decoration: none; transition: background 0.15s; overflow-wrap: anywhere;
}
.chip:hover { background: color-mix(in srgb, var(--c) 24%, var(--vp-c-bg)); }
.chip em { font-style: normal; font-size: 10.5px; margin-left: 6px; padding: 0 5px; border-radius: 4px; border: 1px solid var(--wg-border); color: var(--wg-muted); }

.side { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 12px; margin-top: 16px; }
.sbox { border: 1px solid var(--wg-border); border-radius: 8px; padding: 10px 12px; background: var(--vp-c-bg); }

figcaption { font-size: 12.5px; margin-top: 12px; line-height: 1.7; }

@media (max-width: 560px) {
  .ladder { grid-template-columns: 84px minmax(0, 1fr); column-gap: 8px; }
  .step { padding-left: 8px; }
  .mem b { font-size: 12px; }
}
</style>

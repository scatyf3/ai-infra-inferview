<script setup lang="ts">
// 量化 GEMM 总览：(a) 硬件约束 → (b) W4A16 / (c) FP8 两条路线的数据通路 → (d) 两条路线共用的 scale 规则
// 颜色 = 数据格式：bf16 蓝、int4 橙、fp8 绿，和 QuantRooflineChart 一致
type Fmt = 'bf16' | 'int4' | 'fp8' | 'none'
interface Chip { x: number; y: number; w: number; h?: number; t: string; c: Fmt; left?: boolean }
interface Box { x: number; y: number; w: number; h: number; t?: string; left?: boolean }
interface Arrow { x1: number; y1: number; x2: number; y2: number; t?: string; tx?: number; ty?: number }
interface Mark { x: number; y: number; n: string; c: Fmt }

const W = 700, H = 684

const panels = [
  { y: 8, h: 100, t: '(a) 硬件约束：Tensor Core 算 D = A × B + C，A、B 必须是同一类格式' },
  { y: 118, h: 186, t: '(b) Weight-only（W4A16）：省 decode 时从 HBM 读权重的带宽' },
  { y: 314, h: 150, t: '(c) W8A8（FP8）：X 和 W 都量化，直接用低精度 Tensor Core' },
  { y: 474, h: 200, t: '(d) scale 乘在哪（两条路线共用）：x̄ = s · x_q，只看 s 随不随求和维 K 变' },
]

// 数据通路的列：HBM / smem / 寄存器 / Tensor Core，(b)(c) 上下对齐
const col = { hbm: 24, smem: 173, reg: 322, tc: 556 }
const yb = 156, yc = 352

const boxes: Box[] = [
  { x: col.hbm, y: yb, w: 92, h: 76, t: 'HBM' },
  { x: col.smem, y: yb, w: 92, h: 76, t: 'smem' },
  { x: col.reg, y: yb, w: 176, h: 76, t: '寄存器', left: true },
  { x: col.tc, y: yb, w: 100, h: 76, t: 'Tensor Core' },
  { x: col.hbm, y: yc, w: 92, h: 72, t: 'HBM' },
  { x: col.smem, y: yc, w: 92, h: 72, t: 'smem' },
  { x: col.tc, y: yc, w: 100, h: 72, t: 'Tensor Core' },
  // (d) 两张卡片
  { x: 22, y: 512, w: 316, h: 150 },
  { x: 354, y: 512, w: 324, h: 150 },
]

// (a) 三组 A × B
const pairs = [
  { x: 24, a: 'bf16', b: 'bf16', ca: 'bf16', cb: 'bf16', ok: true },
  { x: 184, a: 'fp8', b: 'fp8', ca: 'fp8', cb: 'fp8', ok: true },
  { x: 344, a: 'int4', b: 'bf16', ca: 'int4', cb: 'bf16', ok: false },
] as const

const chips: Chip[] = [
  ...pairs.flatMap((p) => [
    { x: p.x, y: 50, w: 46, t: p.a, c: p.ca as Fmt },
    { x: p.x + 60, y: 50, w: 46, t: p.b, c: p.cb as Fmt },
  ]),
  // (b)
  { x: col.hbm + 8, y: yb + 26, w: 76, h: 20, t: 'W  int4', c: 'int4' },
  { x: col.hbm + 8, y: yb + 50, w: 76, h: 20, t: 'X  bf16', c: 'bf16' },
  { x: col.smem + 8, y: yb + 26, w: 76, h: 20, t: 'W  int4', c: 'int4' },
  { x: col.smem + 8, y: yb + 50, w: 76, h: 20, t: 'X  bf16', c: 'bf16' },
  { x: col.reg + 8, y: yb + 38, w: 40, t: 'int4', c: 'int4' },
  { x: col.reg + 132, y: yb + 38, w: 38, t: 'bf16', c: 'bf16' },
  { x: col.tc + 10, y: yb + 38, w: 80, t: 'bf16 MMA', c: 'bf16' },
  // (c)
  { x: col.hbm + 8, y: yc + 26, w: 76, h: 20, t: 'W  fp8', c: 'fp8' },
  { x: col.hbm + 8, y: yc + 48, w: 76, h: 20, t: 'X  fp8', c: 'fp8' },
  { x: col.smem + 8, y: yc + 26, w: 76, h: 20, t: 'W  fp8', c: 'fp8' },
  { x: col.smem + 8, y: yc + 48, w: 76, h: 20, t: 'X  fp8', c: 'fp8' },
  { x: col.tc + 10, y: yc + 26, w: 80, t: 'FP8 MMA', c: 'fp8' },
  // (d)
  { x: 32, y: 590, w: 296, h: 26, t: 'epilogue：累加完乘一次 s_x[i] · s_w[j]', c: 'none', left: true },
  { x: 364, y: 584, w: 304, t: 'W4A16：乘进操作数，并进 dequant 的 hfma2', c: 'int4', left: true },
  { x: 364, y: 610, w: 304, t: 'FP8：乘进每组部分和，CUDA core 每 128 个 K 一次', c: 'fp8', left: true },
  { x: 364, y: 636, w: 304, t: 'Blackwell MX / NVFP4：Tensor Core 自己乘', c: 'none', left: true },
]

const mid = (a: number, b: number) => (a + b) / 2
const arrows: Arrow[] = [
  // (b)
  { x1: col.hbm + 92, y1: yb + 46, x2: col.smem, y2: yb + 46, t: 'cp.async', tx: mid(col.hbm + 92, col.smem), ty: yb + 39 },
  { x1: col.smem + 92, y1: yb + 46, x2: col.reg, y2: yb + 46, t: 'lds128', tx: mid(col.smem + 92, col.reg), ty: yb + 39 },
  { x1: col.reg + 48, y1: yb + 49, x2: col.reg + 60, y2: yb + 49 },
  { x1: col.reg + 120, y1: yb + 49, x2: col.reg + 132, y2: yb + 49 },
  { x1: col.reg + 176, y1: yb + 46, x2: col.tc, y2: yb + 46 },
  // (c)
  { x1: col.hbm + 92, y1: yc + 46, x2: col.smem, y2: yc + 46, t: 'TMA', tx: mid(col.hbm + 92, col.smem), ty: yc + 39 },
  { x1: col.smem + 92, y1: yc + 46, x2: col.tc, y2: yc + 46, t: 'wgmma 直接读 smem，不经过寄存器里的 ALU', tx: mid(col.smem + 92, col.tc), ty: yc + 38 },
]

const marks: Mark[] = [
  { x: col.reg + 176, y: yb, n: '1', c: 'int4' },
  { x: col.reg + 120, y: yb + 36, n: '2', c: 'int4' },
  { x: mid(col.smem + 92, col.reg), y: yb + 60, n: '3', c: 'int4' },
  { x: col.hbm + 92, y: yc, n: '4', c: 'fp8' },
]

const notes = [
  { y: 254, n: '1', c: 'int4' as Fmt, t: '在哪转：smem → 寄存器之后。smem 里还是 int4，cp.async 异步搬运也保得住' },
  { y: 272, n: '2', c: 'int4' as Fmt, t: '怎么转：没有 int4 → bf16 指令；lop3 把 4 bit 塞进浮点尾数，hfma2 减偏移（顺手乘 scale）' },
  { y: 290, n: '3', c: 'int4' as Fmt, t: '怎么喂 mma：ldmatrix 拆不开 4 bit，权重离线 repack 成 fragment 顺序' },
  { y: 450, n: '4', c: 'fp8' as Fmt, t: 'X 每步都要在线量化（求 scale 要扫一遍 X），一般和前面的 RMSNorm 融合成一个 kernel' },
]

const fill = (c: Fmt) => (c === 'none' ? 'var(--vp-c-bg-soft)' : `color-mix(in srgb, var(--qo-${c}) 16%, var(--vp-c-bg))`)
const stroke = (c: Fmt) => (c === 'none' ? 'var(--vp-c-text-3)' : `var(--qo-${c})`)
</script>

<template>
  <figure class="widget qo">
    <div class="scroll">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="量化 GEMM 总览：硬件约束、W4A16 和 FP8 两条路线的数据通路、scale 乘在哪">
        <defs>
          <marker id="qo-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" class="head" />
          </marker>
        </defs>

        <g v-for="p in panels" :key="p.t">
          <rect x="8" :y="p.y" width="684" :height="p.h" rx="8" class="panel" />
          <text x="22" :y="p.y + 24" class="ptitle">{{ p.t }}</text>
        </g>

        <!-- (a) -->
        <g v-for="p in pairs" :key="p.x">
          <text :x="p.x + 53" y="66" class="t" text-anchor="middle">×</text>
          <text :x="p.x + 114" y="66" class="t" :class="p.ok ? 'ok' : 'bad'">{{ p.ok ? '✓' : '✗ 没有这种 mma' }}</text>
        </g>
        <text x="22" y="94" class="t">所以 int4 × bf16 只能二选一：(b) 把 W 转回 bf16 再算，或 (c) 把 X 也量化，同格式直接算</text>

        <!-- 框 -->
        <g v-for="(b, i) in boxes" :key="'b' + i">
          <rect :x="b.x" :y="b.y" :width="b.w" :height="b.h" rx="6" class="box" />
          <text v-if="b.t" :x="b.left ? b.x + 8 : b.x + b.w / 2" :y="b.y + 16" class="t head-t" :text-anchor="b.left ? 'start' : 'middle'">{{ b.t }}</text>
        </g>
        <rect :x="col.reg + 60" :y="yb + 36" width="60" height="26" rx="4" class="box strong" />
        <text :x="col.reg + 90" :y="yb + 53" class="t mono" text-anchor="middle">dequant</text>
        <text :x="col.tc + 50" :y="yc + 64" class="t small muted" text-anchor="middle">算力 × 2</text>

        <!-- (d) 卡片里的字 -->
        <text x="32" y="532" class="t head-t">s 不随 K 变</text>
        <text x="32" y="552" class="t small">per-tensor、激活 per-token、权重 per-out-channel</text>
        <text x="32" y="572" class="t small muted">y = s · Σₖ xₖqₖ，s 提到求和外面</text>
        <text x="32" y="638" class="t small muted">两条路线一样，几乎免费</text>
        <text x="364" y="532" class="t head-t">s 随 K 变（沿 K 每 g 个一组）</text>
        <text x="364" y="552" class="t small">per-group / per-block</text>
        <text x="364" y="572" class="t small muted">y = Σ_g s_g · (组内 Σ xq)，只能在主循环里乘</text>

        <g v-for="(c, i) in chips" :key="'c' + i">
          <rect :x="c.x" :y="c.y" :width="c.w" :height="c.h ?? 22" rx="4" :style="{ fill: fill(c.c), stroke: stroke(c.c) }" class="chip" />
          <text :x="c.left ? c.x + 10 : c.x + c.w / 2" :y="c.y + (c.h ?? 22) / 2 + 4" class="t" :class="c.left ? 'small' : 'mono'"
            :text-anchor="c.left ? 'start' : 'middle'">{{ c.t }}</text>
        </g>

        <g v-for="(a, i) in arrows" :key="'a' + i">
          <line :x1="a.x1" :y1="a.y1" :x2="a.x2" :y2="a.y2" class="arrow" marker-end="url(#qo-arrow)" />
          <text v-if="a.t" :x="a.tx" :y="a.ty" class="t small muted" text-anchor="middle">{{ a.t }}</text>
        </g>

        <g v-for="m in marks" :key="'m' + m.n">
          <circle :cx="m.x" :cy="m.y" r="8" :style="{ fill: stroke(m.c) }" class="mark" />
          <text :x="m.x" :y="m.y + 3.5" class="mark-t" text-anchor="middle">{{ m.n }}</text>
        </g>
        <g v-for="n in notes" :key="'n' + n.n">
          <circle cx="28" :cy="n.y - 4" r="7" :style="{ fill: stroke(n.c) }" class="mark" />
          <text x="28" :y="n.y - 0.5" class="mark-t" text-anchor="middle">{{ n.n }}</text>
          <text x="42" :y="n.y" class="t small">{{ n.t }}</text>
        </g>
      </svg>
    </div>
    <figcaption class="muted">
      颜色表示数据格式：<span class="k bf16">bf16</span><span class="k int4">int4</span><span class="k fp8">fp8</span>。
      (b)(c) 的列对齐同一条存储层次；(b) 的 ①②③ 是 W4A16 要回答的三个实现问题，(c) 的 ④ 是 FP8 多出来的代价。
    </figcaption>
  </figure>
</template>

<style scoped>
.qo { --qo-bf16: #2a78d6; --qo-int4: #eb6834; --qo-fp8: #1baf7a; margin: 20px 0; }
:root.dark .qo { --qo-bf16: #3987e5; --qo-int4: #d95926; --qo-fp8: #199e70; }
.scroll { overflow-x: auto; }
svg { display: block; width: 100%; min-width: 600px; height: auto; background: var(--vp-c-bg); border-radius: 8px; }
.panel { fill: none; stroke: var(--wg-border); stroke-width: 1; }
.ptitle { font-size: 13px; font-weight: 600; fill: var(--wg-text); }
.box { fill: var(--vp-c-bg); stroke: var(--vp-c-text-3); stroke-width: 1; }
.box.strong { stroke: var(--vp-c-text-2); stroke-width: 1.2; }
.chip { stroke-width: 1.2; }
.t { font-size: 12px; fill: var(--wg-text); }
.t.small { font-size: 11px; }
.t.muted { fill: var(--wg-muted); }
.t.mono { font-family: var(--vp-font-family-mono); font-size: 11px; }
.head-t { font-weight: 600; font-size: 11.5px; }
.t.ok { fill: var(--qo-fp8); font-weight: 600; }
.t.bad { fill: #dc2626; }
.arrow { stroke: var(--vp-c-text-2); stroke-width: 1.2; }
.head { fill: var(--vp-c-text-2); }
.mark { stroke: var(--vp-c-bg); stroke-width: 1.5; }
.mark-t { font-size: 10px; font-weight: 700; fill: #fff; }
figcaption { font-size: 12.5px; margin-top: 10px; line-height: 1.7; }
.k { display: inline-block; padding: 0 6px; margin: 0 3px; border-radius: 4px; border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 16%, var(--vp-c-bg)); color: var(--wg-text); font-size: 11.5px; }
.k.bf16 { --c: var(--qo-bf16); } .k.int4 { --c: var(--qo-int4); } .k.fp8 { --c: var(--qo-fp8); }
</style>

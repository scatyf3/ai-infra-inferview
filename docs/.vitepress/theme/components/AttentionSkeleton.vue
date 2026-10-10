<script setup lang="ts">
import { ref } from 'vue'

// 六步骨架：每步一个节点，点开看这一步的代码、shape 和要点，链接跳到本页对应小节
// 锚点和 docs/leetgpu/attention.md 的标题一一对应，改标题时要同步改这里
interface Link { text: string; href: string }
interface Step {
  n: number
  name: string
  tensors: string // 这一步之后流向下一步的 tensor
  shape: string
  optional?: string // 骨架里由 if 守着、可能跳过的步
  code: string
  point: string
  links: Link[]
}

const steps: Step[] = [
  {
    n: 1, name: '排 shape', tensors: 'q, k, v', shape: '[..., M, D]',
    code: 'q = Q.reshape(N, h, dk).transpose(0, 1)   # [N, d_model] → [h, N, dk]',
    point: 'head 必须在 seq 前面：matmul 只对最后两维做矩阵乘，前面全当 batch。换维度用 transpose，不用 reshape。',
    links: [
      { text: '换 shape：第 1、6 步不同', href: '#换-shape-第-1、6-步不同' },
      { text: '#12 MHA 自己拆 head', href: '#_12-multi-head-attention-自己拆-head' },
      { text: '#26 Cross-Attention', href: '#_26-multi-head-cross-attention-q-和-k-v-不等长' },
      { text: '#80 GQA', href: '#_80-grouped-query-attention-k-v-head-少于-q-head' },
    ],
  },
  {
    n: 2, name: '位置编码', tensors: 'q, k', shape: '[..., M, D]', optional: 'cos is not None',
    code: 'q, k = rope(q, cos, sin), rope(k, cos, sin)',
    point: '只转 q、k，v 不动。必须在拆 head 之后：第 k 维和第 k + D/2 维配对，D 是单 head 维度。',
    links: [
      { text: '位置编码：第 2 步不同', href: '#位置编码-第-2-步不同' },
      { text: '#61 RoPE 旋转本身', href: '#_61-rotary-positional-embedding-旋转本身' },
      { text: '带 KV cache 时的位置', href: '#带-kv-cache-时的位置' },
      { text: '#115 QKV + RoPE + 写 cache', href: '#_115-fused-qkv-projection-rope-kv-cache-update-decode-的前半步' },
    ],
  },
  {
    n: 3, name: '打分', tensors: 'attn', shape: '[..., M_q, M_k]',
    code: 'attn = q @ k.transpose(-1, -2) / math.sqrt(D)',
    point: 'D 是单 head 维度。点积是 D 项求和，方差 ∝ D，不缩放 softmax 会接近 one-hot。',
    links: [{ text: '#6 Softmax Attention：骨架本身', href: '#_6-softmax-attention-骨架本身' }],
  },
  {
    n: 4, name: 'mask', tensors: 'attn', shape: '[..., M_q, M_k]', optional: 'allowed is not None',
    code: "attn = attn.masked_fill(~allowed, float('-inf'))",
    point: 'i = arange(M)[:, None] 是行（query），j = arange(M)[None, :] 是列（key），每条规则是 i、j 的一个 bool 条件。',
    links: [
      { text: '换 mask：第 4 步不同', href: '#换-mask-第-4-步不同' },
      { text: '#53 Causal', href: '#_53-causal-self-attention-不看未来' },
      { text: '#59 Sliding Window', href: '#_59-sliding-window-只看附近' },
      { text: '#112 Sinks', href: '#_112-attention-with-sinks-因果窗口-常驻开头' },
      { text: '#102 Varlen', href: '#_102-variable-length-causal-多条序列打包' },
      { text: '带 KV cache 时的 mask', href: '#带-kv-cache-时的-mask' },
    ],
  },
  {
    n: 5, name: 'softmax', tensors: 'attn', shape: '[..., M_q, M_k]',
    code: 'attn = torch.softmax(attn, dim=-1)',
    point: '沿 key 方向（最后一维），每个 query 一行归一化。方阵写成 dim=0 不报错，静默算错。',
    links: [{ text: '#6 Softmax Attention：骨架本身', href: '#_6-softmax-attention-骨架本身' }],
  },
  {
    n: 6, name: '乘 V、写回', tensors: 'out', shape: '[..., M_q, D]',
    code: 'output.copy_((attn @ v).transpose(0, 1).reshape(N, d_model))',
    point: 'key 维被求和掉。换回题目的 shape 再写进 output：shape 完全一致才能 out=，否则 copy_。',
    links: [
      { text: '换 shape：第 1、6 步不同', href: '#换-shape-第-1、6-步不同' },
      { text: '#12 MHA 合 head', href: '#_12-multi-head-attention-自己拆-head' },
      { text: '#26 Cross-Attention', href: '#_26-multi-head-cross-attention-q-和-k-v-不等长' },
    ],
  },
]

// 图的几何：六个框横排，q/k 两条线进打分，v 从排 shape 出来走底部直接进第 6 步
const BOX = { y: 30, h: 130, w: 96, gap: 34, x0: 82 }
const boxX = (n: number) => BOX.x0 + (n - 1) * (BOX.w + BOX.gap)
const MID = BOX.y + BOX.h / 2
const Y = { q: 70, k: 115, v: 140, rail: 190 }
const right = (n: number) => boxX(n) + BOX.w
const label: Record<number, { name: string; sub: string; shape: string }> = {
  1: { name: '排 shape', sub: 'head 挪前', shape: '[.., M, D]' },
  2: { name: '位置编码', sub: 'RoPE(q, k)', shape: '[.., M, D]' },
  3: { name: '打分', sub: 'q·kᵀ / √D', shape: '[M_q, M_k]' },
  4: { name: 'mask', sub: '-inf 填充', shape: '[M_q, M_k]' },
  5: { name: 'softmax', sub: 'dim = -1', shape: '[M_q, M_k]' },
  6: { name: '乘 V', sub: 'attn · v', shape: '[M_q, D]' },
}
const vPath = `M ${right(1)} ${Y.v} H ${right(1) + 18} V ${Y.rail} H ${boxX(6) - 20} V ${Y.v} H ${boxX(6)}`

const open = ref<number | null>(null)
const toggle = (n: number) => (open.value = open.value === n ? null : n)
</script>

<template>
  <div class="widget">
    <h4>Attention 六步骨架（点框展开）</h4>
    <div class="scroll">
      <svg :viewBox="`0 0 ${boxX(6) + BOX.w + 50} 205`" role="img" aria-label="attention 六步数据流">
        <defs>
          <marker id="as-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="head" />
          </marker>
        </defs>

        <text x="4" :y="MID + 6" class="io">QKV</text>
        <path :d="`M 44 ${MID} H ${boxX(1)}`" class="edge" marker-end="url(#as-arrow)" />

        <!-- q、k：1 → 2 → 3 -->
        <template v-for="t in (['q', 'k'] as const)" :key="t">
          <path :d="`M ${right(1)} ${Y[t]} H ${boxX(2)}`" class="edge" marker-end="url(#as-arrow)" />
          <path :d="`M ${right(2)} ${Y[t]} H ${boxX(3)}`" class="edge" marker-end="url(#as-arrow)" />
          <text :x="right(1) + 6" :y="Y[t] - 5" class="lbl">{{ t }}</text>
          <text :x="right(2) + 6" :y="Y[t] - 5" class="lbl">{{ t }}</text>
        </template>
        <!-- v：绕过 2–5 -->
        <path :d="vPath" class="edge v" marker-end="url(#as-arrow)" />
        <text :x="(right(2) + boxX(5)) / 2" :y="Y.rail - 6" class="lbl">v（不转、不打分）</text>
        <!-- attn：3 → 4 → 5 → 6 -->
        <path v-for="n in [3, 4, 5]" :key="'a' + n" :d="`M ${right(n)} ${MID - 20} H ${boxX(n + 1)}`" class="edge" marker-end="url(#as-arrow)" />
        <path :d="`M ${right(6)} ${MID} H ${right(6) + 30}`" class="edge" marker-end="url(#as-arrow)" />
        <text :x="right(6) + 36" :y="MID + 5" class="io">O</text>

        <g v-for="s in steps" :key="s.n" class="box" :class="{ active: open === s.n, optional: s.optional }"
           role="button" tabindex="0" :aria-expanded="open === s.n"
           @click="toggle(s.n)" @keydown.enter.prevent="toggle(s.n)" @keydown.space.prevent="toggle(s.n)">
          <rect :x="boxX(s.n)" :y="BOX.y" :width="BOX.w" :height="BOX.h" rx="6" />
          <text :x="boxX(s.n) + 8" :y="BOX.y + 16" class="num">{{ s.n }}</text>
          <text :x="boxX(s.n) + BOX.w / 2" :y="MID - 12" class="name">{{ label[s.n].name }}</text>
          <text :x="boxX(s.n) + BOX.w / 2" :y="MID + 12" class="sub">{{ label[s.n].sub }}</text>
          <text :x="boxX(s.n) + BOX.w / 2" :y="MID + 34" class="shape">{{ label[s.n].shape }}</text>
          <text v-if="s.optional" :x="boxX(s.n) + BOX.w - 8" :y="BOX.y + 16" class="opt">可跳过</text>
        </g>
      </svg>
    </div>
    <p class="muted hint">虚线框由 <code>if</code> 守着：没有位置编码 / mask 时直接跳过。</p>

    <div v-for="s in steps" v-show="open === s.n" :key="s.n" class="detail">
      <div class="head">
        <b>{{ s.n }}. {{ s.name }}</b>
        <span v-if="s.optional" class="muted mono">if {{ s.optional }}</span>
      </div>
      <pre><code>{{ s.code }}</code></pre>
      <p>{{ s.point }}</p>
      <div class="links">
        <span class="muted">跳到：</span>
        <a v-for="l in s.links" :key="l.href + l.text" :href="l.href">{{ l.text }}</a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scroll { overflow-x: auto; }
svg { display: block; width: 100%; min-width: 620px; height: auto; }
.edge { fill: none; stroke: var(--wg-muted); stroke-width: 1.5; }
.edge.v { stroke-dasharray: 5 4; }
.head { fill: var(--wg-muted); }
.io { font-size: 17px; font-weight: 600; font-family: var(--vp-font-family-mono); }
.lbl { font-size: 13px; fill: var(--wg-muted) !important; font-family: var(--vp-font-family-mono); }
.box { cursor: pointer; outline: none; }
.box rect { fill: var(--vp-c-bg); stroke: var(--wg-border); stroke-width: 1.5; transition: fill 0.15s, stroke 0.15s; }
.box.optional rect { stroke-dasharray: 6 4; }
.box:hover rect, .box:focus-visible rect { stroke: var(--vp-c-brand-1); }
.box.active rect { fill: var(--vp-c-brand-soft); stroke: var(--vp-c-brand-1); stroke-width: 2; }
.box text { text-anchor: middle; pointer-events: none; }
.num { text-anchor: start !important; font-size: 13px; font-weight: 700; fill: var(--vp-c-brand-1) !important; font-family: var(--vp-font-family-mono); }
.name { font-size: 17px; font-weight: 600; }
.sub { font-size: 13.5px; }
.shape { font-size: 12.5px; fill: var(--wg-muted) !important; font-family: var(--vp-font-family-mono); }
.opt { text-anchor: end !important; font-size: 11px; fill: var(--wg-muted) !important; }
.hint { font-size: 12px; margin: 6px 0 0; }
.detail { margin-top: 10px; padding: 10px 12px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-bg); border-radius: 0 6px 6px 0; }
.detail .head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.detail pre { margin: 6px 0; padding: 6px 10px; background: var(--vp-c-bg-soft); border-radius: 4px; overflow-x: auto; }
.detail pre code { font-size: 12px; background: none; padding: 0; }
.detail p { margin: 6px 0; line-height: 1.6; }
.links { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; font-size: 13px; }
.links a { padding: 1px 8px; border-radius: 4px; background: var(--vp-c-brand-soft); text-decoration: none; }
</style>

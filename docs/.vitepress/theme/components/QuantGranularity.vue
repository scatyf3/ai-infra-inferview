<script setup lang="ts">
import { computed, ref } from 'vue'

// 图里的小矩阵：X 是 [M, K]，W 是 [K, N]；g 代表真实的 group / block 大小（128、32、16）
const M = 4, K = 8, N = 4, g = 4
const OUTLIER_K = 2 // X 里的 outlier 通道

type Group = [number, number] | null // 组在两个方向上的编号；null 表示不量化
type Mode = {
  label: string
  x: (r: number, c: number) => Group
  w: (r: number, c: number) => Group
  sx: string
  sw: string
  note: string
}

const modes: Record<string, Mode> = {
  tensor: {
    label: 'per-tensor',
    x: () => [0, 0],
    w: () => [0, 0],
    sx: '1 个',
    sw: '1 个',
    note: '整个矩阵共用一个 scale，最省。但 X 里的 outlier 通道把整个 X 的 scale 撑大，所有数都受影响。典型：FP8 W8A8 的 static scale。',
  },
  tokenChannel: {
    label: 'per-token × per-channel',
    x: (r) => [r, 0],
    w: (_, c) => [0, c],
    sx: `[M, 1]，${M} 个`,
    sw: `[1, N]，${N} 个`,
    note: 'X 每行（一个 token）一个 scale，W 每列（一个输出通道）一个 scale。两者都不沿 K 变，能提到 Σₖ 外面，GEMM 做完在 epilogue 里乘 sₓ[i]·s_w[j]。outlier 通道横穿每一行，每个 token 的 scale 都被撑大。典型：W8A8。',
  },
  group: {
    label: 'per-group',
    x: () => null,
    w: (r, c) => [Math.floor(r / g), c],
    sx: '无（bf16）',
    sw: `[K/g, N]，${(K / g) * N} 个`,
    note: `X 保持 bf16 不量化。W 每列沿 K 每 g 个元素一组（图里 g = ${g}，实际常取 128）。scale 沿 K 变，提不出求和，所以 kernel 先把 W 反量化成 bf16 再乘。典型：W4A16（GPTQ / AWQ）。`,
  },
  block: {
    label: 'per-block',
    x: (r, c) => [r, Math.floor(c / g)],
    w: (r, c) => [Math.floor(r / g), c],
    sx: `[M, K/b]，${M * (K / g)} 个`,
    sw: `[K/b, N]，${(K / g) * N} 个`,
    note: `X 和 W 都沿 K 每 b 个元素一块（图里 b = ${g}，MXFP4 是 32，NVFP4 是 16）。outlier 只撑大它所在那一块的 scale。scale 沿 K 变，Tensor Core 每算完一块就乘上这块的两个 scale 再累加，Blackwell 硬件直接支持。典型：MXFP4 / NVFP4 W4A4。`,
  },
}

const mode = ref<keyof typeof modes>('tokenChannel')
const cur = computed(() => modes[mode.value])

const xPalette = ['#3b82f6', '#14b8a6', '#8b5cf6', '#06b6d4', '#22c55e']
const wPalette = ['#f97316', '#ec4899', '#eab308', '#ef4444', '#a855f7']
const color = (grp: Group, pal: string[]) => (grp ? pal[(grp[0] * 2 + grp[1]) % pal.length] : null)
const same = (a: Group, b: Group) => (a === null && b === null) || (!!a && !!b && a[0] === b[0] && a[1] === b[1])

function cells(rows: number, cols: number, f: (r: number, c: number) => Group, pal: string[], outlier: boolean) {
  const out = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const grp = f(r, c)
      const col = color(grp, pal)
      const edge = '2px solid var(--wg-text)'
      out.push({
        key: `${r}-${c}`,
        outlier: outlier && c === OUTLIER_K,
        style: {
          background: col ? `color-mix(in srgb, ${col} 38%, var(--vp-c-bg))` : 'var(--vp-c-bg)',
          borderRight: c < cols - 1 && !same(grp, f(r, c + 1)) ? edge : undefined,
          borderBottom: r < rows - 1 && !same(grp, f(r + 1, c)) ? edge : undefined,
        },
      })
    }
  return out
}

const xCells = computed(() => cells(M, K, cur.value.x, xPalette, true))
const wCells = computed(() => cells(K, N, cur.value.w, wPalette, false))
</script>

<template>
  <div class="widget">
    <h4>量化粒度：哪些元素共用一个 scale</h4>

    <div class="toggle-group modes">
      <button v-for="(m, k) in modes" :key="k" :class="{ active: mode === k }" @click="mode = k">{{ m.label }}</button>
    </div>

    <div class="mats">
      <div class="mat">
        <div class="title"><b>X</b> 激活 <span class="muted">[M, K]</span></div>
        <div class="axes">
          <span class="ylab">M（token）↓</span>
          <div class="grid" :style="{ gridTemplateColumns: `repeat(${K}, 1fr)` }">
            <i v-for="c in xCells" :key="c.key" :style="c.style" :class="{ outlier: c.outlier }" />
          </div>
        </div>
        <div class="xlab">K（输入通道）→</div>
        <div class="scale">sₓ：{{ cur.sx }}</div>
      </div>

      <div class="op">×</div>

      <div class="mat">
        <div class="title"><b>W</b> 权重 <span class="muted">[K, N]</span></div>
        <div class="axes">
          <span class="ylab">K（输入通道）↓</span>
          <div class="grid" :style="{ gridTemplateColumns: `repeat(${N}, 1fr)` }">
            <i v-for="c in wCells" :key="c.key" :style="c.style" />
          </div>
        </div>
        <div class="xlab">N（输出通道）→</div>
        <div class="scale">s_w：{{ cur.sw }}</div>
      </div>
    </div>

    <p class="legend muted">同色、粗线围起来的格子共用一个 scale；白格不量化。<span class="dot" /> 是 X 的 outlier 通道（第 {{ OUTLIER_K + 1 }} 列）。</p>
    <p class="note">{{ cur.note }}</p>
  </div>
</template>

<style scoped>
.modes { flex-wrap: wrap; margin-bottom: 14px; }
.mats { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 18px; }
.mat { display: flex; flex-direction: column; gap: 4px; }
.title { font-size: 13px; }
.axes { display: flex; align-items: center; gap: 6px; }
.ylab { writing-mode: vertical-rl; font-size: 11px; color: var(--wg-muted); }
.grid { display: grid; border: 2px solid var(--wg-text); border-radius: 3px; overflow: hidden; }
.grid i { width: 22px; height: 22px; box-sizing: border-box; border-right: 1px solid var(--wg-border); border-bottom: 1px solid var(--wg-border); position: relative; }
.grid i.outlier::after { content: ''; position: absolute; inset: 7px; border-radius: 50%; background: #dc2626; }
.xlab { font-size: 11px; color: var(--wg-muted); padding-left: 20px; }
.scale { font-size: 12.5px; font-family: var(--vp-font-family-mono); padding-left: 20px; }
.op { font-size: 22px; color: var(--wg-muted); }
.legend { font-size: 12px; margin: 12px 0 4px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #dc2626; margin: 0 2px; }
.note { font-size: 13px; line-height: 1.75; margin: 0; }
</style>

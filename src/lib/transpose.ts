/**
 * 矩阵转置的访存模型（纯函数），给 TransposeViz 用。
 * 缩小版：一个 warp = tile 边长个线程，shared memory 的 bank 数也等于 tile 边长；
 * HBM 的 sector 不缩小，仍是 32 B = 8 个 float。tile = 8 时，按行读 1 个 sector、按列读 8 个，比例和真实的 4 : 32 一样。
 */

export type TransposeMode = 'naive' | 'smem' | 'padded'
/** [row, col] */
export type Cell = [number, number]

/** 一个 32 B sector 装几个 float */
export const SECTOR_FLOATS = 8

/** 行主序 float 矩阵（ncols 列）上一组元素覆盖的 sector 编号（元素下标 / 8），去重 */
export function sectorsOf(cells: Cell[], ncols: number): number[] {
  return [...new Set(cells.map(([r, c]) => Math.floor((r * ncols + c) / SECTOR_FLOATS)))]
}

/**
 * shared memory 一次访问里，每个 bank 上的不同地址数。最大值就是几路冲突；
 * 多个线程访问同一地址算广播，只记一次。pitch 是 tile 每行实际占几个 float（padding 时 = tile + 1）。
 */
export function bankLoad(cells: Cell[], pitch: number, banks: number): number[] {
  const per = Array.from({ length: banks }, () => new Set<number>())
  for (const [r, c] of cells) {
    const a = r * pitch + c
    per[a % banks].add(a)
  }
  return per.map((s) => s.size)
}

export interface TransposeStep {
  /** read：从 in 读；sync：__syncthreads；write：写 out */
  kind: 'read' | 'sync' | 'write'
  /** 第几个 warp（= threadIdx.y） */
  warp: number
  /** 这一步访问的 HBM：哪个矩阵，cells[t] 是 lane t 访问的元素（全局坐标） */
  hbm?: { mat: 'in' | 'out'; cells: Cell[] }
  /** 这一步访问的 shared memory tile：cells[t] 是 lane t 访问的 tile 坐标 */
  smem?: { op: 'store' | 'load'; cells: Cell[] }
  /** HBM 上搬了几个 sector */
  sectors: number
  /** 每个 bank 上的地址数（没访问 shared memory 时为空） */
  banks: number[]
  /** 几路 bank conflict = shared memory 这次访问要分几次完成（没访问时为 0） */
  ways: number
}

/**
 * block (bx, by) 搬 in 的一个 tile × tile 块的全部访存步骤。n 是矩阵边长（行主序，n × n）。
 * - naive：线程 t 读 in[y0+w][x0+t]，直接写 out[x0+t][y0+w]，写是按列的。
 * - smem / padded：阶段 1 按行读 in、存进 tile[w][t]；__syncthreads；阶段 2 取 tile[t][w]、按行写 out[x0+w][y0+t]。
 *   padded 的 tile 每行多 1 个 float，按列取时错开 bank。
 */
export function transposeSteps(mode: TransposeMode, n: number, tile: number, bx: number, by: number): TransposeStep[] {
  const x0 = bx * tile, y0 = by * tile
  const pitch = mode === 'padded' ? tile + 1 : tile
  const lanes = Array.from({ length: tile }, (_, t) => t)
  const steps: TransposeStep[] = []
  const push = (kind: TransposeStep['kind'], warp: number, hbm?: TransposeStep['hbm'], smem?: TransposeStep['smem']) => {
    const banks = smem ? bankLoad(smem.cells, pitch, tile) : []
    steps.push({
      kind, warp, hbm, smem,
      sectors: hbm ? sectorsOf(hbm.cells, n).length : 0,
      banks,
      ways: banks.length ? Math.max(...banks) : 0,
    })
  }

  if (mode === 'naive') {
    for (const w of lanes) {
      push('read', w, { mat: 'in', cells: lanes.map((t): Cell => [y0 + w, x0 + t]) })
      push('write', w, { mat: 'out', cells: lanes.map((t): Cell => [x0 + t, y0 + w]) })
    }
    return steps
  }

  for (const w of lanes)
    push('read', w, { mat: 'in', cells: lanes.map((t): Cell => [y0 + w, x0 + t]) }, { op: 'store', cells: lanes.map((t): Cell => [w, t]) })
  push('sync', -1)
  for (const w of lanes)
    push('write', w, { mat: 'out', cells: lanes.map((t): Cell => [x0 + w, y0 + t]) }, { op: 'load', cells: lanes.map((t): Cell => [t, w]) })
  return steps
}

/** 一个 block 的合计：HBM sector 数，以及 shared memory 访问一共要几次（冲突时一次访问拆成 ways 次） */
export function transposeTotals(steps: TransposeStep[]) {
  return {
    sectors: steps.reduce((s, x) => s + x.sectors, 0),
    smemPasses: steps.reduce((s, x) => s + x.ways, 0),
    smemAccesses: steps.filter((x) => x.smem).length,
  }
}

/**
 * Triton 编程模型的下标计算（纯函数）。
 * 用于可视化：1D program / offs / mask，matmul 的 tile 划分、K 循环和 grouped launch 顺序。
 */

export const cdiv = (a: number, b: number) => Math.floor((a + b - 1) / b)

export interface VecProgram {
  pid: number
  /** offs = pid * BLOCK + arange(0, BLOCK) */
  offs: number[]
  /** mask = offs < n */
  mask: boolean[]
  /** mask 为 True 的 lane 数 */
  valid: number
}

/** 1D kernel：grid = cdiv(n, BLOCK)，每个 program 拿一段下标向量 */
export function vectorPrograms(n: number, block: number): VecProgram[] {
  return Array.from({ length: cdiv(n, block) }, (_, pid) => {
    const offs = Array.from({ length: block }, (_, i) => pid * block + i)
    const mask = offs.map((o) => o < n)
    return { pid, offs, mask, valid: mask.filter(Boolean).length }
  })
}

export interface TileCoord {
  pid: number
  pidM: number
  pidN: number
}

/**
 * 1D pid → 2D tile 坐标，和 Triton matmul 教程的 grouped ordering 一致。
 * groupM = 1 退化为行优先：pid_m = pid // num_pid_n, pid_n = pid % num_pid_n。
 */
export function pidToTile(pid: number, numPidM: number, numPidN: number, groupM: number): TileCoord {
  const numPidInGroup = groupM * numPidN
  const groupId = Math.floor(pid / numPidInGroup)
  const firstPidM = groupId * groupM
  const groupSizeM = Math.min(numPidM - firstPidM, groupM)
  const local = pid % numPidInGroup
  return { pid, pidM: firstPidM + (local % groupSizeM), pidN: Math.floor(local / groupSizeM) }
}

/** 所有 program 的 launch 顺序，下标即 pid */
export function launchOrder(numPidM: number, numPidN: number, groupM: number): TileCoord[] {
  return Array.from({ length: numPidM * numPidN }, (_, pid) => pidToTile(pid, numPidM, numPidN, groupM))
}

/**
 * 同一波并发的 program（pid ∈ [start, start + width)）要读多少条 A 行块、多少条 B 列块。
 * 同一波共享的条带可以命中 L2，条带越少 HBM 流量越小。
 */
export function waveFootprint(order: TileCoord[], start: number, width: number) {
  const wave = order.slice(start, start + width)
  return {
    rowsA: new Set(wave.map((t) => t.pidM)).size,
    colsB: new Set(wave.map((t) => t.pidN)).size,
    programs: wave.length,
  }
}

export interface L2Event extends TileCoord {
  wave: number
  /** 读 A 行条带 / B 列条带时是否命中 L2 */
  aHit: boolean
  bHit: boolean
  /** 到这个 program 为止（含）累计从 HBM 读了多少条条带 */
  hbm: number
}

/**
 * 按 launch 顺序逐个派发 program，模拟 A/B 条带的 L2 命中。
 * 简化：同一波读过的条带留在 L2，换波时清空（上一波的条带视为已被挤出）。
 */
export function l2Replay(order: TileCoord[], width: number): L2Event[] {
  let rows = new Set<number>(), cols = new Set<number>(), hbm = 0
  return order.map((t, i) => {
    if (i % width === 0) { rows = new Set(); cols = new Set() }
    const aHit = rows.has(t.pidM), bHit = cols.has(t.pidN)
    rows.add(t.pidM); cols.add(t.pidN)
    hbm += Number(!aHit) + Number(!bHit)
    return { ...t, wave: Math.floor(i / width), aHit, bHit, hbm }
  })
}

/** 第 idx 块在长度为 size 的维度上覆盖 [start, end)，end 已按 size 截断（越界部分靠 mask） */
export function tileRange(idx: number, block: number, size: number) {
  const start = idx * block
  return { start, end: Math.min(start + block, size), masked: Math.max(0, start + block - size) }
}

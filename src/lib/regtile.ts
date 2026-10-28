/**
 * GEMM 第二级 tiling（寄存器 tile）的指令计数（纯函数）。
 * 每个线程负责 C tile 里 TM × TN 个输出；每个 kk 从 shared memory 读 a[TM]、b[TN]，做 TM·TN 次 FMA。
 */

export interface RegTileCost {
  /** 每个 kk 每线程的 shared memory load 指令数 */
  lds: number
  /** 每个 kk 每线程的 FMA 指令数 */
  fma: number
  /** FMA 占这两类指令的比例 */
  fmaShare: number
  /** 每线程的寄存器：acc + a + b */
  regs: number
}

/** vec = 4 表示用 float4 一次读 4 个连续的数（a 要连续，需要 As 在 shared memory 里转置存） */
export function regTileCost(tm: number, tn: number, vec: 1 | 4 = 1): RegTileCost {
  const lds = Math.ceil(tm / vec) + Math.ceil(tn / vec)
  const fma = tm * tn
  return { lds, fma, fmaShare: fma / (lds + fma), regs: tm * tn + tm + tn }
}

/**
 * 整个 block 在一个 kk 里从 shared memory 读多少个数，以及其中有多少个不同的数。
 * As 的一列（BM 个）被 BN / TN 个线程各读一遍，Bs 的一行（BN 个）被 BM / TM 个线程各读一遍。
 */
export function blockSmemReads(bm: number, bn: number, tm: number, tn: number) {
  const threads = (bm / tm) * (bn / tn)
  return { threads, reads: threads * (tm + tn), distinct: bm + bn }
}

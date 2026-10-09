---
title: CUDA Tiled Matmul
status: draft
tags: [cuda, gemm, handson]
difficulty: 3
order: 10
related: [/gpu/tensor-core-gemm, /gpu/gpu-architecture, /inference/prefill-decode-roofline, /handson/cuda-reduce, /handson/kernel-mindset]
stack: [k-gemm]
leetgpu: [2, 22]
---

# CUDA Tiled Matmul

> shared memory tiling、register tiling、bank conflict · [LeetGPU #2 Matrix Multiplication、#22 GEMM](https://leetgpu.com/challenges)（题面：[#2](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/2_matrix_multiplication)、[#22](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/22_gemm)）

## 一句话结论

朴素 matmul 每个线程算 C 的一个元素，要从全局内存读 2K 个数、只做 K 次乘加，算术强度 0.25 FLOP/B，被带宽卡死。tiled 版本让一个 block 负责 C 的一个 BM×BN 块，沿 K 维每次把 A 的 BM×BK 和 B 的 BK×BN 块搬进 shared memory 给 block 内所有线程复用，全局访存量除以 tile 边长；再让每个线程算 TM×TN 个元素（寄存器 tile），shared memory 读取量再除一次。下一步是 Tensor Core，见 [Tensor Core 与 GEMM tiling](/gpu/tensor-core-gemm)。

## 约定

- 本页统一写 **C[M, N] = A[M, K] · B[K, N]**（BLAS 惯例，K 是归约维），全部 **row-major**：`A[i][k]` 在 `A[i * K + k]`，`B[k][j]` 在 `B[k * N + j]`，`C[i][j]` 在 `C[i * N + j]`。
- **LeetGPU #2 的命名不一样**：它的 A 是 M×N、B 是 N×K、C 是 M×K，即内维叫 N、输出列数叫 K。入口 `extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K)`，调用本页 kernel 时把 N 和 K 对调。fp32 输入输出，容差 1e-4。
- **LeetGPU #22**：A[M, K]、B[K, N]、C[M, N] 都是 fp16，算 `C = α·(A·B) + β·C`，要求 fp32 累加，α、β 是 fp32。入口 `solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)`。
- 计数：一次乘加（MAC）= 2 FLOP。GEMM 的 MAC 数是 M·N·K，FLOP = 2·M·N·K。

## 先算上限

LeetGPU #2 的性能测试换成本页记号是 M = 8192、K = 6144、N = 4096：

| | 公式 | 数值 |
|---|---|---|
| FLOP | 2·M·N·K | 4.12 × 10¹¹ |
| 最少要搬的字节（每个矩阵只读写一次） | 4·(MK + KN + MN) | 436 MB |
| T4 fp32 算力下限（8.1 TFLOPS） | FLOP / 8.1e12 | **50.9 ms** |
| T4 带宽下限（300 GB/s） | bytes / 300e9 | 1.45 ms |

算力下限比带宽下限大 35 倍，所以 GEMM **只要数据复用做够，就是 compute-bound**。问题是"做够"是多少：T4 的 ridge point 是 8.1e12 / 300e9 ≈ 27 FLOP/B（[T4 数据手册](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)），kernel 从 HBM 读的每个字节要配 27 次以上的 FLOP。下面每个版本都算这个比值。

## v0：朴素版，算术强度 0.25

每个线程算一个 `C[row][col]`：

```cpp
__global__ void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // threadIdx.x → 列，见下面"合并访存"
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
// launch: dim3 block(32, 32); dim3 grid(cdiv(N, 32), cdiv(M, 32));
```

**算术强度**：每次循环读 2 个 float（8 B），做 1 次 MAC（2 FLOP），0.25 FLOP/B。不算 cache 的话，T4 上最多 300 GB/s × 0.25 = 75 GFLOPS，是峰值的 1%。

**合并访存**：一个 warp 是 32 个 `threadIdx.x` 连续的线程。`threadIdx.x` 映射到 `col` 时，同一时刻 warp 读 `B[k*N + col .. col+31]`，32 个连续 float = 128 字节，一次事务；`A[row*K + k]` 32 个线程读同一个地址，广播。反过来把 `threadIdx.x` 映射到 `row`，warp 读 A 的 32 行、地址跨步 K×4 字节，每个线程各自一次事务。Simon Boehm 的实测里只改这一处映射，就从 309 到 1987 GFLOPS（作者自测，RTX A6000，4092² fp32，[worklog](https://siboehm.com/articles/22/CUDA-MMM) kernel 1→2）。合并访存的硬件背景见 [GPU 架构](/gpu/gpu-architecture)。

## v1：shared memory tiling

### 为什么能复用

C 的同一行 32 个元素都要读 A 的同一行，同一列 32 个元素都要读 B 的同一列。让一个 32×32 的 block 负责 C 的 32×32 块，沿 K 每次搬 A 的 32×32 和 B 的 32×32 进 shared memory，block 内每个数被用 32 次。

一般地，block 负责 BM×BN，每步沿 K 走 BK：

- 每步从 HBM 读 (BM + BN)·BK 个 float
- 每步做 BM·BN·BK 次 MAC

$$
\text{AI}_{\text{global}} = \frac{2 \cdot BM \cdot BN \cdot BK}{4 \cdot (BM + BN) \cdot BK} = \frac{BM \cdot BN}{2(BM + BN)} \ \text{FLOP/B}
$$

BM = BN = 32 时 AI = 8 FLOP/B，比朴素版高 32 倍，但还没过 T4 的 27。

### 代码

```cpp
template <int TILE>
__global__ void sgemm_smem(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 协作加载：每个线程搬 A、B 各一个数；越界补 0，补的 0 对乘加没有影响
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
        __syncthreads();                       // 等整块搬完才能读
        for (int kk = 0; kk < TILE; ++kk)
            acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();                       // 等所有人算完才能覆盖下一块
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
// launch: dim3 block(32, 32); dim3 grid(cdiv(N, 32), cdiv(M, 32)); sgemm_smem<32><<<grid, block>>>(...)
```

逐项检查：

- **加载合并**：一个 warp 是同一 `ty`、`tx = 0..31`，读 `A[row*K + k0 .. k0+31]` 和 `B[(k0+ty)*N + col .. col+31]`，都是 128 字节连续。
- **bank conflict**：内循环里一个 warp 读 `As[ty][kk]` 是同一个地址（广播），读 `Bs[kk][tx]` 是 32 个连续地址、落在 32 个 bank，都没有冲突。
- **两个 `__syncthreads()` 都不能省**：第一个保证读之前写完，第二个保证下一轮覆盖之前大家都读完。
- **资源**：1024 线程、8 KB shared memory 每 block。

### 还差在哪

内循环每次 MAC 要从 shared memory 读 2 个 float。shared memory 虽然比 HBM 快一个数量级以上，但"每 FMA 两次 smem load"意味着 load 指令和 FMA 一样多，算力单元一半时间在等 load。Boehm 的这一版只到 2980 GFLOPS，cuBLAS 的 12.8%（作者自测）。

## v2：register tiling（2D blocktiling）

### 思路

让每个线程算 C 的 TM×TN 个元素，结果放寄存器 `acc[TM][TN]`。内循环每个 `kk`：

- 从 As 读一列 TM 个数进 `regA`，从 Bs 读一行 TN 个数进 `regB`
- 做 TM×TN 次 MAC：`acc[i][j] += regA[i] * regB[j]`（一个外积）

| | 每个 `kk` 的 smem load | 每个 `kk` 的 MAC | smem load / MAC |
|---|---|---|---|
| v1（TM = TN = 1） | 2 | 1 | 2 |
| v2（TM = TN = 8） | 16 | 64 | **0.25** |

同时 block tile 可以做大：256 个线程 × 64 个结果 = 16384 = 128×128。按上面的公式 BM = BN = 128 时全局 AI = 128·128 / (2·256) = 32 FLOP/B，过了 T4 的 27。

这就是 Boehm 的 kernel 5：BM = BN = 128、BK = 8、TM = TN = 8、256 线程，15972 GFLOPS，cuBLAS 的 68.7%（作者自测）。

### 代码

一个模板同时服务 #2（fp32）和 #22（fp16 输入、fp32 累加、带 α / β）：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }

template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ float from_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f32<__half>(float x) { return __float2half(x); }

// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C，row-major，fp32 累加
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_2d(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
        int M, int N, int K, float alpha, float beta) {
    constexpr int NT = (BM / TM) * (BN / TN);   // 每 block 线程数：128/8 * 128/8 = 256
    __shared__ float As[BK][BM];                // 转置存：As[k][m]，内循环按 m 读
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);           // 本线程负责 block tile 里的第几组 TM 行
    const int tCol = tid % (BN / TN);           // 第几组 TN 列
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[TM][TN] = {};                      // 全 0
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 搬 A 的 BM×BK 块：第 e 个元素是块内 (r, c) = (e / BK, e % BK)，越界补 0
        for (int e = tid; e < BM * BK; e += NT) {
            int r = e / BK, c = e % BK;
            int gr = blockRow + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? to_f32(A[gr * K + gc]) : 0.f;
        }
        // 搬 B 的 BK×BN 块
        for (int e = tid; e < BK * BN; e += NT) {
            int r = e / BN, c = e % BN;
            int gr = k0 + r, gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? to_f32(B[gr * N + gc]) : 0.f;
        }
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            for (int i = 0; i < TM; ++i) regA[i] = As[kk][tRow * TM + i];
            for (int j = 0; j < TN; ++j) regB[j] = Bs[kk][tCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regA[i] * regB[j];     // 外积累加
        }
        __syncthreads();
    }

    // epilogue：alpha、beta、写回、转回 T
    for (int i = 0; i < TM; ++i) {
        int r = blockRow + tRow * TM + i;
        for (int j = 0; j < TN; ++j) {
            int c = blockCol + tCol * TN + j;
            if (r < M && c < N) {
                float out = alpha * acc[i][j];
                if (beta != 0.f) out += beta * to_f32(C[r * N + c]);   // beta = 0 时不读 C
                C[r * N + c] = from_f32<T>(out);
            }
        }
    }
}

template <typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));                       // 256 线程
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);         // x 方向是列块，y 方向是行块
    gemm_2d<T, BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

// ---------- LeetGPU #2：A 是 M×N、B 是 N×K、C 是 M×K ----------
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    launch_gemm<float>(A, B, C, /*M=*/M, /*N=*/K, /*K=*/N, 1.f, 0.f);   // 对调 N 和 K
}

// ---------- LeetGPU #22：单独提交，入口换成这个 ----------
// extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K,
//                       float alpha, float beta) {
//     launch_gemm<__half>(A, B, C, M, N, K, alpha, beta);
// }
```

逐项检查：

- **搬运**：As 块 128×8 = 1024 个数、256 线程，每线程 4 个；Bs 同样 4 个。一个 warp 搬 A 时是 4 行 × 8 个连续 float，每个 32 字节 sector 都用满；搬 B 时是一行里 32 个连续 float。
- **为什么 As 转置存**：内循环同一 warp 的 32 个线程只有 2 个不同的 `tRow`，读 `As[kk][tRow*8 + i]` 是 2 个相距 8 的地址，落在不同 bank、各自广播，无冲突。不转置的话两个地址相距 64 个 float，落在同一 bank，2 路冲突。代价是写 `As[c][r]` 时 warp 内 8 个不同 `c` 落在同一个 bank（地址相距 128），8 路冲突，但写每个 tile 只发生一次，读发生 BK·TM 次。
- **还剩的冲突**：读 `Bs[kk][tCol*8 + j]`，warp 内 16 个 `tCol`，地址相距 8 个 float，bank `8·tCol mod 32` 只有 4 个取值，按标量 load 算是 4 路冲突。Boehm 后续的 kernel 6（转置 As + `float4` 向量化读写）、kernel 10（warptiling）继续提速，到 cuBLAS 的 78% 和 94%（作者自测）；他专门消除 bank conflict 的 kernel 7、8 冲突是消了，但整体反而更慢，没放进正文。
- **`beta == 0` 时不读 C**：#2 的 C 是 `torch.empty` 出来的，里面可能是 NaN，`0 * NaN = NaN`。BLAS 的约定也是 β = 0 时不读 C。
- **#22 的 fp16**：搬进 shared memory 时就转成 fp32，累加和 epilogue 全在 fp32，最后一步才 `__float2half`，符合题目"fp32 累加"的要求。
- **寄存器**：`acc` 64 + `regA` 8 + `regB` 8 = 80 个 float，加上地址变量一般在 128 个寄存器以内；`__launch_bounds__(256)` 告诉编译器按 256 线程分配。

### tile 大小要跟着问题规模变

#22 的性能测试是 M = N = K = 1024。用 128×128 的 block tile 只有 8 × 8 = 64 个 block，分到 T4 的 40 个 SM 上，24 个 SM 拿到 2 个、16 个 SM 只拿到 1 个：负载不均，而且每个 SM 只有 8–16 个 warp 可切换，藏访存延迟的余量很小。换成 BM = BN = 64（TM = TN = 4，同样 256 线程）就是 256 个 block，AI 降到 64·64/(2·128) = 16 FLOP/B，但 SM 都有活干。这正是 autotune 在选的东西。

同一题的 FLOP 是 2·1024³ = 2.15 × 10⁹，T4 CUDA core fp32 下限 0.27 ms；用 Tensor Core（fp16 输入、fp32 累加，65 TFLOPS）下限 33 µs。题目允许 WMMA，想冲排名就要上 Tensor Core，见 [Tensor Core 与 GEMM tiling](/gpu/tensor-core-gemm)。

## 版本对比

| 版本 | 每个 C 元素的全局读（float） | 全局 AI（FLOP/B） | smem load / MAC | Boehm 实测（GFLOPS，作者自测） |
|---|---|---|---|---|
| v0 朴素，合并访存 | 2K | 0.25 | — | 1987 |
| v1 smem，TILE = 32 | 2K / 32 | 8 | 2 | 2980 |
| v2 2D 寄存器 tile，128/128/8/8/8 | K / 64 | 32 | 0.25 | 15972 |
| cuBLAS | | | | 23250 |

（v0 的实测远高于 75 GFLOPS 的理论值，因为 L1 / L2 cache 替你做了一部分复用；前两列是"不算 cache"的口径。Boehm 的测试平台是 RTX A6000，fp32 峰值约 30 TFLOPS。）

## PyTorch 参考与正确性检查

用 torch 把分块逻辑原样写一遍（block tile、沿 K 的步进、越界补 0、外积累加），可以在 CPU 上先确认索引没写错：

```python
import torch

def tiled_matmul(A, B, BM=4, BN=4, BK=2):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros(M, N, dtype=A.dtype)
    for bm in range(0, M, BM):                         # grid.y
        for bn in range(0, N, BN):                     # grid.x
            acc = torch.zeros(BM, BN, dtype=A.dtype)   # 整个 block 的寄存器 tile 拼起来
            for k0 in range(0, K, BK):
                As = torch.zeros(BM, BK, dtype=A.dtype)   # 越界补 0
                Bs = torch.zeros(BK, BN, dtype=A.dtype)
                a = A[bm:bm + BM, k0:k0 + BK]; As[:a.shape[0], :a.shape[1]] = a
                b = B[k0:k0 + BK, bn:bn + BN]; Bs[:b.shape[0], :b.shape[1]] = b
                for kk in range(BK):
                    acc += As[:, kk:kk + 1] * Bs[kk:kk + 1, :]   # 外积：[BM,1] * [1,BN]
            m, n = min(BM, M - bm), min(BN, N - bn)
            C[bm:bm + m, bn:bn + n] = acc[:m, :n]      # 写回判界
    return C

A, B = torch.randn(7, 5), torch.randn(5, 9)            # 故意不整除
assert torch.allclose(tiled_matmul(A, B), A @ B, atol=1e-5)
```

CUDA 版在有 GPU 的机器上用 `load_inline` 对拍（文件里放上面的 `gemm_2d` 和 `launch_gemm`）：

```python
from torch.utils.cpp_extension import load_inline

cuda_src = open("gemm.cu").read() + r"""
#include <torch/extension.h>
torch::Tensor run_f32(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    launch_gemm<float>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, 1.f, 0.f);
    return C;
}
torch::Tensor run_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C, double alpha, double beta) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    launch_gemm<__half>((const __half*)A.data_ptr<at::Half>(), (const __half*)B.data_ptr<at::Half>(),
                        (__half*)C.data_ptr<at::Half>(), M, N, K, (float)alpha, (float)beta);
    return C;
}
"""
mod = load_inline("gemm_ext",
                  cpp_sources="torch::Tensor run_f32(torch::Tensor, torch::Tensor);"
                              "torch::Tensor run_f16(torch::Tensor, torch::Tensor, torch::Tensor, double, double);",
                  cuda_sources=cuda_src, functions=["run_f32", "run_f16"])

for M, K, N in [(1, 3, 1), (2, 2, 2), (7, 30, 13), (129, 257, 130), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda")
    assert torch.allclose(mod.run_f32(A, B), A @ B, rtol=1e-4, atol=1e-4)

    A16, B16 = A.half(), B.half()
    C16 = torch.randn(M, N, device="cuda").half()
    ref = (1.5 * (A16.float() @ B16.float()) + 0.5 * C16.float()).half()
    got = mod.run_f16(A16, B16, C16.clone(), 1.5, 0.5)
    assert torch.allclose(got.float(), ref.float(), rtol=5e-2, atol=5e-2)   # #22 的容差
```

## 面试追问

::: details Q：tile 大小为什么不能一直加大？
三个约束。一是 shared memory：每 block 占 (BM + BN)·BK·4 字节，加 double buffering 再 ×2，SM 上能驻留的 block 变少。二是寄存器：每线程 TM·TN 个累加器，TM = TN = 8 已经 64 个，再大就 spill 到 local memory（实际在 HBM）。三是问题规模：BM、BN 变大时 block 数变少，像上面 1024³ 的例子只有 64 个 block，40 个 SM 分得不均；M、N 不整除时边角的浪费也更大。实际是在复用和并行度之间调甜点，这就是 autotune 做的事。
:::

::: details Q：double buffering 解决什么问题？
v1、v2 都是"搬一块 → 同步 → 算一块 → 同步"，搬的时候算力闲着，算的时候访存闲着。double buffering 开两份 As/Bs，算第 k 块时就把第 k+1 块的 load 发出去。Ampere 起有 `cp.async`，可以从全局内存直接异步拷到 shared memory，不占寄存器；更早的卡（比如 T4 所在的 Turing）只能先 load 进寄存器、算完再写进 smem。
:::

::: details Q：为什么 GEMM 能 compute-bound，而 reduce、norm 不能？
算术强度。GEMM 搬 O(N²) 的数据做 O(N³) 的计算，数据复用随 tile 变大而增大，tile 做到 128×128 时每字节 32 FLOP 就能过 ridge point。reduce / norm 每个元素只参与 O(1) 次运算，算术强度是常数（0.25–1 FLOP/B 量级），再怎么写都是 memory-bound，能做的只有打满带宽和融合。见 [Prefill vs Decode 与 Roofline](/inference/prefill-decode-roofline)。
:::

## 参考

- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)（版本对比表里的全部实测数字）
- Mark Harris, [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)（bank 与 `__syncthreads`）
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)（block / warp / thread 三级 tiling 的工程化形态）
- [Triton 教程 · Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)（同样的 tiling，Triton 写法）

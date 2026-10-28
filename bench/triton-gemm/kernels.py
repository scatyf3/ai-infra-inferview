"""GEMM tiling ladder in Triton: C[M, N] = A[M, K] @ B[K, N], bf16 in/out, fp32 accumulate.

Every version after v0 is the same kernel `matmul_tiled` with one more knob turned:

    v0  naive          one program per output element, inner product along K (no reuse)
    v1  block tile     BM x BN tile per program, operands in shared memory, FMA on CUDA cores
    v2  tensor core    same tile, tl.dot on bf16 -> mma instructions
    v3  grouped order  GROUP_M = 8, so one wave of programs covers a square patch of C (L2 reuse)
    v4  pipelining     num_stages = 3, load tile k+1 while computing tile k
    v5  autotune       pick BM / BN / BK / warps / stages per shape
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_naive(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BK: tl.constexpr):
    # program (i, j) computes C[i, j] alone: reads a row of A and a column of B, uses each element once
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    acc = tl.zeros((BK,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + i * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0)
        b = tl.load(b_ptr + offs_k * stride_bk + j * stride_bn, mask=offs_k < K, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)
    tl.store(c_ptr + i * stride_cm + j * stride_cn, tl.sum(acc).to(tl.bfloat16))


@triton.jit
def matmul_tiled(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GROUP_M: tl.constexpr, TENSOR_CORE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    # GROUP_M = 1 is plain row-major order; larger groups walk down GROUP_M rows before moving right
    num_pid_in_group = GROUP_M * num_pid_n
    group = pid // num_pid_in_group
    first_m = group * GROUP_M
    group_rows = min(num_pid_m - first_m, GROUP_M)
    local = pid % num_pid_in_group
    pid_m = first_m + local % group_rows
    pid_n = local // group_rows

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k0), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k0) & (offs_n[None, :] < N), other=0.0)
        if TENSOR_CORE:
            acc = tl.dot(a, b, acc)                     # bf16 x bf16 -> fp32 on Tensor Cores
        else:
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                         input_precision="ieee")       # fp32 FMA on CUDA cores
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


AUTOTUNE_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3), (256, 128, 32, 8, 3),
        (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (64, 64, 64, 4, 4), (64, 64, 32, 4, 5),
    ]
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_autotuned(a_ptr, b_ptr, c_ptr, M, N, K,
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    # same body as v4; a jit function called from another jit function is inlined
    matmul_tiled(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BM, BN, BK, GROUP_M, True)


def _args(a, b, c):
    M, K = a.shape
    _, N = b.shape
    return (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))


def naive(a, b, BK=128):
    M, N = a.shape[0], b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    matmul_naive[(M * N,)](*_args(a, b, c), BK=BK)
    return c


def tiled(a, b, BM=128, BN=128, BK=32, GROUP_M=1, tensor_core=True, num_warps=4, num_stages=1):
    M, N = a.shape[0], b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_tiled[grid](*_args(a, b, c), BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M,
                       TENSOR_CORE=tensor_core, num_warps=num_warps, num_stages=num_stages)
    return c


def autotuned(a, b):
    M, N = a.shape[0], b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    matmul_autotuned[grid](*_args(a, b, c))
    return c

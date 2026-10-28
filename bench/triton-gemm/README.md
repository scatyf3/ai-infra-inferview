# Triton GEMM tiling ladder

Code for the measured report on [GEMM Tiling](https://scatyf3.github.io/ai-infra-inferview/gpu/tensor-core-gemm#实测-triton-逐级优化-rtx-5070-ti).
Each version turns one more knob of the same kernel; the page explains the why, this folder holds the code and the raw numbers.

| file | what |
|---|---|
| `kernels.py` | v0 naive, `matmul_tiled` (v1–v4, one kernel with knobs), v5 autotuned |
| `bench.py` | correctness check against fp32 `torch.matmul`, then `do_bench` sweeps; writes `results/<gpu>.json` |
| `results/rtx5070ti.json` | raw results the page chart reads |

| version | change | knob in `tiled(...)` |
|---|---|---|
| v0 | one program per output, inner product along K | `naive()` |
| v1 | 64 × 64 block tile, fp32 FMA on CUDA cores | `tensor_core=False` |
| v2 | 128 × 128 × 64 tile on Tensor Cores | `tensor_core=True` |
| v3 | grouped program order for L2 reuse | `GROUP_M=8` |
| v4 | software pipelining | `num_stages=3` |
| v5 | autotune over 8 tile / warp / stage configs | `autotuned()` |

## Run

Windows with an RTX 50-series card needs a CUDA 12.8 build of torch and the `triton-windows` package:

```sh
python -m venv .venv && .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy "triton-windows<3.6"   # the report used 3.5.1
python bench.py --quick   # 4 sizes, ~1 min
python bench.py           # full sweep, ~10 min
```

On Linux, `pip install torch` already brings `triton`.

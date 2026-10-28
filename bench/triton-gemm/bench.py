"""Benchmark the GEMM tiling ladder; writes results/rtx5070ti.json (read by the docs page chart).

    python bench.py              # full run, ~10 min on an RTX 5070 Ti
    python bench.py --quick      # a few sizes, for a smoke test
    python bench.py --only group # rerun one experiment, merge into the json
"""
import argparse
import json
import platform
import subprocess
import threading
import time
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

import kernels as k

LADDER = [
    # id, label, fn, largest size to run (naive and FMA get slow)
    ("v0", "v0 朴素（内积）", lambda a, b: k.naive(a, b), 2048),
    ("v1", "v1 block tile · CUDA core", lambda a, b: k.tiled(a, b, 64, 64, 32, tensor_core=False), 8192),
    ("v2", "v2 + Tensor Core", lambda a, b: k.tiled(a, b, 128, 128, 64, num_warps=8), None),
    ("v3", "v3 + 分组顺序", lambda a, b: k.tiled(a, b, 128, 128, 64, GROUP_M=8, num_warps=8), None),
    ("v4", "v4 + 软件流水", lambda a, b: k.tiled(a, b, 128, 128, 64, GROUP_M=8, num_warps=8, num_stages=3), None),
    ("v5", "v5 autotune", lambda a, b: k.autotuned(a, b), None),
    ("cublas", "cuBLAS（torch.matmul）", lambda a, b: a @ b, None),
]
SIZES = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384]


def tflops(n, ms):
    return 2 * n ** 3 / (ms * 1e-3) / 1e12


def make(n):
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    return a, b


def check(fn, a, b):
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(fn(a, b), ref, rtol=2e-2, atol=1e-1)


def bench(fn, a, b):
    fn(a, b)  # compile / autotune outside the timed region
    return do_bench(lambda: fn(a, b), warmup=50, rep=200, return_mode="median")


def run_ladder(sizes):
    series = []
    for vid, label, fn, cap in LADDER:
        pts = []
        for n in sizes:
            if cap and n > cap:
                continue
            a, b = make(n)
            if n <= 4096:
                check(fn, a, b)
            ms = bench(fn, a, b)
            pts.append({"n": n, "ms": round(ms, 4), "tflops": round(tflops(n, ms), 2)})
            print(f"{vid:7s} n={n:6d} {ms:9.3f} ms {pts[-1]['tflops']:7.2f} TFLOP/s", flush=True)
            del a, b
        series.append({"id": vid, "label": label, "points": pts})
    return series


def run_tiles(n=4096):
    # tile size vs throughput: same kernel as v4, only BM x BN changes
    rows = []
    a, b = make(n)
    for bm, bn, bk, w in [(16, 16, 32, 1), (32, 32, 32, 2), (64, 64, 32, 4), (128, 128, 32, 4), (128, 256, 32, 8)]:
        fn = lambda a, b: k.tiled(a, b, bm, bn, bk, GROUP_M=8, num_warps=w, num_stages=3)
        check(fn, a, b)
        ms = bench(fn, a, b)
        rows.append({"bm": bm, "bn": bn, "bk": bk, "warps": w, "ai": round(bm * bn / (bm + bn), 1),
                     "ms": round(ms, 4), "tflops": round(tflops(n, ms), 2)})
        print("tile", rows[-1], flush=True)
    return {"n": n, "rows": rows}


def run_group(sizes=(2048, 3072, 4096, 6144, 8192, 12288, 16384)):
    # program order vs tile size: row-major (GROUP_M = 1) only hurts once B no longer fits in L2
    # and BM is below the ridge point (row-major re-reads all of B per row of tiles, so AI ~ BM)
    series = []
    for bm, bk, w, st in [(64, 32, 4, 4), (128, 64, 8, 3)]:
        for g in (1, 8):
            pts = []
            for n in sizes:
                a, b = make(n)
                fn = lambda a, b: k.tiled(a, b, bm, bm, bk, GROUP_M=g, num_warps=w, num_stages=st)
                ms = bench(fn, a, b)
                pts.append({"n": n, "ms": round(ms, 4), "tflops": round(tflops(n, ms), 2)})
                print(f"group bm={bm} G={g} n={n} {pts[-1]['tflops']} TFLOP/s", flush=True)
            series.append({"id": f"t{bm}g{g}", "label": f"{bm}×{bm} tile，{'行优先' if g == 1 else '分组 GROUP_M=8'}",
                           "points": pts})
    return {"sizes": list(sizes), "series": series}


def run_stages(n=4096):
    # num_stages under two configs: a 4-warp block that fits several per SM, and an 8-warp block with BK = 64
    rows = []
    a, b = make(n)
    for bm, bn, bk, w in [(128, 128, 32, 4), (128, 128, 64, 8)]:
        for s in (1, 2, 3, 4):
            fn = lambda a, b: k.tiled(a, b, bm, bn, bk, GROUP_M=8, num_warps=w, num_stages=s)
            ms = bench(fn, a, b)
            rows.append({"config": f"{bm}×{bn}×{bk}, {w} warps", "stages": s,
                         "ms": round(ms, 4), "tflops": round(tflops(n, ms), 2)})
            print("stages", rows[-1], flush=True)
    return {"n": n, "rows": rows}


def sm_clock_under_load(seconds=3.0):
    # sample SM clock and power while cuBLAS runs a large GEMM back to back
    samples = []
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits",
                             "-lms", "200"], stdout=subprocess.PIPE, text=True)
    t = threading.Thread(target=lambda: samples.extend(proc.stdout), daemon=True)
    t.start()
    a, b = make(8192)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    proc.terminate()
    t.join(timeout=2)
    vals = [tuple(float(x) for x in s.split(",")) for s in samples if s.strip()]
    vals = vals[len(vals) // 3:]  # skip the ramp-up
    return {"sm_clock_mhz": round(sum(v[0] for v in vals) / len(vals)), "power_w": round(sum(v[1] for v in vals) / len(vals))}


def env():
    p = torch.cuda.get_device_properties(0)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {"gpu": p.name, "sms": p.multi_processor_count, "l2_mb": p.L2_cache_size // 2 ** 20,
            "driver": driver, "torch": torch.__version__, "triton": triton.__version__,
            "os": platform.platform(), "date": time.strftime("%Y-%m-%d")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["ladder", "tiles", "stages", "group"],
                    help="rerun one experiment and merge it into the existing --out file")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "rtx5070ti.json"))
    args = ap.parse_args()
    sizes = [512, 1024, 2048, 4096] if args.quick else SIZES
    if args.only:
        out = json.loads(Path(args.out).read_text(encoding="utf-8"))
        out[args.only] = {"ladder": lambda: run_ladder(sizes), "tiles": run_tiles,
                          "stages": run_stages, "group": run_group}[args.only]()
    else:
        out = {"env": env(), "load": sm_clock_under_load(), "sizes": sizes,
               "ladder": run_ladder(sizes), "tiles": run_tiles(), "stages": run_stages(), "group": run_group()}
        out["load_after"] = sm_clock_under_load()
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out)

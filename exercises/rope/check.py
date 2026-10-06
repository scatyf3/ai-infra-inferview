"""RoPE 三连的本地对拍。

用法（仓库根目录）：
    uv run --no-project --with torch python exercises/rope/check.py p1
    uv run --no-project --with torch python exercises/rope/check.py p2
    uv run --no-project --with torch python exercises/rope/check.py p3

默认检查同目录下的 p1_rope_causal.py / p2_rope_decode.py / p3_gqa_rope.py；
要检查别的文件：check.py p1 --file path/to/mine.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # 让 p2 / p3 能 from p1_rope_causal import apply_rope
import _reference as R  # noqa: E402

FILES = {'p1': 'p1_rope_causal.py', 'p2': 'p2_rope_decode.py', 'p3': 'p3_gqa_rope.py'}
ATOL = 1e-4


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def close(a, b):
    return torch.is_tensor(a) and a.shape == b.shape and torch.allclose(a, b, atol=ATOL, rtol=1e-4)


def fail(msg):
    print(f'✗ {msg}')
    sys.exit(1)


def run(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError as e:
        fail(str(e) or f'{fn.__name__} 还没实现')


def rand_case(g, Hq=None, Hkv=None):
    T = int(torch.randint(1, 17, (1,), generator=g))
    H = Hq or int(torch.randint(1, 5, (1,), generator=g))
    D = 2 * int(torch.randint(1, 9, (1,), generator=g))
    start = int(torch.randint(0, 50, (1,), generator=g))
    pos = torch.arange(start, start + T)
    q = torch.randn(T, H, D, generator=g)
    k = torch.randn(T, Hkv or H, D, generator=g)
    v = torch.randn(T, Hkv or H, D, generator=g)
    return q, k, v, pos


def check_apply_rope(mod, g):
    for _ in range(50):
        x, _, _, pos = rand_case(g)
        got, want = run(mod.apply_rope, x.clone(), pos), R.apply_rope(x, pos)
        if close(got, want):
            continue
        if not torch.is_tensor(got) or got.shape != want.shape:
            fail(f'apply_rope 返回的 shape 是 {getattr(got, "shape", type(got))}，应该和输入一样 {tuple(x.shape)}')
        if close(got, R.apply_rope_interleaved(x, pos)):
            fail('apply_rope 用的是相邻两维配对（GPT-J 风格）；本题约定前后两半配对：(x[i], x[i + D/2])')
        if close(got, R.apply_rope(x, pos - pos[0])):
            fail('apply_rope 没用传进来的 positions，像是默认从 0 开始数了')
        if not torch.allclose(got.norm(dim=-1), x.norm(dim=-1), atol=1e-3):
            fail('apply_rope 改变了向量长度；旋转应该保长，检查 cos / sin 有没有配对写错')
        fail(f'apply_rope 数值不对（最大误差 {(got - want).abs().max():.3g}）')
    print('✓ apply_rope')


def check_p1(mod, g):
    check_apply_rope(mod, g)
    for _ in range(50):
        q, k, v, pos = rand_case(g)
        got, want = run(mod.solve, q.clone(), k.clone(), v.clone(), pos), R.p1_solve(q, k, v, pos)
        if close(got, want):
            continue
        if close(got, R.attention(R.apply_rope(q, pos), k, v)) or close(got, R.attention(q, R.apply_rope(k, pos), v)):
            fail('只转了 q 或只转了 k；两个都要转')
        if close(got, R.attention(R.apply_rope(q, pos), R.apply_rope(k, pos), R.apply_rope(v, pos))):
            fail('v 也被转了；RoPE 只作用在 q、k 上')
        if close(got, R.attention(R.apply_rope(q, pos), R.apply_rope(k, pos), v, causal=False)):
            fail('没加 causal mask')
        fail(f'solve 数值不对（最大误差 {(got - want).abs().max():.3g}）')
    # 性质：所有位置整体平移，scores 不变，所以输出不变
    q, k, v, pos = rand_case(g)
    shift = 1000
    a, b = run(mod.solve, q, k, v, pos), run(mod.solve, q, k, v, pos + shift)
    if not close(a, b):
        fail('位置整体平移后输出变了：分数应该只依赖相对位置 i - j')
    print(f'✓ solve，且位置整体平移 {shift} 后输出不变（只依赖 i - j）')


def check_p2(mod, g):
    for _ in range(50):
        q, k, v, pos = rand_case(g)
        T, H, D = q.shape
        past = T - 1
        # 前 past 个 token 的 cache：k 已经转过
        k_cache = R.apply_rope(k[:past], torch.arange(past))
        v_cache = v[:past].clone()
        args = (q[-1].clone(), k[-1].clone(), v[-1].clone(), k_cache.clone(), v_cache.clone(), past)
        got = run(mod.decode_step, *args)
        want = R.p2_decode_step(*args)
        if not isinstance(got, tuple) or len(got) != 3:
            fail('decode_step 要返回 (out, k_cache_new, v_cache_new)')
        out, kc, vc = got
        if kc is None or kc.shape != (past + 1, H, D):
            fail(f'k_cache_new 的 shape 应该是 {(past + 1, H, D)}，拿到 {getattr(kc, "shape", kc)}')
        # 位置 0 的旋转是恒等变换，past_len = 0 时「没转」和「转对了」分不出来，只在 past_len > 0 时诊断
        if past > 0 and not close(kc[-1], want[1][-1]):
            if close(kc[-1], k[-1]):
                fail('k_cache_new 最后一行没转：cache 里存的是转过的 k')
            for off, name in ((-1, 'past_len - 1'), (1, 'past_len + 1'), (-past, '0')):
                if off != 0 and close(kc[-1], R.apply_rope(k[-1][None], torch.tensor([past + off]))[0]):
                    fail(f'新 token 的位置用成了 {name}；应该是 past_len（cache 里已有 past_len 个，从 0 数）')
        if not close(kc, want[1]):
            fail('k_cache_new 不对：前 past_len 行应该原样保留，最后一行是转过的新 k')
        if not close(vc, want[2]):
            fail('v_cache_new 不对：v 不转，直接拼到末尾')
        if past > 0 and not close(out, want[0]) and close(out, R.attention(q[-1][None], kc, vc, causal=False)[0]):
            fail('q 没转')
        if not close(out, want[0]):
            fail(f'out 数值不对（最大误差 {(out - want[0]).abs().max():.3g}）')
        # 和整段 prefill 的最后一个位置一致
        full = R.p1_solve(q, k, v, torch.arange(T))[-1]
        if not close(out, full):
            fail('decode 结果和整段 prefill 的最后一个位置不一致')
    print('✓ decode_step，且和整段 prefill 的最后一个位置一致')


def check_p3(mod, g):
    for _ in range(50):
        Hkv = int(torch.randint(1, 4, (1,), generator=g))
        group = int(torch.randint(1, 4, (1,), generator=g))
        q, k, v, pos = rand_case(g, Hq=Hkv * group, Hkv=Hkv)
        got, want = run(mod.solve, q.clone(), k.clone(), v.clone(), pos), R.p3_solve(q, k, v, pos)
        if close(got, want):
            continue
        if not torch.is_tensor(got) or got.shape != want.shape:
            fail(f'输出 shape 应该是 {tuple(want.shape)}（和 q 一样），拿到 {getattr(got, "shape", type(got))}')
        if Hkv > 1 and group > 1:
            tiled = R.attention(R.apply_rope(q, pos), R.apply_rope(k, pos).repeat(1, group, 1), v.repeat(1, group, 1))
            if close(got, tiled):
                fail('KV head 用 repeat 平铺了：q head h 应该对应 kv head h // group，用 repeat_interleave')
        fail(f'solve 数值不对（最大误差 {(got - want).abs().max():.3g}）')
    print('✓ solve（GQA 分组和 RoPE 都对）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('problem', choices=FILES)
    ap.add_argument('--file')
    a = ap.parse_args()
    mod = load(a.file or HERE / FILES[a.problem])
    g = torch.Generator().manual_seed(0)
    {'p1': check_p1, 'p2': check_p2, 'p3': check_p3}[a.problem](mod, g)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VLLM_ROOT = str(PROJECT_ROOT / "vllm_stageb_snapshot")
AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, VLLM_ROOT)
sys.path.insert(0, AVELANG_DIR)

from vllm.model_executor.layers.fla.ops import chunk_delta_h as vllm_chunk_delta_h
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk_gdn
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v14_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v14_mfma_layout,
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_chunked_avelang_v14_mfma_layout,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v16_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v16_2wave_mfma_layout,
    qwen_gdn_chunked_avelang_v16_mfma_layout,
)


def patch_vllm_rocm_autotune_configs():
    if getattr(torch.version, "hip", None) is None:
        return
    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    configs = getattr(autotuner, "configs", None)
    if configs:
        filtered = [c for c in configs if getattr(c, "num_stages", None) != 4]
        if len(filtered) != len(configs):
            autotuner.configs = filtered
            autotuner.cache.clear()
            print(f"patched vLLM ROCm autotune configs: {len(configs)} -> {len(filtered)}")


def sync():
    torch.cuda.synchronize()


def l2norm(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 1234, with_initial_state: bool = True):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def call_vllm(q, k, v, g, beta, initial_state, scale):
    return vllm_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )


def time_fn(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def stage_breakdown(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int, *, version: str):
    chunk = 16

    def cumsum():
        return qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)

    g_cumsum = cumsum()

    def kkt():
        return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)

    a = kkt()

    def solve():
        return qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)

    a_solved = solve()

    def wu_fn():
        return qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)

    if version == "v14":
        gdr_fn = qwen_gdn_chunk_gdr_avelang_v14_mfma_layout
    elif version == "v16":
        gdr_fn = qwen_gdn_chunk_gdr_avelang_v16_2wave_mfma_layout
    else:
        raise ValueError(version)

    w, u = wu_fn()

    def gdr():
        return gdr_fn(k, w, u, g_cumsum, initial_state=initial_state, chunk_size=chunk)

    h, vn, _ = gdr()

    def chunk_o():
        return qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)

    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu_fn, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def run_t(t: int, warmup: int, repeat: int):
    q, k, v, g, beta, initial_state = make_inputs(t)
    scale = 128 ** -0.5
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,layout=[B,T,H,D],chunk=16")

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v14 = lambda: qwen_gdn_chunked_avelang_v14_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )
    f_v16 = lambda: qwen_gdn_chunked_avelang_v16_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )

    out_vllm, fs_vllm = f_vllm()
    out_v14, fs_v14 = f_v14()
    out_v16, fs_v16 = f_v16()
    sync()

    stage_warmup = max(2, warmup // 4)
    stage_repeat = max(5, repeat // 4)
    vllm_ms = time_fn(f_vllm, warmup, repeat)
    v14_ms = time_fn(f_v14, warmup, repeat)
    v16_ms = time_fn(f_v16, warmup, repeat)
    stages_v14 = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v14")
    stages_v16 = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v16")
    result = {
        "T": t,
        "vllm_ms": vllm_ms,
        "v14_mfma_ms": v14_ms,
        "v16_mfma_ms": v16_ms,
        "chunk_gdr_speedup_v16_vs_v14": stages_v14["chunk_gdr"] / stages_v16["chunk_gdr"],
        "full_speedup_v16_vs_v14": v14_ms / v16_ms,
        "slowdown_v16_vs_vllm": v16_ms / vllm_ms,
        "v16_vs_v14_out_err": max_err(out_v16, out_v14),
        "v16_vs_v14_fs_err": max_err(fs_v16, fs_v14),
        "v16_vs_vllm_out_err": max_err(out_v16, out_vllm),
        "v16_vs_vllm_fs_err": max_err(fs_v16, fs_vllm),
        "stages_v14": stages_v14,
        "stages_v16": stages_v16,
    }
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    patch_vllm_rocm_autotune_configs()
    rows = [run_t(t, args.warmup, args.repeat) for t in args.T]
    print("summary")
    for r in rows:
        s14 = r["stages_v14"]
        s16 = r["stages_v16"]
        print(
            f"T={r['T']} vLLM={r['vllm_ms']:.4f}ms v14={r['v14_mfma_ms']:.4f}ms "
            f"v16={r['v16_mfma_ms']:.4f}ms full_speedup_v16_vs_v14={r['full_speedup_v16_vs_v14']:.4f} "
            f"chunk_gdr_speedup_v16_vs_v14={r['chunk_gdr_speedup_v16_vs_v14']:.4f} "
            f"slowdown_v16_vs_vllm={r['slowdown_v16_vs_vllm']:.4f} "
            f"v14_w_u={s14['w_u']:.4f}ms v14_chunk_gdr={s14['chunk_gdr']:.4f}ms v14_chunk_o={s14['chunk_o']:.4f}ms "
            f"v16_w_u={s16['w_u']:.4f}ms v16_chunk_gdr={s16['chunk_gdr']:.4f}ms v16_chunk_o={s16['chunk_o']:.4f}ms "
            f"out_err_v14={r['v16_vs_v14_out_err']:.6g} fs_err_v14={r['v16_vs_v14_fs_err']:.6g} "
            f"out_err_vllm={r['v16_vs_vllm_out_err']:.6g} fs_err_vllm={r['v16_vs_vllm_fs_err']:.6g}"
        )


if __name__ == "__main__":
    main()

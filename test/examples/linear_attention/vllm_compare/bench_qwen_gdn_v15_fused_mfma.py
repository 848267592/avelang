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
from qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v15_fused_mfma_layout,
    qwen_gdn_fused_chunk_gdr_o_avelang_v15_mfma_layout,
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


def stage_breakdown_v14(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int):
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

    def wu():
        return qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)

    w, u = wu()

    def gdr():
        return qwen_gdn_chunk_gdr_avelang_v14_mfma_layout(k, w, u, g_cumsum, initial_state=initial_state, chunk_size=chunk)

    h, vn, _ = gdr()

    def chunk_o():
        return qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)

    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def stage_breakdown_v15(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int):
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

    def wu():
        return qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)

    w, u = wu()

    def fused():
        return qwen_gdn_fused_chunk_gdr_o_avelang_v15_mfma_layout(
            q,
            k,
            w,
            u,
            g_cumsum,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk,
        )

    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "fused_chunk_gdr_o": time_fn(fused, warmup, repeat),
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
    f_v15 = lambda: qwen_gdn_chunked_avelang_v15_fused_mfma_layout(
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
    out_v15, fs_v15 = f_v15()
    sync()

    stage_warmup = max(2, warmup // 4)
    stage_repeat = max(5, repeat // 4)
    vllm_ms = time_fn(f_vllm, warmup, repeat)
    v14_ms = time_fn(f_v14, warmup, repeat)
    v15_ms = time_fn(f_v15, warmup, repeat)
    stages_v14 = stage_breakdown_v14(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat)
    stages_v15 = stage_breakdown_v15(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat)
    result = {
        "T": t,
        "vllm_ms": vllm_ms,
        "v14_mfma_ms": v14_ms,
        "v15_fused_mfma_ms": v15_ms,
        "full_speedup_v15_vs_v14": v14_ms / v15_ms,
        "fused_stage_speedup_vs_v14_chunk_gdr_plus_o": (stages_v14["chunk_gdr"] + stages_v14["chunk_o"])
        / stages_v15["fused_chunk_gdr_o"],
        "slowdown_v15_vs_vllm": v15_ms / vllm_ms,
        "v15_vs_v14_out_err": max_err(out_v15, out_v14),
        "v15_vs_v14_fs_err": max_err(fs_v15, fs_v14),
        "v15_vs_vllm_out_err": max_err(out_v15, out_vllm),
        "v15_vs_vllm_fs_err": max_err(fs_v15, fs_vllm),
        "stages_v14": stages_v14,
        "stages_v15": stages_v15,
    }
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()

    patch_vllm_rocm_autotune_configs()
    results = [run_t(t, args.warmup, args.repeat) for t in args.T]
    print("summary")
    for row in results:
        s14 = row["stages_v14"]
        s15 = row["stages_v15"]
        print(
            f"T={row['T']} vLLM={row['vllm_ms']:.4f}ms "
            f"v14={row['v14_mfma_ms']:.4f}ms v15={row['v15_fused_mfma_ms']:.4f}ms "
            f"full_speedup_v15_vs_v14={row['full_speedup_v15_vs_v14']:.4f} "
            f"fused_stage_speedup={row['fused_stage_speedup_vs_v14_chunk_gdr_plus_o']:.4f} "
            f"slowdown_v15_vs_vllm={row['slowdown_v15_vs_vllm']:.4f} "
            f"v14_chunk_gdr={s14['chunk_gdr']:.4f}ms v14_chunk_o={s14['chunk_o']:.4f}ms "
            f"v15_fused_chunk_gdr_o={s15['fused_chunk_gdr_o']:.4f}ms "
            f"v15_w_u={s15['w_u']:.4f}ms "
            f"out_err_v14={row['v15_vs_v14_out_err']:.9g} fs_err_v14={row['v15_vs_v14_fs_err']:.9g}"
        )


if __name__ == "__main__":
    main()

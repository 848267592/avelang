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
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v11_mfma_layout_fixed import qwen_gdn_chunked_avelang_v11_mfma_layout
from qwen_gdn_chunked_avelang_v12_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v12_mfma_layout,
    qwen_gdn_chunk_o_avelang_v12_mfma_layout,
    qwen_gdn_chunked_avelang_v12_mfma_layout,
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
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=True,
        cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False,
    )


def time_fn(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record(); fn(); end.record(); sync()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def stage_breakdown_v12(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int):
    chunk = 16
    def cumsum(): return qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    g_cumsum = cumsum()
    def kkt(): return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a = kkt()
    def solve(): return qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    a_solved = solve()
    def wu(): return qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk, prefer_optimized=True)
    w, u = wu()
    def gdr(): return qwen_gdn_chunk_gdr_avelang_v12_mfma_layout(k, w, u, g_cumsum, initial_state=initial_state, chunk_size=chunk)
    h, vn, _ = gdr()
    def chunk_o(): return qwen_gdn_chunk_o_avelang_v12_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def run_t(t: int, warmup: int, repeat: int):
    q, k, v, g, beta, initial_state = make_inputs(t)
    scale = 128 ** -0.5
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,layout=[B,T,H,D],chunk=16")

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v11 = lambda: qwen_gdn_chunked_avelang_v11_mfma_layout(q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=16, use_mfma_chunk_gdr=True, use_clean_kernel=True, use_update_mfma=True)
    f_v12 = lambda: qwen_gdn_chunked_avelang_v12_mfma_layout(q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=16)

    out_vllm, fs_vllm = f_vllm()
    out_v11, fs_v11 = f_v11()
    out_v12, fs_v12 = f_v12()
    sync()

    result = {
        "T": t,
        "vllm_ms": time_fn(f_vllm, warmup, repeat),
        "v11_update_mfma_ms": time_fn(f_v11, warmup, repeat),
        "v12_mfma_ms": time_fn(f_v12, warmup, repeat),
        "v12_vs_v11_out_err": max_err(out_v12, out_v11),
        "v12_vs_v11_fs_err": max_err(fs_v12, fs_v11),
        "v12_vs_vllm_out_err": max_err(out_v12, out_vllm),
        "v12_vs_vllm_fs_err": max_err(fs_v12, fs_vllm),
        "stages_v12": stage_breakdown_v12(q, k, v, g, beta, initial_state, scale, max(2, warmup // 4), max(5, repeat // 4)),
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
        print(
            f"T={r['T']} vLLM={r['vllm_ms']:.4f}ms v11_update_mfma={r['v11_update_mfma_ms']:.4f}ms "
            f"v12_mfma={r['v12_mfma_ms']:.4f}ms speedup_v12_vs_v11={r['v11_update_mfma_ms']/r['v12_mfma_ms']:.4f} "
            f"vllm_over_v12={r['vllm_ms']/r['v12_mfma_ms']:.4f} "
            f"out_err_v11={r['v12_vs_v11_out_err']:.6g} fs_err_v11={r['v12_vs_v11_fs_err']:.6g}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import qwen_gdn_chunked_avelang_v24_kkt_mfma_layout
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32 as qwen_gdn_fused_chunk_gdr_full_original,
    qwen_gdn_fused_chunk_gdr_full_reference,
)
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp import (
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32 as qwen_gdn_fused_chunk_gdr_full_rewrite,
)
from qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto import (
    MODE_FUSED_PRED_UPDATE_SKELETON,
    qwen_gdn_fused_pred_update_proto_avelang_v29_mfma32,
)
from qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best import (
    qwen_gdn_pred_only_avelang_v29_mfma32_grouped_v4_best,
)


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_v24_inputs(t: int, seed: int = 1234):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def time_fn(fn, warmup: int, repeat: int) -> float:
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


def run_t(t: int, warmup: int, repeat: int) -> dict[str, float]:
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = _make_inputs(t, seed=29000 + t)
    h, fs = qwen_gdn_fused_chunk_gdr_full_original(k, w, u, gdr_decay, gdr_g_last_exp, initial_state)
    h_rewrite, fs_rewrite = qwen_gdn_fused_chunk_gdr_full_rewrite(
        k, w, u, gdr_decay, gdr_g_last_exp, initial_state
    )
    sync()
    h_ref, fs_ref = qwen_gdn_fused_chunk_gdr_full_reference(k, w, u, gdr_decay, gdr_g_last_exp, initial_state)
    sync()
    h_err = (h - h_ref).abs()
    fs_err = (fs - fs_ref).abs()
    rewrite_h_err = (h_rewrite - h).abs()
    rewrite_fs_err = (fs_rewrite - fs).abs()

    # Isolate update correctness: w=0 removes pred feedback while preserving
    # the fused update dataflow.
    w_zero = torch.zeros_like(w)
    h_zero, fs_zero = qwen_gdn_fused_chunk_gdr_full_rewrite(
        k, w_zero, u, gdr_decay, gdr_g_last_exp, initial_state
    )
    sync()
    h_zero_ref, fs_zero_ref = qwen_gdn_fused_chunk_gdr_full_reference(k, w_zero, u, gdr_decay, gdr_g_last_exp, initial_state)
    sync()
    fs_zero_err = (fs_zero - fs_zero_ref).abs()

    f_v29_original = lambda: qwen_gdn_fused_chunk_gdr_full_original(
        k, w, u, gdr_decay, gdr_g_last_exp, initial_state
    )
    f_v29_rewrite = lambda: qwen_gdn_fused_chunk_gdr_full_rewrite(
        k, w, u, gdr_decay, gdr_g_last_exp, initial_state
    )
    f_grouped = lambda: qwen_gdn_pred_only_avelang_v29_mfma32_grouped_v4_best(w, u, initial_state)
    f_skeleton = lambda: qwen_gdn_fused_pred_update_proto_avelang_v29_mfma32(
        w, u, initial_state, variant=MODE_FUSED_PRED_UPDATE_SKELETON
    )

    q, k24, v24, g24, beta24, init24 = make_v24_inputs(t, seed=24000 + t)
    f_v24 = lambda: qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(
        q,
        k24,
        v24,
        g24,
        beta24,
        initial_state=init24,
        scale=128 ** -0.5,
        chunk_size=16,
    )

    row = {
        "T": t,
        "v24_full_ms": time_fn(f_v24, warmup, repeat),
        "v29_grouped_pred_only_ms": time_fn(f_grouped, warmup, repeat),
        "v29_fused_skeleton_ms": time_fn(f_skeleton, warmup, repeat),
        "v29_original_ms": time_fn(f_v29_original, warmup, repeat),
        "v29_kfrag_rewrite_exp_ms": time_fn(f_v29_rewrite, warmup, repeat),
        "h_max_abs": h_err.max().item(),
        "h_mean_abs": h_err.mean().item(),
        "final_state_max_abs": fs_err.max().item(),
        "final_state_mean_abs": fs_err.mean().item(),
        "rewrite_vs_original_h_max_abs": rewrite_h_err.max().item(),
        "rewrite_vs_original_h_mean_abs": rewrite_h_err.mean().item(),
        "rewrite_vs_original_final_state_max_abs": rewrite_fs_err.max().item(),
        "rewrite_vs_original_final_state_mean_abs": rewrite_fs_err.mean().item(),
        "w_zero_final_state_max_abs": fs_zero_err.max().item(),
        "w_zero_final_state_mean_abs": fs_zero_err.mean().item(),
    }
    print(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    rows = [run_t(t, args.warmup, args.repeat) for t in args.T]
    print("summary")
    for row in rows:
        print(
            f"T={row['T']} v24_full={row['v24_full_ms']:.6f} "
            f"grouped_pred_only={row['v29_grouped_pred_only_ms']:.6f} "
            f"fused_skeleton={row['v29_fused_skeleton_ms']:.6f} "
            f"v29_original={row['v29_original_ms']:.6f} "
            f"v29_kfrag_rewrite={row['v29_kfrag_rewrite_exp_ms']:.6f} "
            f"h_max_abs={row['h_max_abs']:.6e} final_state_max_abs={row['final_state_max_abs']:.6e} "
            f"rewrite_vs_original_h_max_abs={row['rewrite_vs_original_h_max_abs']:.6e} "
            f"rewrite_vs_original_final_state_max_abs={row['rewrite_vs_original_final_state_max_abs']:.6e} "
            f"w_zero_final_state_max_abs={row['w_zero_final_state_max_abs']:.6e}"
        )


if __name__ == "__main__":
    main()

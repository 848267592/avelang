#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v22_gdr_ablation_layout_fixed import (
    VARIANT_TO_MODE,
    qwen_gdn_chunk_gdr_avelang_v22_ablation_mfma_layout,
)

ALL_VARIANTS = [
    "v17_predecay",
    "full_baseline",
    "no_h_write",
    "no_vn_write",
    "no_h_no_vn_write",
    "pred_only",
    "update_only_dummy",
    "pred_vn_only_no_update",
    "update_no_state_decay",
    "distributed_vn_vdecay",
]


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 1234, with_initial_state: bool = True):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, v, g, beta, initial_state


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


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def build_gdr_inputs(k, v, g, beta, chunk: int):
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    w, u = qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)
    return g_cumsum, w, u, gdr_decay, gdr_g_last_exp


def make_gdr_fn(variant: str, k, w, u, g_cumsum, gdr_decay, gdr_g_last_exp, initial_state):
    if variant == "v17_predecay":
        return lambda: qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            gdr_decay,
            gdr_g_last_exp,
            initial_state=initial_state,
            chunk_size=16,
        )
    if variant not in VARIANT_TO_MODE:
        raise ValueError(f"unknown variant {variant}")
    return lambda: qwen_gdn_chunk_gdr_avelang_v22_ablation_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        gdr_decay,
        gdr_g_last_exp,
        initial_state=initial_state,
        chunk_size=16,
        variant=variant,
    )


def run_t(t: int, variants: list[str], warmup: int, repeat: int) -> dict:
    k, v, g, beta, initial_state = make_inputs(t, seed=22000 + t)
    g_cumsum, w, u, gdr_decay, gdr_g_last_exp = build_gdr_inputs(k, v, g, beta, 16)
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,chunk=16,chunk_gdr_only")

    fns = {
        variant: make_gdr_fn(variant, k, w, u, g_cumsum, gdr_decay, gdr_g_last_exp, initial_state)
        for variant in variants
    }
    outputs = {variant: fn() for variant, fn in fns.items()}
    sync()

    if "v17_predecay" not in outputs:
        ref = make_gdr_fn("v17_predecay", k, w, u, g_cumsum, gdr_decay, gdr_g_last_exp, initial_state)()
        sync()
    else:
        ref = outputs["v17_predecay"]

    latencies = {variant: time_fn(fn, warmup, repeat) for variant, fn in fns.items()}
    v17_latency = latencies.get("v17_predecay")
    if v17_latency is None:
        v17_latency = time_fn(make_gdr_fn("v17_predecay", k, w, u, g_cumsum, gdr_decay, gdr_g_last_exp, initial_state), warmup, repeat)

    rows = {}
    for variant in variants:
        h, vn, fs = outputs[variant]
        row = {
            "latency_ms": latencies[variant],
            "speedup_vs_v17": v17_latency / latencies[variant],
            "delta_vs_v17": latencies[variant] - v17_latency,
        }
        if variant in {"full_baseline", "distributed_vn_vdecay", "v17_predecay"}:
            row["h_err"] = max_err(h, ref[0])
            row["vn_err"] = max_err(vn, ref[1])
            row["final_state_err"] = max_err(fs, ref[2])
        rows[variant] = row
        print(
            "chunk_gdr_latency,"
            f"T={t},variant={variant},"
            f"latency_ms={row['latency_ms']:.6f},"
            f"speedup_vs_v17={row['speedup_vs_v17']:.4f},"
            f"delta_vs_v17={row['delta_vs_v17']:.6f},"
            f"h_err={row.get('h_err', float('nan')):.9g},"
            f"vn_err={row.get('vn_err', float('nan')):.9g},"
            f"final_state_err={row.get('final_state_err', float('nan')):.9g}"
        )
    return {"T": t, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    unknown = [v for v in args.variants if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    results = [run_t(t, args.variants, args.warmup, args.repeat) for t in args.T]
    print("summary")
    for result in results:
        t = result["T"]
        for variant, row in result["rows"].items():
            print(
                f"T={t} variant={variant} latency={row['latency_ms']:.6f}ms "
                f"speedup_vs_v17={row['speedup_vs_v17']:.4f} "
                f"delta_vs_v17={row['delta_vs_v17']:.6f}ms "
                f"h_err={row.get('h_err', float('nan')):.6g} "
                f"vn_err={row.get('vn_err', float('nan')):.6g} "
                f"final_state_err={row.get('final_state_err', float('nan')):.6g}"
            )


if __name__ == "__main__":
    main()

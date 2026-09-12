#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v12_mfma_layout_fixed import qwen_gdn_chunked_avelang_v12_mfma_layout


def sync():
    torch.cuda.synchronize()


def l2norm(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 1234):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    candidates = [(16, 16), (32, 16), (16, 32), (32, 32)]
    q, k, v, g, beta, initial_state = make_inputs(args.T)
    scale = 128 ** -0.5
    baseline_ms = None
    rows = []
    print(f"target_shape,B=1,T={args.T},Hk=4,Hv=8,K=128,V=128,dtype=BF16")
    for bt, bv in candidates:
        status = "ok"
        ms = None
        try:
            if (bt, bv) != (16, 16):
                raise ValueError("v12 phase-1 kernel is intentionally fixed to BT=16,BV=16; no fallback path is used.")
            fn = lambda: qwen_gdn_chunked_avelang_v12_mfma_layout(
                q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=16
            )
            out, fs = fn()
            sync()
            ms = time_fn(fn, args.warmup, args.repeat)
            baseline_ms = ms if baseline_ms is None else baseline_ms
        except Exception as exc:  # noqa: BLE001 - sweep records failures instead of hiding them.
            status = f"FAILED: {type(exc).__name__}: {exc}"
        speedup = None if ms is None or baseline_ms is None else baseline_ms / ms
        row = {"BT": bt, "BV": bv, "full_ms": ms, "speedup_vs_16x16": speedup, "status": status}
        rows.append(row)
        print(row)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["full_ms"])
        print(f"recommended_tile,BT={best['BT']},BV={best['BV']},full_ms={best['full_ms']:.6f}")
    else:
        print("recommended_tile=none")


if __name__ == "__main__":
    main()

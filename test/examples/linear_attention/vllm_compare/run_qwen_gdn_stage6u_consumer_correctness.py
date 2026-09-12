#!/usr/bin/env python3
"""Diagnostic-only correctness matrix for the Stage 6U C0 W/U consumer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u,
)
from stage2_runner import make_inputs  # noqa: E402


def reference(k, v, g, beta, a_solved):
    t = int(k.shape[1])
    w = torch.empty((1, t, 8, 128), device=k.device, dtype=torch.bfloat16)
    u = torch.empty_like(w)
    for chunk_start in range(0, t, 64):
        aa = a_solved[0, chunk_start : chunk_start + 64].float().permute(1, 0, 2)
        bb = beta[0, chunk_start : chunk_start + 64]
        gg = g[0, chunk_start : chunk_start + 64]
        aw = (aa * bb.transpose(0, 1)[:, None, :] * torch.exp(gg).transpose(0, 1)[:, None, :]).to(torch.bfloat16)
        au = (aa * bb.transpose(0, 1)[:, None, :]).to(torch.bfloat16)
        for head in range(8):
            kk = k[0, chunk_start : chunk_start + 64, head // 2]
            vv = v[0, chunk_start : chunk_start + 64, head]
            w[0, chunk_start : chunk_start + 64, head] = (aw[head].float() @ kk.float()).to(torch.bfloat16)
            u[0, chunk_start : chunk_start + 64, head] = (au[head].float() @ vv.float()).to(torch.bfloat16)
    return w, u


def first_bad(actual, expected):
    bad = torch.nonzero(actual.view(torch.int16) != expected.view(torch.int16), as_tuple=False)
    return "" if bad.numel() == 0 else str(tuple(int(x) for x in bad[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--T", type=int, nargs="+", default=(64, 512, 2048))
    args = parser.parse_args()
    rows = []
    for t in args.T:
        _, k, v, g, beta, _ = make_inputs(t, 2026072300 + t, "random", True)
        torch.manual_seed(2026072400 + t)
        a_solved = (torch.randn((1, t, 8, 64), device=k.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual_w, actual_u = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g, beta, a_solved)
        stream.synchronize()
        expected_w, expected_u = reference(k, v, g, beta, a_solved)
        for name, actual, expected in (("W", actual_w, expected_w), ("U", actual_u, expected_u)):
            diff = (actual.float() - expected.float()).abs()
            rows.append(
                {
                    "diagnostic_only": True,
                    "T": t,
                    "tensor": name,
                    "bf16_mismatch_count": int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item()),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "first_bad_index": first_bad(actual, expected),
                    "finite": bool(torch.isfinite(actual).all().item()),
                    "contiguous": actual.is_contiguous(),
                    "accepted": bool(diff.max().item() <= 0.0078125),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"cases={len(rows)} failed={sum(not row['accepted'] for row in rows)} max_abs={max(row['max_abs'] for row in rows):.9g}")


if __name__ == "__main__":
    main()

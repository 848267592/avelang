#!/usr/bin/env python3
"""P0 BF16 solve-writeback correctness matrix for Stage 6U."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
DEFAULT_OUT = ROOT / "codex_qwen_bt64_bf16_solved_boundary_stage6u"
sys.path.insert(0, str(HERE))

from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (  # noqa: E402
    _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into,
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
    qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u,
)


def make_a(t: int, seed: int, mode: str) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.zeros((1, t, 8, 64), device="cuda", dtype=torch.float32)
    if mode == "zero" or mode == "identity_like":
        return a
    values = torch.randn(a.shape, generator=generator, device=a.device, dtype=a.dtype)
    if mode == "random":
        values.mul_(0.003)
    elif mode == "high_dynamic":
        values.mul_(0.04)
    elif mode == "small":
        values.mul_(1.0e-6)
    elif mode == "cancellation":
        values.mul_(0.01)
        values[..., 1::2].neg_()
    elif mode == "sparse_lower":
        values.mul_(0.01)
        mask = torch.rand(a.shape, generator=generator, device=a.device) < 0.08
        values.masked_fill_(~mask, 0.0)
    else:
        raise ValueError(mode)
    chunks = values.view(1, t // 64, 64, 8, 64)
    lower = torch.tril(torch.ones((64, 64), device=a.device, dtype=torch.bool), diagonal=-1)
    chunks.masked_fill_(~lower[None, None, :, None, :], 0.0)
    return values


def evaluate(t: int, mode: str, seed: int, nondefault: bool, prefill_reuse: bool):
    a = make_a(t, seed, mode)
    expected = qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u(a)
    if prefill_reuse:
        actual = torch.full(a.shape, float("nan"), device=a.device, dtype=torch.bfloat16)
        _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(a, actual)
        torch.cuda.synchronize()
        actual.fill_(float("nan"))
    else:
        actual = torch.empty(a.shape, device=a.device, dtype=torch.bfloat16)
    if nondefault:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(a, actual)
        stream.synchronize()
    elif prefill_reuse:
        _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(a, actual)
        torch.cuda.synchronize()
    else:
        actual = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a)
        torch.cuda.synchronize()
    mismatch_mask = actual.view(torch.int16) != expected.view(torch.int16)
    mismatch = int(mismatch_mask.sum().item())
    delta = (actual.float() - expected.float()).abs()
    tiles = actual.view(1, t // 64, 64, 8, 64).permute(0, 1, 3, 2, 4)
    diagonal = torch.diagonal(tiles, dim1=-2, dim2=-1)
    strict_upper = torch.triu(tiles, diagonal=1)
    return {
        "T": t, "mode": mode, "seed": seed, "nondefault_stream": nondefault,
        "prefill_reuse": prefill_reuse, "bf16_mismatch_count": mismatch,
        "max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item()),
        "diagonal_max_abs_from_one": float((diagonal.float() - 1.0).abs().max().item()),
        "strict_upper_nonzero": int(torch.count_nonzero(strict_upper).item()),
        "nan_count": int(torch.isnan(actual.float()).sum().item()),
        "inf_count": int(torch.isinf(actual.float()).sum().item()),
        "stride": str(tuple(actual.stride())), "accepted": mismatch == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = ("random", "zero", "identity_like", "high_dynamic", "small", "cancellation", "sparse_lower")
    rows = []
    for t in (64, 128, 512, 1024, 2048, 8192):
        for index, mode in enumerate(modes):
            rows.append(evaluate(t, mode, 2026072600 + t + index, mode == "cancellation", mode == "sparse_lower"))
    with (args.out_dir / "producer_p0_correctness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if not row["accepted"]]
    (args.out_dir / "producer_p0_first_divergence.md").write_text(
        "# P0 First Divergence\n\n" + (json.dumps(failed[0], indent=2) if failed else "All P0 outputs are BF16 bit-exact to P-REF.\n")
    )
    source = HERE / "qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py"
    contract = {
        "kernel": "_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u",
        "wrapper": "qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input": "fp32[1,T,8,64]",
        "output": "bf16[1,T,8,64]",
        "workgroup": 256,
        "grid": "(T/64)*8",
        "bit_exact_to_p_ref": not failed,
        "mismatch_count": sum(int(row["bf16_mismatch_count"]) for row in rows),
        "max_abs": max(float(row["max_abs"]) for row in rows),
        "hsaco_sha256": None,
    }
    (args.out_dir / "producer_p0_code_object.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "failed": len(failed), "max_abs": contract["max_abs"]}, indent=2))
    if failed:
        raise AssertionError(failed[0])


if __name__ == "__main__":
    main()

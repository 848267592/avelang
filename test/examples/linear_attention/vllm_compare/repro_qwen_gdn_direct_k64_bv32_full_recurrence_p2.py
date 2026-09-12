#!/usr/bin/env python3
"""P2: nonzero-W multi-chunk feedback audit built from the P1 composition.

This file deliberately has no timing loop.  It replays the P1-passing BT64
composition in recurrence order and records every chunk boundary.  The host
sequence is a *correctness microscope*, not a proposal to split the runtime
recurrence: its only job is to locate the first failing intermediate if P1's
validated single-chunk semantics amplify under feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0 as p0
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1 as p1


LENGTHS = (128, 512, 2048)


def _make_long_case(t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 current-ABI inputs plus a FP32 initial state.

    Values are intentionally nonzero and small.  This keeps the test in the
    numerically relevant BF16 range while exercising every pred/update path.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    k = (torch.randn((1, t, p0.H_K, p0.KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.02).to(torch.bfloat16).contiguous()
    w = (torch.randn((1, t, p0.H_V, p0.KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.015).to(torch.bfloat16).contiguous()
    u = (torch.randn((1, t, p0.H_V, p0.KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.02).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, p0.H_V), device="cuda", dtype=torch.float32, generator=generator) * 0.01).contiguous()
    initial_state = (torch.randn((1, p0.H_V, p0.KDIM, p0.KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.01).contiguous()
    return k, w, u, g, initial_state


def _max_mean(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return float(error.max().item()), float(error.mean().item())


def _chunk_summary(
    *,
    chunk_idx: int,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    actual_state_before: torch.Tensor,
) -> dict[str, Any]:
    row: dict[str, Any] = {"chunk": chunk_idx}
    first_error: dict[str, Any] | None = None
    for name in (
        "raw_acc",
        "pred_partial",
        "pred_f32",
        "pred_bf16",
        "v_new",
        "v_decay",
        "delta",
        "h",
        "state_after",
        "final_state",
    ):
        maximum, mean = _max_mean(actual[name], expected[name])
        row[f"{name}_max_abs"] = maximum
        row[f"{name}_mean_abs"] = mean
        limit = p0.P0_BF16_ATOL if name in {"pred_bf16", "v_new", "v_decay", "h"} else p0.P0_FP32_ATOL
        if name in {"delta", "state_after", "final_state"}:
            limit = p1.STATE_ATOL
        if first_error is None and maximum > limit:
            diff = (actual[name].float() - expected[name].float()).abs()
            flat = int(diff.flatten().argmax().item())
            index = tuple(int(item.item()) for item in torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
            first_error = {
                "stage": name,
                "index": index,
                "expected": float(expected[name][index].float().item()),
                "actual": float(actual[name][index].float().item()),
                "abs_error": maximum,
                "limit": limit,
            }

    # The current C0 contract maintains the recurrence state in FP32.  H is a
    # BF16 pre-update snapshot, rather than the feedback carrier.  Record both
    # facts instead of incorrectly attributing any later error to BF16 state
    # readback.
    snapshot_expected = actual_state_before.to(torch.bfloat16)
    row["h_snapshot_vs_actual_state_before_bf16_max_abs"] = _max_mean(actual["h"][:, 0], snapshot_expected)[0]
    row["state_after_bf16_roundtrip_max_abs"] = _max_mean(
        actual["state_after"], actual["state_after"].to(torch.bfloat16).float()
    )[0]
    row["finite"] = all(bool(torch.isfinite(value).all()) for value in actual.values())
    row["first_error"] = first_error
    row["pass"] = bool(row["finite"]) and first_error is None
    return row


def run_feedback_length(t: int, *, seed: int) -> dict[str, Any]:
    if t % p0.BT:
        raise ValueError(f"T={t} must be divisible by BT={p0.BT}")
    p1._set_c0_lowering()
    k_all, w_all, u_all, g_all, initial_state = _make_long_case(t, seed)
    actual_state = initial_state.clone()
    expected_state = initial_state.clone()
    rows: list[dict[str, Any]] = []
    first_amplified: dict[str, Any] | None = None

    for chunk_idx in range(t // p0.BT):
        start = chunk_idx * p0.BT
        stop = start + p0.BT
        actual_before = actual_state.clone()
        case_actual = p0.Case(
            f"nonzero_w_T{t}_chunk{chunk_idx}",
            w_all[:, start:stop].contiguous(),
            u_all[:, start:stop].contiguous(),
            actual_state,
        )
        case_expected = p0.Case(
            case_actual.name,
            case_actual.w,
            case_actual.u,
            expected_state,
        )
        k = k_all[:, start:stop].contiguous()
        g = g_all[:, start:stop].contiguous()
        actual = p1._run_kernel(case_actual, k, g)
        torch.cuda.synchronize()
        expected = p1._reference(case_expected, k, g)
        torch.cuda.synchronize()
        row = _chunk_summary(
            chunk_idx=chunk_idx,
            actual=actual,
            expected=expected,
            actual_state_before=actual_before,
        )
        rows.append(row)
        if first_amplified is None and not bool(row["pass"]):
            first_amplified = {"chunk": chunk_idx, "first_error": row["first_error"]}
            break
        actual_state = actual["state_after"].detach()
        expected_state = expected["state_after"].detach()

    completed = len(rows)
    return {
        "T": t,
        "chunks_expected": t // p0.BT,
        "chunks_completed": completed,
        "finite": all(bool(row["finite"]) for row in rows),
        "pass": completed == t // p0.BT and all(bool(row["pass"]) for row in rows),
        "first_amplified_chunk": first_amplified,
        "last_chunk": rows[-1] if rows else None,
        "per_chunk": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [run_feedback_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "p2_recurrence_feedback_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({key: value for key, value in result.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("P2 feedback gate failed; do not advance to phase-lifetime or performance work.")


if __name__ == "__main__":
    main()

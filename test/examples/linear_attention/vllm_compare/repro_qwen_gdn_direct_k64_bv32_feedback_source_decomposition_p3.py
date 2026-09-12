#!/usr/bin/env python3
"""P3: source decomposition for the first P2 feedback threshold crossing.

P3 is intentionally a correctness-only control.  It replays the deterministic
P2 input to the state *before* the first failing chunk, then invokes the same
P1 pred-plus-update kernel three times with the only changed state pointer:

* A: actual feedback state;
* B: independent reference feedback state;
* C: BF16(actual state), widened back to FP32.

P1 runs pred before update, and P3 judges only its exported pred/V-new fields;
therefore the later update result cannot affect this pred comparison.  K and
g remain the original target-chunk tensors in every arm.  The script reports
both local kernel-vs-local-reference error and trajectory error relative to
the reference-state arm.  It contains no timing loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0 as p0
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1 as p1
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2


T_DEFAULT = 2048
TARGET_CHUNK_DEFAULT = 23
SEED_BASE_DEFAULT = 20260729


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    error = (actual.float() - expected.float()).abs()
    flat = int(error.flatten().argmax().item())
    index = tuple(int(item.item()) for item in torch.unravel_index(torch.tensor(flat, device=error.device), error.shape))
    return {
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
        "argmax": index,
        "actual_at_argmax": float(actual[index].float().item()),
        "expected_at_argmax": float(expected[index].float().item()),
    }


PRED_FIELDS = ("raw_acc", "pred_partial", "pred_f32", "pred_bf16", "v_new", "v_decay")


def _pred_arm(
    name: str,
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    case = p0.Case(name, w, u, state)
    actual = p1._run_kernel(case, k, g)
    torch.cuda.synchronize()
    expected = p1._reference(case, k, g)
    torch.cuda.synchronize()
    summary = p1._summary(case, actual, expected)
    return actual, expected, summary


def _bf16_state_comparison(actual_state: torch.Tensor, reference_state: torch.Tensor) -> dict[str, Any]:
    actual_bf16 = actual_state.to(torch.bfloat16)
    reference_bf16 = reference_state.to(torch.bfloat16)
    fp32 = _error_metrics(actual_state, reference_state)
    different = actual_bf16.view(torch.int16) != reference_bf16.view(torch.int16)
    count = int(different.sum().item())
    total = different.numel()
    result: dict[str, Any] = {
        "fp32": fp32,
        "bf16_bitwise_equal": count == 0,
        "bf16_different_elements": count,
        "bf16_total_elements": total,
        "bf16_different_fraction": count / total,
        "bf16_value": _error_metrics(actual_bf16.float(), reference_bf16.float()),
    }
    if count:
        flat = int(torch.nonzero(different.flatten(), as_tuple=False)[0].item())
        index = tuple(int(item.item()) for item in torch.unravel_index(torch.tensor(flat, device=different.device), different.shape))
        result["first_bf16_difference"] = {
            "index": index,
            "actual_fp32": float(actual_state[index].item()),
            "reference_fp32": float(reference_state[index].item()),
            "actual_bf16": float(actual_bf16[index].float().item()),
            "reference_bf16": float(reference_bf16[index].float().item()),
        }
    else:
        result["first_bf16_difference"] = None
    return result


def _replay_to_target(
    t: int,
    target_chunk: int,
    seed_base: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    if t % p0.BT:
        raise ValueError(f"T={t} must be divisible by BT={p0.BT}")
    if not 0 < target_chunk < t // p0.BT:
        raise ValueError(f"target chunk {target_chunk} must be in [1, {t // p0.BT - 1}]")
    p1._set_c0_lowering()
    k_all, w_all, u_all, g_all, initial_state = p2._make_long_case(t, seed_base + t)
    actual_state = initial_state.clone()
    reference_state = initial_state.clone()
    replay: list[dict[str, Any]] = []

    for chunk_idx in range(target_chunk):
        start = chunk_idx * p0.BT
        stop = start + p0.BT
        w = w_all[:, start:stop].contiguous()
        u = u_all[:, start:stop].contiguous()
        k = k_all[:, start:stop].contiguous()
        g = g_all[:, start:stop].contiguous()
        actual_case = p0.Case(f"p3_replay_actual_{chunk_idx}", w, u, actual_state)
        reference_case = p0.Case(f"p3_replay_reference_{chunk_idx}", w, u, reference_state)
        actual = p1._run_kernel(actual_case, k, g)
        torch.cuda.synchronize()
        expected = p1._reference(reference_case, k, g)
        torch.cuda.synchronize()
        row = p1._summary(actual_case, actual, expected)
        row["chunk"] = chunk_idx
        replay.append(row)
        actual_state = actual["state_after"].detach()
        reference_state = expected["state_after"].detach()

    start = target_chunk * p0.BT
    stop = start + p0.BT
    return (
        w_all[:, start:stop].contiguous(),
        u_all[:, start:stop].contiguous(),
        k_all[:, start:stop].contiguous(),
        g_all[:, start:stop].contiguous(),
        actual_state,
        reference_state,
        replay,
    )


def run_p3(*, t: int, target_chunk: int, seed_base: int) -> dict[str, Any]:
    w, u, k, g, actual_state, reference_state, replay = _replay_to_target(t, target_chunk, seed_base)
    actual_snapshot = actual_state.to(torch.bfloat16).float().contiguous()

    arm_a, _ref_a, summary_a = _pred_arm("arm_a_actual_state", k=k, w=w, u=u, g=g, state=actual_state)
    arm_b, ref_b, summary_b = _pred_arm("arm_b_reference_state", k=k, w=w, u=u, g=g, state=reference_state)
    arm_c, _ref_c, summary_c = _pred_arm("arm_c_bf16_actual_snapshot", k=k, w=w, u=u, g=g, state=actual_snapshot)

    local = {"A_actual": summary_a, "B_reference": summary_b, "C_bf16_actual_snapshot": summary_c}
    trajectory = {
        "A_actual_vs_reference_trajectory": {name: _error_metrics(arm_a[name], ref_b[name]) for name in PRED_FIELDS},
        "B_reference_vs_reference_trajectory": {name: _error_metrics(arm_b[name], ref_b[name]) for name in PRED_FIELDS},
        "C_snapshot_vs_reference_trajectory": {name: _error_metrics(arm_c[name], ref_b[name]) for name in PRED_FIELDS},
    }
    a_vs_c = {name: _error_metrics(arm_a[name], arm_c[name]) for name in PRED_FIELDS}
    state = _bf16_state_comparison(actual_state, reference_state)
    result = {
        "schema": "qwen-direct-k64-bv32-feedback-source-decomposition-p3-v1",
        "kernel": p1._qwen_gdn_direct_k64_bv32_full_p1_kernel.fn.__name__,
        "T": t,
        "target_chunk": target_chunk,
        "seed_base": seed_base,
        "replay_chunks": replay,
        "state_before_target": state,
        "local_kernel_vs_local_reference": local,
        "trajectory_error": trajectory,
        "A_vs_C_same_bf16_operand_check": a_vs_c,
        "finite": all(bool(summary["finite"]) for summary in local.values()),
        "pass": all(bool(summary["pass"]) for summary in local.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=T_DEFAULT)
    parser.add_argument("--target-chunk", type=int, default=TARGET_CHUNK_DEFAULT)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_feedback_source_decomposition_p3",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_p3(t=args.T, target_chunk=args.target_chunk, seed_base=args.seed_base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "p3_feedback_source_decomposition.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({
            "kernel": result["kernel"],
            "T": result["T"],
            "target_chunk": result["target_chunk"],
            "pass": result["pass"],
            "bf16_bitwise_equal": result["state_before_target"]["bf16_bitwise_equal"],
            "bf16_different_elements": result["state_before_target"]["bf16_different_elements"],
            "a_pred_f32_trajectory_max_abs": result["trajectory_error"]["A_actual_vs_reference_trajectory"]["pred_f32"]["max_abs"],
            "b_pred_f32_local_max_abs": result["local_kernel_vs_local_reference"]["B_reference"]["pred_f32_max_abs"],
            "a_vs_c_pred_f32_max_abs": result["A_vs_C_same_bf16_operand_check"]["pred_f32"]["max_abs"],
        }, sort_keys=True))
    if not bool(result["pass"]):
        raise SystemExit("P3 local pred correctness failed; do not attribute the feedback error to state provenance yet.")


if __name__ == "__main__":
    main()

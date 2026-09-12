#!/usr/bin/env python3
"""Recurrence-only ABI and correctness gate for the v29 compact-K candidate.

This deliberately does not time a full graph.  It gives the overflowing
compact-K schedule exactly the BF16 W/U and FP32 g/initial-state boundary used
by the current-vLLM recurrence, then compares its three recurrence outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
LADDER = HERE.parents[1] / "compile_bug/qwen_mfma32_lowering_ladder"
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(STAGE6A), str(STAGE2), str(HERE.parent)]

import stage6a_full_graph_audit as stage6a  # noqa: E402
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_current_abi_exp import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_current_abi_avelang_v29_mfma32,
    qwen_gdn_fused_chunk_gdr_full_current_abi_reference,
    qwen_gdn_gdr_decay_bt64_reference,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402


BOUNDARY_ATOL = 1.0 / 128.0
FINAL_STATE_ATOL = 2.0e-2


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "exact": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _run_case(t: int, seed: int, case: str) -> dict[str, object]:
    q, k, v, g, beta, initial_state = make_inputs(t, seed, "random", True)
    values = stage6a.vllm_manual_stages((q, k, v, g, beta, initial_state))
    w = values["w"]
    u = values["u"]
    if case == "zero_w":
        w = torch.zeros_like(w)
    elif case != "native_wu":
        raise ValueError(case)

    decay, g_last = qwen_gdn_gdr_decay_bt64_reference(values["g_cumsum"])
    compact = qwen_gdn_fused_chunk_gdr_full_current_abi_avelang_v29_mfma32(
        k, w, u, decay, g_last, initial_state
    )
    compact_ref = qwen_gdn_fused_chunk_gdr_full_current_abi_reference(
        k, w, u, decay, g_last, initial_state
    )
    native = stage6a.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=values["g_cumsum"],
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    h, v_new, final_state = compact
    h_ref, v_new_ref, final_ref = compact_ref
    native_h, native_v_new, native_final = native
    own = {
        "h": _error(h, h_ref),
        "v_new": _error(v_new, v_new_ref),
        "final_state": _error(final_state, final_ref),
    }
    native_error = {
        "h": _error(h, native_h),
        "v_new": _error(v_new, native_v_new),
        "final_state": _error(final_state, native_final),
    }
    native_contract_pass = (
        native_error["h"]["max_abs"] <= BOUNDARY_ATOL
        and native_error["v_new"]["max_abs"] <= BOUNDARY_ATOL
        and native_error["final_state"]["max_abs"] <= FINAL_STATE_ATOL
        and all(value["finite"] for value in native_error.values())
    )
    return {
        "T": t,
        "case": case,
        "input_dtypes": {
            "k": str(k.dtype), "w": str(w.dtype), "u": str(u.dtype),
            "g_cumsum": str(values["g_cumsum"].dtype), "initial_state": str(initial_state.dtype),
        },
        "compact_output_dtypes": {
            "h": str(h.dtype), "v_new": str(v_new.dtype), "final_state": str(final_state.dtype),
        },
        "compact_vs_its_fp32_reference": own,
        "compact_vs_current_vllm_recurrence": native_error,
        "native_contract_pass": native_contract_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=2026072501)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    patch_rocm_autotune()
    if not torch.cuda.is_available():
        raise RuntimeError("requires HIP/CUDA GPU")

    rows = []
    for index, t in enumerate(args.T):
        for case in ("zero_w", "native_wu"):
            row = _run_case(t, args.seed + index * 17 + (case == "native_wu"), case)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    result = {
        "experiment": "v29_compact_k_current_vllm_bf16_abi",
        "thresholds": {"h_vnew_atol": BOUNDARY_ATOL, "final_state_atol": FINAL_STATE_ATOL},
        "rows": rows,
        "replacement_gate_pass": all(row["native_contract_pass"] for row in rows),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"replacement_gate_pass": result["replacement_gate_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()

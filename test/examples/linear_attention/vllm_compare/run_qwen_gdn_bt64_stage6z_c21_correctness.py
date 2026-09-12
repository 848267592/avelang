#!/usr/bin/env python3
"""T=2048 correctness gate for the C21 selected-native pipeline candidate.

This is intentionally a single-length, caller-owned-output gate.  C21 is a
schedule reconstruction experiment, so the reference remains the frozen Z5B
BF16 chunk-o contract rather than a looser floating-point tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
HV = 8
DIM = 128


def _random_inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, t, HV, DIM), device=q.device, dtype=torch.float32) * 0.02).to(
        torch.bfloat16
    )
    h = (
        torch.randn((1, t // BT, HV, DIM, DIM), device=q.device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _one_hot_inputs(t: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    q = torch.zeros((1, t, 4, DIM), device=device, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v_new = torch.zeros((1, t, HV, DIM), device=device, dtype=torch.bfloat16)
    h = torch.zeros((1, t // BT, HV, DIM, DIM), device=device, dtype=torch.bfloat16)
    g = torch.zeros((1, t, HV), device=device, dtype=torch.float32)
    q[0, 9, 0, 5] = 1
    q[0, 37, 0, 33] = -0.5
    k[0, 6, 0, 5] = 1
    k[0, 19, 0, 33] = 0.75
    v_new[0, 6, 0, 11] = 0.5
    v_new[0, 19, 0, 71] = -0.25
    h[0, 0, 0, 11, 5] = 1
    h[0, 0, 0, 71, 33] = -0.5
    return q, k, v_new, h, g


def _pattern_inputs(t: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    q_values = (torch.arange(t * 4 * DIM, device=device, dtype=torch.int32) % 17 - 8).to(torch.float32)
    k_values = (torch.arange(t * 4 * DIM, device=device, dtype=torch.int32) % 23 - 11).to(torch.float32)
    v_values = (torch.arange(t * HV * DIM, device=device, dtype=torch.int32) % 19 - 9).to(torch.float32)
    h_values = (
        torch.arange((t // BT) * HV * DIM * DIM, device=device, dtype=torch.int32) % 29 - 14
    ).to(torch.float32)
    q = (q_values.reshape(1, t, 4, DIM) / 64).to(torch.bfloat16)
    k = (k_values.reshape(1, t, 4, DIM) / 96).to(torch.bfloat16)
    v_new = (v_values.reshape(1, t, HV, DIM) / 80).to(torch.bfloat16)
    h = (h_values.reshape(1, t // BT, HV, DIM, DIM) / 128).to(torch.bfloat16)
    g = (torch.arange(t * HV, device=device, dtype=torch.float32).reshape(1, t, HV) % 13) / 32
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _evaluate(name: str, tensors: tuple[torch.Tensor, ...], *, caller_owned: bool) -> dict[str, object]:
    q, k, v_new, h, g = tensors
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(q, k, v_new, h, g)
    if caller_owned:
        actual = torch.full_like(v_new, float("nan"))
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(
            q, k, v_new, h, g, actual
        )
    else:
        actual = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline(q, k, v_new, h, g)
    torch.cuda.synchronize()
    delta = (expected.float() - actual.float()).abs()
    return {
        "name": name,
        "caller_owned": caller_owned,
        "finite": bool(torch.isfinite(actual).all().item()),
        "no_nan": not bool(torch.isnan(actual).any().item()),
        "bf16_byte_exact_to_z5b": bool(torch.equal(expected, actual)),
        "max_abs": float(delta.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument(
        "--out",
        type=Path,
        default=LADDER / "stage6z_c21_correctness_t2048.json",
    )
    args = parser.parse_args()
    if args.T != 2048:
        raise ValueError("C21-NSM correctness gate is intentionally fixed at T=2048")

    cases: list[tuple[str, tuple[torch.Tensor, ...], bool]] = []
    cases.append(("random", _random_inputs(args.T, 2026082100), False))
    zero_v = _random_inputs(args.T, 2026082200)
    zero_v[2].zero_()
    cases.append(("zero_v_new", zero_v, False))
    cases.append(("nan_prefilled_caller_owned_output", _random_inputs(args.T, 2026082300), True))
    cases.append(("structured_q_k_h_one_hot", _one_hot_inputs(args.T), False))
    cases.append(("token_value_pattern", _pattern_inputs(args.T), False))

    results = [_evaluate(name, tensors, caller_owned=caller_owned) for name, tensors, caller_owned in cases]
    passed = all(
        bool(row["finite"]) and bool(row["no_nan"]) and bool(row["bf16_byte_exact_to_z5b"])
        for row in results
    )
    payload = {
        "schema": "qwen.gfx942.stage6z.c21.correctness.v1",
        "experiment": "C21-NSM Selected-Native Pipeline Reconstruction",
        "T": args.T,
        "reference": "Z5B direct-Q-cache consumer, BF16 byte-exact",
        "cases": results,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

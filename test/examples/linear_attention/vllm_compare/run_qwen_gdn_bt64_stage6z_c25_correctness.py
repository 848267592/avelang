#!/usr/bin/env python3
"""C25 current-ready/next-pending BF16 correctness gate.

C25 changes only compiler-owned producer issue/commit ordering below the
frozen C21 logical chunk-o source.  This runner keeps Z5B as the frozen
caller-owned BF16 reference and deliberately executes every length in a fresh
Python process so compiler/JIT state cannot turn one lowering mode into the
other.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
HK = 4
HV = 8
DIM = 128
_C25_ENV = (
    "AVELANG_STAGE6Z_PENDING_PACKET_INFRA",
    "AVELANG_STAGE6Z_CURRENT_READY_NEXT_PENDING",
)


def _set_mode(c25: bool) -> None:
    if c25:
        os.environ[_C25_ENV[0]] = "c24"
        os.environ[_C25_ENV[1]] = "c25"
    else:
        for key in _C25_ENV:
            os.environ.pop(key, None)


def _random_inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, t, HV, DIM), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, DIM, DIM), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _structured_inputs(t: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    q = torch.zeros((1, t, HK, DIM), device=device, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v_new = torch.zeros((1, t, HV, DIM), device=device, dtype=torch.bfloat16)
    h = torch.zeros((1, t // BT, HV, DIM, DIM), device=device, dtype=torch.bfloat16)
    g = torch.zeros((1, t, HV), device=device, dtype=torch.float32)
    coordinates = ((9, 5, 1.0), (37, 33, -0.5), (53, 71, 0.25))
    for token, feature, value in coordinates:
        token %= t
        q[0, token, 0, feature] = value
        k[0, (token * 3 + 7) % t, 0, feature] = value * 0.75
        v_new[0, token, 0, (feature * 5 + 11) % DIM] = value * 0.5
        h[0, token // BT, 0, (feature * 5 + 11) % DIM, feature] = value
    return q, k, v_new, h, g


def _pattern_inputs(t: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    q = ((torch.arange(t * HK * DIM, device=device, dtype=torch.int32) % 17 - 8).to(torch.float32) / 64).reshape(1, t, HK, DIM).to(torch.bfloat16)
    k = ((torch.arange(t * HK * DIM, device=device, dtype=torch.int32) % 23 - 11).to(torch.float32) / 96).reshape(1, t, HK, DIM).to(torch.bfloat16)
    v_new = ((torch.arange(t * HV * DIM, device=device, dtype=torch.int32) % 19 - 9).to(torch.float32) / 80).reshape(1, t, HV, DIM).to(torch.bfloat16)
    h = ((torch.arange((t // BT) * HV * DIM * DIM, device=device, dtype=torch.int32) % 29 - 14).to(torch.float32) / 128).reshape(1, t // BT, HV, DIM, DIM).to(torch.bfloat16)
    g = (torch.arange(t * HV, device=device, dtype=torch.float32).reshape(1, t, HV) % 13) / 32
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _run_case(name: str, tensors: tuple[torch.Tensor, ...], *, caller_owned: bool) -> dict[str, object]:
    q, k, v_new, h, g = tensors
    _set_mode(False)
    reference = torch.empty_like(v_new)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(q, k, v_new, h, g, reference)
    torch.cuda.synchronize()

    _set_mode(True)
    actual = torch.full_like(v_new, float("nan")) if caller_owned else torch.empty_like(v_new)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(q, k, v_new, h, g, actual)
    torch.cuda.synchronize()
    delta = (reference.float() - actual.float()).abs()
    return {
        "name": name,
        "caller_owned": caller_owned,
        "finite": bool(torch.isfinite(actual).all().item()),
        "no_nan": not bool(torch.isnan(actual).any().item()),
        "bf16_byte_exact_to_z5b": bool(torch.equal(reference, actual)),
        "max_abs": float(delta.max().item()),
    }


def _worker(t: int) -> dict[str, object]:
    cases: list[tuple[str, tuple[torch.Tensor, ...], bool]] = [
        ("random", _random_inputs(t, 2026082500 + t), False),
        ("zero_v_new", _random_inputs(t, 2026082600 + t), False),
        ("nan_prefilled_caller_owned_output", _random_inputs(t, 2026082700 + t), True),
        ("structured_q_h_k", _structured_inputs(t), False),
        ("token_value_pattern", _pattern_inputs(t), False),
    ]
    cases[1][1][2].zero_()
    rows = [_run_case(name, tensors, caller_owned=caller_owned) for name, tensors, caller_owned in cases]
    passed = all(bool(row["finite"]) and bool(row["no_nan"]) and bool(row["bf16_byte_exact_to_z5b"]) for row in rows)
    return {
        "T": t,
        "chunks": t // BT,
        "reference": "frozen Z5B direct-Q-cache caller-owned BF16 output",
        "c25_mode": {"pending_packet_infra": "c24", "current_ready_next_pending": "c25"},
        "cases": rows,
        "passed": passed,
    }


def _parent(args: argparse.Namespace) -> None:
    per_t: list[dict[str, object]] = []
    for t in args.T:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(t)]
        completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"C25 correctness T={t} failed:\n{completed.stdout}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        per_t.append(json.loads(lines[-1]))
    payload = {
        "schema": "qwen.gfx942.stage6z.c25.current_ready_next_pending.correctness.v1",
        "fresh_process_per_length": True,
        "lengths": per_t,
        "passed": all(bool(row["passed"]) for row in per_t),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--T", type=int, nargs="+", default=[64, 2048, 8192])
    parser.add_argument("--out", type=Path, default=LADDER / "stage6z_c25_correctness.json")
    args = parser.parse_args()
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("every T must be a positive multiple of 64")
    if args.worker:
        if len(args.T) != 1:
            raise ValueError("worker accepts exactly one T")
        payload = _worker(args.T[0])
        print(json.dumps(payload, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(1)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

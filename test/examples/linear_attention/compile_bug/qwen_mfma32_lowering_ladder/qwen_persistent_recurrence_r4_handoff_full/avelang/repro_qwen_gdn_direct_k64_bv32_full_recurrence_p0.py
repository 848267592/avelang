#!/usr/bin/env python3
"""P0: Direct-K64 BV32 nonzero-W MFMA32 pred mapping audit.

This is deliberately a *single-chunk pred-only* diagnostic.  It is not a
split recurrence benchmark and it never executes an update kernel.  The
purpose is to establish the exact lane/fragment mapping of the BF16 current
recurrence ABI before composing pred with the validated C0 update suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 128
H_K = 4
H_V = 8
WORKGROUP = 128
GRID = H_V * (KDIM // BV)

P0_FP32_ATOL = 5.0e-5
P0_BF16_ATOL = 1.0 / 128.0


@avelang.jit
def _qwen_gdn_direct_k64_bv32_pred_mapping_p0_kernel(
    w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16),
    initial_state_ptr: al.Pointer(al.f32),
    raw_acc_ptr: al.Pointer(al.f32),
    pred_partial_ptr: al.Pointer(al.f32),
    pred_f32_ptr: al.Pointer(al.f32),
    pred_bf16_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
):
    """One BT64 pred tile for every (value head, BV32) CTA.

    The MFMA32 operand staging and accumulator unpack are intentionally the
    same K-split form previously used by v29 pred-only.  The public boundary
    changes only to the current recurrence ABI: W/U are BF16 and input state
    is FP32 but is converted to BF16 before it feeds the MFMA operand.
    """

    w = al.make_tensor(
        w_ptr,
        al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    raw_acc = al.make_tensor(
        raw_acc_ptr,
        al.f32,
        al.make_layout((H_V, 4, 2, 2, 64, 16), (4 * 2 * 2 * 64 * 16, 2 * 2 * 64 * 16, 2 * 64 * 16, 64 * 16, 16, 1)),
    )
    pred_partial_out = al.make_tensor(
        pred_partial_ptr,
        al.f32,
        al.make_layout((H_V, 4, 2, 2, 32, 32), (4 * 2 * 2 * 32 * 32, 2 * 2 * 32 * 32, 2 * 32 * 32, 32 * 32, 32, 1)),
    )
    pred_f32 = al.make_tensor(
        pred_f32_ptr,
        al.f32,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    pred_bf16 = al.make_tensor(
        pred_bf16_ptr,
        al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    v_new = al.make_tensor(
        v_new_ptr,
        al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    # MFMA32 ownership is wave-local: the workgroup wave selects K64, while
    # lane 0:31 selects the logical row and lane 32:63 selects the second
    # packed 16-byte operand vector for that row.
    lane_row = lane & 31
    mfma_lane_group = lane >> 5
    program_id = al.block_id(0)
    value_head_idx = program_id >> 2
    value_base = (program_id & 3) * BV

    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))

    for rep_state in al.range(32):
        linear = tid + rep_state * WORKGROUP
        k_half = linear // (BV * 64)
        rem = linear - k_half * (BV * 64)
        local_v = rem // 64
        local_k = rem - local_v * 64
        state_bf16[k_half, local_v, local_k] = al.convert(
            initial_state[0, value_head_idx, value_base + local_v, k_half * 64 + local_k], al.bf16
        )
    al.syncthreads()

    for token_tile in al.range(2):
        token_base = token_tile * 32
        for rep_w in al.range(32):
            linear = tid + rep_w * WORKGROUP
            k_half = linear // (32 * 64)
            rem = linear - k_half * (32 * 64)
            token_off = rem // 64
            local_k = rem - token_off * 64
            w_bf16[k_half, token_off, local_k] = w[
                0, token_base + token_off, value_head_idx, k_half * 64 + local_k
            ]
        al.syncthreads()

        pred_acc = al.full((16,), 0.0, al.f32)
        for kpack in al.range(4):
            k_vec = kpack * 2 + mfma_lane_group
            w_words = w_vec[wave_id, lane_row, k_vec]
            state_words = state_vec[wave_id, lane_row, k_vec]
            w_frag = al.view(w_words, al.Tensor((2, 4, 1), al.bf16))
            state_frag = al.view(state_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[0], w_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[1], w_frag[1], pred_acc)

        for acc_i in al.range(16):
            raw_acc[value_head_idx, program_id & 3, token_tile, wave_id, lane, acc_i] = pred_acc[acc_i]
            out_row = lane_row
            out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
        al.syncthreads()

        for rep_dump in al.range(16):
            linear = tid + rep_dump * WORKGROUP
            k_half = linear // (32 * BV)
            rem = linear - k_half * (32 * BV)
            token_off = rem // BV
            local_v = rem - token_off * BV
            pred_partial_out[value_head_idx, program_id & 3, token_tile, k_half, token_off, local_v] = pred_partial[
                k_half, token_off, local_v
            ]

        for rep_out in al.range(8):
            linear = tid + rep_out * WORKGROUP
            token_off = linear // BV
            local_v = linear - token_off * BV
            token_idx = token_base + token_off
            global_v = value_base + local_v
            pred = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
            pred_f32[0, token_idx, value_head_idx, global_v] = pred
            pred_bf16[0, token_idx, value_head_idx, global_v] = al.convert(pred, al.bf16)
            v_new[0, token_idx, value_head_idx, global_v] = al.convert(
                al.convert(u[0, token_idx, value_head_idx, global_v], al.f32) - pred, al.bf16
            )
        al.syncthreads()


@dataclass(frozen=True)
class Case:
    name: str
    w: torch.Tensor
    u: torch.Tensor
    initial_state: torch.Tensor


def _zeros() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = torch.zeros((1, BT, H_V, KDIM), device="cuda", dtype=torch.bfloat16)
    u = torch.zeros_like(w)
    initial_state = torch.zeros((1, H_V, KDIM, KDIM), device="cuda", dtype=torch.float32)
    return w, u, initial_state


def _machine_cases(seed: int) -> list[Case]:
    cases: list[Case] = []
    w, u, state = _zeros()
    state[0, 0, 5, 73] = 1.0
    w[0, 11, 0, 73] = 1.0
    cases.append(Case("onehot_state_w_zero_u", w, u, state))

    w, u, state = _zeros()
    state[0, 0, 5, 73] = 1.0
    w[0, 11, 0, 73] = 1.0
    u[0, 11, 0, 5] = 1.0
    cases.append(Case("onehot_state_w_onehot_u", w, u, state))

    w, u, state = _zeros()
    # Every K feature is assigned a unique token. This catches both K halves
    # and gives a machine-readable all-128-feature scan in one launch.
    for k_idx in range(KDIM):
        state[0, 0, 9, k_idx] = 1.0
        w[0, k_idx & 63, 0, k_idx] = 1.0
    cases.append(Case("single_head_value_k_feature_scan", w, u, state))

    w, u, state = _zeros()
    for value_idx in range(BV):
        k_idx = (value_idx * 3) & 127
        token_idx = (value_idx * 5) & 63
        state[0, 0, value_idx, k_idx] = 1.0 if value_idx % 2 == 0 else -1.0
        w[0, token_idx, 0, k_idx] = 1.0
    cases.append(Case("sparse_diagonal", w, u, state))

    w, u, state = _zeros()
    for value_idx in range(BV):
        k_idx = (value_idx * 7 + 3) & 127
        token_idx = (value_idx * 11 + 1) & 63
        state[0, 0, value_idx, k_idx] = 1.0
        w[0, token_idx, 0, k_idx] = 1.0
        u[0, token_idx, 0, value_idx] = -0.5
    cases.append(Case("permutation", w, u, state))

    generator = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((1, BT, H_V, KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.015).to(torch.bfloat16)
    u = (torch.randn((1, BT, H_V, KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.015).to(torch.bfloat16)
    state = torch.randn((1, H_V, KDIM, KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.015
    cases.append(Case("random_low_amplitude_nonzero_w", w.contiguous(), u.contiguous(), state.contiguous()))
    return cases


def _reference(case: Case) -> dict[str, torch.Tensor]:
    state_bf16 = case.initial_state.to(torch.bfloat16).float()
    partial = torch.empty((H_V, 4, 2, 2, 32, 32), device="cuda", dtype=torch.float32)
    pred = torch.empty((1, BT, H_V, KDIM), device="cuda", dtype=torch.float32)
    for head in range(H_V):
        for v_block in range(4):
            value_base = v_block * BV
            for token_tile in range(2):
                token_base = token_tile * 32
                for k_half in range(2):
                    partial[head, v_block, token_tile, k_half] = (
                        case.w[0, token_base : token_base + 32, head, k_half * 64 : (k_half + 1) * 64].float()
                        @ state_bf16[0, head, value_base : value_base + BV, k_half * 64 : (k_half + 1) * 64].t()
                    )
                pred[0, token_base : token_base + 32, head, value_base : value_base + BV] = (
                    partial[head, v_block, token_tile, 0] + partial[head, v_block, token_tile, 1]
                )
    pred_bf16 = pred.to(torch.bfloat16)
    v_new = (case.u.float() - pred).to(torch.bfloat16)

    lane = torch.arange(64, device="cuda")[:, None]
    acc = torch.arange(16, device="cuda")[None, :]
    lane_row = lane & 31
    mfma_lane_group = lane >> 5
    token = lane_row + torch.zeros_like(acc)
    local_v = ((acc >> 2) * 8) + mfma_lane_group * 4 + (acc & 3)
    raw = torch.empty((H_V, 4, 2, 2, 64, 16), device="cuda", dtype=torch.float32)
    for head in range(H_V):
        for v_block in range(4):
            for token_tile in range(2):
                for k_half in range(2):
                    raw[head, v_block, token_tile, k_half] = partial[head, v_block, token_tile, k_half][token, local_v]
    return {
        "raw_acc": raw,
        "pred_partial": partial,
        "pred_f32": pred,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
    }


def _run_kernel(case: Case) -> dict[str, torch.Tensor]:
    raw_acc = torch.empty((H_V, 4, 2, 2, 64, 16), device="cuda", dtype=torch.float32)
    pred_partial = torch.empty((H_V, 4, 2, 2, 32, 32), device="cuda", dtype=torch.float32)
    pred_f32 = torch.empty((1, BT, H_V, KDIM), device="cuda", dtype=torch.float32)
    pred_bf16 = torch.empty((1, BT, H_V, KDIM), device="cuda", dtype=torch.bfloat16)
    v_new = torch.empty_like(case.u)
    _qwen_gdn_direct_k64_bv32_pred_mapping_p0_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
        case.w, case.u, case.initial_state, raw_acc, pred_partial, pred_f32, pred_bf16, v_new, num_warps=2
    )
    return {
        "raw_acc": raw_acc,
        "pred_partial": pred_partial,
        "pred_f32": pred_f32,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
    }


def _first_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any] | None:
    for name in ("raw_acc", "pred_partial", "pred_f32", "pred_bf16", "v_new"):
        diff = (actual[name].float() - expected[name].float()).abs()
        limit = P0_BF16_ATOL if name in {"pred_bf16", "v_new"} else P0_FP32_ATOL
        maximum = float(diff.max().item())
        if maximum > limit:
            flat = int(diff.flatten().argmax().item())
            index = list(torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
            index_int = tuple(int(item.item()) for item in index)
            return {
                "stage": name,
                "index": index_int,
                "expected": float(expected[name][index_int].float().item()),
                "actual": float(actual[name][index_int].float().item()),
                "abs_error": maximum,
                "limit": limit,
            }
    return None


def _summary(case: Case, actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {"case": case.name}
    for name in ("raw_acc", "pred_partial", "pred_f32", "pred_bf16", "v_new"):
        error = (actual[name].float() - expected[name].float()).abs()
        result[f"{name}_max_abs"] = float(error.max().item())
        result[f"{name}_mean_abs"] = float(error.mean().item())
    result["finite"] = all(bool(torch.isfinite(value).all()) for value in actual.values())
    result["first_error"] = _first_error(actual, expected)
    result["pass"] = result["finite"] and result["first_error"] is None
    return result


def _write_mapping(case: Case, actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], out_dir: Path) -> None:
    """Write exact accumulator ownership for head 0 / BV block 0.

    The static rule is valid for every CTA; the CSV keeps one representative
    CTA readable while retaining expected/actual values from the all-feature
    scan.  The full raw tensors remain available through the runner JSON.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    static = {
        "case": case.name,
        "cta": {"value_head": 0, "value_block": 0, "value_base": 0},
        "raw_acc_shape": [H_V, 4, 2, 2, 64, 16],
        "status": "P0-corrected MFMA32 unpack rule, checked against empirical raw accumulator data",
        "formula": {
            "k_half": "wave_id",
            "token": "lane & 31",
            "local_value": "((acc_i >> 2) * 8) + ((lane >> 5) * 4) + (acc_i & 3)",
            "logical_pred": "dot(W[token, head, Khalf*64:(Khalf+1)*64], H[head, value_base+local_value, same K half])",
        },
    }
    (out_dir / "lane_to_logical_pred_mapping.json").write_text(json.dumps(static, indent=2) + "\n")
    with (out_dir / "lane_to_logical_pred_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["wave", "lane", "acc_i", "token", "local_value", "k_begin", "k_end", "expected_partial", "actual_raw_acc", "abs_error"],
        )
        writer.writeheader()
        for wave in range(2):
            for lane in range(64):
                for acc_i in range(16):
                    token = lane & 31
                    local_value = ((acc_i >> 2) * 8) + ((lane >> 5) * 4) + (acc_i & 3)
                    exp = float(expected["raw_acc"][0, 0, 0, wave, lane, acc_i].item())
                    got = float(actual["raw_acc"][0, 0, 0, wave, lane, acc_i].item())
                    writer.writerow({
                        "wave": wave,
                        "lane": lane,
                        "acc_i": acc_i,
                        "token": token,
                        "local_value": local_value,
                        "k_begin": wave * 64,
                        "k_end": (wave + 1) * 64,
                        "expected_partial": exp,
                        "actual_raw_acc": got,
                        "abs_error": abs(got - exp),
                    })


def _write_case_raw_provenance(
    case: Case,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    out_dir: Path,
) -> None:
    """Record the raw-accumulator ownership implied by the source unpack rule.

    This is intentionally emitted even when the rule is wrong.  It lets the
    first divergence be inspected without attempting to infer a corrected
    hardware accumulator layout from a failed source hypothesis.
    """

    if case.name == "random_low_amplitude_nonzero_w":
        return
    path = out_dir / f"{case.name}_raw_acc_provenance.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "head", "value_block", "token_tile", "wave", "lane", "acc_i",
                "logical_token", "logical_local_value", "logical_k_begin", "logical_k_end",
                "expected_raw_acc", "actual_raw_acc", "abs_error",
            ],
        )
        writer.writeheader()
        for head in range(H_V):
            for value_block in range(4):
                for token_tile in range(2):
                    for wave in range(2):
                        for lane in range(64):
                            for acc_i in range(16):
                                token = lane & 31
                                local_value = ((acc_i >> 2) * 8) + ((lane >> 5) * 4) + (acc_i & 3)
                                exp = float(expected["raw_acc"][head, value_block, token_tile, wave, lane, acc_i].item())
                                got = float(actual["raw_acc"][head, value_block, token_tile, wave, lane, acc_i].item())
                                if exp != 0.0 or got != 0.0:
                                    writer.writerow({
                                        "head": head,
                                        "value_block": value_block,
                                        "token_tile": token_tile,
                                        "wave": wave,
                                        "lane": lane,
                                        "acc_i": acc_i,
                                        "logical_token": token_tile * 32 + token,
                                        "logical_local_value": local_value,
                                        "logical_k_begin": wave * 64,
                                        "logical_k_end": (wave + 1) * 64,
                                        "expected_raw_acc": exp,
                                        "actual_raw_acc": got,
                                        "abs_error": abs(got - exp),
                                    })


def _write_empirical_mapping(out_dir: Path) -> dict[str, Any]:
    """Map direct-K64 raw accumulator ownership without trusting source unpack.

    A shift run assigns one unique BF16 W amplitude to each token and a unique
    K element to each token.  Across 32 shifts, each of the 64 token rows is
    paired with every BV32 value.  The actual/expected ratio is recorded
    verbatim, so a duplicated fragment remains visible in the artifact.
    """

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for k_half in range(2):
        for shift in range(BV):
            w, u, state = _zeros()
            amplitudes: list[float] = []
            for token in range(BT):
                value = (token + shift) % BV
                amplitude = 1.0 + token / 64.0
                k_idx = k_half * 64 + token
                amplitudes.append(amplitude)
                w[0, token, 0, k_idx] = amplitude
                state[0, 0, value, k_idx] = 1.0
            actual = _run_kernel(Case(f"empirical_h{k_half}_s{shift}", w, u, state))
            torch.cuda.synchronize()
            raw = actual["raw_acc"][0, 0, :, k_half]
            partial = actual["pred_partial"][0, 0, :, k_half]
            pred = actual["pred_f32"][0, :, 0, :BV]
            raw_indices = torch.nonzero(raw).tolist()
            for token_tile, lane, acc_i in raw_indices:
                actual_value = float(raw[token_tile, lane, acc_i].item())
                candidates = [abs(actual_value - amplitude) for amplitude in amplitudes]
                logical_token = min(range(BT), key=lambda index: candidates[index])
                confidence = candidates[logical_token]
                if confidence > 1.0e-3:
                    unresolved += 1
                logical_value = (logical_token + shift) % BV
                partial_hits = torch.nonzero((partial - actual_value).abs() < 1.0e-5).tolist()
                pred_hits = torch.nonzero((pred - actual_value).abs() < 1.0e-5).tolist()
                rows.append({
                    "k_half": k_half,
                    "shift": shift,
                    "logical_token_from_unique_w": logical_token,
                    "logical_value_from_state": logical_value,
                    "logical_k": k_half * 64 + logical_token,
                    "raw_token_tile": token_tile,
                    "raw_wave": k_half,
                    "raw_lane": lane,
                    "raw_acc_i": acc_i,
                    "expected_partial_value": amplitudes[logical_token],
                    "actual_raw_value": actual_value,
                    "raw_over_expected_ratio": actual_value / amplitudes[logical_token],
                    "matching_partial_coordinates": partial_hits,
                    "matching_pred_coordinates": pred_hits,
                    "unique_value_match_residual": confidence,
                })
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (out_dir / "empirical_lane_to_logical_pred_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "qwen-p0-empirical-mfma32-mapping-v1",
        "coverage": {"k_halves": 2, "shifts": BV, "logical_rows_per_shift": BT},
        "unique_value_encoding": "W[token, k_half*64+token] = 1 + token/64; state[value=(token+shift)%32, same_k] = 1",
        "actual_to_logical_match": "nearest encoded W amplitude; actual/expected ratio is recorded without normalization",
        "row_count": len(rows),
        "unresolved_unique_value_matches": unresolved,
        "csv": "empirical_lane_to_logical_pred_mapping.csv",
    }
    (out_dir / "empirical_lane_to_logical_pred_mapping.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_p0(*, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mapping_written = False
    for case in _machine_cases(seed):
        actual = _run_kernel(case)
        torch.cuda.synchronize()
        expected = _reference(case)
        torch.cuda.synchronize()
        results.append(_summary(case, actual, expected))
        _write_case_raw_provenance(case, actual, expected, out_dir)
        if case.name == "single_head_value_k_feature_scan":
            _write_mapping(case, actual, expected, out_dir)
            mapping_written = True
    if not mapping_written:
        raise RuntimeError("P0 mapping case was not executed")
    empirical = _write_empirical_mapping(out_dir)
    results.append({"case": "empirical_lane_mapping", "pass": empirical["unresolved_unique_value_matches"] == 0, **empirical})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p0_pred_mapping_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p0",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = run_p0(seed=args.seed, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    if not all(bool(row["pass"]) for row in rows):
        raise SystemExit("P0 pred mapping gate failed; P1/update integration and all benchmarking are blocked.")


if __name__ == "__main__":
    main()

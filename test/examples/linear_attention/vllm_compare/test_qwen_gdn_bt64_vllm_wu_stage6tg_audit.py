"""Regression checks for the audit-only Stage 6T-Golden evidence bundle."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
OUT = LADDER / "codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_fused_wu_eager_stage6t import _validate_wu_inputs  # noqa: E402


def _read_json(relative: str) -> dict[str, object]:
    return json.loads((OUT / relative).read_text())


def _csv(relative: str) -> list[dict[str, str]]:
    with (OUT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage6tg_authoritative_contract_is_eager_and_graph_free() -> None:
    contract = _read_json("eager_public_api_contract.json")
    assert contract["timing_contract"] == "eager_public_api"
    assert contract["cuda_graph_used"] is False

    forbidden = ("torch.cuda.CUDAGraph", "torch.cuda.graph(", ".replay()")
    for source in (
        HERE / "audit_qwen_gdn_bt64_vllm_wu_stage6tg.py",
        HERE / "qwen_gdn_bt64_fused_wu_eager_stage6t.py",
    ):
        text = source.read_text()
        assert not any(token in text for token in forbidden), source


def test_stage6tg_actual_vllm_capture_is_repeatable_at_t2048() -> None:
    rows = _csv("specialization_matrix.csv")
    by_t = {int(row["T"]): row for row in rows}
    assert set(by_t) == {512, 2048, 8192, 16384}
    assert by_t[2048]["symbol"] == "recompute_w_u_fwd_kernel"
    assert by_t[2048]["num_warps"] == "4"
    assert by_t[2048]["num_stages"] == "2"
    assert by_t[2048]["workgroup"] == "256"
    assert by_t[2048]["cta"] == "256"

    first = _read_json("vllm_actual/by_t/T2048/capture_result.json")
    second = _read_json("repeatability/run2/vllm_actual/by_t/T2048/capture_result.json")
    first_config = {key: value for key, value in first["runtime_selected_config"].items() if key != "repr"}
    second_config = {key: value for key, value in second["runtime_selected_config"].items() if key != "repr"}
    assert first_config == second_config
    assert first["result"]["hsaco_sha256"] == second["result"]["hsaco_sha256"]


def test_stage6tg_counter_normalization_and_hsaco_evidence() -> None:
    decision = _read_json("final_decision.json")
    assert decision["f1_mfma_per_dispatch_t2048"] == 524288
    assert decision["vllm_mfma_per_dispatch_t2048"] == 32768
    assert decision["f1_mfma_per_chunk_head"] == 2048
    assert decision["vllm_mfma_per_chunk_head"] == 128
    assert decision["f1_duplicate_factor"] == 16.0
    assert decision["vllm_duplicate_factor"] == 1.0

    expected_hash = str(decision["actual_vllm_wu_hsaco_sha256"])
    actual_hash = (OUT / "vllm_actual/by_t/T2048/sha256.txt").read_text().strip().split()[0]
    assert actual_hash == expected_hash


def test_stage6tg_public_correctness_and_baseline_direction() -> None:
    correctness = _read_json("correctness_summary.json")
    assert correctness["timing_contract"] == "eager_public_api"
    assert correctness["cuda_graph_used"] is False
    assert correctness["public_full_correct"] is True
    assert float(correctness["max_output_abs"]) <= 1.0 / 128.0
    assert float(correctness["max_final_state_abs"]) <= 2.0e-2

    rows = _csv("eager_baseline_summary.csv")
    aggregate = {
        (int(row["T"]), row["implementation"]): float(row["event_median_ms"])
        for row in rows
        if row["session"] == "aggregate"
    }
    assert aggregate[(2048, "f1")] > aggregate[(2048, "stage6s")]
    assert aggregate[(8192, "f1")] < aggregate[(8192, "stage6s")]
    assert aggregate[(16384, "f1")] < aggregate[(16384, "stage6s")]


def test_stage6tg_f1_rejects_invalid_dtype_and_chunk_shape() -> None:
    k = torch.empty((1, 64, 4, 128), dtype=torch.float32)
    v = torch.empty((1, 64, 8, 128), dtype=torch.bfloat16)
    g = torch.empty((1, 64, 8), dtype=torch.float32)
    beta = torch.empty((1, 64, 8), dtype=torch.float32)
    a_solved = torch.empty((1, 64, 8, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16 k/v"):
        _validate_wu_inputs(k, v, g, beta, a_solved, chunk_size=64)
    with pytest.raises(ValueError, match="only supports chunk_size=64"):
        _validate_wu_inputs(k, v, g, beta, a_solved, chunk_size=32)

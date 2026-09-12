#!/usr/bin/env python3
"""CUDA smoke tests for the audit-only current-vLLM recurrence bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage6r_recurrence_body_benchmark import Bridge, CURRENT, external_call  # noqa: E402

LADDER = HERE.parent
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
sys.path.insert(0, str(STAGE6A))
import stage6a_full_graph_audit as stage6a  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HIP/CUDA")


def test_current_actual_vllm_bridge_is_bit_exact_on_current_stream() -> None:
    patch_rocm_autotune()
    inputs = stage6a.fixed_inputs(512)
    _, k, _, _, _, h0 = inputs
    values = stage6a.vllm_manual_stages(inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        native = chunk_gated_delta_rule_fwd_h(k, values["w"], values["u"], values["g_cumsum"], None, h0, True, 64, True, None)
        _, run = external_call(Bridge(), "actual", CURRENT / "vllm/kernel.hsaco",
                               "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", (4, 8), 128, 40960,
                               k, values["u"], values["w"], values["g_cumsum"], h0, torch.bfloat16)
        bridged = run()
    stream.synchronize()
    assert all(torch.equal(actual, expected) for actual, expected in zip(bridged, native))


def test_current_actual_bridge_rejects_wrong_value_dtype() -> None:
    patch_rocm_autotune()
    inputs = stage6a.fixed_inputs(64)
    _, k, _, _, _, h0 = inputs
    values = stage6a.vllm_manual_stages(inputs)
    with pytest.raises(ValueError, match="matched v/w/v_new dtype"):
        external_call(Bridge(), "invalid", CURRENT / "vllm/kernel.hsaco",
                      "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", (4, 8), 128, 40960,
                      k, values["u"].float(), values["w"], values["g_cumsum"], h0, torch.bfloat16)

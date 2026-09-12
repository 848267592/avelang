"""Focused correctness guards for the opt-in Stage 6S boundary experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (
    qwen_gdn_bt64_stage6s_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages,
    qwen_gdn_full_bt64_stage6s_current_asm_stages,
    stage6s_contract,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout


HERE = Path(__file__).resolve()
LADDER = HERE.parents[1] / "compile_bug/qwen_mfma32_lowering_ladder"
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
sys.path.insert(0, str(STAGE6A))
import stage6a_full_graph_audit as stage6a  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HIP/CUDA")


def _inputs(t: int, state: bool = True) -> tuple[torch.Tensor, ...]:
    patch_rocm_autotune()
    return stage6a.make_inputs(t, 2026071700 + t, "random", state)


@pytest.mark.parametrize("t", (64, 512))
def test_stage6s_bridge_matches_native_vllm_on_identical_bf16_boundary(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t)
    values = stage6a.vllm_manual_stages((q, k, v, g, beta, h0))
    native = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=values["w"],
        u=values["u"],
        g=values["g_cumsum"],
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=None,
    )
    bridged = qwen_gdn_bt64_stage6s_recurrence_bridge(k, values["w"], values["u"], values["g_cumsum"], h0)
    torch.cuda.synchronize()
    assert all(torch.equal(actual, expected) for actual, expected in zip(bridged, native))


@pytest.mark.parametrize("t,state", ((64, True), (512, False)))
def test_stage6s_full_boundary_contract_stays_within_public_tolerance(t: int, state: bool) -> None:
    q, k, v, g, beta, h0 = _inputs(t, state)
    output, final_state = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    native_output, native_state = stage6a.vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )
    stages = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages(q, k, v, g, beta, initial_state=h0)
    torch.cuda.synchronize()
    assert stages["w"].dtype == torch.float32 and stages["w_bf16"].dtype == torch.bfloat16
    assert stages["u"].dtype == torch.float32 and stages["u_bf16"].dtype == torch.bfloat16
    assert stages["v_new_bf16"].dtype == torch.bfloat16 and stages["v_new"].dtype == torch.float32
    assert float((output.float() - native_output.float()).abs().max()) <= 1.0 / 128.0
    assert float((final_state.float() - native_state.float()).abs().max()) <= 2.0e-2


def test_stage6s_rejects_invalid_boundary_without_fallback() -> None:
    q, k, v, g, beta, h0 = _inputs(64)
    values = stage6a.vllm_manual_stages((q, k, v, g, beta, h0))
    with pytest.raises(ValueError, match="BF16 k/w/u"):
        qwen_gdn_bt64_stage6s_recurrence_bridge(k, values["w"].float(), values["u"], values["g_cumsum"], h0)
    with pytest.raises(ValueError, match="divisible"):
        qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q[:, :63], k[:, :63], v[:, :63], g[:, :63], beta[:, :63], initial_state=h0)
    assert stage6s_contract()["hsaco_sha256"] == "632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e"


@pytest.mark.parametrize("t", (64, 512))
def test_stage6s_captured_hierarchical_solve_matches_v18_contract(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t)
    stages = qwen_gdn_full_bt64_stage6s_current_asm_stages(q, k, v, g, beta, initial_state=h0)
    expected = qwen_gdn_solve_avelang_v18_bt64_layout(stages["a"])
    torch.cuda.synchronize()
    delta = (stages["a_solved"] - expected).abs()
    assert float(delta.max()) <= 1.0e-5


def test_stage6s_nondefault_stream_and_graph_replay() -> None:
    q, k, v, g, beta, h0 = _inputs(64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, initial_state=h0)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, initial_state=h0)
        graph.replay()
    stream.synchronize()
    assert all(value.isfinite().all() for value in result if value is not None)

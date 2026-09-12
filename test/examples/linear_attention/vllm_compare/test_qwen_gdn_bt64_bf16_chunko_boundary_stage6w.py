"""Correctness gates for the opt-in Stage 6W BF16 chunk-o boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w,
    qwen_gdn_full_bt64_stage6w_bf16_chunko_eager,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_chunk_o_bt64_mfma_v2_s0  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def _chunk_o_inputs(t: int, seed: int):
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    chunks = t // 64
    torch.manual_seed(seed + 17)
    v_new_bf16 = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h_bf16 = (torch.randn((1, chunks, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, v_new_bf16, h_bf16, g


@pytest.mark.parametrize("t", (64, 512, 2048))
def test_stage6w_chunko_matches_existing_bf16_boundary(t: int):
    q, k, v_new_bf16, h_bf16, g = _chunk_o_inputs(t, 2026074000 + t)
    expected = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new_bf16.float(), h_bf16, g).to(torch.bfloat16)
    actual = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new_bf16, h_bf16, g)
    torch.cuda.synchronize()
    mismatch = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    mean_abs = float((actual.float() - expected.float()).abs().mean().item())
    print(f"Stage6W chunk-o T={t}: mismatch={mismatch} max_abs={max_abs:.9g} mean_abs={mean_abs:.9g}")
    assert mismatch == 0
    assert actual.dtype == torch.bfloat16
    assert actual.stride() == expected.stride()


@pytest.mark.parametrize(
    ("t", "mode", "with_initial_state"),
    ((64, "random", True), (512, "high_dynamic", True), (2048, "neutral_gate", False), (8192, "cancellation", True)),
)
def test_stage6w_full_eager_correctness(t: int, mode: str, with_initial_state: bool):
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, 2026074100 + t, mode, with_initial_state)
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    u1_output, u1_state = qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **kwargs)
    actual_output, actual_state = qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(q, k, v, g, beta, **kwargs)
    ref_output, ref_state = vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    assert actual_state is not None and u1_state is not None and ref_state is not None
    equivalence = int((actual_output.view(torch.int16) != u1_output.view(torch.int16)).sum().item())
    output_max = float((actual_output.float() - ref_output.float()).abs().max().item())
    state_max = float((actual_state - ref_state).abs().max().item())
    print(
        f"Stage6W full T={t}: U1_mismatch={equivalence} "
        f"vllm_output={output_max:.9g} vllm_state={state_max:.9g}"
    )
    assert equivalence == 0
    assert torch.equal(actual_state, u1_state)
    assert output_max <= OUTPUT_ATOL
    assert state_max <= STATE_ATOL


def test_stage6w_rejects_fp32_v_new_without_fallback():
    q, k, v_new_bf16, h_bf16, g = _chunk_o_inputs(64, 2026074299)
    with pytest.raises(ValueError):
        qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new_bf16.float(), h_bf16, g)

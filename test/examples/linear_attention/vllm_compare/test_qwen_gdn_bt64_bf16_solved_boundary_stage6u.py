"""Correctness gates for the opt-in Stage 6U solved-boundary experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager,
    qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager,
    qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u,
)
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (  # noqa: E402
    _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into,
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
    qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


@pytest.mark.parametrize("t", (64, 128, 512))
def test_stage6u_p0_matches_explicit_cast_bitwise(t: int):
    torch.manual_seed(2026072000 + t)
    a = torch.randn((1, t, 8, 64), device="cuda", dtype=torch.float32) * 0.003
    expected = qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u(a)
    actual = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a)
    torch.cuda.synchronize()
    mismatch = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
    print(f"P0 T={t}: mismatch={mismatch} max_abs={(actual.float() - expected.float()).abs().max().item():.9g}")
    assert mismatch == 0
    assert actual.stride() == expected.stride()
    tiles = actual.view(1, t // 64, 64, 8, 64).permute(0, 1, 3, 2, 4)
    assert torch.count_nonzero(torch.triu(tiles, diagonal=1)).item() == 0


def test_stage6u_p0_prefill_reuse_and_nondefault_stream():
    torch.manual_seed(2026072064)
    a = torch.randn((1, 64, 8, 64), device="cuda", dtype=torch.float32) * 0.003
    expected = qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u(a)
    out = torch.full(a.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(a, out)
    stream.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    out.fill_(float("nan"))
    _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.isfinite(out).all()


def _c0_reference(k, v, g, beta, a_solved):
    t = k.shape[1]
    w = torch.empty((1, t, 8, 128), device=k.device, dtype=torch.bfloat16)
    u = torch.empty_like(w)
    for chunk_start in range(0, t, 64):
        for head in range(8):
            aa = a_solved[0, chunk_start : chunk_start + 64, head].float()
            bb = beta[0, chunk_start : chunk_start + 64, head]
            gg = g[0, chunk_start : chunk_start + 64, head]
            aw = (aa * bb[None, :] * torch.exp(gg)[None, :]).to(torch.bfloat16)
            au = (aa * bb[None, :]).to(torch.bfloat16)
            kk = k[0, chunk_start : chunk_start + 64, head // 2]
            vv = v[0, chunk_start : chunk_start + 64, head]
            w[0, chunk_start : chunk_start + 64, head] = (aw.float() @ kk.float()).to(torch.bfloat16)
            u[0, chunk_start : chunk_start + 64, head] = (au.float() @ vv.float()).to(torch.bfloat16)
    return w, u


def test_stage6u_c0_isolated_math_and_layout():
    q, k, v, g, beta, _ = make_inputs(64, 2026072077, "random", True)
    del q
    a = torch.randn((1, 64, 8, 64), device=k.device, dtype=torch.bfloat16) * 0.02
    actual_w, actual_u = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g, beta, a)
    expected_w, expected_u = _c0_reference(k, v, g, beta, a)
    torch.cuda.synchronize()
    w_max = float((actual_w.float() - expected_w.float()).abs().max().item())
    u_max = float((actual_u.float() - expected_u.float()).abs().max().item())
    print(f"C0: W max_abs={w_max:.9g}, U max_abs={u_max:.9g}")
    assert w_max <= 0.0078125
    assert u_max <= 0.0078125
    assert actual_w.stride() == (64 * 8 * 128, 8 * 128, 128, 1)
    assert actual_u.stride() == actual_w.stride()
    assert torch.isfinite(actual_w).all() and torch.isfinite(actual_u).all()


@pytest.mark.parametrize(
    ("t", "mode", "with_initial_state"),
    ((64, "random", True), (128, "neutral_gate", False), (512, "high_dynamic", True)),
)
def test_stage6u_complete_eager_public_correctness(t: int, mode: str, with_initial_state: bool):
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, 2026072100 + t, mode, with_initial_state)
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    calls = {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **kwargs),
        "u0": lambda: qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager(q, k, v, g, beta, **kwargs),
        "u1": lambda: qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **kwargs),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }
    values = {name: fn() for name, fn in calls.items()}
    torch.cuda.synchronize()
    ref_output, ref_state = values["vllm"]
    assert ref_state is not None
    for name in ("stage6s", "u0", "u1"):
        output, state = values[name]
        assert state is not None
        output_max = float((output.float() - ref_output.float()).abs().max().item())
        state_max = float((state - ref_state).abs().max().item())
        print(f"{name} T={t}: output={output_max:.9g}, state={state_max:.9g}")
        assert output_max <= OUTPUT_ATOL
        assert state_max <= STATE_ATOL
    assert torch.equal(values["u0"][0].view(torch.int16), values["u1"][0].view(torch.int16))
    assert torch.equal(values["u0"][1], values["u1"][1])


def test_stage6u_rejects_unsupported_contract_without_fallback():
    _, k, v, g, beta, _ = make_inputs(64, 2026072199, "random", True)
    a = torch.zeros((1, 64, 8, 64), device=k.device, dtype=torch.float32)
    with pytest.raises(ValueError):
        qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g, beta, a)
    with pytest.raises(ValueError):
        qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a.to(torch.bfloat16))

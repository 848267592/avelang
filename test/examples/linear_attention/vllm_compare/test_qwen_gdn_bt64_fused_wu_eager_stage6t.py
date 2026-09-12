"""Public-API correctness coverage for the opt-in Stage 6T experiment."""

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
from qwen_gdn_bt64_fused_wu_eager_stage6t import (  # noqa: E402
    qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager,
    qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def _public_calls(inputs: tuple[torch.Tensor, ...]):
    q, k, v, g, beta, h0 = inputs
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **kwargs),
        "f0": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, **kwargs),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **kwargs),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def _assert_close(name: str, actual: tuple[torch.Tensor, torch.Tensor | None], reference: tuple[torch.Tensor, torch.Tensor | None]):
    output, state = actual
    expected_output, expected_state = reference
    assert state is not None and expected_state is not None
    output_max = float((output.float() - expected_output.float()).abs().max().item())
    state_max = float((state - expected_state).abs().max().item())
    print(f"{name}: output_max_abs={output_max:.9g} final_state_max_abs={state_max:.9g}")
    assert output_max <= OUTPUT_ATOL
    assert state_max <= STATE_ATOL


@pytest.mark.parametrize(
    ("t", "mode", "with_initial_state"),
    ((64, "random", True), (128, "neutral_gate", False), (512, "high_dynamic", True)),
)
def test_stage6t_public_full_correctness(t: int, mode: str, with_initial_state: bool):
    patch_rocm_autotune()
    calls = _public_calls(make_inputs(t, 2026071700 + t, mode, with_initial_state))
    results = {name: fn() for name, fn in calls.items()}
    torch.cuda.synchronize()
    for name in ("stage6s", "f0", "f1"):
        _assert_close(name, results[name], results["vllm"])
    f0_output, f0_state = results["f0"]
    f1_output, f1_state = results["f1"]
    assert f0_state is not None and f1_state is not None
    assert float((f0_output.float() - f1_output.float()).abs().max().item()) == 0.0
    assert float((f0_state - f1_state).abs().max().item()) == 0.0


def test_stage6t_public_full_nondefault_stream():
    patch_rocm_autotune()
    calls = _public_calls(make_inputs(64, 2026071711, "cancellation", True))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results = {name: fn() for name, fn in calls.items()}
    stream.synchronize()
    for name in ("stage6s", "f0", "f1"):
        _assert_close(name, results[name], results["vllm"])


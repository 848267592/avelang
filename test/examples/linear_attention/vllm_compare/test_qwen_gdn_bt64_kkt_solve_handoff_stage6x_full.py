"""Full graph semantic gate: Stage 6X X2 must preserve Stage 6W exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import qwen_gdn_full_bt64_stage6w_bf16_chunko_eager  # noqa: E402
from qwen_gdn_bt64_kkt_solve_handoff_stage6x_full import qwen_gdn_full_bt64_stage6x_kkt_solve_eager  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


@pytest.mark.parametrize("t", (64, 128, 512, 2048, 8192))
@pytest.mark.parametrize("mode", ("random", "high_dynamic", "cancellation", "neutral_gate"))
@pytest.mark.parametrize("with_initial_state", (False, True))
def test_stage6x_full_is_bit_exact_to_stage6w(
    t: int, mode: str, with_initial_state: bool
) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 2026072700 + t, mode, True)
    kwargs = dict(
        initial_state=h0 if with_initial_state else None,
        output_final_state=True,
        scale=128 ** -0.5,
    )
    expected_output, expected_state = qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(q, k, v, g, beta, **kwargs)
    actual_output, actual_state = qwen_gdn_full_bt64_stage6x_kkt_solve_eager(q, k, v, g, beta, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(actual_output, expected_output)
    assert actual_state is not None and expected_state is not None
    assert torch.equal(actual_state, expected_state)
    print(f"stage6x_full,T={t},mode={mode},initial={with_initial_state},output/state=bit_exact")

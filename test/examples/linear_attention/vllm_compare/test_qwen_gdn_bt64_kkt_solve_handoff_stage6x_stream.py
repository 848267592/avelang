"""Non-default-stream contract test for the Stage 6X full public API."""

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


@pytest.mark.parametrize("t", (64, 2048))
def test_stage6x_full_non_default_stream_matches_stage6w(t: int) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 2026073000 + t, "random", True)
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    expected_out, expected_state = qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(q, k, v, g, beta, **kwargs)
    stream = torch.cuda.Stream(device=q.device)
    with torch.cuda.stream(stream):
        actual_out, actual_state = qwen_gdn_full_bt64_stage6x_kkt_solve_eager(q, k, v, g, beta, **kwargs)
    stream.synchronize()
    assert torch.equal(actual_out, expected_out)
    assert actual_state is not None and expected_state is not None
    assert torch.equal(actual_state, expected_state)

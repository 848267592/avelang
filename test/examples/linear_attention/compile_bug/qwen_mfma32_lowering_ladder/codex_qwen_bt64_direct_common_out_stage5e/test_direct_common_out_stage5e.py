"""Correctness and API gates for the audit-only Stage 5E wrappers."""

from __future__ import annotations

import pytest
import torch

from direct_common_out_harness import SOLVE_A, SOLVE_B, Stage5ERunner, patch_rocm_autotune
from qwen_gdn_bt64_solve_direct_out_stage5e_audit import (
    qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit,
    qwen_gdn_solve_v18_bt64_direct_out_audit,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1


@pytest.fixture(autouse=True)
def _gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    patch_rocm_autotune()


@pytest.mark.parametrize("t", [64, 512, 2048])
def test_direct_out_is_bitwise_identical_and_reusable(t: int) -> None:
    runner = Stage5ERunner(t, 20260717 + t)
    runner.prepare_local_matrix()
    a = runner.buffers.a
    common = runner.solved_common
    pointer = common.data_ptr()

    expected_v18 = qwen_gdn_solve_avelang_v18_bt64_layout(a)
    actual_v18 = qwen_gdn_solve_v18_bt64_direct_out_audit(a, common)
    torch.cuda.synchronize()
    assert actual_v18.data_ptr() == pointer
    assert torch.equal(actual_v18.view(torch.int32), expected_v18.view(torch.int32))

    expected_v1 = qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    actual_v1 = qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, common)
    torch.cuda.synchronize()
    assert actual_v1.data_ptr() == pointer
    assert torch.equal(actual_v1.view(torch.int32), expected_v1.view(torch.int32))

    qwen_gdn_solve_v18_bt64_direct_out_audit(a, common)
    torch.cuda.synchronize()
    assert torch.equal(common.view(torch.int32), expected_v18.view(torch.int32))


@pytest.mark.parametrize("t", [64, 512, 2048])
def test_direct_common_full_preserves_original_graph(t: int) -> None:
    runner = Stage5ERunner(t, 20260727 + t)
    runner.launch_full(SOLVE_A)
    expected_output, expected_state = runner.output_snapshot()
    runner.launch_full_direct(SOLVE_A)
    actual_output, actual_state = runner.output_snapshot()
    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state)

    runner.launch_full_direct(SOLVE_B)
    output_v1, state_v1 = runner.output_snapshot()
    assert (output_v1.float() - expected_output.float()).abs().max().item() <= 0.0078125
    assert (state_v1 - expected_state).abs().max().item() <= 0.02


def test_direct_out_rejects_invalid_contracts() -> None:
    a = torch.zeros((1, 64, 8, 64), dtype=torch.float32, device="cuda")
    out = torch.empty_like(a)
    with pytest.raises(ValueError, match="alias"):
        qwen_gdn_solve_v18_bt64_direct_out_audit(a, a)
    with pytest.raises(ValueError, match="dtype"):
        qwen_gdn_solve_v18_bt64_direct_out_audit(a, out.to(torch.float16))
    noncontiguous = torch.empty((1, 64, 64, 8), dtype=torch.float32, device="cuda").transpose(2, 3)
    with pytest.raises(ValueError, match="contiguous"):
        qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, noncontiguous)
    with pytest.raises(ValueError, match="shape"):
        qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out[:, :32])


def test_direct_out_uses_current_stream() -> None:
    runner = Stage5ERunner(64, 20260737)
    runner.prepare_local_matrix()
    expected = qwen_gdn_solve_avelang_v18_bt64_layout(runner.buffers.a)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qwen_gdn_solve_v18_bt64_direct_out_audit(runner.buffers.a, runner.solved_common)
    stream.synchronize()
    assert torch.equal(runner.solved_common.view(torch.int32), expected.view(torch.int32))


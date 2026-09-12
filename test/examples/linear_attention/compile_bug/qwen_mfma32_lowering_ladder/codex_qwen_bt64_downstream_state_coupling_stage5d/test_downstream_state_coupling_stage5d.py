"""Correctness gates for the audit-only Stage 5D fixed-buffer graph."""

from __future__ import annotations

import pytest
import torch

from downstream_state_coupling_harness import SOLVE_A, SOLVE_B, Stage5DRunner, patch_rocm_autotune


@pytest.fixture(autouse=True)
def _gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    patch_rocm_autotune()


@pytest.mark.parametrize("t", [64, 512, 8192])
def test_fixed_buffer_graphs_and_canonical_control(t: int) -> None:
    runner = Stage5DRunner(t, 20260760 + t)
    runner.launch_full(SOLVE_A)
    output_a, state_a = runner.output_snapshot()
    runner.launch_full(SOLVE_B)
    output_b, state_b = runner.output_snapshot()
    assert (output_a.float() - output_b.float()).abs().max().item() <= 1.0e-3
    assert (state_a - state_b).abs().max().item() <= 1.0e-3

    runner.prepare_canonical()
    runner.launch_tail(runner.buffers.canonical)
    output_c, state_c = runner.output_snapshot()
    assert torch.equal(output_a, output_c)
    assert torch.equal(state_a, state_c)


def test_same_pointer_control_reuses_exact_pointer() -> None:
    runner = Stage5DRunner(512, 20261272)
    runner.prepare_local_matrix()
    pointers = []
    for solve_impl in (SOLVE_A, SOLVE_B):
        solved = runner.launch_solve(solve_impl)
        runner.buffers.canonical.copy_(solved)
        pointers.append(runner.buffers.canonical.data_ptr())
        runner.launch_tail(runner.buffers.canonical)
    torch.cuda.synchronize()
    assert pointers[0] == pointers[1]


def test_cache_controls_preserve_visible_results() -> None:
    runner = Stage5DRunner(512, 20261273)
    runner.prepare_canonical()
    runner.launch_tail(runner.buffers.canonical)
    expected_output, expected_state = runner.output_snapshot()

    runner.perturb_cache(64)
    runner.launch_tail(runner.buffers.canonical)
    cold_output, cold_state = runner.output_snapshot()
    runner.prime_downstream_inputs(runner.buffers.canonical)
    runner.launch_tail(runner.buffers.canonical)
    prime_output, prime_state = runner.output_snapshot()
    assert torch.equal(expected_output, cold_output)
    assert torch.equal(expected_state, cold_state)
    assert torch.equal(expected_output, prime_output)
    assert torch.equal(expected_state, prime_state)

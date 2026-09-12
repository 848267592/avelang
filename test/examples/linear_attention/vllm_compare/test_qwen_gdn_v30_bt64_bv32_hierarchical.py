import pytest
import torch

from qwen_gdn_chunked_avelang_v30_bt64_bv32_hierarchical import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical,
    qwen_gdn_fused_chunk_gdr_full_reference,
)


@pytest.mark.parametrize("num_tokens", [64, 512])
def test_v30_hierarchical_w_zero_update_matches_reference(num_tokens: int) -> None:
    """The V16/token16/K16 update order preserves the BT64 recurrence."""

    k, w, u, decay, g_last, initial_state = _make_inputs(num_tokens, seed=20260711)
    w.zero_()
    h, final_state = qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(
        k, w, u, decay, g_last, initial_state
    )
    h_ref, state_ref = qwen_gdn_fused_chunk_gdr_full_reference(
        k, w, u, decay, g_last, initial_state
    )
    torch.cuda.synchronize()
    print(
        f"T={num_tokens} h_max_abs={(h - h_ref).abs().max().item():.8e} "
        f"state_max_abs={(final_state - state_ref).abs().max().item():.8e}"
    )
    # The per-token16 state writeback changes BF16 accumulation order across
    # BT64 chunks.  T=512 establishes the bounded 2.29e-5 rounding delta.
    torch.testing.assert_close(h, h_ref, rtol=0.0, atol=3e-5)
    torch.testing.assert_close(final_state, state_ref, rtol=0.0, atol=3e-5)


def test_v30_nonzero_w_mismatch_is_visible() -> None:
    """Guard the known D1 semantic failure from being mistaken for success."""

    k, w, u, decay, g_last, initial_state = _make_inputs(64, seed=17)
    _, final_state = qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(
        k, w, u, decay, g_last, initial_state
    )
    _, state_ref = qwen_gdn_fused_chunk_gdr_full_reference(k, w, u, decay, g_last, initial_state)
    torch.cuda.synchronize()
    error = (final_state - state_ref).abs().max().item()
    print(f"T=64 nonzero_w_state_max_abs={error:.8e}")
    assert error > 1e-3

import pytest
import torch

from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import _make_inputs
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp import (
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32,
)
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_pred_streaming_exp import (
    qwen_gdn_fused_chunk_gdr_full_kfrag_pred_streaming_avelang_v29_mfma32,
)


@pytest.mark.parametrize("num_tokens", [64, 512, 1024, 2048])
def test_kfrag_pred_streaming_is_bit_exact_to_kfrag_rewrite(num_tokens: int) -> None:
    k, w, u, decay, g_last, initial_state = _make_inputs(num_tokens, seed=20260711)
    h_rewrite, state_rewrite = qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32(
        k, w, u, decay, g_last, initial_state
    )
    h_stream, state_stream = qwen_gdn_fused_chunk_gdr_full_kfrag_pred_streaming_avelang_v29_mfma32(
        k, w, u, decay, g_last, initial_state
    )
    h_error = (h_stream - h_rewrite).abs()
    state_error = (state_stream - state_rewrite).abs()
    print(
        f"T={num_tokens} h_max_abs={h_error.max().item():.8e} "
        f"state_max_abs={state_error.max().item():.8e}"
    )
    torch.testing.assert_close(h_stream, h_rewrite, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_stream, state_rewrite, rtol=0.0, atol=0.0)

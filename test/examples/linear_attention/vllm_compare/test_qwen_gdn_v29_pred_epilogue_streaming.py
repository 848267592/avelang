import pytest
import torch

from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (
    _make_inputs,
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32,
)
from qwen_gdn_chunked_avelang_v29_mfma32_pred_epilogue_streaming_exp import (
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32_pred_epilogue_streaming,
)


@pytest.mark.parametrize("num_tokens", [64, 512])
def test_pred_epilogue_streaming_preserves_original_v29(num_tokens: int) -> None:
    k, w, u, decay, g_last, initial_state = _make_inputs(num_tokens, seed=20260711)
    h_original, state_original = qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32(
        k, w, u, decay, g_last, initial_state
    )
    h_stream, state_stream = qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32_pred_epilogue_streaming(
        k, w, u, decay, g_last, initial_state
    )
    h_error = (h_stream - h_original).abs()
    state_error = (state_stream - state_original).abs()
    print(
        f"T={num_tokens} h_max_abs={h_error.max().item():.8e} "
        f"h_mean_abs={h_error.mean().item():.8e} "
        f"state_max_abs={state_error.max().item():.8e} "
        f"state_mean_abs={state_error.mean().item():.8e}"
    )
    torch.testing.assert_close(h_stream, h_original, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_stream, state_original, rtol=0.0, atol=0.0)

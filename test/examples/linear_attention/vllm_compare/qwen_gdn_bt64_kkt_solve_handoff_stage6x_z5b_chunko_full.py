"""Opt-in X2 full graph with only Stage 6W chunk-o replaced by Stage 6Z Z5B.

This is intentionally a narrow integration candidate.  Cumsum, fused
KKT-to-solve, fused W/U, and the hash-guarded current-vLLM recurrence bridge
are byte-for-byte the same public stages used by the established X2 graph.
Only the final chunk-o dispatch changes.
"""

from __future__ import annotations

import torch

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import _require_target
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import qwen_gdn_bt64_stage6s_recurrence_bridge
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u
from qwen_gdn_bt64_kkt_solve_handoff_stage6x import qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import BT, K_DIM
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone


def qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the X2 full Eager graph with Z5B as its only substituted stage."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a_solved_bf16 = qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(k, g_cumsum, beta)
    w_bf16, u_bf16 = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g_cumsum, beta, a_solved_bf16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    output_bf16 = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(
        q, k, v_new_bf16, h_bf16, g_cumsum, scale=scale
    )
    return output_bf16, final_state if output_final_state else None


__all__ = ["qwen_gdn_full_bt64_stage6x_z5b_chunko_eager"]

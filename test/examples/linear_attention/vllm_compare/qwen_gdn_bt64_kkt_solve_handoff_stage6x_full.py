"""Opt-in Stage 6X full Eager graph with CTA-local KKT-to-solve handoff.

This module replaces exactly the Stage 6W sequence ``KKT FP32 global ->
Stage 6U solve``.  All remaining Stage 6W producers, recurrence ABI and
BF16 chunk-o boundary are intentionally reused unchanged.
"""

from __future__ import annotations

import torch

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (
    _require_target,
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w,
)
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import qwen_gdn_bt64_stage6s_recurrence_bridge
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u
from qwen_gdn_bt64_kkt_solve_handoff_stage6x import qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import BT, K_DIM
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone


def qwen_gdn_full_bt64_stage6x_kkt_solve_eager(
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
    """Stage 6W graph with X2 replacing only the KKT-to-solve global boundary."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a_solved_bf16 = qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(k, g_cumsum, beta)
    w_bf16, u_bf16 = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g_cumsum, beta, a_solved_bf16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    output_bf16 = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(
        q, k, v_new_bf16, h_bf16, g_cumsum, scale=scale
    )
    return output_bf16, final_state if output_final_state else None


__all__ = ["qwen_gdn_full_bt64_stage6x_kkt_solve_eager"]

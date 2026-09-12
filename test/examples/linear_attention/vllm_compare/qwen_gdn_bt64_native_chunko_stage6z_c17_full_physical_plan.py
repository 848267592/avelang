"""C17 experimental full-region ChunkOPhysicalPlan entry point.

This module deliberately reuses the already validated BDV2 logical full-scope
source contract.  It does not introduce another Qwen kernel or a second
source schedule.  The only new choice is the compiler gate consumed by
``lower_qwen_block_dot_pass``:

    one ChunkOPhysicalPlan -> H/K producer + MFMA consumers

The public production selector is untouched.  The wrapper fixes the C17
machine contract to gfx942/WG256/specialized/p2-first-class and fails closed
if the compiler gate is not available.
"""

from __future__ import annotations

import os

import torch

from qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope import (
    BT,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into,
)


def _enable_c17() -> None:
    os.environ["AVELANG_STAGE6Z_FULL_PHYSICAL_PLAN"] = "c17"


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan_launch_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> None:
    _enable_c17()
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
        q,
        k,
        v_new_bf16,
        h_bf16,
        g,
        output_bf16,
        scale=scale,
        chunk_size=chunk_size,
        lowering="specialized",
        planner="bdv2_p1_affine",
        preservation="p2_first_class",
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    output_bf16 = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan_launch_into(
        q,
        k,
        v_new_bf16,
        h_bf16,
        g,
        output_bf16,
        scale=scale,
        chunk_size=chunk_size,
    )
    return output_bf16


__all__ = [
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c17_full_physical_plan_launch_into",
]

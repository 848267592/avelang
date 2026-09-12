#!/usr/bin/env python3
"""Replay one pre-captured Stage 6A graph for rocprof trace/counter capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage6a_full_graph_audit import (  # noqa: E402
    BT,
    IMPL_A,
    IMPL_B,
    IMPLEMENTATIONS,
    CapturedGraph,
    fixed_inputs,
    full_call,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1  # noqa: E402
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402


def vllm_prefix(inputs: tuple[torch.Tensor, ...], through: str) -> dict[str, torch.Tensor]:
    """Materialize only the canonical dependencies required by one body.

    This is deliberately for profiler setup, not for the ABBA timing harness.
    It avoids compiling an unrelated complete Avelang graph before profiling a
    one-kernel body while retaining the real BT64 shapes, dtypes, and layouts.
    """
    q, k, v, g, beta, initial_state = inputs
    values: dict[str, torch.Tensor] = {}
    if through == "cast":
        values["output_fp32"] = torch.empty_like(q, dtype=torch.float32)
        return values
    values["g_cumsum"] = chunk_local_cumsum(g, chunk_size=BT)
    if through == "cumsum":
        return values
    values["a"] = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=values["g_cumsum"], output_dtype=torch.float32)
    if through == "kkt":
        return values
    values["a_solved"] = solve_tril(A=values["a"], output_dtype=k.dtype)
    if through == "solve":
        return values
    values["w"], values["u"] = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=values["a_solved"], g_cumsum=values["g_cumsum"], cu_seqlens=None
    )
    if through == "wu":
        return values
    values["h_bf16"], values["v_new"], values["final_state"] = chunk_gated_delta_rule_fwd_h(
        k=k, w=values["w"], u=values["u"], g=values["g_cumsum"], initial_state=initial_state,
        output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )
    return values


def body_call(implementation: str, stage: str, inputs: tuple[torch.Tensor, ...]):
    q, k, v, g, beta, initial_state = inputs
    if implementation == IMPL_B and stage == "cast":
        return None
    values = vllm_prefix(inputs, stage)
    if implementation == IMPL_A:
        # vLLM's stage contract retains these values as BF16.  The Stage 4
        # Avelang ABI intentionally consumes FP32.  This one-time setup cast
        # is outside the captured target body and preserves the target's
        # real input dtype/layout for resource collection.
        for name in ("a_solved", "w", "u", "v_new"):
            if name in values:
                values[name] = values[name].float().contiguous()
        if stage == "cumsum":
            return lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
        if stage == "kkt":
            return lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, values["g_cumsum"], beta)
        if stage == "solve":
            return lambda: qwen_gdn_solve_bt64_hierarchical_fp32_v1(values["a"])
        if stage == "wu":
            return lambda: qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, values["g_cumsum"], beta, values["a_solved"])
        if stage == "recurrence":
            return lambda: qwen_gdn_bt64_gfx942_asm_v0(k, values["w"], values["u"], values["g_cumsum"], initial_state)
        if stage == "chunk_o":
            return lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, values["v_new"], values["h_bf16"], values["g_cumsum"])
        if stage == "cast":
            return lambda: values["output_fp32"].to(q.dtype)
    if stage == "cumsum":
        return lambda: chunk_local_cumsum(g, chunk_size=BT)
    if stage == "kkt":
        return lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=values["g_cumsum"], output_dtype=torch.float32)
    if stage == "solve":
        return lambda: solve_tril(A=values["a"], output_dtype=k.dtype)
    if stage == "wu":
        return lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=values["a_solved"], g_cumsum=values["g_cumsum"], cu_seqlens=None)
    if stage == "recurrence":
        return lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=values["w"], u=values["u"], g=values["g_cumsum"], initial_state=initial_state,
            output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
        )
    if stage == "chunk_o":
        return lambda: chunk_fwd_o(
            q=q, k=k, v=values["v_new"], h=values["h_bf16"], g=values["g_cumsum"], scale=128 ** -0.5,
            chunk_size=BT,
        )
    raise ValueError(f"unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("full", "body"), required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--replay", type=int, default=3)
    args = parser.parse_args()
    if args.scope == "body" and args.stage is None:
        raise ValueError("--stage is required for body capture")
    patch_rocm_autotune()
    inputs = fixed_inputs(args.T)
    if args.scope == "full":
        graph = CapturedGraph.create(args.implementation, full_call(args.implementation, inputs))
    else:
        fn = body_call(args.implementation, args.stage, inputs)
        if fn is None:
            print(json.dumps({"scope": args.scope, "stage": args.stage, "implementation": args.implementation,
                              "T": args.T, "not_materialized": True}))
            return
        graph = CapturedGraph.create(f"{args.implementation}:{args.stage}", fn)
    torch.cuda.synchronize()
    for _ in range(args.replay):
        graph.replay()
    torch.cuda.synchronize()
    print(json.dumps({"scope": args.scope, "stage": args.stage, "implementation": args.implementation,
                      "T": args.T, "replay": args.replay, "output_ptrs": graph.capture_output_ptrs}))


if __name__ == "__main__":
    main()

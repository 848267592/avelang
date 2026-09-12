#!/usr/bin/env python3
"""Unified post-conv packed KDA decode correctness and latency benchmark.

The contract deliberately starts after causal convolution and ends before
RMSNorm:

  packed post-conv QKV + raw gate + raw beta + state
      -> packed recurrent KDA -> output + updated state

For SGLang, the optional CUDA JIT packed-decode fast path is bypassed and the
same local ``@triton.jit`` kernel is launched directly.  This keeps the MI300X
baseline ordinary Triton KDA at every batch size.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from bench_common import (
    DEFAULT_RESULTS_DIR,
    HEAD_DIM,
    LOWER_BOUND,
    assert_close,
    bench_cuda,
    effective_state_tbps,
    make_kda_decode_inputs,
    result_fields,
    result_row,
    save_result,
    sync,
    torch_kda_reference,
)


SGLANG_COMMIT = "908226fea2df861769e2720161a75649ae4c6f92"
VLLM_COMMIT = "40e6042ec83eb8f2971f21043a5da40496bd188a"
Runner = Callable[[dict[str, torch.Tensor | float | int]], tuple[torch.Tensor, torch.Tensor]]


def _decode_tensors(
    inp: dict[str, torch.Tensor | float | int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, float]:
    mixed_qkv = inp["mixed_qkv"]
    raw_g = inp["raw_g"]
    raw_beta = inp["raw_beta"]
    A_log = inp["A_log"]
    dt_bias = inp["dt_bias"]
    state_pool = inp["state_pool"]
    state_indices = inp["state_indices"]
    assert all(
        isinstance(x, torch.Tensor)
        for x in (mixed_qkv, raw_g, raw_beta, A_log, dt_bias, state_pool, state_indices)
    )
    return (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state_pool,
        state_indices,
        int(inp["B"]),
        int(inp["H"]),
        int(inp["K"]),
        int(inp["V"]),
        float(inp["scale"]),
    )


def launch_sglang_packed_decode_triton(
    *,
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    B: int,
    H: int,
    K: int,
    V: int,
    scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch SGLang's production Triton packed KDA decode kernel directly.

    The upstream Python wrapper switches to an optional CUDA JIT implementation
    for sufficiently large batches.  Directly launching its Triton kernel
    preserves the exact recurrent/L2Norm math while ensuring this MI300X
    baseline never measures that non-Triton fast path.
    """
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode_kernel,
    )

    if K != HEAD_DIM or V != HEAD_DIM:
        raise ValueError(f"packed Triton baseline requires K=V={HEAD_DIM}")
    if out.shape != (B, 1, H, V):
        raise ValueError(f"unexpected output shape {tuple(out.shape)}")
    if state_pool.shape[1:] != (H, V, K):
        raise ValueError(f"unexpected state shape {tuple(state_pool.shape)}")

    # These are the local wrapper's launch parameters for K=V=128.  Q/K are
    # normalized in the Triton kernel, so the post-conv input remains raw.
    fused_recurrent_kda_packed_decode_kernel[(V // 32, B, H)](
        mixed_qkv=mixed_qkv,
        a=raw_g,
        b=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        o=out,
        h0=state_pool,
        ht=state_pool,
        ssm_state_indices=state_indices,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=raw_g.stride(0),
        stride_b_tok=raw_beta.stride(0),
        stride_init_state_token=state_pool.stride(0),
        stride_final_state_token=state_pool.stride(0),
        stride_indices_seq=state_indices.stride(0),
        H=H,
        HV=H,
        K=K,
        V=V,
        BK=HEAD_DIM,
        BV=32,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        USE_LOWER_BOUND=True,
        num_warps=1,
        num_stages=3,
    )
    return out


@torch.inference_mode()
def run_sglang(inp: dict[str, torch.Tensor | float | int]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state_pool,
        state_indices,
        B,
        H,
        K,
        V,
        scale,
    ) = _decode_tensors(inp)
    state_work = state_pool.clone()
    out = torch.empty((B, 1, H, V), device=mixed_qkv.device, dtype=torch.bfloat16)
    launch_sglang_packed_decode_triton(
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_work,
        state_indices=state_indices,
        B=B,
        H=H,
        K=K,
        V=V,
        scale=scale,
        out=out,
    )
    return out[:, 0].contiguous(), state_work.index_select(0, state_indices.long()).contiguous()


@torch.inference_mode()
def run_vllm(inp: dict[str, torch.Tensor | float | int]) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )

    (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state_pool,
        state_indices,
        B,
        H,
        K,
        _,
        scale,
    ) = _decode_tensors(inp)
    state_work = state_pool.clone()
    # vLLM's packed AMD wrapper has [1, B, H, K] and [1, B, H] inputs,
    # whereas the unified external contract keeps them compact/2-D.
    out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        raw_g=raw_g.view(B, H, K).unsqueeze(0).contiguous(),
        raw_beta=raw_beta.view(B, H).unsqueeze(0).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=state_work,
        state_indices=state_indices,
        scale=scale,
    )
    if tuple(out.shape) != (1, B, H, int(inp["V"])):
        raise RuntimeError(f"unexpected vLLM packed-decode layout: {tuple(out.shape)}")
    return out[0].contiguous(), state_work.index_select(0, state_indices.long()).contiguous()


def get_runner(backend: str) -> Runner:
    if backend == "sglang":
        return run_sglang
    if backend == "vllm":
        return run_vllm
    if backend == "avelang":
        raise NotImplementedError("Avelang packed-decode adapter is intentionally reserved for later")
    raise ValueError(f"unknown backend: {backend}")


def unpack_for_reference(
    inp: dict[str, torch.Tensor | float | int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixed_qkv, raw_g, raw_beta, _, _, state_pool, state_indices, B, H, K, V, _ = _decode_tensors(inp)
    q_width, k_width, v_width = H * K, H * K, H * V
    q, k, v = torch.split(mixed_qkv, [q_width, k_width, v_width], dim=-1)
    return (
        q.view(B, 1, H, K),
        k.view(B, 1, H, K),
        v.view(B, 1, H, V),
        raw_g.view(B, 1, H, K),
        raw_beta.view(B, 1, H),
        state_pool.index_select(0, state_indices.long()).contiguous(),
    )


def make_bench_invoker(
    backend: str, inp: dict[str, torch.Tensor | float | int]
) -> tuple[Callable[[], None], Callable[[], object]]:
    (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        state_template,
        state_indices,
        B,
        H,
        K,
        V,
        scale,
    ) = _decode_tensors(inp)
    state_work = state_template.clone()

    def prepare() -> None:
        state_work.copy_(state_template)

    if backend == "sglang":
        out_work = torch.empty((B, 1, H, V), device=mixed_qkv.device, dtype=torch.bfloat16)

        def invoke() -> torch.Tensor:
            return launch_sglang_packed_decode_triton(
                mixed_qkv=mixed_qkv,
                raw_g=raw_g,
                raw_beta=raw_beta,
                A_log=A_log,
                dt_bias=dt_bias,
                state_pool=state_work,
                state_indices=state_indices,
                B=B,
                H=H,
                K=K,
                V=V,
                scale=scale,
                out=out_work,
            )

        return prepare, invoke

    if backend == "vllm":
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            fused_recurrent_kda_packed_decode,
        )

        raw_g_vllm = raw_g.view(B, H, K).unsqueeze(0).contiguous()
        raw_beta_vllm = raw_beta.view(B, H).unsqueeze(0).contiguous()

        def invoke() -> tuple[torch.Tensor, torch.Tensor]:
            return fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed_qkv,
                raw_g=raw_g_vllm,
                raw_beta=raw_beta_vllm,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=LOWER_BOUND,
                initial_state=state_work,
                state_indices=state_indices,
                scale=scale,
            )

        return prepare, invoke

    if backend == "avelang":
        raise NotImplementedError("Avelang packed-decode adapter is intentionally reserved for later")
    raise ValueError(f"unknown backend: {backend}")


@torch.inference_mode()
def correctness(
    *, backend: str, B: int, H: int, atol: float, rtol: float, result_dir: str
) -> None:
    inp = make_kda_decode_inputs(B=B, H=H)
    out, state = get_runner(backend)(inp)
    sync()
    q, k, v, raw_g, raw_beta, initial_state = unpack_for_reference(inp)
    expected_out, expected_state = torch_kda_reference(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=inp["A_log"],  # type: ignore[arg-type]
        dt_bias=inp["dt_bias"],  # type: ignore[arg-type]
        initial_state=initial_state,
        l2norm_round_to_bf16=False,
    )
    out_metrics = assert_close(name="decode output", actual=out, expected=expected_out[:, 0], atol=atol, rtol=rtol)
    state_metrics = assert_close(name="decode final state", actual=state, expected=expected_state, atol=atol, rtol=rtol)
    row = result_row(
        phase="decode",
        mode="correctness",
        backend=backend,
        backend_path=(
            "sglang.fla.fused_recurrent_kda_packed_decode_kernel direct ordinary Triton"
            if backend == "sglang"
            else "vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent_kda_packed_decode"
        ),
        source_commit=SGLANG_COMMIT if backend == "sglang" else VLLM_COMMIT,
        B=B,
        H=H,
        K=HEAD_DIM,
        V=HEAD_DIM,
        dtype="bf16",
        state_dtype="fp32",
        lower_bound=LOWER_BOUND,
        l2norm="in-kernel FP32 packed-decode norm",
        output_shape=list(out.shape),
        state_shape=list(state.shape),
        output_error=out_metrics,
        state_error=state_metrics,
        atol=atol,
        rtol=rtol,
        passed=True,
    )
    save_result(row, result_dir=result_dir, stem=f"correctness_decode_{backend}")


@torch.inference_mode()
def benchmark(
    *,
    backend: str,
    B: int,
    H: int,
    warmup: int,
    rep: int,
    result_dir: str,
    result_stem: str,
    round_id: int,
) -> None:
    inp = make_kda_decode_inputs(B=B, H=H)
    prepare, invoke = make_bench_invoker(backend, inp)
    result = bench_cuda(invoke, prepare=prepare, warmup=warmup, rep=rep)
    row = result_row(
        phase="decode",
        mode="bench",
        round_id=round_id,
        backend=backend,
        source_commit=SGLANG_COMMIT if backend == "sglang" else VLLM_COMMIT,
        B=B,
        H=H,
        K=HEAD_DIM,
        V=HEAD_DIM,
        dtype="bf16",
        state_dtype="fp32",
        lower_bound=LOWER_BOUND,
        state_reset_copy_excluded=True,
        warmup=warmup,
        rep=rep,
        state_tbps=effective_state_tbps(B=B, H=H, K=HEAD_DIM, V=HEAD_DIM, median_us=result.median_us),
        **result_fields(result),
    )
    save_result(row, result_dir=result_dir, stem=result_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["sglang", "vllm", "avelang"])
    parser.add_argument("--mode", choices=["correctness", "bench"], default="correctness")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--result-stem", default=None)
    parser.add_argument("--round-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for B in args.batch_sizes:
        if args.mode == "correctness":
            correctness(backend=args.backend, B=B, H=args.heads, atol=args.atol, rtol=args.rtol, result_dir=args.result_dir)
        else:
            benchmark(
                backend=args.backend,
                B=B,
                H=args.heads,
                warmup=args.warmup,
                rep=args.rep,
                result_dir=args.result_dir,
                result_stem=args.result_stem or f"benchmark_decode_{args.backend}",
                round_id=args.round_id,
            )


if __name__ == "__main__":
    main()

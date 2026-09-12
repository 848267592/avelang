#!/usr/bin/env python3
"""Unified ordinary-KDA prefill correctness and latency benchmark.

This file deliberately does not invoke an upstream benchmark.  It adapts the
two locked local source snapshots to one boundary:

  q/k/v + raw_g + raw_beta + A_log + dt_bias + initial_state
      -> ordinary KDA prefill -> output + final_state

The vLLM adapter explicitly disables its ROCm fused chunk path on gfx942.
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
    make_kda_prefill_inputs,
    result_fields,
    result_row,
    save_result,
    sync,
    torch_kda_reference,
)


SGLANG_COMMIT = "908226fea2df861769e2720161a75649ae4c6f92"
VLLM_COMMIT = "40e6042ec83eb8f2971f21043a5da40496bd188a"
Runner = Callable[[dict[str, torch.Tensor | float | int]], tuple[torch.Tensor, torch.Tensor]]


def _prefill_initial_state(inp: dict[str, torch.Tensor | float | int]) -> torch.Tensor:
    state_pool = inp["state_pool"]
    state_index = inp["state_index"]
    assert isinstance(state_pool, torch.Tensor) and isinstance(state_index, torch.Tensor)
    return state_pool.index_select(0, state_index.long()).clone().contiguous()


@torch.inference_mode()
def run_sglang(inp: dict[str, torch.Tensor | float | int]) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang ordinary Triton ``chunk_kda`` with the serving-path arguments."""
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    q, k, v = inp["q"], inp["k"], inp["v"]
    raw_g, raw_beta = inp["raw_g"], inp["raw_beta"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    state_pool, state_index, cu_seqlens = (
        inp["state_pool"],
        inp["state_index"],
        inp["cu_seqlens"],
    )
    assert all(
        isinstance(x, torch.Tensor)
        for x in (q, k, v, raw_g, raw_beta, A_log, dt_bias, state_pool, state_index, cu_seqlens)
    )
    scale = float(inp["scale"])

    # chunk_kda writes the selected recurrent row in place and can reuse v as
    # its output storage, so both are copied for a correctness invocation.
    state_work = state_pool.clone()
    out = chunk_kda(
        q=q,
        k=k,
        v=v.clone(),
        g=raw_g,
        beta=raw_beta,
        scale=scale,
        initial_state=state_work,
        initial_state_indices=state_index,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        use_qk_l2norm_in_kernel=True,
        beta_is_raw=True,
    )
    return out, state_work.index_select(0, state_index.long()).contiguous()


@torch.inference_mode()
def run_vllm(inp: dict[str, torch.Tensor | float | int]) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM AMD KDA prefill, explicitly forced to its vendored Triton fallback."""
    from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill

    q, k, v = inp["q"], inp["k"], inp["v"]
    raw_g, raw_beta = inp["raw_g"], inp["raw_beta"]
    A_log, dt_bias, cu_seqlens = inp["A_log"], inp["dt_bias"], inp["cu_seqlens"]
    assert all(
        isinstance(x, torch.Tensor)
        for x in (q, k, v, raw_g, raw_beta, A_log, dt_bias, cu_seqlens)
    )
    out, final_state = chunk_kda_prefill(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=dt_bias,
        scale=float(inp["scale"]),
        initial_state=_prefill_initial_state(inp),
        output_final_state=True,
        lower_bound=LOWER_BOUND,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        # gfx942 baseline: never select the gfx950 ROCm fused chunk route.
        use_fused_chunk=False,
    )
    if final_state is None:
        raise RuntimeError("vLLM Triton prefill unexpectedly returned no final state")
    return out, final_state.contiguous()


def get_runner(backend: str) -> Runner:
    if backend == "sglang":
        return run_sglang
    if backend == "vllm":
        return run_vllm
    if backend == "avelang":
        raise NotImplementedError("Avelang prefill adapter is intentionally reserved for later")
    raise ValueError(f"unknown backend: {backend}")


def make_bench_invoker(
    backend: str, inp: dict[str, torch.Tensor | float | int]
) -> tuple[Callable[[], None], Callable[[], object]]:
    """Create a timer-safe invocation with state/input resets outside events."""
    q, k, v_template = inp["q"], inp["k"], inp["v"]
    raw_g, raw_beta = inp["raw_g"], inp["raw_beta"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    assert all(isinstance(x, torch.Tensor) for x in (q, k, v_template, raw_g, raw_beta, A_log, dt_bias))
    v_work = v_template.clone()
    scale = float(inp["scale"])

    if backend == "sglang":
        from sglang.kernels.ops.attention.fla.kda import chunk_kda

        state_template, state_index, cu_seqlens = (
            inp["state_pool"],
            inp["state_index"],
            inp["cu_seqlens"],
        )
        assert all(isinstance(x, torch.Tensor) for x in (state_template, state_index, cu_seqlens))
        state_work = state_template.clone()

        def prepare() -> None:
            state_work.copy_(state_template)
            v_work.copy_(v_template)

        def invoke() -> torch.Tensor:
            return chunk_kda(
                q=q,
                k=k,
                v=v_work,
                g=raw_g,
                beta=raw_beta,
                scale=scale,
                initial_state=state_work,
                initial_state_indices=state_index,
                cu_seqlens=cu_seqlens,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=LOWER_BOUND,
                use_qk_l2norm_in_kernel=True,
                beta_is_raw=True,
            )

        return prepare, invoke

    if backend == "vllm":
        from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill

        cu_seqlens = inp["cu_seqlens"]
        assert isinstance(cu_seqlens, torch.Tensor)
        state_template = _prefill_initial_state(inp)
        state_work = state_template.clone()

        def prepare() -> None:
            state_work.copy_(state_template)
            v_work.copy_(v_template)

        def invoke() -> tuple[torch.Tensor, torch.Tensor | None]:
            return chunk_kda_prefill(
                q=q,
                k=k,
                v=v_work,
                raw_g=raw_g,
                raw_beta=raw_beta,
                A_log=A_log,
                g_bias=dt_bias,
                scale=scale,
                initial_state=state_work,
                output_final_state=True,
                lower_bound=LOWER_BOUND,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                use_fused_chunk=False,
            )

        return prepare, invoke

    if backend == "avelang":
        raise NotImplementedError("Avelang prefill adapter is intentionally reserved for later")
    raise ValueError(f"unknown backend: {backend}")


@torch.inference_mode()
def correctness(
    *, backend: str, T: int, H: int, atol: float, rtol: float, result_dir: str
) -> None:
    inp = make_kda_prefill_inputs(T=T, H=H)
    out, state = get_runner(backend)(inp)
    sync()
    expected_out, expected_state = torch_kda_reference(
        q=inp["q"],  # type: ignore[arg-type]
        k=inp["k"],  # type: ignore[arg-type]
        v=inp["v"],  # type: ignore[arg-type]
        raw_g=inp["raw_g"],  # type: ignore[arg-type]
        raw_beta=inp["raw_beta"],  # type: ignore[arg-type]
        A_log=inp["A_log"],  # type: ignore[arg-type]
        dt_bias=inp["dt_bias"],  # type: ignore[arg-type]
        initial_state=_prefill_initial_state(inp),
        l2norm_round_to_bf16=True,
    )
    out_metrics = assert_close(name="prefill output", actual=out, expected=expected_out, atol=atol, rtol=rtol)
    state_metrics = assert_close(name="prefill final state", actual=state, expected=expected_state, atol=atol, rtol=rtol)
    row = result_row(
        phase="prefill",
        mode="correctness",
        backend=backend,
        backend_path=(
            "sglang.fla.chunk_kda ordinary Triton"
            if backend == "sglang"
            else "vllm.amd.chunk_kda_prefill(use_fused_chunk=False) vendored Triton"
        ),
        source_commit=SGLANG_COMMIT if backend == "sglang" else VLLM_COMMIT,
        T=T,
        H=H,
        K=HEAD_DIM,
        V=HEAD_DIM,
        dtype="bf16",
        state_dtype="fp32",
        lower_bound=LOWER_BOUND,
        l2norm="separate Triton prefill norm; BF16-rounded reference",
        output_shape=list(out.shape),
        state_shape=list(state.shape),
        output_error=out_metrics,
        state_error=state_metrics,
        atol=atol,
        rtol=rtol,
        passed=True,
    )
    save_result(row, result_dir=result_dir, stem=f"correctness_prefill_{backend}")


@torch.inference_mode()
def benchmark(
    *,
    backend: str,
    T: int,
    H: int,
    warmup: int,
    rep: int,
    result_dir: str,
    result_stem: str,
    round_id: int,
) -> None:
    inp = make_kda_prefill_inputs(T=T, H=H)
    prepare, invoke = make_bench_invoker(backend, inp)
    result = bench_cuda(invoke, prepare=prepare, warmup=warmup, rep=rep)
    row = result_row(
        phase="prefill",
        mode="bench",
        round_id=round_id,
        backend=backend,
        source_commit=SGLANG_COMMIT if backend == "sglang" else VLLM_COMMIT,
        T=T,
        H=H,
        K=HEAD_DIM,
        V=HEAD_DIM,
        dtype="bf16",
        state_dtype="fp32",
        lower_bound=LOWER_BOUND,
        state_reset_copy_excluded=True,
        warmup=warmup,
        rep=rep,
        **result_fields(result),
    )
    save_result(row, result_dir=result_dir, stem=result_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["sglang", "vllm", "avelang"])
    # Correctness remains the default so a bare invocation cannot launch a long benchmark.
    parser.add_argument("--mode", choices=["correctness", "bench"], default="correctness")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--correctness-seq-lens", type=int, nargs="+", default=[1, 16, 64, 127, 128])
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
    if args.mode == "correctness":
        for T in args.correctness_seq_lens:
            correctness(backend=args.backend, T=T, H=args.heads, atol=args.atol, rtol=args.rtol, result_dir=args.result_dir)
    else:
        for T in args.seq_lens:
            benchmark(
                backend=args.backend,
                T=T,
                H=args.heads,
                warmup=args.warmup,
                rep=args.rep,
                result_dir=args.result_dir,
                result_stem=args.result_stem or f"benchmark_prefill_{args.backend}",
                round_id=args.round_id,
            )


if __name__ == "__main__":
    main()

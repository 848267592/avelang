#!/usr/bin/env python3
"""AITER FlashKDA adapter for the shared Kimi-K3 prefill contract.

This adapter deliberately lives beside (rather than inside) the existing
SGLang/vLLM benchmark.  It reuses ``bench_common`` for input construction,
the FP32 recurrence reference, error metrics, and GPU-event timing.

Only the public AITER ``chunk_kimi_delta_attn`` entry point is used.  A
preflight monkeypatch proves that the call reaches ``flash_kda_fwd`` before
any correctness or timed benchmark is accepted; a silent fallback is an
error.  The monkeypatch is removed before timing starts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from bench_common import (
    DEFAULT_RESULTS_DIR,
    HEAD_DIM,
    LOWER_BOUND,
    assert_close,
    bench_cuda,
    error_metrics,
    make_kda_prefill_inputs,
    result_fields,
    require_rocm_cuda,
    torch_kda_reference,
)


AITER_COMMIT = "7bb44998274fa679ece0a37bf8426ab780b1837d"
AITER_BACKEND_PATH = (
    "aiter.ops.triton.kimi_delta_attn.chunk_kimi_delta_attn; "
    "merged FlashKDA-style two-kernel Triton path"
)
DEFAULT_SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _initial_state(inp: dict[str, torch.Tensor | float | int]) -> torch.Tensor:
    state_pool = inp["state_pool"]
    state_index = inp["state_index"]
    assert isinstance(state_pool, torch.Tensor)
    assert isinstance(state_index, torch.Tensor)
    # This is deliberately outside timed invocation.  AITER receives the
    # benchmark's [N,H,V,K] layout and is told state_v_first=True.
    return state_pool.index_select(0, state_index.long()).clone().contiguous()


def _aiter_call(
    inp: dict[str, torch.Tensor | float | int],
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    q, k, v = inp["q"], inp["k"], inp["v"]
    raw_g, raw_beta = inp["raw_g"], inp["raw_beta"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    cu_seqlens = inp["cu_seqlens"]
    assert all(
        isinstance(x, torch.Tensor)
        for x in (q, k, v, raw_g, raw_beta, A_log, dt_bias, cu_seqlens)
    )
    if initial_state is None:
        initial_state = _initial_state(inp)

    # Do not add copies/transposes/dtype conversions here.  All tensors come
    # from make_kda_prefill_inputs and reset buffers are prepared before the
    # timing event by make_bench_invoker().
    out, final_state = chunk_kimi_delta_attn(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=float(inp["scale"]),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        chunk_size=32,
        cu_seqlens=cu_seqlens,
    )
    if final_state is None:
        raise RuntimeError("AITER FlashKDA returned no final state")
    return out, final_state


def _assert_flash_capability(inp: dict[str, torch.Tensor | float | int]) -> None:
    """Reject an ineligible call before it can enter correctness/timing."""
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_fwd
    from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
        flash_kda_supported,
    )

    q, v, A_log = inp["q"], inp["v"], inp["A_log"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(A_log, torch.Tensor)
    if not chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA:
        raise RuntimeError(
            "AITER FlashKDA dispatch is disabled; set "
            "CHUNK_DELTA_ATTN_USE_FLASH_KDA=1 before importing AITER."
        )
    if not flash_kda_supported(
        q=q,
        v=v,
        chunk_size=32,
        safe_gate=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        A_log=A_log,
    ):
        raise RuntimeError(
            "AITER FlashKDA capability check failed for this shape; refusing "
            "to benchmark the ordinary fallback pipeline."
        )


@torch.inference_mode()
def verify_flash_dispatch(inp: dict[str, torch.Tensor | float | int]) -> None:
    """Prove the public wrapper actually calls ``flash_kda_fwd``.

    The probe is outside any reported timing and uses a private, process-local
    monkeypatch only for call-counting.  The original function is restored in
    ``finally`` before the benchmark starts.
    """
    _assert_flash_capability(inp)
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_fwd

    calls: list[bool] = []
    real_flash_kda_fwd = chunk_fwd.flash_kda_fwd

    def counting_flash_kda_fwd(**kwargs: Any):
        calls.append(True)
        return real_flash_kda_fwd(**kwargs)

    chunk_fwd.flash_kda_fwd = counting_flash_kda_fwd
    try:
        _aiter_call(inp)
        torch.cuda.synchronize()
    finally:
        chunk_fwd.flash_kda_fwd = real_flash_kda_fwd

    if len(calls) != 1:
        raise RuntimeError(
            "AITER public wrapper did not dispatch exactly once to flash_kda_fwd; "
            f"observed calls={len(calls)}. No fallback result is accepted."
        )


@torch.inference_mode()
def run_correctness(
    *,
    T: int,
    H: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    inp = make_kda_prefill_inputs(T=T, H=H)
    verify_flash_dispatch(inp)
    out, state = _aiter_call(inp)
    torch.cuda.synchronize()
    expected_out, expected_state = torch_kda_reference(
        q=inp["q"],  # type: ignore[arg-type]
        k=inp["k"],  # type: ignore[arg-type]
        v=inp["v"],  # type: ignore[arg-type]
        raw_g=inp["raw_g"],  # type: ignore[arg-type]
        raw_beta=inp["raw_beta"],  # type: ignore[arg-type]
        A_log=inp["A_log"],  # type: ignore[arg-type]
        dt_bias=inp["dt_bias"],  # type: ignore[arg-type]
        initial_state=_initial_state(inp),
        lower_bound=LOWER_BOUND,
        l2norm_round_to_bf16=True,
    )
    out_metrics = error_metrics(out, expected_out)
    state_metrics = error_metrics(state, expected_state)
    assert_close(
        name="AITER prefill output",
        actual=out,
        expected=expected_out,
        atol=atol,
        rtol=rtol,
    )
    assert_close(
        name="AITER prefill final state",
        actual=state,
        expected=expected_state,
        atol=atol,
        rtol=rtol,
    )
    return {
        "T": T,
        "H": H,
        "K": HEAD_DIM,
        "V": HEAD_DIM,
        "passed": True,
        "output_shape": list(out.shape),
        "state_shape": list(state.shape),
        "output_error": out_metrics,
        "state_error": state_metrics,
        "state_case": "zero_initial_state",
    }


@torch.inference_mode()
def run_nonzero_state_correctness(*, H: int, T: int, atol: float, rtol: float) -> dict[str, Any]:
    inp = make_kda_prefill_inputs(T=T, H=H, seed=314159)
    state_pool = inp["state_pool"]
    assert isinstance(state_pool, torch.Tensor)
    torch.manual_seed(271828)
    state_pool[1].normal_(mean=0.0, std=0.1)
    verify_flash_dispatch(inp)
    out, state = _aiter_call(inp)
    torch.cuda.synchronize()
    expected_out, expected_state = torch_kda_reference(
        q=inp["q"],  # type: ignore[arg-type]
        k=inp["k"],  # type: ignore[arg-type]
        v=inp["v"],  # type: ignore[arg-type]
        raw_g=inp["raw_g"],  # type: ignore[arg-type]
        raw_beta=inp["raw_beta"],  # type: ignore[arg-type]
        A_log=inp["A_log"],  # type: ignore[arg-type]
        dt_bias=inp["dt_bias"],  # type: ignore[arg-type]
        initial_state=_initial_state(inp),
        lower_bound=LOWER_BOUND,
        l2norm_round_to_bf16=True,
    )
    out_metrics = error_metrics(out, expected_out)
    state_metrics = error_metrics(state, expected_state)
    assert_close(
        name="AITER nonzero-state prefill output",
        actual=out,
        expected=expected_out,
        atol=atol,
        rtol=rtol,
    )
    assert_close(
        name="AITER nonzero-state prefill final state",
        actual=state,
        expected=expected_state,
        atol=atol,
        rtol=rtol,
    )
    return {
        "T": T,
        "H": H,
        "K": HEAD_DIM,
        "V": HEAD_DIM,
        "passed": True,
        "output_shape": list(out.shape),
        "state_shape": list(state.shape),
        "output_error": out_metrics,
        "state_error": state_metrics,
        "state_case": "nonzero_initial_state",
    }


def make_bench_invoker(
    inp: dict[str, torch.Tensor | float | int],
) -> tuple[Any, Any]:
    """Return prepare/invoke closures with all reset/copy work pre-event."""
    verify_flash_dispatch(inp)
    q, k, v_template = inp["q"], inp["k"], inp["v"]
    raw_g, raw_beta = inp["raw_g"], inp["raw_beta"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    cu_seqlens = inp["cu_seqlens"]
    assert all(
        isinstance(x, torch.Tensor)
        for x in (q, k, v_template, raw_g, raw_beta, A_log, dt_bias, cu_seqlens)
    )
    state_template = _initial_state(inp)
    v_work = v_template.clone()
    state_work = state_template.clone()

    def prepare() -> None:
        state_work.copy_(state_template)
        v_work.copy_(v_template)

    def invoke() -> tuple[torch.Tensor, torch.Tensor | None]:
        # No tensor construction, reset, transpose, or dtype conversion here.
        # AITER receives contiguous buffers in the contract's [N,H,V,K] layout.
        from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

        return chunk_kimi_delta_attn(
            q=q,
            k=k,
            v=v_work,
            g=raw_g,
            beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=float(inp["scale"]),
            initial_state=state_work,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            state_v_first=True,
            chunk_size=32,
            cu_seqlens=cu_seqlens,
        )

    return prepare, invoke


@torch.inference_mode()
def run_benchmark_round(*, T: int, H: int, warmup: int, rep: int) -> dict[str, Any]:
    inp = make_kda_prefill_inputs(T=T, H=H)
    prepare, invoke = make_bench_invoker(inp)
    result = bench_cuda(invoke, prepare=prepare, warmup=warmup, rep=rep)
    return {
        "T": T,
        "H": H,
        "K": HEAD_DIM,
        "V": HEAD_DIM,
        "round_id": None,
        "backend": "aiter",
        "backend_path": AITER_BACKEND_PATH,
        "source_commit": AITER_COMMIT,
        "dtype": "bf16",
        "state_dtype": "fp32",
        "lower_bound": LOWER_BOUND,
        "chunk_size": 32,
        "state_v_first": True,
        "state_reset_copy_excluded": True,
        "warmup": warmup,
        "rep": rep,
        **result_fields(result),
    }


def _jsonable_environment() -> dict[str, Any]:
    return {
        "hardware": "AMD Instinct MI300X",
        "arch": "gfx942",
        "aiter_commit": AITER_COMMIT,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": __import__("triton").__version__,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "contract": {
            "B": 1,
            "H": 12,
            "K": 128,
            "V": 128,
            "dtype": "bf16",
            "state_dtype": "fp32",
            "lower_bound": -5.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["correctness", "bench"], required=True)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--correctness-seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--result-path", default=str(DEFAULT_RESULTS_DIR / "aiter_kimi_prefill.json"))
    parser.add_argument("--nonzero-state-T", type=int, default=128)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    require_rocm_cuda()
    if args.heads != 12:
        raise ValueError("This AITER Kimi contract requires H=12")

    if args.mode == "correctness":
        rows: list[dict[str, Any]] = []
        for T in args.correctness_seq_lens:
            row = run_correctness(T=T, H=args.heads, atol=args.atol, rtol=args.rtol)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
        nonzero = run_nonzero_state_correctness(
            H=args.heads,
            T=args.nonzero_state_T,
            atol=args.atol,
            rtol=args.rtol,
        )
        rows.append(nonzero)
        print(json.dumps(nonzero, sort_keys=True))
        print("AITER FlashKDA correctness: PASS")
        return

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    rows: list[dict[str, Any]] = []
    for T in args.seq_lens:
        for round_id in range(1, args.rounds + 1):
            row = run_benchmark_round(T=T, H=args.heads, warmup=args.warmup, rep=args.rep)
            row["round_id"] = round_id
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    payload = {
        "stage": "Stage 5B AITER Kimi Prefill",
        "environment": _jsonable_environment(),
        "implementation": AITER_BACKEND_PATH,
        "flash_path_verified": True,
        "correctness_required_before_bench": True,
        "timing_contract": {"warmup": args.warmup, "rep": args.rep, "rounds": args.rounds},
        "rows": rows,
    }
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {result_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fair BT64 recurrent-step gate for the external gfx942 assembly experiment.

This is intentionally a one-chunk workload.  It uses the exact BT64 reference
used by the v31 experiment and invokes the existing v31 P16 implementation
unchanged.  The vLLM call is best-effort: an unavailable snapshot is reported
as unavailable rather than substituted with a different implementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
VLLM_COMPARE = HERE.parents[2] / "vllm_compare"
PROJECT_ROOT = HERE.parents[5]
VLLM_ROOT = PROJECT_ROOT / "vllm_stageb_snapshot"
sys.path.insert(0, str(VLLM_COMPARE))
if VLLM_ROOT.is_dir():
    sys.path.insert(0, str(VLLM_ROOT))

from qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred,
    qwen_gdn_fused_chunk_gdr_full_reference,
    qwen_gdn_gdr_decay_bt64_reference,
)


BT = 64
HKEY = 4
HVALUE = 8
KDIM = 128
VDIM = 128


def _quantiles_ms(fn: Callable[[], object], warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, round((len(samples) - 1) * 0.10))],
        "p90_ms": samples[min(len(samples) - 1, round((len(samples) - 1) * 0.90))],
        "repeat": float(repeat),
    }


def make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    k = (torch.randn((1, BT, HKEY, KDIM), device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w = (torch.randn((1, BT, HVALUE, KDIM), device="cuda", dtype=torch.float32) * 0.04).contiguous()
    u = (torch.randn((1, BT, HVALUE, VDIM), device="cuda", dtype=torch.float32) * 0.04).contiguous()
    g = (torch.randn((1, BT, HVALUE), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    state = (torch.randn((1, HVALUE, VDIM, KDIM), device="cuda", dtype=torch.float32) * 0.03).contiguous()
    decay, scale = qwen_gdn_gdr_decay_bt64_reference(g)
    return k, w, u, g, decay, scale, state


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | list[int] | None]:
    delta = (actual.float() - expected.float()).abs()
    index = (delta == delta.max()).nonzero()
    denom = expected.float().abs().clamp_min(1.0e-7)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_rel": float((delta / denom).max()),
        "first_max_index": index[0].tolist() if index.numel() else None,
    }


def reference_intermediates(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, decay: torch.Tensor,
                            state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pred[T,Hv,V], BF16-rounded v_decay[T,Hv,V], update[Hv,V,K]."""
    pred = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    update = torch.empty_like(state[0])
    for value_head in range(HVALUE):
        key_head = value_head // 2
        p = w[0, :, value_head].to(torch.bfloat16).float() @ state[0, value_head].to(torch.bfloat16).float().t()
        pred[0, :, value_head] = p
        vd = ((u[0, :, value_head] - p) * decay[0, 0, value_head].unsqueeze(-1)).to(torch.bfloat16).float()
        v_decay[0, :, value_head] = vd
        update[value_head] = vd.t() @ k[0, :, key_head].float()
    return pred, v_decay, update


def try_vllm(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, state: torch.Tensor,
             reference_state: torch.Tensor, reference_vnew: torch.Tensor, warmup: int, repeat: int) -> dict[str, object]:
    try:
        from vllm.model_executor.layers.fla.ops import chunk_delta_h as vllm_chunk_delta_h  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific discovery
        return {"available": False, "reason": repr(exc)}
    try:
        def run():
            return vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g, gk=None, initial_state=state,
                output_final_state=True, chunk_size=BT, save_new_value=True,
                cu_seqlens=None,
            )

        h, v_new, final_state = run()
        torch.cuda.synchronize()
        row: dict[str, object] = {
            "available": True,
            "launch": {"chunk_size": BT, "head_first": False, "initial_state": True},
            "h_shape": list(h.shape),
            "v_new_shape": list(v_new.shape) if v_new is not None else None,
            "final_state_shape": list(final_state.shape) if final_state is not None else None,
            **_quantiles_ms(run, warmup, repeat),
        }
        if v_new is not None:
            row["v_new_error_vs_reference"] = error_stats(v_new, reference_vnew)
        if final_state is not None:
            row["state_error_vs_reference"] = error_stats(final_state, reference_state)
        return row
    except Exception as exc:  # pragma: no cover - environment-specific runtime
        return {"available": False, "reason": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    k, w, u, g, decay, scale, state = make_inputs(args.seed)
    ref_h, ref_state = qwen_gdn_fused_chunk_gdr_full_reference(k, w, u, decay, scale, state)
    ref_pred, ref_v_decay, ref_update = reference_intermediates(k, w, u, decay, state)
    p16_h, p16_state = qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred(
        k, w, u, decay, scale, state
    )
    torch.cuda.synchronize()

    result: dict[str, object] = {
        "target": {"arch": torch.cuda.get_device_properties(0).gcnArchName, "BT": BT, "BV": 32, "K": KDIM},
        "seed": args.seed,
        "input_manifest": {
            "k": {"shape": list(k.shape), "dtype": str(k.dtype)},
            "w": {"shape": list(w.shape), "dtype": str(w.dtype), "mfma_input_cast": "bf16"},
            "u": {"shape": list(u.shape), "dtype": str(u.dtype)},
            "g": {"shape": list(g.shape), "dtype": str(g.dtype)},
            "state": {"shape": list(state.shape), "dtype": str(state.dtype)},
            "decay": {"shape": list(decay.shape), "dtype": str(decay.dtype)},
            "scale": {"shape": list(scale.shape), "dtype": str(scale.dtype)},
        },
        "reference": {"h_shape": list(ref_h.shape), "state_shape": list(ref_state.shape)},
        "reference_intermediates": {
            "pred_shape": list(ref_pred.shape),
            "v_decay_shape": list(ref_v_decay.shape),
            "update_shape": list(ref_update.shape),
            "v_decay_cast": "fp32 correction times fp32 decay, then bf16 then fp32",
        },
        "p16_v31": {
            "h_error": error_stats(p16_h, ref_h),
            "state_error": error_stats(p16_state, ref_state),
            "launch": {"grid": 32, "block": 128, "waves": 2},
            **_quantiles_ms(lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred(
                k, w, u, decay, scale, state
            ), args.warmup, args.repeat),
        },
        "triton_vllm": try_vllm(k, w, u, g, state, ref_state, u - ref_pred, args.warmup, args.repeat),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Direct vLLM Triton chunk_delta_h profiling workload.

This calls the real vLLM `chunk_gated_delta_rule_fwd_h` wrapper for the fixed
Qwen GDN shape, without running the full GDN pipeline.  The script is designed
to be wrapped by rocprofv3 with a regex that matches the generated Triton
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` kernel.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[5]
VLLM_ROOT = str(PROJECT_ROOT / "vllm_stageb_snapshot")
AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, VLLM_ROOT)
sys.path.insert(0, AVELANG_DIR)

from vllm.model_executor.layers.fla.ops import chunk_delta_h as vllm_chunk_delta_h  # noqa: E402


def patch_vllm_rocm_autotune_configs(
    *,
    force_bv: int | None = None,
    force_num_warps: int | None = None,
    force_num_stages: int | None = None,
) -> None:
    if getattr(torch.version, "hip", None) is None:
        return

    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    configs = getattr(autotuner, "configs", None)
    if not configs:
        return

    filtered = [config for config in configs if getattr(config, "num_stages", None) != 4]
    if force_bv is not None:
        filtered = [config for config in filtered if getattr(config, "kwargs", {}).get("BV") == force_bv]
    if force_num_warps is not None:
        filtered = [config for config in filtered if getattr(config, "num_warps", None) == force_num_warps]
    if force_num_stages is not None:
        filtered = [config for config in filtered if getattr(config, "num_stages", None) == force_num_stages]
    if not filtered:
        raise ValueError("forced Triton config filter removed all autotune configs")
    if len(filtered) != len(configs):
        autotuner.configs = filtered
        cache = getattr(autotuner, "cache", None)
        if hasattr(cache, "clear"):
            cache.clear()
        print(
            "patched_vllm_rocm_autotune_configs="
            f"{len(configs)}->{len(filtered)},disabled_num_stages=4,"
            f"force_bv={force_bv},force_num_warps={force_num_warps},force_num_stages={force_num_stages}"
        )


def describe_autotune_configs() -> None:
    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    configs = getattr(autotuner, "configs", None) or []
    print(f"triton_autotune_config_count={len(configs)}")
    for idx, config in enumerate(configs):
        kwargs = getattr(config, "kwargs", {})
        print(
            "triton_autotune_config,"
            f"idx={idx},BV={kwargs.get('BV')},"
            f"num_warps={getattr(config, 'num_warps', None)},"
            f"num_stages={getattr(config, 'num_stages', None)}"
        )


def describe_autotune_cache() -> None:
    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    cache = getattr(autotuner, "cache", None)
    print(f"triton_autotune_cache_type={type(cache).__name__}")
    try:
        items = list(cache.items()) if hasattr(cache, "items") else []
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"triton_autotune_cache_error={exc!r}")
        return
    print(f"triton_autotune_cache_entries={len(items)}")
    for key, value in items[:8]:
        kwargs = getattr(value, "kwargs", {}) if value is not None else {}
        print(
            "triton_autotune_cache_entry,"
            f"key={key!r},BV={kwargs.get('BV')},"
            f"num_warps={getattr(value, 'num_warps', None)},"
            f"num_stages={getattr(value, 'num_stages', None)},"
            f"value={value!r}"
        )


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, with_initial_state: bool, seed: int):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    w = (torch.randn(b, t, hv, kdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    u = (torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, w, u, g, initial_state


def time_fn(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=280000)
    parser.add_argument("--with-initial-state", dest="with_initial_state", action="store_true")
    parser.add_argument("--without-initial-state", dest="with_initial_state", action="store_false")
    parser.add_argument("--save-new-value", dest="save_new_value", action="store_true")
    parser.add_argument("--no-save-new-value", dest="save_new_value", action="store_false")
    parser.add_argument("--output-final-state", dest="output_final_state", action="store_true")
    parser.add_argument("--no-output-final-state", dest="output_final_state", action="store_false")
    parser.add_argument("--force-bv", type=int, default=None)
    parser.add_argument("--force-num-warps", type=int, default=None)
    parser.add_argument("--force-num-stages", type=int, default=None)
    parser.set_defaults(with_initial_state=False, save_new_value=True, output_final_state=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    patch_vllm_rocm_autotune_configs(
        force_bv=args.force_bv,
        force_num_warps=args.force_num_warps,
        force_num_stages=args.force_num_stages,
    )
    describe_autotune_configs()

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        "target=vllm_direct_chunk_delta_h,"
        f"B=1,T={args.T},Hk=4,Hv=8,K=128,V=128,"
        f"chunk_size=64,head_first=False,initial_state={args.with_initial_state},"
        f"save_new_value={args.save_new_value},output_final_state={args.output_final_state}"
    )

    k, w, u, g, initial_state = make_inputs(args.T, args.with_initial_state, args.seed + args.T)

    def fn():
        return vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=None,
            initial_state=initial_state,
            output_final_state=args.output_final_state,
            chunk_size=64,
            save_new_value=args.save_new_value,
            cu_seqlens=None,
        )

    # Run once before timing/profiling to trigger Triton JIT/autotune.
    h, v_new, final_state = fn()
    torch.cuda.synchronize()
    describe_autotune_cache()

    checksum = h.float().abs().mean()
    if v_new is not None:
        checksum = checksum + v_new.float().abs().mean()
    if final_state is not None:
        checksum = checksum + final_state.float().abs().mean()

    latency_ms = time_fn(fn, warmup=args.warmup, repeat=args.repeat)
    print(f"result,T={args.T},vllm_chunk_delta_h_ms={latency_ms:.6f},checksum={float(checksum):.9g}")


if __name__ == "__main__":
    main()

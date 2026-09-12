#!/usr/bin/env python3
"""Correctness and timing driver for the extracted selected Triton HSACO.

The target kernel is deliberately launched only through the C++ HIP module
harness. Triton is used separately to generate the comparison outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
PROJECT_ROOT = AUDIT.parents[5]
VLLM_ROOT = PROJECT_ROOT / "vllm_stageb_snapshot"
VLLM_COMPARE = PROJECT_ROOT / "test/examples/linear_attention/vllm_compare"
sys.path.insert(0, str(VLLM_COMPARE))
if VLLM_ROOT.is_dir():
    sys.path.insert(0, str(VLLM_ROOT))

from qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_reference,
    qwen_gdn_gdr_decay_bt64_reference,
)
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402


BT, HK, HV, KDIM, VDIM = 64, 4, 8, 128, 128


def tensor_to_file(tensor: torch.Tensor, path: Path) -> None:
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).cpu().numpy().tofile(path)
    else:
        tensor.cpu().numpy().tofile(path)


def file_to_tensor(path: Path, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    np_dtype = np.uint16 if dtype == torch.bfloat16 else np.float32
    array = np.fromfile(path, dtype=np_dtype).reshape(shape).copy()
    tensor = torch.from_numpy(array)
    return tensor.view(torch.bfloat16) if dtype == torch.bfloat16 else tensor


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    maximum = delta.max()
    index = (delta == maximum).nonzero()
    return {
        "max_abs": float(maximum),
        "mean_abs": float(delta.mean()),
        "max_rel": float((delta / expected.float().abs().clamp_min(1e-7)).max()),
        "first_max_index": index[0].tolist() if index.numel() else None,
    }


def make_case(seed: int, mode: str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    k = (torch.randn((1, BT, HK, KDIM), device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w = (torch.randn((1, BT, HV, KDIM), device="cuda", dtype=torch.float32) * 0.04).contiguous()
    v = (torch.randn((1, BT, HV, VDIM), device="cuda", dtype=torch.float32) * 0.04).contiguous()
    g = (torch.randn((1, BT, HV), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    h0 = (torch.randn((1, HV, VDIM, KDIM), device="cuda", dtype=torch.float32) * 0.03).contiguous()
    if mode == "w_zero":
        w.zero_()
    elif mode == "zero_state":
        h0.zero_()
    elif mode == "scale_one":
        g.zero_()
    elif mode == "high_dynamic":
        w.mul_(5.0)
        v.mul_(8.0)
        h0.mul_(6.0)
        g.copy_(torch.linspace(-8.0, 0.0, BT, device="cuda", dtype=torch.float32).view(1, BT, 1).expand_as(g))
    elif mode == "cancellation":
        w.mul_(0.2)
        h0.mul_(0.2)
        pred = torch.empty_like(v)
        for head in range(HV):
            pred[0, :, head] = w[0, :, head].to(torch.bfloat16).float() @ h0[0, head].to(torch.bfloat16).float().t()
        v.copy_(pred + torch.randn_like(v) * 1e-5)
    return k, w, v, g, h0


def wrapper(k: torch.Tensor, w: torch.Tensor, v: torch.Tensor, g: torch.Tensor, h0: torch.Tensor):
    return chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=v, g=g, gk=None, initial_state=h0,
        output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )


def benchmark(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    times.sort()
    return {"median_ms": statistics.median(times), "p10_ms": times[(len(times)-1)//10], "p90_ms": times[(len(times)-1)*9//10]}


def direct_triton_preallocated(k: torch.Tensor, w: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                               h0: torch.Tensor):
    """Return a no-allocation call through the same selected Triton JIT kernel."""
    h = torch.empty((1, 1, HV, VDIM, KDIM), device="cuda", dtype=torch.bfloat16)
    v_new = torch.empty_like(v)
    ht = torch.empty((1, HV, VDIM, KDIM), device="cuda", dtype=torch.float32)
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = kernel.fn
    autotuner.configs = [config for config in autotuner.configs if config.num_stages != 4]

    def run():
        return kernel[(4, 8)](
            k=k, v=v, w=w, v_new=v_new, g=g, gk=None, h=h, h0=h0, ht=ht,
            cu_seqlens=None, chunk_offsets=None, T=BT, H=HV, Hg=HK,
            K=KDIM, V=VDIM, BT=BT,
        )

    run()
    torch.cuda.synchronize()
    return run, h, v_new, ht


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsaco", type=Path, default=AUDIT / "extracted/original_triton_kernel.hsaco")
    parser.add_argument("--harness", type=Path, default=HERE / "harness")
    parser.add_argument("--out", type=Path, default=AUDIT)
    parser.add_argument("--cases", type=int, default=55)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    case_root = args.out / "standalone_harness/cases"
    case_root.mkdir(parents=True, exist_ok=True)
    modes = ["w_zero", "zero_state", "scale_one", "high_dynamic", "cancellation"]
    names = [f"random_{index:02d}" for index in range(max(0, args.cases - len(modes)))] + modes
    rows: list[dict[str, object]] = []
    benchmark_row: dict[str, object] | None = None

    for index, name in enumerate(names):
        mode = name if name in modes else "random"
        k, w, v, g, h0 = make_case(20260712 + index, mode)
        h_ref, state_ref = qwen_gdn_fused_chunk_gdr_full_reference(
            k, w, v, *qwen_gdn_gdr_decay_bt64_reference(g), h0
        )
        h_wrapper, vnew_wrapper, state_wrapper = wrapper(k, w, v, g, h0)
        direct_run, h_direct, vnew_direct, state_direct = direct_triton_preallocated(k, w, v, g, h0)
        torch.cuda.synchronize()
        input_dir, output_dir = case_root / name / "in", case_root / name / "out"
        input_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_file(k, input_dir / "k_bf16.bin"); tensor_to_file(v, input_dir / "v_fp32.bin")
        tensor_to_file(w, input_dir / "w_fp32.bin"); tensor_to_file(g, input_dir / "g_fp32.bin")
        tensor_to_file(h0, input_dir / "h0_fp32.bin")
        result = subprocess.run([str(args.harness), str(args.hsaco), str(input_dir), str(output_dir), "0", "0"],
                                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        launch = json.loads(result.stdout)
        h_standalone = file_to_tensor(output_dir / "h_bf16.bin", torch.bfloat16, tuple(h_wrapper.shape))
        vnew_standalone = file_to_tensor(output_dir / "v_new_fp32.bin", torch.float32, tuple(vnew_wrapper.shape))
        state_standalone = file_to_tensor(output_dir / "ht_fp32.bin", torch.float32, tuple(state_wrapper.shape))
        rows.append({
            "case": name, "mode": mode, "standalone_launch": launch,
            "h_standalone_vs_wrapper": stats(h_standalone, h_wrapper.cpu()),
            "vnew_standalone_vs_wrapper": stats(vnew_standalone, vnew_wrapper.cpu()),
            "state_standalone_vs_wrapper": stats(state_standalone, state_wrapper.cpu()),
            "h_wrapper_vs_reference": stats(h_wrapper, h_ref.to(torch.bfloat16)),
            "state_wrapper_vs_reference": stats(state_wrapper, state_ref),
            "h_direct_vs_wrapper": stats(h_direct, h_wrapper),
            "vnew_direct_vs_wrapper": stats(vnew_direct, vnew_wrapper),
            "state_direct_vs_wrapper": stats(state_direct, state_wrapper),
        })
        if index == 0:
            if args.repeat > 0:
                standalone_timing = subprocess.run([str(args.harness), str(args.hsaco), str(input_dir), str(output_dir),
                                                    str(args.warmup), str(args.repeat)], check=True, text=True,
                                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                benchmark_row = {
                    "wrapper": benchmark(lambda: wrapper(k, w, v, g, h0), args.warmup, args.repeat),
                    "triton_direct_preallocated": benchmark(direct_run, args.warmup, args.repeat),
                    "standalone": json.loads(standalone_timing.stdout),
                    "case": name,
                }

    correctness = {"seed": 20260712, "case_count": len(rows), "cases": rows}
    (args.out / "standalone_harness/correctness_results.json").write_text(json.dumps(correctness, indent=2) + "\n")
    with (args.out / "standalone_harness/correctness_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["case", "mode", "h_max_abs", "vnew_max_abs", "state_max_abs"])
        for row in rows:
            writer.writerow([row["case"], row["mode"], row["h_standalone_vs_wrapper"]["max_abs"],
                             row["vnew_standalone_vs_wrapper"]["max_abs"], row["state_standalone_vs_wrapper"]["max_abs"]])
    (args.out / "standalone_benchmark.json").write_text(json.dumps(benchmark_row, indent=2) + "\n")
    print(json.dumps({"case_count": len(rows), "benchmark": benchmark_row}, indent=2))


if __name__ == "__main__":
    main()

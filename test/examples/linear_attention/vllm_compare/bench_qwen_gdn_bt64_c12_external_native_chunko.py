#!/usr/bin/env python3
"""C12 external-native upper-bound and Z5B chunk-o body comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARTIFACT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2), str(ARTIFACT)]

from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from qwen_gdn_bt64_c12_external_native_chunko import launch_into as external_launch_into  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT, HK, HV, K, V = 64, 4, 8, 128, 128
ARMS = ("z5b", "external_native", "native")


def _inputs(t: int, seed: int):
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return tuple(x.contiguous() for x in (q, k, v_new, h, g))


def _native(tensors, output):
    q, k, v_new, h, g = tensors
    t = int(q.shape[1])
    return chunk_o.chunk_fwd_kernel_o[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)](
        q, k, v_new, h, g, output, None, None, K ** -0.5, T=t, H=HV, Hg=HK, K=K, V=V, BT=BT
    )


def _launch(arm, tensors, outputs):
    if arm == "z5b":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(
            *tensors, outputs[arm]
        )
    if arm == "external_native":
        return lambda: external_launch_into(*tensors, outputs[arm])
    return lambda: _native(tensors, outputs[arm])


def _q(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    hips = []
    walls = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        end.synchronize()
        hips.append(float(start.elapsed_time(end)))
        walls.append((time.perf_counter_ns() - wall_start) / 1e6)
    return {
        "samples": repeat,
        "hip_ms": {"median": statistics.median(hips), "p10": _q(hips, .1), "p90": _q(hips, .9)},
        "wall_ms": {"median": statistics.median(walls), "p10": _q(walls, .1), "p90": _q(walls, .9)},
    }


def _worker(args):
    tensors = _inputs(args.T, 2026081100 + args.T)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in ARMS}
    # Warm the public native selector outside the measured body.  It also
    # validates that the chosen public kernel has the same input contract.
    native_result = chunk_o.chunk_fwd_o(
        q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4], scale=K ** -0.5, chunk_size=BT
    )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(native_result).all().item()):
        raise RuntimeError("native public chunk-o produced non-finite output")
    del native_result
    launches = {arm: _launch(arm, tensors, outputs) for arm in ARMS}
    order = args.order.split(",")
    for arm in order:
        launches[arm]()
    torch.cuda.synchronize()
    reference = outputs["native"].clone()
    correctness = {}
    for arm in ARMS:
        if not bool(torch.isfinite(outputs[arm]).all().item()):
            raise RuntimeError(f"non-finite output for {arm}")
        correctness[arm] = {
            "max_abs_vs_native": float((outputs[arm].float() - reference.float()).abs().max().item()),
            "bf16_equal_vs_native": bool(torch.equal(outputs[arm], reference)),
        }
    timings = {arm: _measure(launches[arm], args.warmup, args.repeat) for arm in order}
    print(json.dumps({
        "scope": "c12_external_native_chunk_o_body",
        "fresh_process_worker": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "T": args.T,
        "chunks": args.T // BT,
        "order": order,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "correctness": correctness,
        "arms": timings,
    }, sort_keys=True))


def _parent(args):
    orders = [
        "z5b,external_native,native",
        "native,external_native,z5b",
        "external_native,z5b,native",
        "native,z5b,external_native",
        "z5b,native,external_native",
        "external_native,native,z5b",
        "native,z5b,external_native",
    ]
    rows = []
    for session in range(args.sessions):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(args.T),
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--order", orders[session % len(orders)]]
        completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"session {session} failed:\n{completed.stdout}")
        row = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
        row["session"] = session
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    medians = {arm: [float(row["arms"][arm]["hip_ms"]["median"]) for row in rows] for arm in ARMS}
    summary = {arm: {"session_medians_ms": medians[arm], "median_ms": statistics.median(medians[arm]),
                     "mean_ms": statistics.fmean(medians[arm])} for arm in ARMS}
    payload = {"T": args.T, "sessions": rows, "summary": summary,
               "paired_external_minus_native_us": [(a - n) * 1000 for a, n in zip(medians["external_native"], medians["native"])],
               "paired_z5b_minus_external_us": [(z - a) * 1000 for z, a in zip(medians["z5b"], medians["external_native"])],
               "contract": {"warmup": args.warmup, "repeat": args.repeat, "fresh_process": True,
                            "current_stream": True, "cuda_graph_used": False, "caller_owned_output": True}}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--order", default="z5b,external_native,native")
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

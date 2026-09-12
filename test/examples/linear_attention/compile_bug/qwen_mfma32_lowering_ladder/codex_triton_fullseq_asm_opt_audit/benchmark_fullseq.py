#!/usr/bin/env python3
"""Three-session, no-allocation long-sequence device-event benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
VLLM_COMPARE = HERE.parents[4] / "vllm_compare"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(VLLM_COMPARE))
from capture_long_sequence import BT, HV, VDIM, KDIM, call, make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402

spec = importlib.util.spec_from_file_location("external_full", HERE / "avelang_external_full/qwen_gdn_bt64_gfx942_external_full.py")
assert spec and spec.loader
external = importlib.util.module_from_spec(spec); spec.loader.exec_module(external)


def event_stats(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(repeat):
        start.record(); fn(); end.record(); torch.cuda.synchronize(); values.append(float(start.elapsed_time(end)))
    values.sort()
    return {"median_ms": statistics.median(values), "p10_ms": values[(len(values) - 1) // 10], "p90_ms": values[(len(values) - 1) * 9 // 10]}


def direct_triton(k, w, u, g, h0):
    t = k.shape[1]; h = torch.empty((1, t // BT, HV, VDIM, KDIM), device="cuda", dtype=torch.bfloat16); v_new = torch.empty_like(u); ht = torch.empty((1, HV, VDIM, KDIM), device="cuda")
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    def run():
        kernel[(4, 8)](k=k, v=u, w=w, v_new=v_new, g=g, gk=None, h=h, h0=h0, ht=ht, cu_seqlens=None, chunk_offsets=None, T=t, H=8, Hg=4, K=128, V=128, BT=64)
    run(); torch.cuda.synchronize()
    return run


def external_run(k, w, u, g, h0, hsaco: Path):
    t = k.shape[1]; h = torch.empty((1, t // BT, HV, VDIM, KDIM), device="cuda", dtype=torch.bfloat16); v_new = torch.empty_like(u); ht = torch.empty((1, HV, VDIM, KDIM), device="cuda")
    digest = __import__("hashlib").sha256(hsaco.read_bytes()).hexdigest()
    def run():
        os.environ["QWEN_TRITON_FULLSEQ_HSACO_PATH"] = str(hsaco)
        os.environ["QWEN_TRITON_FULLSEQ_HSACO_SHA256"] = digest
        external.qwen_gdn_bt64_gfx942_external_full_preallocated(k, w, u, g, h0, h, v_new, ht)
    run(); torch.cuda.synchronize()
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    patch_rocm_autotune()
    original = HERE / "golden_fullseq/shared/original.hsaco"; rebuilt = HERE / "golden_fullseq/shared/rebuilt.hsaco"
    os.environ["QWEN_TRITON_FULLSEQ_EXTERNAL_BRIDGE"] = str(HERE / "avelang_external_full/libqwen_triton_external_full_bridge.so")
    rows = []
    for t in args.T:
        values = make_inputs(t, 20260717 + t)
        variants = [("vllm_direct_preallocated", direct_triton(*values), None)]
        for label, hsaco in (("extracted_external", original), ("rebuilt_external", rebuilt)):
            variants.append((label, external_run(*values, hsaco), str(hsaco)))
        for label, fn, path in variants:
            sessions = [event_stats(fn, args.warmup, args.repeat) for _ in range(args.sessions)]
            rows.append({"T": t, "variant": label, "hsaco": path, "sessions": sessions, "median_of_session_medians_ms": statistics.median(item["median_ms"] for item in sessions)})
    with (args.out / "fullseq_benchmark.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["T", "variant", "session", "median_ms", "p10_ms", "p90_ms", "median_of_session_medians_ms"])
        for row in rows:
            for index, item in enumerate(row["sessions"]): writer.writerow([row["T"], row["variant"], index, item["median_ms"], item["p10_ms"], item["p90_ms"], row["median_of_session_medians_ms"]])
    (args.out / "fullseq_benchmark.json").write_text(json.dumps({"warmup": args.warmup, "repeat": args.repeat, "sessions": args.sessions, "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()

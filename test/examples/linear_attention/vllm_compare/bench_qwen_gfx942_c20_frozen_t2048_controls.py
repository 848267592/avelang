#!/usr/bin/env python3
"""C20 T=2048 control using the already captured immutable code objects.

The current C18/C19 source contracts cannot be re-JITed by the active binding
for the Phase-C V logical operand.  This driver therefore replays the exact
T=2048 HSACOs recorded by C18/P2/C19 identity artifacts.  It is intentionally
separate from the source sweep and labels every arm as frozen-code replay.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128
T = 2048
GRID_X = (T // BT) * HV * 2
BLOCK_X = 256
ARMS = ("z5b", "p2_frozen_hsaco", "c18_frozen_hsaco", "c19_frozen_hsaco", "native_selected")

FROZEN = {
    "p2_frozen_hsaco": {
        "path": LADDER / "codex_qwen_bt64_stage6z_bdv2_p2_machine_specialized/bdv2_specialized.hsaco",
        "symbol": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope",
        "sha256": "f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947",
    },
    "c18_frozen_hsaco": {
        "path": LADDER / "codex_qwen_gfx942_c18_full_physical_region_t2048/exact_lto/linked.hsaco",
        "symbol": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c18_full_physical_region",
        "sha256": "a4af6a4c7b972400ec5f6ce7a9a732e54b7916edf10ce9d157f078c0c32c94fe",
    },
    "c19_frozen_hsaco": {
        "path": LADDER / "codex_qwen_gfx942_c19_full_physical_region_t2048/machine/c19_full_physical_region.hsaco",
        "symbol": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c19_full_physical_region",
        "sha256": "14ffa78200ff995448c8abbe1c5484375aae1a2f75c8412a937b33d4aa628907",
    },
}


def _inputs(seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(T, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, T, HV, V), device=q.device) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, T // BT, HV, V, K), device=q.device) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _config_summary() -> dict[str, object]:
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    best = getattr(tuner, "best_config", None)
    return {
        "kwargs": dict(getattr(best, "kwargs", {})) if best is not None else {},
        "num_warps": int(getattr(best, "num_warps", -1)) if best is not None else -1,
        "num_stages": int(getattr(best, "num_stages", -1)) if best is not None else -1,
        "num_ctas": int(getattr(best, "num_ctas", -1)) if best is not None else -1,
    }


def _native_direct(tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    q, k, v_new, h, g = tensors
    chunk_o.chunk_fwd_kernel_o[
        lambda meta: ((V + meta["BV"] - 1) // meta["BV"], T // BT, HV)
    ](q, k, v_new, h, g, output, None, None, K**-0.5, T=T, H=HV, Hg=HK, K=K, V=V, BT=BT)


def _load_bridge(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.c20_avelang_launch.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        *([ctypes.c_void_p] * 6),
    ]
    lib.c20_avelang_launch.restype = ctypes.c_int
    lib.c20_avelang_last_error.restype = ctypes.c_char_p
    return lib


def _launch_frozen(lib: ctypes.CDLL, arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    item = FROZEN[arm]
    path = Path(item["path"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise RuntimeError(f"frozen HSACO hash mismatch for {arm}: {actual} != {item['sha256']}")
    q, k, v_new, h, g = tensors
    values = [ctypes.c_void_p(value.data_ptr()) for value in (q, k, v_new, h, g, output)]
    status = lib.c20_avelang_launch(
        str(path).encode(), str(item["symbol"]).encode(),
        int(torch.cuda.current_stream(q.device).cuda_stream),
        GRID_X, 1, 1, BLOCK_X, *values,
    )
    if status:
        message = lib.c20_avelang_last_error()
        detail = message.decode() if message else f"status={status}"
        raise RuntimeError(f"frozen arm {arm} failed: {detail}")


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure(fn, warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(values),
        "p25_ms": _quantile(values, 0.25),
        "p75_ms": _quantile(values, 0.75),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": repeat,
    }


def _worker(args: argparse.Namespace) -> None:
    tensors = _inputs(2026081000 + T)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in ARMS}
    bridge = _load_bridge(Path(args.bridge))
    public = chunk_o.chunk_fwd_o(q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4], scale=K**-0.5, chunk_size=BT)
    torch.cuda.synchronize()
    del public
    native_config = _config_summary()

    def launch(arm: str) -> None:
        if arm == "z5b":
            qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, outputs[arm])
        elif arm == "native_selected":
            _native_direct(tensors, outputs[arm])
        else:
            _launch_frozen(bridge, arm, tensors, outputs[arm])

    order = args.order.split(",")
    for arm in order:
        launch(arm)
    torch.cuda.synchronize()
    measurements = {arm: _measure(lambda arm=arm: launch(arm), args.warmup, args.repeat) for arm in order}
    finite = {arm: bool(torch.isfinite(outputs[arm]).all().item()) for arm in ARMS}
    exact_vs_z5b = {arm: bool(torch.equal(outputs[arm], outputs["z5b"])) for arm in ARMS if arm != "z5b"}
    print(json.dumps({
        "T": T, "scope": "C20_T2048_frozen_code_object_replay", "order": order,
        "warmup": args.warmup, "repeat": args.repeat, "native_selection": native_config,
        "measurements": measurements, "finite": finite, "exact_vs_z5b": exact_vs_z5b,
        "frozen_code_replay": {arm: {k: str(v) for k, v in item.items()} for arm, item in FROZEN.items()},
    }, sort_keys=True))


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z5b,p2_frozen_hsaco,c18_frozen_hsaco,c19_frozen_hsaco,native_selected",
        "native_selected,c19_frozen_hsaco,c18_frozen_hsaco,p2_frozen_hsaco,z5b",
        "p2_frozen_hsaco,z5b,native_selected,c19_frozen_hsaco,c18_frozen_hsaco",
        "c18_frozen_hsaco,native_selected,z5b,c19_frozen_hsaco,p2_frozen_hsaco",
        "c19_frozen_hsaco,c18_frozen_hsaco,p2_frozen_hsaco,native_selected,z5b",
        "native_selected,z5b,c19_frozen_hsaco,p2_frozen_hsaco,c18_frozen_hsaco",
        "c18_frozen_hsaco,p2_frozen_hsaco,c19_frozen_hsaco,z5b,native_selected",
    ]
    raw = []
    for session in range(args.sessions):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--bridge", args.bridge,
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--order", orders[session]]
        completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"C20 frozen session {session} failed:\n{completed.stdout}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        row = json.loads(lines[-1]); row["session"] = session; raw.append(row)
        print(json.dumps(row, sort_keys=True))
    medians = {arm: [float(row["measurements"][arm]["median_ms"]) for row in raw] for arm in ARMS}
    summary = [{"arm": arm, "session_medians_ms": medians[arm], "median_of_session_medians_ms": statistics.median(medians[arm])} for arm in ARMS]
    paired = {f"{arm}_minus_z5b_us": [(x-y)*1e3 for x,y in zip(medians[arm], medians["z5b"])] for arm in ARMS if arm != "z5b"}
    payload = {"T": T, "contract": {"sessions": args.sessions, "warmup": args.warmup, "repeat": args.repeat, "fresh_process": True, "current_stream": True, "cuda_graph_used": False, "preallocated_outputs": True, "scope": "frozen HSACO arms for C18/P2/C19; source Z5B and actual selected native"}, "raw": raw, "summary": summary, "paired_differences_us": paired}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--order", default=ARMS[0])
    parser.add_argument("--out", type=Path, default=LADDER / "codex_qwen_gfx942_c20_formal_performance/stage6z_c20_t2048_frozen_controls.json")
    args = parser.parse_args()
    if args.worker: _worker(args)
    else: _parent(args)


if __name__ == "__main__":
    main()

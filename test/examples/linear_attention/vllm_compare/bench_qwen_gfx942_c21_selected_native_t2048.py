#!/usr/bin/env python3
"""Formal T=2048 body benchmark for C21-NSM.

The C19 and C21 arms replay hash-guarded code objects through the existing
minimal HIP launcher.  Z5B and native stay source/public-selector arms.  Each
parent session is a new Python process so its module/JIT state cannot hide a
launch or compilation difference inside another session.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import statistics
import subprocess
import sys
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


BT = 64
HK = 4
HV = 8
K = 128
V = 128
T = 2048
GRID_X = (T // BT) * HV * 2
BLOCK_X = 256
ARMS = ("z5b", "c19_frozen", "c21_frozen", "native_selected")

FROZEN = {
    "c19_frozen": {
        "path": LADDER / "codex_qwen_gfx942_c19_full_physical_region_t2048/machine/c19_full_physical_region.hsaco",
        "symbol": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c19_full_physical_region",
        "sha256": "14ffa78200ff995448c8abbe1c5484375aae1a2f75c8412a937b33d4aa628907",
    },
    "c21_frozen": {
        "path": LADDER / "codex_qwen_gfx942_c21_selected_native_pipeline/machine/c21_selected_native_pipeline.hsaco",
        "symbol": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline",
        "sha256": "d7d047cb3b9eb84d2e7542086644571d0f715c54e73542cdd831c7f86324f2d8",
    },
    "native_selected": {
        "path": LADDER / "codex_qwen_gfx942_c21_selected_native/selected/chunk_fwd_kernel_o.hsaco",
        "symbol": "chunk_fwd_kernel_o",
        "sha256": "cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d",
    },
}


def _inputs(seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(T, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, T, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, T // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _load_bridge(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.c20_avelang_launch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        *([ctypes.c_void_p] * 6),
    ]
    lib.c20_avelang_launch.restype = ctypes.c_int
    lib.c20_avelang_last_error.restype = ctypes.c_char_p
    return lib


def _load_native_bridge(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.c21_native_launch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        *([ctypes.c_void_p] * 6),
        ctypes.c_float,
        ctypes.c_int32,
    ]
    lib.c21_native_launch.restype = ctypes.c_int
    lib.c21_native_last_error.restype = ctypes.c_char_p
    return lib


def _launch_frozen(lib: ctypes.CDLL, arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    item = FROZEN[arm]
    path = Path(item["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError(f"{arm} code object hash mismatch: {digest} != {item['sha256']}")
    pointers = [ctypes.c_void_p(value.data_ptr()) for value in (*tensors, output)]
    status = lib.c20_avelang_launch(
        str(path).encode(),
        str(item["symbol"]).encode(),
        int(torch.cuda.current_stream(tensors[0].device).cuda_stream),
        GRID_X,
        1,
        1,
        BLOCK_X,
        *pointers,
    )
    if status:
        detail = lib.c20_avelang_last_error()
        raise RuntimeError(detail.decode() if detail else f"{arm} bridge status={status}")


def _launch_native_selected(
    lib: ctypes.CDLL, tensors: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    item = FROZEN["native_selected"]
    path = Path(item["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise RuntimeError(f"native selected code object hash mismatch: {digest} != {item['sha256']}")
    q, k, v_new, h, g = tensors
    status = lib.c21_native_launch(
        str(path).encode(),
        str(item["symbol"]).encode(),
        int(torch.cuda.current_stream(q.device).cuda_stream),
        2,
        T // BT,
        HV,
        BLOCK_X,
        *[ctypes.c_void_p(value.data_ptr()) for value in (q, k, v_new, h, g, output)],
        ctypes.c_float(K**-0.5),
        ctypes.c_int32(T),
    )
    if status:
        detail = lib.c21_native_last_error()
        raise RuntimeError(detail.decode() if detail else f"native selected bridge status={status}")


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure(launch, warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return {
        "median_ms": statistics.median(values),
        "p25_ms": _quantile(values, 0.25),
        "p75_ms": _quantile(values, 0.75),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": repeat,
    }


def _worker(args: argparse.Namespace) -> None:
    tensors = _inputs(2026082400)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in ARMS}
    bridge = _load_bridge(Path(args.bridge))
    native_bridge = _load_native_bridge(Path(args.native_bridge))
    selection = {
        "identity": "fresh C21 capture replay; avoids current-cache stage-3 reselection",
        "kwargs": {"BK": 32, "BV": 64},
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
        "hsaco_sha256": FROZEN["native_selected"]["sha256"],
    }

    def launch(arm: str) -> None:
        if arm == "z5b":
            qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, outputs[arm])
        elif arm == "native_selected":
            _launch_native_selected(native_bridge, tensors, outputs[arm])
        else:
            _launch_frozen(bridge, arm, tensors, outputs[arm])

    order = args.order.split(",")
    for arm in order:
        launch(arm)
    torch.cuda.synchronize()
    finite = {arm: bool(torch.isfinite(outputs[arm]).all().item()) for arm in ARMS}
    exact = {arm: bool(torch.equal(outputs[arm], outputs["z5b"])) for arm in ARMS if arm != "z5b"}
    measurements = {arm: _measure(lambda arm=arm: launch(arm), args.warmup, args.repeat) for arm in order}
    print(json.dumps({
        "T": T,
        "order": order,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "native_selection": selection,
        "finite": finite,
        "exact_vs_z5b": exact,
        "measurements": measurements,
        "frozen": {arm: {key: str(value) for key, value in item.items()} for arm, item in FROZEN.items()},
    }, sort_keys=True))


def _bootstrap(values: list[float], seed: int) -> list[float]:
    if not values:
        return []
    randomizer = random.Random(seed)
    medians = []
    for _ in range(20_000):
        medians.append(statistics.median([randomizer.choice(values) for _ in values]))
    return [_quantile(medians, 0.025), _quantile(medians, 0.975)]


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z5b,c19_frozen,c21_frozen,native_selected",
        "native_selected,c21_frozen,c19_frozen,z5b",
        "c19_frozen,z5b,native_selected,c21_frozen",
        "c21_frozen,native_selected,z5b,c19_frozen",
        "z5b,c21_frozen,native_selected,c19_frozen",
        "native_selected,c19_frozen,c21_frozen,z5b",
        "c19_frozen,c21_frozen,z5b,native_selected",
    ]
    raw = []
    for session in range(args.sessions):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--bridge", str(args.bridge),
            "--native-bridge", str(args.native_bridge),
            "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--order", orders[session % len(orders)],
        ]
        completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"C21 formal session {session} failed:\n{completed.stdout}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        row = json.loads(lines[-1])
        row["session"] = session
        raw.append(row)
        print(json.dumps(row, sort_keys=True))

    session_medians = {arm: [float(row["measurements"][arm]["median_ms"]) for row in raw] for arm in ARMS}
    medians = {arm: statistics.median(values) for arm, values in session_medians.items()}
    paired = {
        arm: [(current - baseline) * 1e3 for current, baseline in zip(session_medians[arm], session_medians["z5b"])]
        for arm in ARMS if arm != "z5b"
    }
    z5b = medians["z5b"]
    c21 = medians["c21_frozen"]
    native = medians["native_selected"]
    gap_denominator = z5b - native
    captured_gap = (z5b - c21) / gap_denominator if gap_denominator > 0 else None
    payload = {
        "schema": "qwen.gfx942.stage6z.c21.formal_body_t2048.v1",
        "experiment": "C21-NSM Selected-Native Pipeline Reconstruction",
        "T": T,
        "contract": {
            "sessions": args.sessions,
            "fresh_python_process_per_session": True,
            "current_hip_stream": True,
            "cuda_graph_used": False,
            "caller_owned_preallocated_output": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "metric": "HIP-event body latency; median of session medians",
        },
        "raw_sessions": raw,
        "summary": {
            arm: {
                "session_medians_ms": session_medians[arm],
                "median_of_session_medians_ms": medians[arm],
                "p25_session_median_ms": _quantile(session_medians[arm], 0.25),
                "p75_session_median_ms": _quantile(session_medians[arm], 0.75),
            }
            for arm in ARMS
        },
        "paired_minus_z5b_us": {
            arm: {"samples": samples, "median_us": statistics.median(samples), "bootstrap_ci95_us": _bootstrap(samples, 20260824 + index)}
            for index, (arm, samples) in enumerate(paired.items())
        },
        "derived": {
            "speedup_vs_z5b": z5b / c21,
            "c21_over_native": c21 / native,
            "captured_gap_fraction": captured_gap,
            "captured_gap_percent": captured_gap * 100 if captured_gap is not None else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--bridge",
        type=Path,
        default=LADDER / "codex_qwen_gfx942_c20_formal_performance/libc20_avelang_hsaco_bridge.so",
    )
    parser.add_argument(
        "--native-bridge",
        type=Path,
        default=HERE / "libc21_selected_native_hsaco_bridge.so",
    )
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--order", default=ARMS[0])
    parser.add_argument("--out", type=Path, default=LADDER / "stage6z_c21_formal_body_t2048.json")
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

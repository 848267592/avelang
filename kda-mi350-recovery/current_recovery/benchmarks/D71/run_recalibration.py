#!/usr/bin/env python3
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path("/workspace/project/avelang")
OUT = ROOT / "test/examples/kda/reports/D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION"
C68_DIR = ROOT / "test/examples/kda/reports/C_20260925_68_AITER_SHORT_TRUE_BEST_ENVELOPE"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["CHUNK_DELTA_ATTN_TRITON_AUTOTUNE"] = "1"
os.environ["TRITON_CACHE_DIR"] = str(OUT / "triton_cache")
sys.path.insert(0, str(C68_DIR))

import torch
import run_sweep as c68

T_VALUES = (32, 64, 127, 128, 256, 384, 511, 512, 1024, 2048, 4096)
EXPECTED_COMMIT = "7bb44998274fa679ece0a37bf8426ab780b1837d"
EXPECTED = {
    32: ("BW16_W2_S1", "1c3f0303f2ea6ae902219be2ebfe9066ecb6a602a7b3b1808b2935da2b35be91", 128, 4096),
    64: ("BW16_W2_S2", "0038c4046fe0a7badc0b27836482d31c5e1e0bd604d9f35b80e6851b9dfd0416", 128, 30720),
    "ge127": ("BW16_W4_S2", "85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16", 256, 30720),
}
WARMUPS = 50
REPEATS = 100
ROUNDS = 3
TOL = 0.03

HIP = ctypes.CDLL("libamdhip64.so")
HIP.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
HIP.hipModuleLoad.restype = ctypes.c_int
HIP.hipModuleGetFunction.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
]
HIP.hipModuleGetFunction.restype = ctypes.c_int
HIP.hipModuleLaunchKernel.restype = ctypes.c_int
module_cache = {}


def save_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cfg_key(cfg: dict) -> str:
    return c68.config_key(cfg)


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    p25, p50, p75, p95 = np.percentile(a, [25, 50, 75, 95])
    return {
        "median_us": float(p50),
        "mean_us": float(np.mean(a)),
        "p25_us": float(p25),
        "p75_us": float(p75),
        "p95_us": float(p95),
        "min_us": float(np.min(a)),
        "max_us": float(np.max(a)),
        "mad_us": float(np.median(np.abs(a - p50))),
        "stddev_us": float(np.std(a)),
    }


def output_metrics(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    rel = delta / expected.float().abs().clamp_min(1.0)
    return {
        "max_abs": float(delta.max().item()),
        "max_rel": float(rel.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def direct_load(path: Path):
    module = ctypes.c_void_p()
    rc = HIP.hipModuleLoad(ctypes.byref(module), str(path).encode())
    if rc != 0:
        raise RuntimeError(f"STOP: hipModuleLoad failed for {path}: rc={rc}")
    fn = ctypes.c_void_p()
    rc = HIP.hipModuleGetFunction(
        ctypes.byref(fn), module, b"_flash_kda_segment_kernel"
    )
    if rc != 0:
        raise RuntimeError(f"STOP: hipModuleGetFunction failed for {path}: rc={rc}")
    return module, fn


def latest_object(cfg):
    rows = c68.scan_objects(c68.candidate_list())[cfg_key(cfg)]
    if not rows:
        raise RuntimeError(f"STOP: no compiled HSACO found for {cfg_key(cfg)}")
    rows = sorted(rows, key=lambda row: row["mtime"], reverse=True)
    path = Path(rows[0]["hsaco"])
    meta = json.loads(path.with_suffix(".json").read_text())
    actual_sha = sha256(path)
    if actual_sha != rows[0]["hsaco_sha256"]:
        raise RuntimeError(f"STOP: object hash changed while resolving {path}")
    if meta.get("target", {}).get("arch") != "gfx942":
        raise RuntimeError(f"STOP: wrong HSACO target in {path}: {meta.get('target')}")
    if int(meta.get("num_warps", -1)) != cfg["W"]:
        raise RuntimeError(f"STOP: num_warps mismatch in {path}")
    if int(meta.get("num_stages", -1)) != cfg["S"]:
        raise RuntimeError(f"STOP: num_stages mismatch in {path}")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "shared": int(meta.get("shared", 0)),
        "num_warps": int(meta["num_warps"]),
        "num_stages": int(meta["num_stages"]),
        "target": meta.get("target"),
        "cache_hash": meta.get("hash"),
    }


def struct_pack(pointers, nt):
    import struct
    ptrs = [int(t.data_ptr()) for t in pointers]
    return struct.pack("=QQQQQQQQQQQQQQQQIIQQ", *ptrs, nt, 0, 0, 0)


def build_launch(case, cfg, obj):
    common = case["alive"][4]
    num_segs = int(case["segments"])
    pointers = [
        common["ws_kd"], common["ws_qd"], common["ws_kr"], common["ws_gt"],
        common["ws_inv_mqk"], common["v_input"], common["beta_raw"],
        common["out"], common["h_in"], common["final_state"],
        common["seg_chunk_base"], common["seg_nchunks"], common["seg_tok_base"],
        common["seg_tok_end"], common["seg_seq"], common["seg_is_last"],
    ]
    packed = struct_pack(pointers, int(case["NT"]))
    kernarg = (ctypes.c_char * len(packed)).from_buffer_copy(packed)
    kernarg_size = ctypes.c_size_t(len(packed))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(1),
        ctypes.cast(ctypes.pointer(kernarg), ctypes.c_void_p),
        ctypes.c_void_p(2),
        ctypes.cast(ctypes.pointer(kernarg_size), ctypes.c_void_p),
        ctypes.c_void_p(3),
    )
    module, fn = module_cache[obj["sha256"]]
    grid = (
        int((c68.V + cfg["BW"] - 1) // cfg["BW"]),
        int(num_segs * c68.HEADS),
        1,
    )
    block = int(cfg["W"] * 64)
    dyn_lds = int(obj["shared"])

    def launch():
        rc = HIP.hipModuleLaunchKernel(
            fn, grid[0], grid[1], grid[2],
            block, 1, 1, dyn_lds,
            ctypes.c_void_p(0), ctypes.c_void_p(0), extra
        )
        if rc != 0:
            raise RuntimeError(f"hipModuleLaunchKernel({cfg_key(cfg)}) failed: rc={rc}")

    return launch, grid, block, dyn_lds


def print_identity(T, cfg, obj, reference_class, grid, block, dyn_lds):
    print(f"T = {T}")
    print("Shape = B1/H12/K128/V128/C32")
    print("GPU = MI300X gfx942")
    print(f"AITER commit = {EXPECTED_COMMIT}")
    print(f"Reference class = {reference_class}")
    print(f"BW = {cfg['BW']}")
    print(f"num_warps = {cfg['W']}")
    print(f"num_stages = {cfg['S']}")
    print("num_ctas = 1")
    print(f"grid = {grid}")
    print(f"block = {block}")
    print(f"dynamic LDS = {dyn_lds}")
    print(f"HSACO SHA256 = {obj['sha256']}")


def event_sample(mode, case, cfg, launch, start, stop):
    if mode == "C68":
        case["reset"]()
        start.record()
        case["invoke"](cfg, warmup=False)
        stop.record()
    elif mode == "D70":
        start.record()
        case["reset"]()
        launch()
        stop.record()
    elif mode == "CANONICAL":
        case["reset"]()
        start.record()
        launch()
        stop.record()
    else:
        raise ValueError(mode)
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0)


def compare_and_pass(actual, expected):
    om = output_metrics(actual[0], expected[0])
    sm = output_metrics(actual[1], expected[1])
    return {
        "output": om,
        "final_state": sm,
        "pass": bool(
            torch.isfinite(actual[0]).all().item()
            and torch.isfinite(actual[1]).all().item()
            and om["max_abs"] <= TOL
            and sm["max_abs"] <= TOL
        ),
    }


def assert_static_identity(T, key, obj, grid, block, dyn_lds):
    expected = EXPECTED[32] if T == 32 else EXPECTED[64] if T == 64 else EXPECTED["ge127"]
    expected_key, expected_sha, expected_block, expected_lds = expected
    if key != expected_key:
        raise RuntimeError(f"STOP: frozen config map mismatch at T={T}: {key} != {expected_key}")
    if obj["sha256"] != expected_sha:
        raise RuntimeError(f"STOP: frozen HSACO mismatch at T={T}: {obj['sha256']} != {expected_sha}")
    if grid != (8, 12, 1) or block != expected_block or dyn_lds != expected_lds:
        raise RuntimeError(
            f"STOP: frozen launch mismatch at T={T}: grid={grid}, block={block}, LDS={dyn_lds}"
        )


def main():
    actual_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/aiter"), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"STOP: AITER commit mismatch: {actual_commit}")

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    if "MI300X" not in device_name.upper():
        raise RuntimeError(f"STOP: expected MI300X, got {device_name}")
    stream = int(torch.cuda.current_stream().cuda_stream)
    if stream != 0:
        raise RuntimeError(f"STOP: expected default HIP stream 0, got {stream}")
    if (c68.BATCH, c68.HEADS, c68.K, c68.V, c68.C) != (1, 12, 128, 128, 32):
        raise RuntimeError("STOP: C68 module workload shape mismatch")
    source_configs = [
        {"BW": c.kwargs["BW"], "W": c.num_warps, "S": c.num_stages, "num_ctas": c.num_ctas}
        for c in c68.flash._K2_CONFIGS
    ]
    configs = c68.candidate_list()
    if source_configs != configs:
        raise RuntimeError(f"STOP: source candidate list mismatch: {source_configs}")

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "container": "ljd_sglang_kda",
        "repo_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "aiter_commit": actual_commit,
        "gpu": device_name,
        "gfx": "gfx942",
        "gpu_index": 0,
        "stream_handle": stream,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "python": platform.python_version(),
        "workload": {"B": 1, "H": 12, "K": 128, "V": 128, "C": 32},
        "T_values": list(T_VALUES),
        "protocol": {"warmups": WARMUPS, "repeats_per_round": REPEATS, "rounds": ROUNDS},
        "candidate_space": configs,
    }
    save_json("environment_manifest.json", env)
    print(json.dumps({
        "IDENTITY_GATE": "PASS", "GPU": device_name, "gfx": "gfx942",
        "AITER commit": actual_commit, "stream": stream, "shape": env["workload"]
    }, indent=2), flush=True)

    cases = {}
    selections = {}
    correctness_rows = []
    candidate_manifest = {}
    for T in T_VALUES:
        inp = c68.make_kda_prefill_inputs(T=T, H=c68.HEADS, seed=6800 + T)
        torch.manual_seed(91000 + T)
        state = torch.randn(
            (1, c68.HEADS, c68.K, c68.V),
            device="cuda", dtype=torch.float32
        ).mul_(0.1).contiguous()
        nt = (T + c68.C - 1) // c68.C
        ws = c68.allocate_workspace(nt, inp["q"].device)
        # K1 preparation and synchronization are before all K2 HIP events.
        c68.prepare_workspace(inp, ws, nt)
        torch.cuda.synchronize()
        case = c68.make_case(inp, ws, state)
        cases[T] = case

        # Exact forced AITER BW16/W2/S2 numerical baseline for this T.
        base_cfg = {"BW": 16, "W": 2, "S": 2, "num_ctas": 1}
        case["invoke"](base_cfg, warmup=True)
        torch.cuda.synchronize()
        base_obj = latest_object(base_cfg)
        module_cache[base_obj["sha256"]] = direct_load(Path(base_obj["path"]))
        base_launch, _, _, _ = build_launch(case, base_cfg, base_obj)
        case["reset"]()
        base_launch()
        torch.cuda.synchronize()
        base_output = case["out"].clone()
        base_state = case["final_state"].clone()

        t_manifest = {}
        t_correctness = []
        for cfg in configs:
            key = cfg_key(cfg)
            if key == cfg_key(base_cfg):
                case["reset"]()
                case["invoke"](cfg, warmup=False)
                torch.cuda.synchronize()
                jit_o = case["out"].clone()
                jit_s = case["final_state"].clone()
                obj = base_obj
            else:
                case["reset"]()
                case["invoke"](cfg, warmup=True)
                torch.cuda.synchronize()
                jit_o = case["out"].clone()
                jit_s = case["final_state"].clone()
                obj = latest_object(cfg)
                module_cache[obj["sha256"]] = direct_load(Path(obj["path"]))

            launch, grid, block, dyn_lds = build_launch(case, cfg, obj)
            case["reset"]()
            launch()
            torch.cuda.synchronize()
            direct_o = case["out"].clone()
            direct_s = case["final_state"].clone()
            parity_o = output_metrics(direct_o, jit_o)
            parity_s = output_metrics(direct_s, jit_s)
            if parity_o["max_abs"] > 1e-7 or parity_s["max_abs"] > 1e-7:
                raise RuntimeError(
                    f"STOP: direct HSACO/JIT parity failed at T={T} {key}: "
                    f"{parity_o}, {parity_s}, object={obj['sha256']}"
                )

            correct = compare_and_pass((direct_o, direct_s), (base_output, base_state))
            gate = {
                "T": T, "config": cfg, "sha256": obj["sha256"],
                "grid": list(grid), "block": block, "dynamic_lds": dyn_lds,
                "correctness": correct, "valid": bool(correct["pass"]),
                "hsaco_path": obj["path"], "cache_hash": obj["cache_hash"],
            }
            t_manifest[key] = gate
            t_correctness.append(gate)
            selections[(T, key)] = {
                "cfg": cfg, "obj": obj, "launch": launch,
                "grid": grid, "block": block, "dyn_lds": dyn_lds,
                "correctness": correct, "valid": bool(correct["pass"]),
            }

        winner_key = EXPECTED[32][0] if T == 32 else EXPECTED[64][0] if T == 64 else EXPECTED["ge127"][0]
        frozen = t_manifest[winner_key]
        assert_static_identity(
            T, winner_key, {"sha256": frozen["sha256"]},
            tuple(frozen["grid"]), frozen["block"], frozen["dynamic_lds"]
        )
        if not frozen["valid"]:
            raise RuntimeError(f"STOP: frozen C68 candidate failed correctness at T={T}")
        candidate_manifest[str(T)] = t_manifest
        correctness_rows.append({"T": T, "baseline": "forced AITER BW16/W2/S2", "rows": t_correctness})
        print(f"PREPARED T={T}: compiled/correctness checked {len(configs)} candidates", flush=True)
        save_json("candidate_manifest.json", candidate_manifest)
        save_json("correctness.json", correctness_rows)

    # Phase 1: boundary calibration on the exact frozen candidate for each T.
    boundary_raw = {}
    boundary_summaries = {}
    modes = ("C68", "D70", "CANONICAL")
    orders = (
        ("C68", "D70", "CANONICAL"),
        ("CANONICAL", "D70", "C68"),
        ("D70", "C68", "CANONICAL"),
    )
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for T in T_VALUES:
        frozen_key = EXPECTED[32][0] if T == 32 else EXPECTED[64][0] if T == 64 else EXPECTED["ge127"][0]
        selected = selections[(T, frozen_key)]
        cfg, obj, launch = selected["cfg"], selected["obj"], selected["launch"]
        print_identity(T, cfg, obj, "AITER_TRUE_BEST boundary calibration", selected["grid"], selected["block"], selected["dyn_lds"])
        for mode in modes:
            for _ in range(WARMUPS):
                cases[T]["reset"]()
                if mode == "C68":
                    cases[T]["invoke"](cfg, warmup=False)
                else:
                    launch()
            torch.cuda.synchronize()
        samples = {mode: [] for mode in modes}
        for order in orders:
            for mode in order:
                for _ in range(REPEATS):
                    samples[mode].append(event_sample(mode, cases[T], cfg, launch, start, stop))
        boundary_raw[str(T)] = {
            "config": cfg, "hsaco_sha256": obj["sha256"],
            "warmups_per_mode": WARMUPS, "repeats_per_round": REPEATS, "rounds": ROUNDS,
            "samples_us": samples,
        }
        summary = {mode: summarize(samples[mode]) for mode in modes}
        canonical = summary["CANONICAL"]["median_us"]
        summary["delta_vs_canonical_us"] = {
            mode: summary[mode]["median_us"] - canonical for mode in modes
        }
        summary["delta_vs_canonical_pct"] = {
            mode: (summary[mode]["median_us"] / canonical - 1.0) * 100.0 for mode in modes
        }
        summary["D70_minus_C68_us"] = summary["D70"]["median_us"] - summary["C68"]["median_us"]
        boundary_summaries[str(T)] = summary
        save_json("timing_boundary_raw.json", {
            "protocol_actions": {
                "C68": {
                    "before_event": [
                        "K1/workspace preparation completed and synchronized before timing.",
                        "HSACO specialization compiled/cached before timing; 50 warmups outside measured events.",
                        "state_work.copy_(state_init) queued on default stream before start.record().",
                    ],
                    "inside_event": [
                        "HIP start event; forced Triton JITFunction.run dispatch (warmup=False) enqueues K2; HIP stop event.",
                        "No reset, K1/workspace prep, tensor memset/copy, compilation, or autotune inside.",
                    ],
                    "after_event": ["stop.synchronize(); read HIP event elapsed time."],
                },
                "D70": {
                    "before_event": [
                        "K1/workspace preparation completed and synchronized before timing.",
                        "Same HSACO module loaded and kernarg built before timing; 50 warmups outside measured events.",
                        "No state reset before start.record() for measured samples.",
                    ],
                    "inside_event": [
                        "HIP start event; state_work.copy_(state_init) on default stream; direct hipModuleLaunchKernel; HIP stop event.",
                        "No K1/workspace prep, JIT, compile, or autotune inside.",
                    ],
                    "after_event": ["stop.synchronize(); read HIP event elapsed time."],
                },
                "CANONICAL": {
                    "before_event": [
                        "K1/workspace preparation completed and synchronized before timing.",
                        "Same HSACO module loaded and kernarg built before timing; 50 warmups outside measured events.",
                        "state_work.copy_(state_init) queued on default stream before start.record().",
                    ],
                    "inside_event": [
                        "HIP start event; direct hipModuleLaunchKernel for K2 only; HIP stop event.",
                        "No state reset, workspace prep, tensor copy/memset, JIT, compilation, or autotune inside.",
                    ],
                    "after_event": ["stop.synchronize(); read HIP event elapsed time."],
                },
            },
            "raw": boundary_raw,
            "summary": boundary_summaries,
        })
        print(f"BOUNDARY T={T}: " + json.dumps(summary, sort_keys=True), flush=True)

    # Phase 3: direct K2-only sweep of all 12 legal AITER candidates.
    benchmark_raw = {
        "protocol": {
            "scope": "K2 only",
            "timer": "HIP events on default stream; direct HIP module launch",
            "state_reset": "same initial state restored before every measured event",
            "warmups": WARMUPS,
            "measured_repeats_per_round": REPEATS,
            "interleaved_rounds": ROUNDS,
            "K1_workspace_prep": "outside all K2 HIP events; prepared once per T",
            "jit_compile_autotune": "completed before timing; none inside events",
            "us_per_chunk": "total K2 GPU latency / ceil(T/32)",
        },
        "raw_samples_us": {},
        "round_stats": [],
        "aggregate": {},
    }
    rows_by_T = {}
    sweep_orders = (
        list(configs),
        list(reversed(configs)),
        list(configs[4:] + configs[:4]),
    )
    for T in T_VALUES:
        case = cases[T]
        by_key = {}
        for cfg in configs:
            key = cfg_key(cfg)
            selected = selections[(T, key)]
            if not selected["valid"]:
                continue
            print_identity(T, cfg, selected["obj"], "AITER_CANDIDATE_SWEEP", selected["grid"], selected["block"], selected["dyn_lds"])
            for _ in range(WARMUPS):
                case["reset"]()
                selected["launch"]()
            torch.cuda.synchronize()
            by_key[key] = []

        for rid, order in enumerate(sweep_orders, 1):
            for cfg in order:
                key = cfg_key(cfg)
                selected = selections[(T, key)]
                if not selected["valid"]:
                    continue
                values = []
                for _ in range(REPEATS):
                    case["reset"]()
                    start.record()
                    selected["launch"]()
                    stop.record()
                    stop.synchronize()
                    values.append(float(start.elapsed_time(stop) * 1000.0))
                by_key[key].extend(values)
                summary = summarize(values)
                benchmark_raw["round_stats"].append({
                    "T": T, "round": rid, "config": cfg, "sha256": selected["obj"]["sha256"],
                    "grid": list(selected["grid"]), "block": selected["block"],
                    "dynamic_lds": selected["dyn_lds"], **summary,
                    "us_per_chunk": summary["median_us"] / ((T + 31) // 32),
                })
            benchmark_raw["raw_samples_us"][str(T)] = {
                cfg_key(c): by_key.get(cfg_key(c), []) for c in configs
            }
            save_json("benchmark_raw.json", benchmark_raw)
            print(f"BENCHMARK T={T} round={rid} complete", flush=True)

        eligible = []
        for cfg in configs:
            key = cfg_key(cfg)
            selected = selections[(T, key)]
            if not selected["valid"]:
                continue
            stats = summarize(by_key[key])
            row = {
                "T": T, "NT": (T + 31) // 32, "config": cfg,
                "config_key": key, "hsaco_sha256": selected["obj"]["sha256"],
                "hsaco_path": selected["obj"]["path"],
                "grid": list(selected["grid"]), "block": selected["block"],
                "dynamic_lds": selected["dyn_lds"],
                "correctness": selected["correctness"],
                **stats,
                "us_per_chunk": stats["median_us"] / ((T + 31) // 32),
            }
            eligible.append(row)
        eligible.sort(key=lambda r: r["median_us"])
        if len(eligible) < 2:
            raise RuntimeError(f"STOP: fewer than 2 correctness-passing candidates at T={T}")
        best, second = eligible[:2]
        rows_by_T[str(T)] = {
            "winner": best,
            "second": second,
            "second_gap_percent": (second["median_us"] / best["median_us"] - 1.0) * 100.0,
            "ranking": eligible,
        }
        benchmark_raw["aggregate"][str(T)] = rows_by_T[str(T)]
        save_json("benchmark_raw.json", benchmark_raw)
        print(
            f"WINNER T={T}: {best['config_key']} {best['hsaco_sha256']} "
            f"median={best['median_us']:.6f} us; second={second['config_key']} "
            f"{second['median_us']:.6f} us gap={rows_by_T[str(T)]['second_gap_percent']:.3f}%",
            flush=True,
        )

    old_c68 = json.loads((C68_DIR / "sweep_results.json").read_text())
    old_c68_map = {
        str(row["T"]): {
            "config": row["winner"]["config"],
            "config_key": cfg_key(row["winner"]["config"]),
            "median_us": row["winner"]["median_us"],
        }
        for row in old_c68["aggregates"]
    }
    confirmed = {}
    for T in T_VALUES:
        frozen_key = EXPECTED[32][0] if T == 32 else EXPECTED[64][0] if T == 64 else EXPECTED["ge127"][0]
        expected_sha = EXPECTED[32][1] if T == 32 else EXPECTED[64][1] if T == 64 else EXPECTED["ge127"][1]
        selected = selections[(T, frozen_key)]
        win = rows_by_T[str(T)]["winner"]
        confirmed[str(T)] = {
            "expected_frozen_config": frozen_key,
            "expected_frozen_sha256": expected_sha,
            "compiled_frozen_sha256": selected["obj"]["sha256"],
            "canonical_winner_config": win["config_key"],
            "canonical_winner_sha256": win["hsaco_sha256"],
            "c68_historical_winner_config": old_c68_map[str(T)]["config_key"],
            "c68_historical_median_us": old_c68_map[str(T)]["median_us"],
            "config_map_confirmed": win["config_key"] == frozen_key,
            "winner_identity_matches_frozen": win["config_key"] == frozen_key and win["hsaco_sha256"] == expected_sha,
            "historical_c68_config_match": old_c68_map[str(T)]["config_key"] == frozen_key,
        }
    save_json("confirmed_true_best_map.json", confirmed)
    save_json("benchmark_raw.json", benchmark_raw)

    with (OUT / "benchmark_raw.csv").open("w", newline="") as f:
        fields = [
            "T", "NT", "config_key", "BW", "W", "S", "num_ctas",
            "hsaco_sha256", "hsaco_path", "grid", "block", "dynamic_lds",
            "correctness_pass", "median_us", "mean_us", "p25_us", "p75_us",
            "p95_us", "min_us", "max_us", "mad_us", "stddev_us", "us_per_chunk",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for T in T_VALUES:
            for row in rows_by_T[str(T)]["ranking"]:
                cfg = row["config"]
                writer.writerow({
                    "T": T, "NT": row["NT"], "config_key": row["config_key"],
                    "BW": cfg["BW"], "W": cfg["W"], "S": cfg["S"],
                    "num_ctas": cfg["num_ctas"], "hsaco_sha256": row["hsaco_sha256"],
                    "hsaco_path": row["hsaco_path"], "grid": row["grid"],
                    "block": row["block"], "dynamic_lds": row["dynamic_lds"],
                    "correctness_pass": row["correctness"]["pass"],
                    "median_us": row["median_us"], "mean_us": row["mean_us"],
                    "p25_us": row["p25_us"], "p75_us": row["p75_us"],
                    "p95_us": row["p95_us"], "min_us": row["min_us"],
                    "max_us": row["max_us"], "mad_us": row["mad_us"],
                    "stddev_us": row["stddev_us"], "us_per_chunk": row["us_per_chunk"],
                })

    report = []
    report.append("# D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION")
    report.append("")
    report.append("SCOPE: K2_ONLY")
    report.append("")
    report.append(f"Identity: MI300X gfx942; B1/H12/K128/V128/C32; AITER {actual_commit}; default stream handle {stream}.")
    report.append("")
    report.append("## C68_PROTOCOL")
    report.append("")
    report.append("Before event: K1/workspace preparation and synchronization; specialization compile/cache plus 50 warmups; restore initial state on default stream.")
    report.append("Inside event: HIP start event, Python forced Triton JITFunction.run dispatch (warmup=False) and K2 launch, HIP stop event. No reset or K1 work is inside.")
    report.append("After event: synchronize stop event and read elapsed time.")
    report.append("")
    report.append("## D70_PROTOCOL")
    report.append("")
    report.append("Before event: K1/workspace preparation and synchronization; exact HSACO module and kernarg buffer prepared; 50 warmups. No state reset before event start.")
    report.append("Inside event: HIP start event, state_work.copy_(state_init), direct hipModuleLaunchKernel of the same HSACO, HIP stop event.")
    report.append("After event: synchronize stop event and read elapsed time.")
    report.append("")
    report.append("## DISCREPANCY_ROOT_CAUSE")
    report.append("")
    report.append("Same T, state, prepared workspace, config, HSACO, process, and default stream were measured with three placements. C68-minus-canonical measures the in-event Triton host-dispatch enqueue gap; D70-minus-canonical measures the in-event state-copy contribution. These conclusions use the paired measurements below.")
    report.append("")
    report.append("| T | C68 median us | D70 median us | Canonical median us | C68 - canonical us | D70 - canonical us | C68 vs canonical % | D70 vs canonical % |")
    report.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for T in T_VALUES:
        s = boundary_summaries[str(T)]
        report.append(
            f"| {T} | {s['C68']['median_us']:.4f} | {s['D70']['median_us']:.4f} | {s['CANONICAL']['median_us']:.4f} | "
            f"{s['delta_vs_canonical_us']['C68']:.4f} | {s['delta_vs_canonical_us']['D70']:.4f} | "
            f"{s['delta_vs_canonical_pct']['C68']:.2f}% | {s['delta_vs_canonical_pct']['D70']:.2f}% |"
        )
    report.append("")
    report.append("## CANONICAL_K2_TIMING_BOUNDARY")
    report.append("")
    report.append("K1/workspace preparation once per T, outside all events; compile/cache and HIP module load outside events; 50 K2 warmups; before each sample restore the same initial state on default stream; event contains direct launch of only the K2 HSACO; stop-event synchronization follows. Timing uses 100 samples per candidate per round and 3 interleaved rounds. K2 us/chunk = total K2 GPU time / ceil(T/32).")
    report.append("")
    report.append("## CONFIRMED_TRUE_BEST_MAP")
    report.append("")
    report.append("| T | BW/W/S | HSACO SHA256 | total K2 median us | us/chunk | second best | second us | gap % |")
    report.append("|---:|---|---|---:|---:|---|---:|---:|")
    for T in T_VALUES:
        r = rows_by_T[str(T)]
        b, s = r["winner"], r["second"]
        cfg = b["config"]
        report.append(
            f"| {T} | {cfg['BW']}/{cfg['W']}/{cfg['S']} | {b['hsaco_sha256']} | "
            f"{b['median_us']:.4f} | {b['us_per_chunk']:.4f} | {s['config_key']} | "
            f"{s['median_us']:.4f} | {r['second_gap_percent']:.2f}% |"
        )
    report.append("")
    report.append("## CHANGES_VS_C68")
    report.append("")
    all_same = all(confirmed[str(T)]["config_map_confirmed"] for T in T_VALUES)
    report.append("C68 CONFIG MAP confirmed; only the absolute timing golden changed." if all_same else "C68 CONFIG MAP changed for one or more T; see the row-by-row controlled ranking and identity map.")
    report.append("")
    report.append("| T | C68 config | C68 historical median us | canonical winner | canonical median us | winner identity matches frozen |")
    report.append("|---:|---|---:|---|---:|---|")
    for T in T_VALUES:
        old = old_c68_map[str(T)]
        new = rows_by_T[str(T)]["winner"]
        report.append(
            f"| {T} | {old['config_key']} | {old['median_us']:.4f} | {new['config_key']} | "
            f"{new['median_us']:.4f} | {confirmed[str(T)]['winner_identity_matches_frozen']} |"
        )
    report.append("")
    report.append("Raw per-candidate samples, correctness metrics, identity manifests, and generated CSV are saved beside this report. Existing C68 and D70 artifacts were not edited.")
    (OUT / "D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION.md").write_text("\n".join(report) + "\n")
    print("D71 COMPLETE", flush=True)
    print(f"REPORT={OUT / 'D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION.md'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, hashlib, subprocess, platform, socket, itertools, random, statistics
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/workspace/project/avelang")
OUT = ROOT / "test/examples/kda/reports/D_20260927_73_AITER_PREFILL_THREE_WAY_TRUE_BEST"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "objects").mkdir(exist_ok=True)
os.environ["CHUNK_DELTA_ATTN_TRITON_AUTOTUNE"] = "0"
os.environ["CHUNK_DELTA_ATTN_USE_FLASH_KDA"] = "1"
os.environ["TRITON_CACHE_DIR"] = str(OUT / "triton_cache")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(ROOT / "kda_baseline"))
sys.path.insert(0, str(ROOT / "third_party/aiter"))
sys.path.insert(0, str(ROOT / "test/examples/kda/reports/C_20260925_68_AITER_SHORT_TRUE_BEST_ENVELOPE"))

import numpy as np
import torch
import triton
from bench_common import make_kda_prefill_inputs
import run_sweep as c68
import aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda as flash
import aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_fwd as chunk_fwd
from aiter.ops.triton.kimi_delta_attn.chunk_delta_attn import chunk_kimi_delta_attn

TS = (32, 64, 127, 128, 256, 384, 511, 512, 1024, 2048, 4096)
K1_CFGS = [{"W": w, "S": s} for w in (2, 4) for s in (1, 2)]
K2_CFGS = [{"BW": bw, "W": w, "S": s, "num_ctas": 1}
           for bw in (16, 32, 64) for w in (2, 4) for s in (1, 2)]
SCREEN_WARMUPS = 10
SCREEN_REPEATS = 10
SCREEN_ROUNDS = 3
CONFIRM_WARMUPS = 50
CONFIRM_REPEATS = 100
CONFIRM_ROUNDS = 3
PATH_WARMUPS = 50
PATH_REPEATS = 100
PATH_ROUNDS = 3
TOL = 0.03
EXPECTED_COMMIT = "7bb44998274fa679ece0a37bf8426ab780b1837d"
K1_JIT = flash._flash_kda_prepare_kernel.fn.fn
K2_AUTOTUNER = flash._flash_kda_segment_kernel.fn
K2_JIT = K2_AUTOTUNER.fn

def dump(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

def sha_bytes(b):
    return hashlib.sha256(bytes(b)).hexdigest()

def config_key1(c):
    return f"W{c['W']}_S{c['S']}"

def config_key2(c):
    return f"BW{c['BW']}_W{c['W']}_S{c['S']}"

def md(x):
    return getattr(x, "metadata", None)

def record_kernel(kind, T, cfg, kernel, grid, extra=None):
    if kernel is None:
        raise RuntimeError(f"missing compiled kernel for {kind} T={T} cfg={cfg}")
    asm = getattr(kernel, "asm", {}) or {}
    hsaco = asm.get("hsaco")
    if not hsaco:
        raise RuntimeError(f"missing HSACO from {kind} T={T} cfg={cfg}")
    digest = sha_bytes(hsaco)
    m = md(kernel)
    nw = int(getattr(m, "num_warps", cfg.get("W", 0)))
    ns = int(getattr(m, "num_stages", cfg.get("S", 0)))
    shared = int(getattr(m, "shared", 0))
    triton_hash = str(getattr(kernel, "hash", "") or getattr(m, "hash", ""))
    outpath = OUT / "objects" / (digest + ".hsaco")
    if not outpath.exists():
        outpath.write_bytes(bytes(hsaco))
    isa = asm.get("amdgcn")
    if isa:
        (OUT / "objects" / (digest + ".isa")).write_text(str(isa))
    info = {
        "kernel_symbol": kind,
        "T": int(T),
        "config": dict(cfg),
        "grid": [int(x) for x in grid],
        "block": nw * 64,
        "num_warps": nw,
        "num_stages": ns,
        "num_ctas": int(cfg.get("num_ctas", 1)),
        "dynamic_shared_bytes": shared,
        "hsaco_path": str(outpath),
        "hsaco_sha256": digest,
        "triton_metadata_hash": triton_hash,
        "workspace_abi_id": ABI_ID,
    }
    if extra:
        info.update(extra)
    meta_path = outpath.with_suffix(".json")
    if not meta_path.exists():
        meta_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info

def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "median_us": float(np.percentile(a, 50)),
        "mean_us": float(np.mean(a)),
        "p25_us": float(np.percentile(a, 25)),
        "p75_us": float(np.percentile(a, 75)),
        "p95_us": float(np.percentile(a, 95)),
        "min_us": float(np.min(a)),
        "max_us": float(np.max(a)),
        "mad_us": float(np.median(np.abs(a - np.median(a)))),
        "stddev_us": float(np.std(a)),
        "n": int(a.size),
    }

def metrics(actual, expected):
    d = (actual.float() - expected.float()).abs()
    rel = d / expected.float().abs().clamp_min(1.0)
    return {
        "max_abs": float(d.max().item()),
        "max_rel": float(rel.max().item()),
        "mean_abs": float(d.mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }

def compare_pair(actual, expected):
    om = metrics(actual[0], expected[0])
    sm = metrics(actual[1], expected[1])
    return {
        "output": om,
        "final_state": sm,
        "pass": bool(om["finite"] and sm["finite"] and
                     om["max_abs"] <= TOL and sm["max_abs"] <= TOL),
        "tolerance": {"max_abs": TOL, "basis": "D71 controlled K2 gate"},
    }

def workspace_abi(nt):
    return {
        "layout": [
            {"name": "ws_kd", "shape": [12*nt,32,128], "dtype": "bf16", "contiguous": True},
            {"name": "ws_qd", "shape": [12*nt,32,128], "dtype": "bf16", "contiguous": True},
            {"name": "ws_kr", "shape": [12*nt,32,128], "dtype": "bf16", "contiguous": True},
            {"name": "ws_gt", "shape": [12*nt,128], "dtype": "fp32", "contiguous": True},
            {"name": "ws_inv_mqk", "shape": [12*nt,64,32], "dtype": "fp16", "contiguous": True},
        ],
        "producer": "_flash_kda_prepare_kernel",
        "consumer": "_flash_kda_segment_kernel",
        "nt": int(nt),
        "descriptor": "one segment covering all NT chunks for controlled two-kernel Prefill",
        "state_layout": "[B,H,K,V] fp32 input; final_state same layout/dtype",
    }

ABI_DESC = {
    "K1_outputs": [
        ("ws_kd", "bf16", "[H*NT,C,K]"),
        ("ws_qd", "bf16", "[H*NT,C,K]"),
        ("ws_kr", "bf16", "[H*NT,C,K]"),
        ("ws_gt", "fp32", "[H*NT,K]"),
        ("ws_inv_mqk", "fp16", "[H*NT,2*C,C]"),
    ],
    "K2_reads": ["ws_kd", "ws_qd", "ws_kr", "ws_gt", "ws_inv_mqk"],
    "workload": {"B": 1, "H": 12, "K": 128, "V": 128, "C": 32},
    "layout": "contiguous",
}
ABI_ID = hashlib.sha256(json.dumps(ABI_DESC, sort_keys=True).encode()).hexdigest()

def run_k1(T, inp, ws, cfg):
    nt = triton.cdiv(T, 32)
    beta = inp["raw_beta"].float().contiguous()
    return K1_JIT[(nt, 12)](
        q=inp["q"], k=inp["k"], g_raw=inp["raw_g"], beta_raw=beta,
        A_log=inp["A_log"], dt_bias=inp["dt_bias"].view(-1),
        ws_kd=ws["kd"], ws_qd=ws["qd"], ws_kr=ws["kr"],
        ws_gt=ws["gt"], ws_inv_mqk=ws["inv_mqk"],
        cu_seqlens=None, chunk_indices=None,
        scale=128**-0.5, lower_bound=-5.0,
        T=int(T), NT=int(nt), TOTAL_TILES=int(nt),
        H=12, K=128, C=32, BC=16, NUM_DOUBLING=3, NUM_MERGE=1,
        IS_VARLEN=False, HAS_BIAS=True,
        num_warps=int(cfg["W"]), num_stages=int(cfg["S"]), num_ctas=1,
    )

def call_public(inp, state):
    return chunk_kimi_delta_attn(
        inp["q"], inp["k"], inp["v"], inp["raw_g"], inp["raw_beta"],
        A_log=inp["A_log"], dt_bias=inp["dt_bias"].view(-1),
        initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    )

def capture_public(T, inp, state):
    records = []
    old1, old2 = K1_JIT.run, K2_JIT.run
    oldscan = None
    scan_obj = getattr(flash, "_flash_kda_seg_scan_kernel", None)
    scan_jit = scan_obj
    if hasattr(scan_obj, "fn"):
        scan_jit = scan_obj.fn
        if hasattr(scan_jit, "fn"):
            scan_jit = scan_jit.fn
    if scan_jit is not None and hasattr(scan_jit, "run"):
        oldscan = scan_jit.run

    def wrap(kind, original):
        def tracked(*args, **kwargs):
            kernel = original(*args, **kwargs)
            cfg = {
                "BW": kwargs.get("BW"),
                "W": int(getattr(md(kernel), "num_warps", 0)),
                "S": int(getattr(md(kernel), "num_stages", 0)),
                "num_ctas": int(kwargs.get("num_ctas", 1)),
            }
            for key in ("W", "HAS_V", "COMPUTE_OUTPUT", "STORE_H_OUT",
                        "STORE_FINAL", "INIT_IDENTITY", "NUM_SEGS_CLASS"):
                if key in kwargs:
                    cfg[key] = kwargs[key]
            grid = kwargs.get("grid", None)
            if callable(grid):
                try:
                    grid = grid(md(kernel))
                except Exception:
                    grid = repr(grid)
            if isinstance(grid, int):
                grid = [grid, 1, 1]
            elif isinstance(grid, tuple):
                grid = list(grid) + [1] * (3-len(grid))
            elif isinstance(grid, list):
                grid = grid + [1] * (3-len(grid))
            else:
                grid = repr(grid)
            rec = record_kernel(kind, T, cfg, kernel,
                                grid if isinstance(grid, list) else [0,0,0],
                                extra={"captured_grid_raw": grid})
            rec["launch_kwargs"] = {
                k: (v if isinstance(v, (str, int, float, bool)) or v is None else repr(v))
                for k, v in kwargs.items()
                if k in ("BW", "W", "HAS_V", "COMPUTE_OUTPUT", "STORE_H_OUT",
                         "STORE_FINAL", "INIT_IDENTITY", "NUM_SEGS_CLASS",
                         "num_warps", "num_stages", "num_ctas")
            }
            records.append(rec)
            return kernel
        return tracked
    try:
        K1_JIT.run = wrap("_flash_kda_prepare_kernel", old1)
        K2_JIT.run = wrap("_flash_kda_segment_kernel", old2)
        if oldscan is not None:
            scan_jit.run = wrap("_flash_kda_seg_scan_kernel", oldscan)
        got = call_public(inp, state)
        torch.cuda.synchronize()
    finally:
        K1_JIT.run = old1
        K2_JIT.run = old2
        if oldscan is not None:
            scan_jit.run = oldscan
    return got, records

def event_time_chain(T, inp, ws, case, k1cfg, k2cfg, start, stop):
    case["reset"]()
    start.record()
    run_k1(T, inp, ws, k1cfg)
    case["invoke"](k2cfg, warmup=False)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0)

def event_time_k1(T, inp, ws, cfg, start, stop):
    run_k1(T, inp, ws, cfg)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0)

def event_time_k2(case, cfg, start, stop):
    case["invoke"](cfg, warmup=False)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0)

def public_event(inp, state_work, state, start, stop):
    state_work.copy_(state)
    start.record()
    out = call_public(inp, state_work)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0), out

def run():
    actual_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/aiter"), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"STOP identity gate: AITER commit {actual_commit}")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    if "MI300X" not in device_name.upper():
        raise RuntimeError(f"STOP identity gate: wrong GPU {device_name}")
    props = torch.cuda.get_device_properties(0)
    stream = int(torch.cuda.current_stream().cuda_stream)
    if (c68.BATCH,c68.HEADS,c68.K,c68.V,c68.C) != (1,12,128,128,32):
        raise RuntimeError("STOP identity gate: wrong workload in C68 helper")
    source_k2 = [{"BW": int(c.kwargs["BW"]), "W": int(c.num_warps),
                  "S": int(c.num_stages), "num_ctas": int(c.num_ctas)}
                 for c in flash._K2_CONFIGS]
    if {config_key2(c) for c in source_k2} != {config_key2(c) for c in K2_CFGS}:
        raise RuntimeError(f"STOP source candidate mismatch: {source_k2}")
    default_k1 = [{"W":int(c.num_warps),"S":int(c.num_stages)}
                  for c in flash._flash_kda_prepare_kernel.fn.configs]
    default_k2 = [{"BW":int(c.kwargs.get("BW",-1)),"W":int(c.num_warps),
                   "S":int(c.num_stages),"num_ctas":int(c.num_ctas)}
                  for c in K2_AUTOTUNER.configs]
    if {config_key1(c) for c in K1_CFGS} - {config_key1(c) for c in default_k1}:
        raise RuntimeError(f"STOP formal K1 source candidate set mismatch: {default_k1}")
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "container": "ljd_sglang_kda",
        "repo_head": subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        "aiter_commit": actual_commit, "gpu": device_name,
        "gfx": "gfx942", "gpu_properties": str(props),
        "torch": torch.__version__, "hip": getattr(torch.version,"hip",None),
        "python": platform.python_version(), "default_stream_handle": stream,
        "workload": {"B":1,"H":12,"K":128,"V":128,"C":32,"T":list(TS)},
        "env": {k:os.environ.get(k) for k in ("CHUNK_DELTA_ATTN_TRITON_AUTOTUNE","CHUNK_DELTA_ATTN_USE_FLASH_KDA","TRITON_CACHE_DIR")},
        "source_defined_K1": K1_CFGS, "source_defined_K2": source_k2,
        "AITER_PUBLIC_default_K1_configs": default_k1,
        "AITER_PUBLIC_default_K2_configs": default_k2,
        "workspace_abi": ABI_DESC, "workspace_abi_id": ABI_ID,
        "protocol": {
            "screen":{"warmups":SCREEN_WARMUPS,"repeats_per_round":SCREEN_REPEATS,"rounds":SCREEN_ROUNDS},
            "winner_confirmation":{"warmups_per_candidate_per_round":CONFIRM_WARMUPS,"repeats_per_round":CONFIRM_REPEATS,"rounds":CONFIRM_ROUNDS},
            "three_path":{"warmups_per_path_per_round":PATH_WARMUPS,"repeats_per_round":PATH_REPEATS,"rounds":PATH_ROUNDS},
            "event_boundary":"reset state outside event; event contains K1 then K2 on same default stream, no host sync between launches",
        },
        "correctness_gate":{"max_abs":TOL,"reference":"same T, seed, public AITER semantic output/state; additionally compare public FlashKDA to default AITER pipeline"},
    }
    dump("environment_manifest.json", env)
    print("D73 IDENTITY_GATE PASS", flush=True)

    c69 = json.loads((ROOT/"test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_true_best_results_final.json").read_text())
    k1map = {int(r["T"]): r["true_best"] for r in c69["public_vs_true_best"]}
    d71 = ROOT/"test/examples/kda/reports/D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION"
    k2cand = json.loads((d71/"candidate_manifest.json").read_text())
    k2best = json.loads((d71/"confirmed_true_best_map.json").read_text())
    d72 = json.loads((ROOT/"test/examples/kda/reports/D_20260927_72_K2_T32_TRUE_BEST_RESOURCE_GOLDEN/D_20260927_72_K2_T32_TRUE_BEST_RESOURCE_GOLDEN.json").read_text())
    k2map = {}
    for T in TS:
        if T == 32:
            ident = d72["identity"]
            cfg = {"BW":16,"W":2,"S":2,"num_ctas":1}
            sha = ident["hsaco_sha256"]
            t_hash = ident.get("triton_hash") or ident.get("triton_metadata_hash")
            grid, block = [8,12,1], 128
        else:
            row = k2best[str(T)]
            key = row["canonical_winner_config"]
            cand = k2cand[str(T)][key]
            cfg = cand["config"]
            sha = cand["sha256"]
            t_hash = cand.get("cache_hash")
            grid, block = cand["grid"], cand["block"]
        k2map[T] = {"config":cfg,"hsaco_sha256":sha,"triton_metadata_hash":t_hash,
                    "grid":grid,"block":block,"source":"D72" if T==32 else "D71"}
    standalone = {}
    for T in TS:
        k1 = k1map[T]
        k1cfgs = str(k1["config"]).lower().replace("-","_")
        m = __import__("re").search(r"w([24])_s([12])", k1cfgs)
        if not m: raise RuntimeError(f"unparseable K1 winner T={T}: {k1}")
        standalone[T] = {
            "K1":{"W":int(m.group(1)),"S":int(m.group(2)),
                  "hsaco_sha256":k1["hsaco_sha256"],"triton_metadata_hash":k1["triton_hash"],
                  "grid":k1["grid"],"block":k1["block"],"source":"C69/C71"},
            "K2":k2map[T],
        }
    dump("standalone_winner_source_maps.json", standalone)

    results = {}
    candidate_manifest = {}
    correctness_all = {}
    public_all = {}
    path_raw = {}
    k1k2_raw = {}
    for T in TS:
        print(f"D73 START T={T}", flush=True)
        torch.manual_seed(91000 + T)
        inp = make_kda_prefill_inputs(T=T, H=12, seed=6800+T)
        state = torch.randn((1,12,128,128), device="cuda", dtype=torch.float32).mul_(0.1).contiguous()
        nt = triton.cdiv(T,32)
        ws = c68.allocate_workspace(nt, inp["q"].device)
        case = c68.make_case(inp, ws, state)
        state_work = case["alive"][3]
        beta = inp["raw_beta"].float().contiguous()

        # Public dispatcher warm call resolves its default config; trace is then separate.
        flash.CHUNK_DELTA_ATTN_USE_FLASH_KDA = True
        chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA = True
        public_first = call_public(inp,state)
        torch.cuda.synchronize()
        pgot, ptrace = capture_public(T,inp,state)
        public_out = (pgot[0].clone(), pgot[1].clone() if pgot[1] is not None else None)
        if public_out[1] is None:
            raise RuntimeError(f"STOP public correctness: no final state T={T}")
        # Compare the public FlashKDA route to AITER's default semantic pipeline.
        chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA = False
        reference = call_public(inp,state)
        torch.cuda.synchronize()
        ref = (reference[0].clone(), reference[1].clone() if reference[1] is not None else None)
        chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA = True
        public_correct = compare_pair(public_out,ref)
        if not public_correct["pass"]:
            print(f"STOP public semantic correctness T={T}: {public_correct}",flush=True)
        public_all[str(T)] = {
            "path":"AITER_PUBLIC_PREFILL","entrypoint":"chunk_kimi_delta_attn",
            "config_forcing":"none","correctness_vs_default_AITER_pipeline":public_correct,
            "launches":ptrace,
            "default_pipeline_oracle":"CHUNK_DELTA_ATTN_USE_FLASH_KDA=0, same API options and inputs",
        }
        dump("aiter_public_provenance.json", public_all)
        if not public_correct["pass"]:
            results[str(T)] = {"status":"STOP_PUBLIC_CORRECTNESS","public":public_all[str(T)]}
            continue

        # Compile/execute every formal K1 candidate and every legal K2 candidate.
        k1objects, k2objects = {}, {}
        for c1 in K1_CFGS:
            case["reset"]()
            k = run_k1(T,inp,ws,c1)
            torch.cuda.synchronize()
            grid=(nt,12,1)
            rec=record_kernel("_flash_kda_prepare_kernel",T,c1,k,grid)
            k1objects[config_key1(c1)] = rec
        for c2 in K2_CFGS:
            case["reset"]()
            # K2 compile/launch consumes workspace produced by the exact chosen K1 below.
            k1c = standalone[T]["K1"]
            k1cfg={"W":k1c["W"],"S":k1c["S"]}
            run_k1(T,inp,ws,k1cfg)
            k = case["invoke"](c2,warmup=False)
            torch.cuda.synchronize()
            grid=(triton.cdiv(128,c2["BW"]),case["segments"]*12,1)
            rec=record_kernel("_flash_kda_segment_kernel",T,c2,k,grid,
                              extra={"has_v":True,"compute_output":True,"num_segments":case["segments"]})
            k2objects[config_key2(c2)] = rec

        # Strictly validate the latest standalone winners before using factorized pair.
        k1expected=standalone[T]["K1"]; k2expected=standalone[T]["K2"]
        k1actual=k1objects[config_key1(k1expected)]
        k2actual=k2objects[config_key2(k2expected["config"])]
        identity_checks={
            "K1":{"config_match":True,"expected_sha256":k1expected["hsaco_sha256"],
                  "actual_sha256":k1actual["hsaco_sha256"],
                  "sha_match":k1actual["hsaco_sha256"]==k1expected["hsaco_sha256"],
                  "expected_triton_hash":k1expected["triton_metadata_hash"],
                  "actual_triton_hash":k1actual["triton_metadata_hash"]},
            "K2":{"config_match":True,"expected_sha256":k2expected["hsaco_sha256"],
                  "actual_sha256":k2actual["hsaco_sha256"],
                  "sha_match":k2actual["hsaco_sha256"]==k2expected["hsaco_sha256"],
                  "expected_triton_hash":k2expected["triton_metadata_hash"],
                  "actual_triton_hash":k2actual["triton_metadata_hash"],
                  "grid_match":k2actual["grid"]==k2expected["grid"],
                  "block_match":k2actual["block"]==k2expected["block"]},
        }
        identity_pass=identity_checks["K1"]["sha_match"] and identity_checks["K2"]["sha_match"] and identity_checks["K2"]["grid_match"] and identity_checks["K2"]["block_match"]
        if not identity_pass:
            print(f"STOP standalone identity mismatch T={T}: {identity_checks}",flush=True)
            results[str(T)]={"status":"STOP_STANDALONE_IDENTITY_MISMATCH","identity_checks":identity_checks}
            candidate_manifest[str(T)]={"K1":k1objects,"K2":k2objects}
            dump("candidate_manifest.json",candidate_manifest)
            dump("identity_checks.json",{**json.loads((OUT/"identity_checks.json").read_text()) if (OUT/"identity_checks.json").exists() else {},str(T):identity_checks})
            continue

        # Gate every compatible K1 x K2 chain against the same public semantic oracle.
        pairs=[]
        rows=[]
        for c1 in K1_CFGS:
            for c2 in K2_CFGS:
                label=config_key1(c1)+"__"+config_key2(c2)
                try:
                    case["reset"]()
                    k1=run_k1(T,inp,ws,c1)
                    k2=case["invoke"](c2,warmup=False)
                    torch.cuda.synchronize()
                    actual=(case["out"].clone(),case["final_state"].clone())
                    gate=compare_pair(actual,public_out)
                    row={"pair":label,"K1":k1objects[config_key1(c1)],"K2":k2objects[config_key2(c2)],
                         "correctness_vs_public":gate,"valid":gate["pass"]}
                    if gate["pass"]: pairs.append((dict(c1),dict(c2),label))
                    rows.append(row)
                except Exception as e:
                    rows.append({"pair":label,"K1":k1objects.get(config_key1(c1)),
                                 "K2":k2objects.get(config_key2(c2)),"valid":False,
                                 "invalid_reason":repr(e)})
        correctness_all[str(T)]={"T":T,"reference":"AITER_PUBLIC_PREFILL output + final_state",
                                 "workspace_abi_id":ABI_ID,"candidate_pairs":rows}
        candidate_manifest[str(T)]={"K1":k1objects,"K2":k2objects}
        dump("candidate_manifest.json",candidate_manifest)
        dump("correctness.json",correctness_all)
        validlabels={x[2] for x in pairs}
        print(f"D73 CORRECTNESS T={T} valid_pairs={len(pairs)}/48",flush=True)
        if not pairs:
            results[str(T)]={"status":"STOP_NO_VALID_JOINT_PAIR","identity_checks":identity_checks}
            continue

        # Identical screen schedule for every valid pair, interleaved per repeat.
        start=torch.cuda.Event(enable_timing=True); stop=torch.cuda.Event(enable_timing=True)
        for c1,c2,label in pairs:
            for _ in range(SCREEN_WARMUPS):
                case["reset"](); run_k1(T,inp,ws,c1); case["invoke"](c2,warmup=False)
        torch.cuda.synchronize()
        screen={x[2]:[] for x in pairs}
        for rnd in range(SCREEN_ROUNDS):
            order=list(pairs); random.Random(7300+T+rnd).shuffle(order)
            for rep in range(SCREEN_REPEATS):
                for c1,c2,label in order:
                    screen[label].append(event_time_chain(T,inp,ws,case,c1,c2,start,stop))
        screen_stats={label:stats(v) for label,v in screen.items()}
        ranked=sorted(screen_stats,key=lambda x:screen_stats[x]["median_us"])
        cutoff=screen_stats[ranked[0]]["median_us"]*1.02
        contenders=[x for x in ranked if screen_stats[x]["median_us"]<=cutoff]
        for x in ranked[:2]:
            if x not in contenders:contenders.append(x)
        # Winner confirmation: every round has 50 warmups and 100 measured repeats.
        confirm={label:[] for label in contenders}; confirm_rounds={}
        for rnd in range(CONFIRM_ROUNDS):
            order=[next(p for p in pairs if p[2]==label) for label in contenders]
            if rnd%2: order.reverse()
            else: order=order[rnd%len(order):]+order[:rnd%len(order)]
            for c1,c2,label in order:
                for _ in range(CONFIRM_WARMUPS):
                    case["reset"](); run_k1(T,inp,ws,c1); case["invoke"](c2,warmup=False)
                torch.cuda.synchronize()
                vals=[]
                for _ in range(CONFIRM_REPEATS):
                    vals.append(event_time_chain(T,inp,ws,case,c1,c2,start,stop))
                confirm[label].extend(vals)
                confirm_rounds.setdefault(label,[]).append(stats(vals))
        cstats={label:stats(v) for label,v in confirm.items()}
        ranked_confirm=sorted(cstats,key=lambda x:cstats[x]["median_us"])
        winner=ranked_confirm[0]
        winner_pair=next(p for p in pairs if p[2]==winner)
        runner=ranked_confirm[1] if len(ranked_confirm)>1 else None
        joint={
            "T":T,"status":"PASS","screen":screen_stats,
            "screen_order":ranked,"confirmation_candidates":contenders,
            "confirmation":{k:{"pooled":cstats[k],"per_round":confirm_rounds[k]} for k in confirm},
            "winner":winner,"winner_config":{"K1":winner_pair[0],"K2":winner_pair[1]},
            "winner_latency_us":cstats[winner]["median_us"],
            "runner_up":runner,
            "runner_up_latency_us":cstats[runner]["median_us"] if runner else None,
            "runner_up_gap_pct":((cstats[runner]["median_us"]/cstats[winner]["median_us"]-1)*100) if runner else None,
            "valid_pair_count":len(pairs),
            "identity_checks":identity_checks,
        }
        # Factorized pair is fixed by standalone source maps; its controlled latency comes from measured pair data.
        fact_label=config_key1({"W":k1expected["W"],"S":k1expected["S"]})+"__"+config_key2(k2expected["config"])
        fact_screen=screen_stats.get(fact_label)
        fact_confirm=cstats.get(fact_label)
        if fact_confirm is None:
            # It was screened but outside 2%; still confirm this required path exactly.
            c1={"W":k1expected["W"],"S":k1expected["S"]}; c2=k2expected["config"]
            vals=[]; rounds=[]
            for rnd in range(CONFIRM_ROUNDS):
                for _ in range(CONFIRM_WARMUPS):
                    case["reset"]();run_k1(T,inp,ws,c1);case["invoke"](c2,warmup=False)
                torch.cuda.synchronize(); one=[]
                for _ in range(CONFIRM_REPEATS):one.append(event_time_chain(T,inp,ws,case,c1,c2,start,stop))
                vals.extend(one);rounds.append(stats(one))
            confirm[fact_label]=vals; confirm_rounds[fact_label]=rounds; cstats[fact_label]=stats(vals)
            fact_confirm=cstats[fact_label]
        factorized={
            "label":fact_label,"K1":k1expected,"K2":k2expected,
            "screen":fact_screen,"confirmation":fact_confirm,
        }
        same_pair=(fact_label==winner)
        factorized["vs_joint"]={
            "same_pair":same_pair,
            "factorized_e2e_us":fact_confirm["median_us"],
            "joint_e2e_us":cstats[winner]["median_us"],
            "delta_us":fact_confirm["median_us"]-cstats[winner]["median_us"],
            "delta_pct":(fact_confirm["median_us"]/cstats[winner]["median_us"]-1)*100,
            "K1_changed":not same_pair and winner_pair[0]!=factorized["K1"],
            "K2_changed":not same_pair and winner_pair[1]!=factorized["K2"]["config"],
        }
        # Store interaction diagnostic against separately measured standalone kernel time later.
        results[str(T)]={"status":"SWEEP_PASS","identity_checks":identity_checks,
                         "joint":joint,"factorized":factorized}
        k1k2_raw[str(T)]={"T":T,"screen_samples_us":screen,"confirmation_samples_us":confirm,
                           "confirmation_rounds":confirm_rounds}
        dump("joint_sweep_raw.json",k1k2_raw)
        dump("correctness.json",correctness_all)
        dump("candidate_manifest.json",candidate_manifest)
        dump("joint_results.json",results)
        print(f"D73 JOINT T={T} winner={winner} median={cstats[winner]['median_us']:.4f} us factored={fact_label}",flush=True)

        # Measure all three principal paths on same stream/inputs and canonical event boundary.
        k1joint=winner_pair[0]; k2joint=winner_pair[1]
        path_samples={"AITER_PUBLIC_PREFILL":[],"AITER_FACTORIZED_BEST_PREFILL":[],"AITER_TRUE_BEST_PREFILL":[]}
        path_rounds={k:[] for k in path_samples}
        # warm each path before final measurement, outside the recorded event
        path_cfg={
            "AITER_FACTORIZED_BEST_PREFILL":(dict(k1expected and {"W":k1expected["W"],"S":k1expected["S"]}),dict(k2expected["config"])),
            "AITER_TRUE_BEST_PREFILL":(dict(k1joint),dict(k2joint)),
        }
        for name,(a,b) in path_cfg.items():
            for _ in range(PATH_WARMUPS):
                case["reset"]();run_k1(T,inp,ws,a);case["invoke"](b,warmup=False)
        # public has its own allocation/setup, warm 50 times after tracing
        for _ in range(PATH_WARMUPS):
            state_work.copy_(state); call_public(inp,state_work)
        torch.cuda.synchronize()
        for rnd in range(PATH_ROUNDS):
            order=["AITER_PUBLIC_PREFILL","AITER_FACTORIZED_BEST_PREFILL","AITER_TRUE_BEST_PREFILL"]
            if rnd%2:order.reverse()
            for name in order:
                vals=[]
                for _ in range(PATH_REPEATS):
                    if name=="AITER_PUBLIC_PREFILL":
                        us,_=public_event(inp,state_work,state,start,stop)
                    else:
                        a,b=path_cfg[name]
                        us=event_time_chain(T,inp,ws,case,a,b,start,stop)
                    vals.append(us);path_samples[name].append(us)
                path_rounds[name].append(stats(vals))
        # Isolated K1 and one-segment K2 timings for the exact three path identities.
        selected_k1={}
        for name,cfg in [("PUBLIC",ptrace[0]["config"] if ptrace else {"W":0,"S":0}),
                         ("FACTORIZED",{"W":k1expected["W"],"S":k1expected["S"]}),
                         ("TRUE_BEST",dict(k1joint))]:
            if cfg.get("W",0):selected_k1[name]=cfg
        isolated={}
        for name,cfg in selected_k1.items():
            vals=[]
            for _ in range(50):run_k1(T,inp,ws,cfg)
            torch.cuda.synchronize()
            for rnd in range(3):
                one=[]
                for _ in range(100):
                    case["reset"]()
                    start.record();run_k1(T,inp,ws,cfg);stop.record();stop.synchronize()
                    one.append(float(start.elapsed_time(stop)*1000.0))
                vals.extend(one)
            isolated["K1_"+name]={"config":cfg,"stats":stats(vals)}
        for name,cfg in [("FACTORIZED",dict(k2expected["config"])),("TRUE_BEST",dict(k2joint))]:
            vals=[]
            for _ in range(50):
                case["reset"]();run_k1(T,inp,ws,k1expected if name=="FACTORIZED" else k1joint);case["invoke"](cfg,warmup=False)
            torch.cuda.synchronize()
            for rnd in range(3):
                one=[]
                for _ in range(100):
                    case["reset"]();run_k1(T,inp,ws,k1expected if name=="FACTORIZED" else k1joint)
                    start.record();case["invoke"](cfg,warmup=False);stop.record();stop.synchronize()
                    one.append(float(start.elapsed_time(stop)*1000.0))
                vals.extend(one)
            isolated["K2_"+name]={"config":cfg,"stats":stats(vals),"timed":"one-segment direct K2"}
        public_launches=ptrace
        path_raw[str(T)]={
            "samples_us":path_samples,"round_medians":path_rounds,
            "pooled":{k:stats(v) for k,v in path_samples.items()},
            "K1_K2_standalone":isolated,
            "public_launches":public_launches,
            "factorized_pair":fact_label,"joint_pair":winner,
            "factorized_joint_same_pair":same_pair,
        }
        dump("three_path_raw.json",path_raw)
        dump("joint_results.json",results)
        print(f"D73 THREE_PATH T={T} public={stats(path_samples['AITER_PUBLIC_PREFILL'])['median_us']:.4f} fact={stats(path_samples['AITER_FACTORIZED_BEST_PREFILL'])['median_us']:.4f} joint={stats(path_samples['AITER_TRUE_BEST_PREFILL'])['median_us']:.4f} us",flush=True)

    dump("correctness.json",correctness_all)
    dump("candidate_manifest.json",candidate_manifest)
    dump("aiter_public_provenance.json",public_all)
    dump("joint_sweep_raw.json",k1k2_raw)
    dump("joint_results.json",results)
    dump("three_path_raw.json",path_raw)
    print("D73 PHASES 0-5 COMPLETE",flush=True)

if __name__ == "__main__":
    run()

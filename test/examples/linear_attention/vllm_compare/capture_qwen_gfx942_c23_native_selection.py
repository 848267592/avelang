#!/usr/bin/env python3
"""Fresh public-selector identity capture for the C23 native control."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be a positive multiple of 64")

    out = args.out_dir.resolve()
    cache = out / "triton_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)

    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    stage2 = repo / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
    sys.path[:0] = [str(here), str(stage2)]
    import torch
    from stage2_runner import make_inputs
    from vllm.model_executor.layers.fla.ops import chunk_o

    q, k, _, g, _, _ = make_inputs(args.T, 2026081200 + args.T, "random", True)
    torch.manual_seed(2026081300 + args.T)
    v_new = (torch.randn((1, args.T, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, args.T // 64, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    tuner.cache.clear()
    output = chunk_o.chunk_fwd_o(q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous(), chunk_size=64)
    torch.cuda.synchronize()
    selected = list(tuner.cache.values())
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one selected config, found {len(selected)}")
    config = selected[0]
    config_json = {
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": int(getattr(config, "num_warps", -1)),
        "num_stages": int(getattr(config, "num_stages", -1)),
        "num_ctas": int(getattr(config, "num_ctas", -1)),
        "T": args.T,
        "finite": bool(torch.isfinite(output).all().item()),
        "selection_method": "fresh Python process; public chunk_fwd_o call after in-memory tuner cache.clear()",
    }
    (out / "selected_config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n")

    # A cache populated by autotuning contains many candidates with the same
    # warp/stage pair. Compile exactly the Config selected by the public call
    # into an empty cache instead of guessing from artifact mtimes.
    exact_cache = out / "selected_config_cache"
    exact_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(exact_cache)
    tuner.fn.device_caches.clear()
    exact_output = torch.empty_like(v_new)
    selected_kwargs = dict(config_json["kwargs"])
    selected_grid = ((v_new.shape[-1] + selected_kwargs["BV"] - 1) // selected_kwargs["BV"], args.T // 64, q.shape[0] * v_new.shape[-2])
    tuner.fn[selected_grid](
        q.contiguous(),
        k.contiguous(),
        v_new.contiguous(),
        h.contiguous(),
        g.contiguous(),
        exact_output,
        None,
        None,
        k.shape[-1] ** -0.5,
        T=args.T,
        H=v_new.shape[-2],
        Hg=q.shape[-2],
        K=q.shape[-1],
        V=v_new.shape[-1],
        BT=64,
        USE_G=True,
        IS_VARLEN=False,
        num_warps=config_json["num_warps"],
        num_stages=config_json["num_stages"],
        num_ctas=config_json["num_ctas"],
        **selected_kwargs,
    )
    torch.cuda.synchronize()
    hits = list(exact_cache.rglob("chunk_fwd_kernel_o.json"))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one direct-selected cache artifact, found {len(hits)}")
    candidate = hits[0].parent
    selected_dir = out / "selected"
    shutil.copytree(candidate, selected_dir, dirs_exist_ok=True)
    print(json.dumps({**config_json, "selected_dir": str(selected_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

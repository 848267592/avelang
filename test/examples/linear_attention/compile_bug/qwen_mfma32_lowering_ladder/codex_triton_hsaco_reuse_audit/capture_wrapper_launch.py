#!/usr/bin/env python3
"""Run one exact wrapper launch and save the actual pointer values."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "vllm_stageb_snapshot"))
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    torch.manual_seed(20260712)
    k = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn(1, 64, 8, 128, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(1, 64, 8, 128, device="cuda", dtype=torch.float32).contiguous()
    g = torch.randn(1, 64, 8, device="cuda", dtype=torch.float32).contiguous()
    h0 = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float32).contiguous()
    h, v_new, ht = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k, w, v, g, None, h0, True, 64, True, None
    )
    torch.cuda.synchronize()
    out.write_text(json.dumps({
        "k": hex(k.data_ptr()), "v": hex(v.data_ptr()), "w": hex(w.data_ptr()),
        "v_new": hex(v_new.data_ptr()), "g": hex(g.data_ptr()), "h": hex(h.data_ptr()),
        "h0": hex(h0.data_ptr()), "ht": hex(ht.data_ptr()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

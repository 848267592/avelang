#!/usr/bin/env python3
"""Bit-exact extracted/rebuilt long-length smoke checks against vLLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
from capture_long_sequence import call, make_inputs, patch_rocm_autotune  # noqa: E402
from run_fullseq_correctness import metrics, run_harness, write_inputs  # noqa: E402


def main() -> None:
    patch_rocm_autotune(); rows = []
    for t in (8192, 16384):
        values = make_inputs(t, 20260719 + t); expected = call(*values); torch.cuda.synchronize()
        in_dir = AUDIT / f"standalone_fullseq/long_smoke/t{t}/in"; write_inputs(in_dir, values)
        for variant in ("original", "rebuilt"):
            actual = run_harness(HERE / "fullseq_harness", AUDIT / f"golden_fullseq/shared/{variant}.hsaco", t, in_dir, AUDIT / f"standalone_fullseq/long_smoke/t{t}/{variant}")
            output = {name: metrics(got, want.cpu()) for name, got, want in zip(("h", "v_new", "final_state"), actual, expected)}
            if any(value["max_abs"] != 0 for value in output.values()): raise AssertionError(f"{t} {variant}: {output}")
            rows.append({"T": t, "variant": variant, "outputs": output})
    (AUDIT / "golden_fullseq/long_length_smoke.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()

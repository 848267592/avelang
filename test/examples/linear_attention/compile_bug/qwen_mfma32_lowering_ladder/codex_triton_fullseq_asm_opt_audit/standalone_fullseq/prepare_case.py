#!/usr/bin/env python3
"""Materialize deterministic host input binaries for the HIP full-sequence harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
from capture_long_sequence import make_inputs  # noqa: E402
from run_fullseq_correctness import write_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_inputs(args.out, make_inputs(args.T, args.seed + args.T))


if __name__ == "__main__": main()

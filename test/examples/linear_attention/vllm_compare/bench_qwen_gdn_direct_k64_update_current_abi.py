"""Body benchmark for the direct-K64 current-ABI recurrence-update repro."""

from __future__ import annotations

import argparse
import json

from repro_qwen_gdn_direct_k64_update_current_abi import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", nargs="+", type=int, default=[512, 1024, 2048, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    rows = [
        run_case(t, seed=args.seed, warmup=args.warmup, repeat=args.repeat, check=not args.no_check)
        for t in args.T
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

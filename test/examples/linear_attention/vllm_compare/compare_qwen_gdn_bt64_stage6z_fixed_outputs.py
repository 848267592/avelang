#!/usr/bin/env python3
"""Offline comparison of CPU snapshots emitted by the fresh-process driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ATOL = 1.0 / 128.0


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z2", type=Path, required=True)
    parser.add_argument("--z3", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    values = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in (("z2", args.z2), ("z3", args.z3), ("reference", args.reference))
    }
    result = {
        "z3_vs_z2_byte_exact": bool(torch.equal(values["z3"], values["z2"])),
        "z2_vs_reference_max_abs": max_abs(values["z2"], values["reference"]),
        "z3_vs_reference_max_abs": max_abs(values["z3"], values["reference"]),
        "reference_contract_pass": max_abs(values["z2"], values["reference"]) <= ATOL
        and max_abs(values["z3"], values["reference"]) <= ATOL,
        "finite": all(bool(torch.isfinite(value).all().item()) for value in values.values()),
    }
    result["pass"] = all(
        (
            result["z3_vs_z2_byte_exact"],
            result["reference_contract_pass"],
            result["finite"],
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

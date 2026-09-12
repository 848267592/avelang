#!/usr/bin/env python3
"""Flatten read-only amd-smi observations collected by Stage 5E."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def metric_value(value: object) -> float | str | None:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value if isinstance(value, (int, float, str)) else None


def nested(data: dict[str, object], *keys: str) -> object:
    value: object = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "telemetry")
    parser.add_argument("--out", type=Path, default=HERE / "clock_power_observation.csv")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.csv")):
        with path.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                try:
                    gpu = json.loads(raw["payload_json"])["gpu_data"][0]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                gfx_clocks = []
                for index in range(8):
                    value = metric_value(nested(gpu, "clock", f"gfx_{index}", "clk"))
                    if isinstance(value, (int, float)):
                        gfx_clocks.append(float(value))
                throttle = nested(gpu, "throttle")
                active = []
                if isinstance(throttle, dict):
                    active = [key for key, value in throttle.items() if key.endswith("_violation_status") and value == "ACTIVE"]
                rows.append({
                    "source": path.name,
                    "time_ns": raw["time_ns"],
                    "solve_impl": raw["solve_impl"],
                    "control": raw["control"],
                    "sequence": raw.get("sequence", "single"),
                    "gfx_activity_percent": metric_value(nested(gpu, "usage", "gfx_activity")),
                    "umc_activity_percent": metric_value(nested(gpu, "usage", "umc_activity")),
                    "socket_power_w": metric_value(nested(gpu, "power", "socket_power")),
                    "gfx_clock_mhz_median_xcp": statistics.median(gfx_clocks) if gfx_clocks else None,
                    "gfx_clock_mhz_min_xcp": min(gfx_clocks) if gfx_clocks else None,
                    "gfx_clock_mhz_max_xcp": max(gfx_clocks) if gfx_clocks else None,
                    "memory_clock_mhz": metric_value(nested(gpu, "clock", "mem_0", "clk")),
                    "hotspot_c": metric_value(nested(gpu, "temperature", "hotspot")),
                    "memory_temp_c": metric_value(nested(gpu, "temperature", "mem")),
                    "perf_level": nested(gpu, "perf_level"),
                    "active_throttle_flags": ";".join(active),
                })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "N/A"}])


if __name__ == "__main__":
    main()

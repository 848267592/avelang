#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from repro_v26_mfma_state_operand import (
    BT,
    KQ,
    NT,
    VARIANT_CONSTANT,
    VARIANT_LDS,
    VARIANT_PERSISTENT,
    VARIANT_UPDATE_ONLY,
    VARIANTS,
    run_repro_v26_mfma_state_operand,
)


def make_inputs(bv: int, seed: int):
    torch.manual_seed(seed + bv)
    # Keep values small so the artificial recurrence does not overflow while
    # still keeping data-dependent math alive for the backend.
    w = (torch.randn((NT, BT, KQ), device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    k = (torch.randn((NT, BT, KQ), device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    fake_v = (torch.randn((NT, BT, bv), device="cuda", dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    return w, k, fake_v


def time_fn(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def checksum(x: torch.Tensor) -> float:
    torch.cuda.synchronize()
    return float(x.float().abs().mean().item())


def run_case(bv: int, variant: str, args: argparse.Namespace) -> dict[str, object]:
    w, k, fake_v = make_inputs(bv, args.seed)

    def fn():
        return run_repro_v26_mfma_state_operand(
            w,
            k,
            fake_v,
            bv=bv,
            variant=variant,
            num_blocks=args.num_blocks,
        )

    out = fn()
    torch.cuda.synchronize()
    ms = time_fn(fn, args.warmup, args.repeat)
    row = {
        "bv": bv,
        "variant": variant,
        "num_blocks": args.num_blocks,
        "ms": ms,
        "checksum": checksum(out),
    }
    print(
        f"result,bv={bv},variant={variant},num_blocks={args.num_blocks},"
        f"ms={ms:.6f},checksum={row['checksum']:.8g}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bv", type=int, choices=[16, 32], nargs="*", default=[32, 16])
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=260260)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape,BT={BT},Kq={KQ},NT={NT},BV={args.bv},num_blocks={args.num_blocks}")
    print("variants=persistent,lds,constant,update_only")
    print("variant_meaning=persistent_reg_state_to_b_operand,lds_state_to_b_operand,constant_b_operand,update_only_no_pred")

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    rows = [run_case(bv, variant, args) for bv in args.bv for variant in variants]

    print("summary")
    print("BV,variant,num_blocks,ms,checksum")
    for row in rows:
        print(f"{row['bv']},{row['variant']},{row['num_blocks']},{row['ms']:.6f},{row['checksum']:.8g}")


if __name__ == "__main__":
    main()

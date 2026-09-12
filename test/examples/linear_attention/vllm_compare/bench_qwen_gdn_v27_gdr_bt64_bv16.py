#!/usr/bin/env python3
"""Chunk_gdr-only benchmark comparing v23/v24, v26 BV32, and v27 BV16."""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import qwen_gdn_gdr_decay_avelang_v17
from qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout,
)
from qwen_gdn_chunked_avelang_v26_gdr_bt64_regstate_layout_fixed import (
    MODE_FULL_V26C,
    qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout,
    qwen_gdn_chunk_gdr_torch_ref_bt64_regstate,
)
from qwen_gdn_chunked_avelang_v27_gdr_bt64_bv16_regstate_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v27_bt64_bv16_regstate_mfma_layout,
)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, with_initial_state: bool, seed: int):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    k = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    w = (torch.randn(b, t, hv, kdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    u = (torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, w, u, g, initial_state


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


def max_err(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    return max_abs, max_rel


def run_case(t: int, args: argparse.Namespace) -> dict[str, object]:
    k, w, u, g, initial_state = make_inputs(t, args.with_initial_state, seed=args.seed + t)
    print(
        f"shape,T={t},qwen_tp4_per_rank,B=1,Hk=4,Hv=8,K=128,V=128,"
        f"k={tuple(k.shape)},w={tuple(w.shape)},u={tuple(u.shape)},g={tuple(g.shape)},"
        f"initial_state={initial_state is not None}"
    )

    gdr_decay16, gdr_g_last_exp16 = qwen_gdn_gdr_decay_avelang_v17(g, chunk_size=16)

    def v23_fn():
        return qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout(
            k,
            w,
            u,
            g,
            gdr_decay16,
            gdr_g_last_exp16,
            initial_state=initial_state,
            chunk_size=16,
        )

    def v26_fn():
        return qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=64,
            variant=MODE_FULL_V26C,
        )

    def v27_fn():
        return qwen_gdn_chunk_gdr_avelang_v27_bt64_bv16_regstate_mfma_layout(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=64,
        )

    v23_fn()
    h26, vn26, fs26 = v26_fn()
    h27, vn27, fs27 = v27_fn()
    torch.cuda.synchronize()

    h26_abs = vn26_abs = fs26_abs = h27_abs = vn27_abs = fs27_abs = float("nan")
    if args.check_ref:
        h_ref, vn_ref, fs_ref = qwen_gdn_chunk_gdr_torch_ref_bt64_regstate(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=64,
        )
        h26_abs, _ = max_err(h26, h_ref)
        vn26_abs, _ = max_err(vn26, vn_ref)
        fs26_abs, _ = max_err(fs26, fs_ref)
        h27_abs, _ = max_err(h27, h_ref)
        vn27_abs, _ = max_err(vn27, vn_ref)
        fs27_abs, _ = max_err(fs27, fs_ref)

    v23_ms = time_fn(v23_fn, args.warmup, args.repeat)
    v26_ms = time_fn(v26_fn, args.warmup, args.repeat)
    v27_ms = time_fn(v27_fn, args.warmup, args.repeat)

    result = {
        "T": t,
        "v23_v24_ms": v23_ms,
        "v26_ms": v26_ms,
        "v27_ms": v27_ms,
        "v27_speedup_vs_v23": v23_ms / v27_ms,
        "v27_speedup_vs_v26": v26_ms / v27_ms,
        "h26_abs": h26_abs,
        "vn26_abs": vn26_abs,
        "fs26_abs": fs26_abs,
        "h27_abs": h27_abs,
        "vn27_abs": vn27_abs,
        "fs27_abs": fs27_abs,
    }
    print(
        f"result,T={t},v23_v24_ms={v23_ms:.6f},v26_bv32_ms={v26_ms:.6f},v27_bv16_ms={v27_ms:.6f},"
        f"v27_speedup_vs_v23={v23_ms / v27_ms:.4f},v27_speedup_vs_v26={v26_ms / v27_ms:.4f},"
        f"h26_abs={h26_abs:.6g},vn26_abs={vn26_abs:.6g},fs26_abs={fs26_abs:.6g},"
        f"h27_abs={h27_abs:.6g},vn27_abs={vn27_abs:.6g},fs27_abs={fs27_abs:.6g}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=270000)
    parser.add_argument("--without-initial-state", dest="with_initial_state", action="store_false")
    parser.add_argument("--no-check-ref", dest="check_ref", action="store_false")
    parser.set_defaults(with_initial_state=True, check_ref=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("target=Qwen3Next TP4 per-rank,B=1,Hk=4,Hv=8,K=128,V=128,BF16")
    print("benchmark=chunk_gdr_only,v23_v24=BT16/BV16,v26=BT64/BV32,v27=BT64/BV16")

    rows = [run_case(t, args) for t in args.T]

    print("summary_table")
    print("T,v23_v24_ms,v26_bv32_ms,v27_bv16_ms,v27_speedup_vs_v23,v27_speedup_vs_v26,h27_abs,vn27_abs,final27_abs")
    for r in rows:
        print(
            f"{r['T']},{r['v23_v24_ms']:.6f},{r['v26_ms']:.6f},{r['v27_ms']:.6f},"
            f"{r['v27_speedup_vs_v23']:.6f},{r['v27_speedup_vs_v26']:.6f},"
            f"{r['h27_abs']:.6g},{r['vn27_abs']:.6g},{r['fs27_abs']:.6g}"
        )


if __name__ == "__main__":
    main()

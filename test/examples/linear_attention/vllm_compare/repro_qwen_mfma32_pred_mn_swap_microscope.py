#!/usr/bin/env python3
"""Recover gfx942 MFMA32 BF16 A/B fragment and accumulator ownership.

This is deliberately one MFMA instruction, one wave, and no recurrence state.
It proves the physical source slots used by
``v_mfma_f32_32x32x8_bf16`` before they are used to repair the State-KV pred
producer.  A and B are passed as explicit [lane, bf16-slot] inputs and all
16 accumulator values/lane are returned verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import avelang
import avelang.language as al


WAVE = 64
FRAG = 4
ACC = 16


@avelang.jit
def _mfma32_pred_mn_microscope_kernel(
    a_in_ptr: al.Pointer(al.bf16),
    b_in_ptr: al.Pointer(al.bf16),
    a_seen_ptr: al.Pointer(al.bf16),
    b_seen_ptr: al.Pointer(al.bf16),
    acc_out_ptr: al.Pointer(al.f32),
):
    """One physical MFMA with direct, observable lane fragment inputs."""
    a_in = al.make_tensor(a_in_ptr, al.bf16,
        al.make_layout((WAVE, FRAG), (FRAG, 1)))
    b_in = al.make_tensor(b_in_ptr, al.bf16,
        al.make_layout((WAVE, FRAG), (FRAG, 1)))
    a_seen = al.make_tensor(a_seen_ptr, al.bf16,
        al.make_layout((WAVE, FRAG), (FRAG, 1)))
    b_seen = al.make_tensor(b_seen_ptr, al.bf16,
        al.make_layout((WAVE, FRAG), (FRAG, 1)))
    acc_out = al.make_tensor(acc_out_ptr, al.f32,
        al.make_layout((WAVE, ACC), (ACC, 1)))

    lane = al.thread_id(0)
    # The current AST frontend requires literal static dimensions here.
    a_frag = al.full((4,), 0.0, al.bf16)
    b_frag = al.full((4,), 0.0, al.bf16)
    for slot in al.range(4):
        a_frag[slot] = a_in[lane, slot]
        b_frag[slot] = b_in[lane, slot]
        a_seen[lane, slot] = a_frag[slot]
        b_seen[lane, slot] = b_frag[slot]

    acc = al.full((16,), 0.0, al.f32)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)
    for acc_i in al.range(16):
        acc_out[lane, acc_i] = acc[acc_i]


def _output_coord(lane: int, acc_i: int) -> tuple[int, int]:
    """Candidate physical accumulator map to verify with the microscope."""
    return lane & 31, (acc_i // 4) * 8 + (lane // 32) * 4 + (acc_i & 3)


def _launch(
    a: torch.Tensor, b: torch.Tensor, a_seen: torch.Tensor,
    b_seen: torch.Tensor, out: torch.Tensor,
) -> None:
    _mfma32_pred_mn_microscope_kernel[lambda: ((1, 1, 1), (WAVE, 1, 1))](
        a, b, a_seen, b_seen, out, num_warps=1)


def recover() -> dict[str, Any]:
    """Recover and validate all 256 A/B lane slots using recognizable data."""
    device = "cuda"
    a = torch.zeros((WAVE, FRAG), dtype=torch.bfloat16, device=device)
    # Each B source has a distinct exactly representable BF16 tag.
    b = torch.arange(1, WAVE * FRAG + 1, dtype=torch.float32,
                     device=device).reshape(WAVE, FRAG).to(torch.bfloat16)
    a_seen = torch.empty_like(a)
    b_seen = torch.empty_like(b)
    out = torch.empty((WAVE, ACC), dtype=torch.float32, device=device)

    _launch(a, b, a_seen, b_seen, out)
    torch.cuda.synchronize()
    echo_ok = bool(torch.equal(a, a_seen) and torch.equal(b, b_seen))
    if not echo_ok:
        raise RuntimeError("MFMA microscope fragment echo changed before the intrinsic")

    source_rows: list[dict[str, Any]] = []
    all_ok = True
    for src in range(WAVE * FRAG):
        a.zero_()
        a.view(-1)[src] = 1.0
        _launch(a, b, a_seen, b_seen, out)
        torch.cuda.synchronize()
        host = out.cpu()
        nonzero = (host.abs() > 0.5).nonzero(as_tuple=False).tolist()
        lane = src // FRAG
        slot = src % FRAG
        # The literal ROCm wrapper takes B first and A second.  With the
        # first (formal ``a``) operand activated, this source owns one B[K,N]
        # and therefore affects every M at a fixed N.
        expected_n = lane & 31
        expected_k = (lane >> 5) * 4 + slot
        by_m: dict[int, int] = {}
        cols: set[int] = set()
        for out_lane, acc_i in nonzero:
            m, n = _output_coord(out_lane, acc_i)
            cols.add(n)
            by_m[m] = int(round(float(host[out_lane, acc_i]))) - 1
        expected_a_by_m = {
            m: (m + 32 * (lane >> 5)) * FRAG + slot for m in range(32)
        }
        valid = (
            len(nonzero) == 32
            and cols == {expected_n}
            and by_m == expected_a_by_m
        )
        all_ok = all_ok and valid
        source_rows.append({
            "arg0_b_source": [lane, slot],
            "observed_n": sorted(cols),
            "observed_arg1_a_source_by_m": [by_m.get(m) for m in range(32)],
            "expected": {"n": expected_n, "k": expected_k,
                         "arg1_a_source_by_m": [expected_a_by_m[m] for m in range(32)]},
            "pass": valid,
        })

    # This formula describes every input physical slot.  It is intentionally
    # named by *argument position*, because the generic primitive's source
    # signature labels do not communicate the ROCm B/A physical order.
    return {
        "kernel": "one v_mfma_f32_32x32x8_bf16",
        "fragment_echo_ok": echo_ok,
        "all_256_a_slots_pass": all_ok,
        "arg0_owner": "arg0 is B[lane,slot] -> (k=4*(lane//32)+slot, n=lane%32)",
        "arg1_owner": "arg1 is A[lane,slot] -> (m=lane%32, k=4*(lane//32)+slot)",
        "acc_owner": "acc[lane,i] -> (m=lane%32, n=8*(i//4)+4*(lane//32)+(i%4))",
        "source_rows": source_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = recover()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "fragment_echo_ok": result["fragment_echo_ok"],
        "all_256_a_slots_pass": result["all_256_a_slots_pass"],
        "arg0_owner": result["arg0_owner"],
        "arg1_owner": result["arg1_owner"],
        "acc_owner": result["acc_owner"],
    }, indent=2))
    if not result["all_256_a_slots_pass"]:
        raise SystemExit("MFMA32 ownership formula mismatch")


if __name__ == "__main__":
    main()

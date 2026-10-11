#!/usr/bin/env python3
"""Write the B70 read-only exact layout/ownership ledger."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "test/examples/kda/reports/B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY"

MARKDOWN = r'''# B70 BW16/W4/S2 exact `#mma1` layout recovery

Status: read-only forensic recovery. No A69, A56, A68, K1, AITER source, or
production object was modified. Target: commit
`7bb44998274fa679ece0a37bf8426ab780b1837d`, pinned HSACO
`85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16`.

## MMA1_LOGICAL_MAP

The B70 Kd and INV probes compile the target transposed MFMA encoding:

```text
#mma1 = amd_mfma<version=3, warpsPerCTA=[2,2], instrShape=[16,16,16], isTransposed=true>
```

For physical wave `w=0..3`, lane `l=0..63`, and returned FP32 element
`f=0..3`:

```text
row = 16 * floor(w / 2) + (l & 15)
col = 4 * (l >> 4) + f
```

Waves 0/1 carry the first row band and waves 2/3 the second. The logical
store predicate is active only for even waves (`w & 1 == 0`); odd-wave MFMA
fragments are still computed but are not stored. Evidence: B70 Kd LLIR
lines 10-18,129-135 and ISA lines 123-159; analogous INV LLIR lines 10-23,89-97.

For `#mma` Kr/H (`warpsPerCTA=[4,1]`):

```text
row = 16 * w + (l & 15)
col = 4 * (l >> 4) + f
```

Evidence: B70 Kr LLIR lines 10-22,110-113 and ISA lines 134-146.

## ORDINARY_V_CARRIER_MAP

The ordinary-V primitive uses a four-FP32 destination bank `v[d:d+3]`; `f`
is the contiguous register index. BF16/FP16 input vectors contain four
16-bit values packed as two 32-bit VGPR words:
`(x0 | x1<<16)`, `(x2 | x3<<16)`. B67 proves literal-zero initialization,
tied continuation (`dest=C`), AGPR=0, and no inter-MFMA drain requirement.

Pinned physical chains:

| phase | ISA evidence |
|---|---|
| Kd/H1,H2 | 454-499, destinations `v[96:99]` then `v[18:21]` |
| Qd | 787-840 and 934-955 |
| INV | 576-578, `v[18:21]` then `v[26:29]` |
| Mqk | 601-603, tied `v[94:97]` |
| Kr/H | 710-724 and 1111-1117 |

Source packet ownership from pinned TTGIR:

| family | per-wave/lane packet | dot operand |
|---|---|---|
| Kd/Qd | `row=8*w+l//8`, `col=8*(l%8)+q`, `q=0..7` | `dot_op(opIdx=0,parent=#mma1,kWidth=4)` |
| INV/Mqk | `row=8*w+l//8`, `col=4*(l%8)+q`, `q=0..3` | `dot_op(opIdx=0,parent=#mma1,kWidth=4)` |
| U | `row=8*w+l//8`, `col=2*(l%8)+q`, `q=0..1` | `#mma1 -> shared4 -> dot_op(opIdx=1)` |
| Kr | `row0=8*(l%8)+q`, `col=8*w+l//8`, `q=0..7` | `dot_op(opIdx=0,parent=#mma,kWidth=4)` |

## LDS_WAVE_MAP

The 30720-byte local allocation follows the pinned TTGIR allocation order:

| byte base | role | bytes | encoding |
|---:|---|---:|---|
| 0,4096,8192,12288 | Kd1,Kd2,Qd1,Qd2 | 4096 each | `#shared`, vec4/perPhase1/maxPhase16/order[1,0] |
| 16384,18432 | INV,Mqk | 2048 each | `#shared1`, vec4/perPhase2/maxPhase8/order[1,0] |
| 20480,24576 | Kr1,Kr2 | 4096 each | `#shared2`, vec1/perPhase1/maxPhase1/order[0,1] |
| 26624 | H BF16 overlay | 2048 | `#shared3`, vec4/perPhase1/maxPhase16/order[0,1] |
| 28672 | U/overlay temporal slot | 1024+ alias | `#shared4`, vec4/perPhase2/maxPhase8/order[0,1] |

Pinned ISA representative reads/writes: `ds_read2st64_b64` at 2278,
22B4-22C4, 259C-25C0, 2AAC-2AC4, 2EC8-2EE0; `ds_read_b64/u16` at
22E8-247C, 25C8-26D4, 2B0C-2B24, 2D74-2DE8; writes at 2290-23A0,
243C,2574-258C,268C-26A4,2908-2960,2ACC-2AF0,2C08-2C20,2E24-2E84.
The frozen steady-state synchronization contract is 12 barriers and 36
waitcnt per chunk.

## PERMUTE_MAP

B70 isolated exact exchanges:

```text
Kd:  v_perm_b32 ISA lines 117-122,135-141; MFMA 123,137,145,153
INV: v_perm_b32 ISA lines 115-120; MFMA 121,128
Kr:  ds_bpermute_b32 ISA lines 96,111-113;
     v_perm_b32 lines 116-123; MFMA 134,141
```

The pinned AITER object has the corresponding `v_perm_b32` sequences around
275C/2780 and 2994-29D8, plus the Kr bpermute exchange. B70 Kr LLIR lines
37-60 give the exact bpermute address/select sequence before the tied MFMA.

## A69_REQUIRED_CHANGES

1. Replace A69's two-wave row-major carrier with the paired-wave `#mma1`
   carrier above; changing only `wave < 2` to `wave < 4` is invalid.
2. Preserve blocked1/blocked4 packets and `#shared/#shared1` swizzles for
   Kd/Qd/INV/Mqk.
3. Treat post-INV U as a `#mma1` result before `#shared4` overlay.
4. Use the `#mma` Kr/H carrier and exact bpermute/permute exchange.

No A69 source change is made by B70.

## PROBE_STATUS

All three B70 probes compiled and executed with finite output. Their TTGIR,
LLIR, AMDGCN, HSACO, and metadata are copied into this directory. The logical
`#mma1`/`#mma` output carriers are machine-derived. The report is a frozen
forensic map and required-change ledger, not a production promotion.
'''


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "B_20260927_70_LAYOUT_RECOVERY.md").write_text(MARKDOWN)
    report = {
        "artifact": "B_20260927_70_BW16_W4_MMA1_LAYOUT_RECOVERY",
        "read_only": True,
        "MMA1_LOGICAL_MAP": {
            "status": "RECOVERED",
            "formula": "row=16*floor(wave/2)+(lane&15), col=4*(lane>>4)+frag",
            "store_active": "wave&1==0",
            "evidence": "B70 mma1 Kd/INV LLIR and AMDGCN artifacts",
        },
        "ORDINARY_V_CARRIER_MAP": {
            "status": "RECOVERED",
            "f32_dest": "v[d:d+3], frag 0..3",
            "f16_bf16_input": "two packed 32-bit words: x0|x1<<16, x2|x3<<16",
            "continuation": "destination bank tied to C bank",
        },
        "LDS_WAVE_MAP": {
            "status": "RECOVERED_SLOT_AND_OPCODE_LEVEL",
            "pool_bytes": 30720,
            "barriers_per_chunk": 12,
            "waitcnt_per_chunk": 36,
        },
        "PERMUTE_MAP": {
            "status": "RECOVERED_PHYSICAL_SEQUENCE",
            "evidence": "B70 probe ISA plus pinned AITER ISA",
        },
        "A69_REQUIRED_CHANGES": "Implement the recovered carriers through an Avelang layout/fragment representation before enabling waves 2/3; no A69 change made.",
        "PROBE_STATUS": "PASS: isolated B70 Kd/INV/Kr probes compiled, executed finite, and copied TTGIR/LLIR/AMDGCN/HSACO artifacts.",
    }
    (OUT_DIR / "B_20260927_70_LAYOUT_RECOVERY.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

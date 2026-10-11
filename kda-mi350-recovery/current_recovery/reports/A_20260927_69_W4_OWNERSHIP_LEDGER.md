# A69 BW16/W4/S2 ownership ledger

This is the pre-implementation ledger for `A_20260927_69_BW16_PERSISTENT_W4_REBUILD`. It is derived from the pinned AITER source, TTGIR, ISA, and the already-frozen C67 ledger. A56 is not used to infer machine layout.

Pinned machine identity: commit `7bb44998274fa679ece0a37bf8426ab780b1837d`, HSACO SHA256 `85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16`, grid `(8,12,1)`, block 256, four waves, BW16, stages2.

The exact tensor layouts are in `golden_reference.ttgir:1-13`: Kd/Qd use `blocked1` (`[1,8]`, `[8,8]`, `[4,1]`); INV/Mqk use `blocked4` (`[1,4]`, `[8,8]`, `[4,1]`); V/U use `blocked3` (`[1,2]`, `[8,8]`, `[4,1]`); Kr uses `blocked` (`[8,1]`, `[8,8]`, `[1,4]`); results use `mma1` `[2,2]` or `mma` `[4,1]`.

## Four-wave roles

With `wave = lane_id // 64` and `lane = lane_id % 64`:

| phase/layout | wave 0 | wave 1 | wave 2 | wave 3 | per-lane packet |
|---|---|---|---|---|---|
| Kd/Qd `blocked1` | rows 0–7 | rows 8–15 | rows 16–23 | rows 24–31 | `row=8w+lane//8`, `col0=8(lane%8)`, 8 values |
| INV/Mqk `blocked4` | rows 0–7 | rows 8–15 | rows 16–23 | rows 24–31 | `row=8w+lane//8`, `col0=4(lane%8)`, 4 values |
| V/U `blocked3` | rows 0–7 | rows 8–15 | rows 16–23 | rows 24–31 | `row=8w+lane//8`, `col0=2(lane%8)`, 2 values |
| Kr `blocked` | columns 0–7 | columns 8–15 | columns 16–23 | columns 24–31 | `row0=8(lane%8)`, `col=8w+lane//8`, 8 values |

The `mma1` result has the exact encoded warp arrangement `[2,2]`; the pinned artifacts do not support replacing that MFMA fragment layout with a guessed scalar 16x8-per-wave table. The `mma` result is `[4,1]`, so its four wave coordinates are the four consecutive H-fragment coordinates. The TTGIR evidence is `golden_reference.ttgir:7-8,426-479,497-509`.

## Physical MFMA continuation

The pinned ISA records the operand/accumulator banks that must be preserved by the lowering:

- Kd first chain: `v[110:111] × v[18:19] -> v[96:99]`, then the exact tied continuation sequence at ISA lines 461, 466, 472, 479, 484, 490, 499.
- INV: `v[22:23] × v[122:123] -> v[18:21]`, followed by `v[24:25] × v[124:125] + v[18:21] -> v[26:29]` at ISA lines 576–578.
- Mqk: `v[104:105] × v[110:111] -> v[94:97]`, followed by `v[100:101] × v[106:107] + v[94:97] -> v[94:97]` at ISA lines 601–603.
- Kr/H continuation uses the tied BF16 accumulator banks shown at ISA lines 710–724 and 1111–1117.

These are physical bank contracts, not a row-major reinterpretation of the four FP32 registers. A69 must compare raw per-lane payloads at the phase boundaries against the logical packet layouts above and keep the physical continuation tied.

## LDS contract

The exact shared layouts are `#shared` for Kd/Qd, `#shared1` for INV/Mqk, `#shared3` for H BF16, `#shared4` for U BF16, and `#shared2` for Kr (`golden_reference.ttgir:9-13`). The frozen C67 ledger gives 30 writes, 44 reads, 12 barriers, 36 waitcnt, and 12 `v_perm` per chunk, with 30720 bytes dynamic LDS. These existing C67 numbers are reused as the comparison target; they are not regenerated.

## A68 structural deltas to remove

A68 guards the Kd, INV, Mqk, and Qd MFMA bodies with `wave < 2` and therefore omits waves 2/3. Its host wrapper launches `_bw16_k2_chunk` once per chunk and synchronizes after every launch (`A_20260925_68_BW16_PRODUCTION_CLOSURE_PROFILE.py:205-246,294-337,377-421,435-481,613-630`). A69 therefore changes only these structural boundaries: four-wave ownership and one device-side chunk loop, while retaining A68’s already-passing math formulas.

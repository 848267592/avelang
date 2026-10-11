# B70 BW16/W4/S2 exact `#mma1` layout recovery

Status: read-only forensic recovery. No A69, A56, A68, K1, AITER source, or
production object was modified. The pinned object is commit
`7bb44998274fa679ece0a37bf8426ab780b1837d`, HSACO
`85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16`.

## MMA1_LOGICAL_MAP

The isolated B70 Kd and INV probes compile the same gfx942 transposed MFMA
encoding as the pinned target:

```text
#mma1 = amd_mfma<version=3, warpsPerCTA=[2,2], instrShape=[16,16,16], isTransposed=true>
```

For a physical wave `w = 0..3`, lane `l = 0..63`, and returned FP32 element
`f = 0..3`, the exact logical result carrier is:

```text
row = 16 * floor(w / 2) + (l & 15)
col = 4 * (l >> 4) + f
```

Waves 0 and 1 carry the same logical row band; waves 2 and 3 carry the second
row band. The pinned/Triton store predicate is active only for even waves
(`w & 1 == 0`); odd-wave fragments are still produced by MFMA but are not
stored by the logical result store. Evidence is B70 Kd LLIR lines 10-18 and
129-135, B70 Kd ISA lines 123-159, and the analogous INV LLIR lines 10-23 and
89-97.

For the `#mma` Kr/H result (`warpsPerCTA=[4,1]`), the exact carrier is:

```text
row = 16 * w + (l & 15)
col = 4 * (l >> 4) + f
```

Evidence: B70 Kr LLIR lines 10-22 and 110-113, and B70 Kr ISA lines 134-146.

This separates the pinned four-wave result carrier from the A69 two-wave
row-major mapping (`row = 16*w + 4*(l>>4)+f`, `col=l&15`).

## ORDINARY_V_CARRIER_MAP

The ordinary-V MFMA primitive has the same physical four-FP32 destination
carrier: contiguous `v[d:d+3]`, with `f=0..3` corresponding to the four
returned FP32 registers. B67 proves the continuation contract:

```text
zero:         v[d:d+3] = mfma(a, b, 0)
continuation: v[d:d+3] = mfma(a, b, v[d:d+3])
```

BF16/FP16 input vectors are four 16-bit elements packed as two 32-bit VGPR
words: `(x0 | x1<<16)` and `(x2 | x3<<16)`. The B67 tied-bank audit proves
`AGPR=0`, exact tied continuation, and no inter-MFMA drain requirement.

The pinned ISA carrier examples are:

| phase | physical chain evidence |
|---|---|
| Kd/H1, Kd/H2 | ISA 454-499, destinations `v[96:99]` then `v[18:21]` |
| Qd | ISA 787-840 and 934-955 |
| INV | ISA 576-578, `v[18:21]` then `v[26:29]` |
| Mqk | ISA 601-603, tied `v[94:97]` |
| Kr/H | ISA 710-724 and 1111-1117 |

The exact dot-operand packet ownership is:

| logical family | source packet per wave/lane | pinned dot operand |
|---|---|---|
| Kd/Qd | `row=8*w+l//8`, `col=8*(l%8)+q`, `q=0..7` | `dot_op(opIdx=0,parent=#mma1,kWidth=4)` |
| INV/Mqk | `row=8*w+l//8`, `col=4*(l%8)+q`, `q=0..3` | `dot_op(opIdx=0,parent=#mma1,kWidth=4)` |
| U | `row=8*w+l//8`, `col=2*(l%8)+q`, `q=0..1` | `#mma1 -> shared4 -> dot_op(opIdx=1)` |
| Kr | `row0=8*(l%8)+q`, `col=8*w+l//8`, `q=0..7` | `dot_op(opIdx=0,parent=#mma,kWidth=4)` |

The exact B70 Kd/INV source layouts and their generated dot operands are
captured in the copied TTGIR/LLIR/ISA files; the pinned target's logical
versions are the `#blocked1/#blocked4/#blocked3/#blocked` encodings in
`golden_reference.ttgir:1-8,183-192,298-309,347-367,497-509`.

## LDS_WAVE_MAP

The 30720-byte allocation follows the TTGIR local-allocation order. Byte slots
and logical roles are:

| slot base | role | bytes | shared encoding |
|---:|---|---:|---|
| 0 | Kd1 | 4096 | `#shared`, vec4/perPhase1/maxPhase16/order[1,0] |
| 4096 | Kd2 | 4096 | `#shared` |
| 8192 | Qd1 | 4096 | `#shared` |
| 12288 | Qd2 | 4096 | `#shared` |
| 16384 | INV | 2048 | `#shared1`, vec4/perPhase2/maxPhase8/order[1,0] |
| 18432 | Mqk | 2048 | `#shared1` |
| 20480 | Kr1 | 4096 | `#shared2`, vec1/perPhase1/maxPhase1/order[0,1] |
| 24576 | Kr2 | 4096 | `#shared2` |
| 26624 | H BF16 overlay | 2048 | `#shared3`, vec4/perPhase1/maxPhase16/order[0,1] |
| 28672 | U/overlay FP16/BF16 | 1024/temporal alias | `#shared4`, vec4/perPhase2/maxPhase8/order[0,1] |

The pinned ISA verifies these bases: INV reads at offsets 16384/16512 and
24576; H reads at 26624; U/overlay reads/writes at 28672/28736; Kr packet
traffic uses the 20480/24576 slots. Representative exact instructions are
`ds_read2st64_b64` at ISA 2278, 22B4-22C4, 259C-25C0, 2AAC-2AC4, and 2EC8-2EE0;
`ds_read_b64/u16` at 22E8-247C, 25C8-26D4, 26B4-26D4, 2B0C-2B24,
2D74-2DE8; and `ds_write_b16/b64/b128` at 2290-23A0, 243C, 2574-258C,
268C-26A4, 2908-2960, 2ACC-2AF0, 2C08-2C20, and 2E24-2E84.

Wave packet ownership is the table above; all four waves participate in source
packet production. The `#mma1` result has the paired-wave carrier described in
MMA1_LOGICAL_MAP, while `#mma` Kr/H has one distinct 16-row band per wave.
The frozen steady-state synchronization contract is 12 barriers and 36
waitcnt per chunk.

## PERMUTE_MAP

The isolated B70 Kd/INV probes expose the exact operand exchange before MFMA:

```text
Kd:  v_perm_b32 lines 117-122, 135-141; MFMA lines 123,137,145,153
INV: v_perm_b32 lines 115-120;        MFMA lines 121,128
Kr:  ds_bpermute_b32 lines 96,111-113;
     v_perm_b32 lines 116-123;        MFMA lines 134,141
```

The pinned AITER machine has the same exchange families in its exact ISA:
`v_perm_b32` around 275C/2780 and 2994-29D8, and the Kr `ds_bpermute_b32`
exchange around the Kr preparation/consume sequence. The B70 LLIR gives the
address/data order for Kr directly: the bpermute destinations are formed from
the lane-group address `%42`, `%45=%42^8`, followed by the conditional swaps
at LLIR lines 56-60. This is the required exchange before the two tied Kr
MFMA chains.

## A69_REQUIRED_CHANGES

The current A69 source must change only after carrying these contracts into an
Avelang representation:

1. Replace the A69 `wave < 2` two-wave result mapping with the exact paired-wave
   `#mma1` carrier above; waves 2/3 cannot be enabled by changing the predicate
   alone.
2. Stage Kd/Qd through blocked1 and INV/Mqk through blocked4, preserving the
   dot-op operand packets and `#shared/#shared1` swizzles.
3. Treat post-INV U as a `#mma1` result carrier before `#shared4` overlay; do
   not reinterpret it as the A69 row-major vector map.
4. Use the `#mma` Kr/H result carrier and the exact bpermute/permute exchange
   before the tied BF16 continuation.

No A69 source change is made by B70.

## PROBE_STATUS

The three isolated B70 probes compiled and executed with finite output. They
copied TTGIR, LLIR, AMDGCN, HSACO, and metadata into this B70 report directory.
The logical `#mma1` and `#mma` output carriers are now machine-derived, not
guessed. The remaining implementation gap is Avelang's ability to express the
pinned dot-op/shared conversion and paired-wave carrier while retaining the
ordinary-V primitive. This report is therefore a frozen forensic map and
required-change ledger, not a production promotion.

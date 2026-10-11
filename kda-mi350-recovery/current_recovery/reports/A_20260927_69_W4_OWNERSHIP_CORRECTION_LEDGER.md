# A69 four-wave ownership correction ledger

This ledger records the first machine divergence found after the A69 semantic
and persistence gates. It uses the pinned C67/AITER TTGIR and ISA only for
machine mapping; it does not alter or regenerate those references.

## Current A69 guards

The current A69 source still predicates the ordinary-V MFMA/result bodies with
`wave < 2` at lines 133, 158, 177, 201, 209, 236, 248, 285, 311, 329, and
344. Kr is the only main phase that already runs for all four waves. The
resulting T4096 profile is 14 MFMA per wave/chunk instead of the C67 target 24:
the four-wave average is `(24 + 24 + 4 + 4) / 4`.

## Pinned target contract

The frozen TTGIR defines:

| phase | pinned layout | per-wave ownership |
|---|---|---|
| Kd/Qd | `#blocked1`, `#mma1` | `row = 8*wave + lane//8`, `col0 = 8*(lane%8)`; dot result is `32x16` under `#mma1` |
| INV/Mqk | `#blocked4`, `#mma1` | `row = 8*wave + lane//8`, `col0 = 4*(lane%8)`; dot result is `32x16` under `#mma1` |
| U | `#blocked3`, `#mma1` | `row = 8*wave + lane//8`, `col0 = 2*(lane%8)`; dot result is `32x16` under `#mma1` |
| Kr | `#blocked`, `#mma` | `row0 = 8*(lane%8)`, `col = 8*wave + lane//8`; result is `64x16` under `#mma` |

Evidence: `C_20260925_67_BW16_W4_MACHINE_GOLDEN_AUDIT/golden_reference.ttgir:1-8,183-192,298-309,347-367,497-509` and the pinned MFMA sequences in `golden_reference.isa:454-499,552-603,710-724,787-840,934-955`.

## A69-to-target mismatch

The current scalar Avelang implementation feeds the ordinary-V primitive with
`a_row = wave*16 + (lane&15)`, `out_row_base = wave*16 + 4*(lane>>4)`, and a
four-FP32 result vector. That is the previously validated two-wave row-major
fragment, not the pinned `#mma1` four-wave distribution. Simply changing the
guards to `wave < 4` would read rows 32-63 from a 32-row Kd/Qd packet and would
not provide the encoded `#mma1` result ownership. It is therefore not a safe
ownership fix.

The pinned artifact exposes the exact physical MFMA continuation banks, but it
does not expose a scalar per-lane mapping from the `#mma1` dot result to the
ordinary-V four-register return value. The C67 ledger explicitly disallows
replacing that encoded result ownership with a guessed scalar 16x8 map.

## Bounded correction decision

No kernel source change is applied from this ledger. The concrete blocker is
the missing Avelang representation/conversion for the pinned
`dot_op<parent=#mma1>` fragment. Removing `wave < 2` without that conversion
would be an unverified layout guess and would violate the first-divergence
method. The current A69 correctness, resource, benchmark, and profile reports
remain the pre-correction revision; the next implementation must first add or
reuse an exact `#mma1` fragment carrier, then re-run T32 strict tracing before
any performance measurement.

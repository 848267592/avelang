# A70 BW16 TRUE W4 production closure

## CORRECTNESS

PASS against public AITER production semantics at the established production tolerance (`atol=rtol=3e-2`) for `T=32,64,127,128,256,512,1024,2048,4096`.

- T32 strict trace: all Kd/pred, u_pre, INV/U, Qd, Mqk, output accumulator, H1 and H2 stages pass `1e-5`.
- T64/NT2: complete H1/H2 recurrent backedge passes strict `1e-5` for both chunks; output is finite and passes.
- T127: nonzero initial state passes; output/final state are finite.
- No NaN/Inf; every correctness case uses grid `[8,12,1]`, block `[256,1,1]`.
- First A70-only fix: Qd blocked1 continuation consumers changed from `base + offset` to the B70-required `base ^ offset`; no Kd/pred/u_pre/INV/Mqk redesign was made.

## TRUE_FOUR_WAVE_OWNERSHIP

PASS, debug-only and excluded from the production HSACO. At `T4096/NT128`, Kd, INV, Mqk and Qd counters are `[128,128,128,128]` for waves 0–3. Raw INV markers are finite on all four waves. The final production loop contains 24 MFMA/chunk: 20 BF16 + 4 FP16.

## FINAL_HSACO

- Source: `A_20260927_70_BW16_TRUE_W4_REBUILD.py`
- Source SHA256: `a496c2b2cefa4bf0ff475049876d6db34091fe64e73a586498e83a61c1174801`
- T4096 HSACO SHA256: `2fac8d38f007c2c53c13ba7021f0ead713274b0dd0acaf92a7859e6231ef434a`
- Symbol: `_a70_bw16_k2_persistent`

## RESOURCES

Final A70 hard gates pass: AGPR=0, no VGPR/SGPR spills, private/scratch=0, occupancy=2 blocks/CU, MFMA=20 BF16+4 FP16.

| metric | A70 | C67 BW16/W4 golden |
|---|---:|---:|
| ordinary VGPR | 94 | 126 |
| AGPR | 0 | 0 |
| unified span | 94 | 126 |
| accum_offset | not emitted; AGPR=0 | 128 |
| SGPR numbered / total | 32 / 38 | 84 / 90 |
| highest physical VGPR / SGPR | v93 / s31 | v125 / frozen ledger |
| LDS | 30720 B | 30720 B |
| occupancy | 2 blocks/CU | 2 blocks/CU |
| MFMA | 24 = 20 BF16 + 4 FP16 | 24 = 20 BF16 + 4 FP16 |
| VMEM loads / stores | 18 / 1 | 12 / 1 |
| LDS reads / writes | 58 / 64 | 44 / 30 |
| barriers | 20 | 12 |
| waitcnt | 56 | 36 |
| v_perm / DPP | 10 / 0 | 12 / frozen |

The C67 values are read-only frozen ledger values; no AITER resource audit was regenerated.

## K2_KERNEL_TRUE_BEST_PERFORMANCE

Protocol: same process/default stream, 50 warmups, 100 samples per round, 3 interleaved rounds; event contains only the already-built direct K2 launch and state restore is before the event. Public dispatch/autotune ran before timing and selected the object recorded per T.

`us/chunk`; public and W4 AITER HSACO hash is the pinned `85470da...` object.

| T | A70 | AITER public BW/W/S | AITER public | AITER W4 | A56 BW64 | gap to fastest AITER |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 4.221 | BW16/W4/S2 | 2.388 | 2.418 | 9.094 | 76.74% |
| 512 | 5.000 | BW16/W4/S2 | 2.003 | 2.008 | 8.473 | 149.62% |
| 1024 | 4.793 | BW16/W4/S2 | 1.790 | 1.790 | 8.168 | 167.76% |
| 2048 | 4.697 | BW16/W4/S2 | 1.683 | 1.683 | 8.013 | 179.06% |
| 4096 | 4.655 | BW16/W4/S2 | 1.630 | 1.631 | 8.640 | 185.53% |

The exact public-selected object hash was `85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16`; A56 was read-only at `55081383ac7d4f07090c4ec6e33de14ac6ba69865641b70debf752dd8b6a2943`.

## GAP_TO_D71

Against the fastest known pinned AITER W4 direct K2 row, A70 is slower by `74.55%, 149.00%, 167.76%, 179.06%, 185.47%` at `T=256,512,1024,2048,4096`, respectively. The public dispatch selected the same BW16/W4/S2 object; it is reported separately from the exact machine-reference row.

## PROFILE_DELTA

A70 T4096/NT128 was profiled with the existing C67 counter families; AITER was not re-profiled. Per logical chunk:

| metric | A70 | C67 BW16/W4 | delta | likely effect |
|---|---:|---:|---:|---|
| MFMA | 9216 | 9216 | 0 | not the gap |
| BF16 MFMA | 7680 | 7680 | 0 | not the gap |
| FP16 MFMA | 1536 | 1536 | 0 | not the gap |
| VALU | 77112 | 69702 | +7410 | extra vector/address work |
| VMEM read | 5382 | 4620 | +762 | extra global reads |
| LDS instructions | 44544 | 28446 | +16098 | extra LDS traffic |
| LDS bank conflict | 41472 | 21558 | +19914 | bank serialization |
| LDS wait | 49475.7 | 28551.0 | +20924.6 | synchronization exposure |
| wait-any | 764359.7 | 158453.6 | +605906.1 | dominant dependency/issue stalls |
| issue wait | 58851.7 | 37276.3 | +21575.3 | extra issue stalls |
| wave cycles | 998522.4 | 318342.9 | +680179.6 | dominant residency cost |
| TCC requests | 29422.1 | 29439.0 | -17.0 | HBM/cache traffic comparable |

## NEXT_SINGLE_BOTTLENECK

The single highest-priority next region is the temporal H/U overlay plus blocked1 Qd/Kd packet staging and its barrier/wait chain: source lines 113–171 and 261–311, ISA loop `0x1cfc–0x2af4` (especially repeated staging around `0x2094–0x2a38`). MFMA counts and TCC/HBM traffic already match C67; the measured excess is LDS traffic/bank conflicts and wait/barrier exposure. Stop here; no speculative tuning was started.

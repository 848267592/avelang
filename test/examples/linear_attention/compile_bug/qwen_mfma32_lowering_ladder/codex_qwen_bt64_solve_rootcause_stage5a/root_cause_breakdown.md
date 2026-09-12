# Root-Cause Breakdown

## Measured facts at T=2048

The body-only measurement is the primary comparison: allocation and vLLM
upper-triangular zero preparation are excluded.

| metric | v18 | vLLM selected BT64 | ratio/delta |
|:--|--:|--:|--:|
| body HIP median | 0.122422 ms | 0.035613 ms | 3.44x |
| public-wrapper median | 0.125386 ms | 0.053179 ms | 2.36x |
| body-fit intercept | 103.069 us | 33.866 us | +69.203 us |
| body-fit slope/chunk | 0.895079 us | 0.087651 us | 10.21x |
| rocprof trace | 110.224 us | 23.875 us | 4.62x |
| workgroup / waves | 128 / 2 | 256 / 4 | 2x waves |
| CTA count | 256 | 256 | equal |
| VGPR / AccVGPR / SGPR | 96 / 128 / 112 | 56 / 16 / 48 | lower vLLM pressure |
| LDS block / scratch | 17,920 B / 0 | 0 B / 0 | no scratch in either |
| occupancy percent | 4.733 | 6.991 | vLLM higher |
| MFMA | 0 | 65,536 | vLLM uses FP32 MFMA |
| VALU | 2,695,424 | 1,419,264 | v18 1.90x |
| SALU | 2,535,680 | 570,368 | v18 4.45x |
| VMEM | 32,768 | 135,168 | vLLM 4.12x |
| LDS instructions | 1,177,856 | 397,312 | v18 2.96x |

Counters are separate rocprof dispatch measurements and must not be added to
HIP-event totals. They support classification, rather than a precise
percentage attribution of elapsed time.

## Classification

1. **Primary: serial recurrence plus lack of block-dot MFMA.** v18 has a
   63-stage row chain and 189 recurrence barriers. vLLM replaces it with four
   16x16 local inverse tiles plus a shallow lower-block DAG and actual MFMA.
   The large SALU, LDS, and slope gaps, together with a 3.44x body-only
   difference at T=2048, support this as the dominant cause.
2. **Secondary: fixed body overhead and ownership/resource pressure.** The
   fitted intercept differs by 69.2 us, v18 has two rather than four waves per
   CTA, and resource/occupancy values are worse. This helps explain short
   sequences, but cannot explain the widening gap at 128/256 chunks.
3. **Not primary: final global stores or generic global traffic.** The
   load/store-only and no-store diagnostics are small relative to actual v18;
   moreover vLLM has more VMEM instructions yet is faster.
4. **Not demonstrated: compiler resource cliff, spilling, or an assembly
   requirement.** Both final code objects have zero private segment and zero
   VGPR spill count. v18's resource use is higher, but there is no scratch
   cliff analogous to the older full-v29 experiments.
5. **Not a correctness/numerical-policy advantage.** The same input object,
   FP32 output contract, strict-lower rule, and 54-case authority check pass.

## Why nearly flat T=512 to T=2048, then rising

The GPU can concurrently schedule all 64 CTAs at T=512 and all 256 CTAs at
T=2048, so fixed per-launch/per-CTA cost and machine parallelism hide much of
the total chunk count. Once the grid grows (T=8192 has 1024 CTAs), the v18
body rises to 0.229501 ms while vLLM is 0.043484 ms. The fitted per-chunk
slope is therefore the more useful long-sequence evidence.

This is an inference from the same-device sweep and grid sizes; it is not a
claim that a single hardware occupancy threshold was directly measured.

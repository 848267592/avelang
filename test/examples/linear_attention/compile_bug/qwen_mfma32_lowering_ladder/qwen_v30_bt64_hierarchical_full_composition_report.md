# Qwen v30 Hierarchical BT64/BV32 Full-Composition Experiment

## Summary

v30 D1 validates the intended register-lifetime hypothesis but fails the
semantic and throughput gates. It keeps one BT64 recurrence chunk and one
128-thread workgroup, yet serializes the active computation as V16, token16,
and K16 microtiles. Exact LTO MIR shows that this removes the v29 full-rewrite
spill cliff completely:

| metric | v29 rewrite + pred streaming C | v30 D1 |
|:--|--:|--:|
| spill words | 151 | 0 |
| private scratch | 600 B | 0 B |
| rocprof AccVGPR | 384 | 164 |
| highest physical AGPR | 254 | 12 |
| high AGPR copy count | 166 | 0 |

However, D1's nonzero-W final state is not equivalent to the BT64 reference
or original v29 (`T=64` maximum reference error `0.307391`). Its T=2048
chunk_gdr latency is `2.327338 ms`, much slower than original v29
(`0.836063 ms`), the full rewrite+streaming experiment (`1.271430 ms`), and
the known v24 chunk_gdr context (`~0.349 ms`). D1 is therefore an
experimental negative result, not a production candidate.

## Why Retain BT64

BT64 reduces recurrence windows relative to BT16. The experiment does **not**
turn BT64 into four independent recurrence chunks and does not add a kernel
launch. `num_chunks = T / 64` remains unchanged. V16/token16/K16 are only
machine-level active working tiles inside one BT64 recurrence step.

The relevant distinction is:

```text
macro state recurrence: BT64, BV32, one CTA owns the state
active working set:     V16, token16, K16, one block at a time
```

## D1 Schedule

The new kernel is
[`qwen_gdn_chunked_avelang_v30_bt64_bv32_hierarchical.py`](../../vllm_compare/qwen_gdn_chunked_avelang_v30_bt64_bv32_hierarchical.py).

For each BT64 chunk it performs:

```text
write h from chunk-start state
for V16 in [0:16, 16:32]:
    stage this V16's old state as the MFMA32 operand
    for token16 in four token tiles:
        stage only this token16 W microtile
        MFMA32 pred, lane-major serialize accumulator
        inverse-map and consume only the active 16x16 quadrant
        form V16 x token16 v_decay in LDS
        for K16 in eight output tiles:
            stage one K16 x token16 tile
            one MFMA16 update accumulator, immediately write state
```

MFMA32 structurally requires 32 rows. D1 duplicates the active 16 rows in
each pred operand and consumes only the first 16x16 output quadrant. This
preserves the MFMA32 instruction geometry but multiplies pred-side work; it
is the direct reason D1 must be judged by real counters rather than by its
register result alone.

Old full-width `pred_partial[2,32,32]`, `v_decay[32,64]`, and
`k_all_t[128,64]` materializations do not exist in D1. The pred accumulator
uses the verified lane-major `[2,64,16]` serialization.

## Correctness

The focused pytest is
[`test_qwen_gdn_v30_bt64_bv32_hierarchical.py`](../../vllm_compare/test_qwen_gdn_v30_bt64_bv32_hierarchical.py).

```text
3 passed in 10.42s
```

The passing checks intentionally separate the parts that are and are not
valid:

| case | result |
|:--|:--|
| `w=0`, T=64 | h exact; final state max abs `3.81e-6` |
| `w=0`, T=512 | h/final state max abs `2.29e-5` |
| nonzero W, T=64 | final-state max abs vs reference `0.307391` |

The small `w=0` T=512 difference is a bounded BF16 accumulation-order change
from writing state after each token16 update. The nonzero-W difference is not
acceptable. Comparison of T=64 with original v29 shows original itself also
has its known nonzero-W error, but D1 is not bit-exact to original (`0.350167`
max final-state difference), so this experiment cannot claim to merely inherit
that known v29 issue.

## Exact LTO, MIR, and ISA

Exact artifacts are under:

- `rocprof_outputs/qwen_v30_hierarchical/d1_link/`
- `rocprof_outputs/qwen_v30_hierarchical/exact_lto_d1/`
- `codex_v30_hierarchical_audit/d1_spill_vregs.{json,csv}`

The post-greedy MIR has no `SI_SPILL_AV32_SAVE` or
`SI_SPILL_AV64_SAVE`. Code-object metadata confirms:

| field | D1 |
|:--|--:|
| `vgpr_spill_count` | 0 |
| `private_segment_fixed_size` | 0 |
| `sgpr_spill_count` | 0 |
| code-object `agpr_count` | 16 |
| highest physical AGPR in MIR | 12 |
| high AGPR copies (`>=100`) | 0 |
| static `scratch_load` / `scratch_store` | 0 / 0 |

The high-level serial schedule therefore survived generic lowering well enough
to remove the previous mixed RA cliff. No compiler-owned V16 block op was
implemented: its stated purpose was to prevent premature address/fragment
formation, and exact MIR shows D1 already has no spill, no scratch, and no
high-AGPR parking. Adding such an op would not address the actual D1 failure.

Static ISA still uses both expected MFMA forms:

| mnemonic | count |
|:--|--:|
| `v_mfma_f32_32x32x8_bf16` | 16 |
| `v_mfma_f32_16x16x16_bf16` | 8 |

## Rocprof

| metric | T=2048 | T=8192 |
|:--|--:|--:|
| trace median us | `3052.015` | `9120.654` |
| workgroup / grid | `128 / 4096` | `128 / 4096` |
| LDS block | `41984 B` | `41984 B` |
| scratch | `0 B` | `0 B` |
| VGPR / AccVGPR / SGPR | `100 / 164 / 112` | `100 / 164 / 112` |
| MFMA | `393216` | `1572864` |
| VALU | `19162944` | `76560192` |
| SALU | `3376512` | `13471104` |
| VMEM | `923648` | `3682304` |
| LDS instructions | `2180288` | `8711360` |
| occupancy percent | `0.6384%` | `0.6516%` |

The compiler/lowering result is positive; the execution result is not. D1
uses much more VALU/SALU/LDS/VMEM than v29 because it duplicates token/V rows
to satisfy MFMA32 and serializes K16 staging/update inside every V16 and
token16 step. At T=2048, its MFMA count (`393216`) is also higher than the
original v29's `294912`.

## Long-Sequence Benchmark

These results are genuine measured D1 chunk_gdr timings, but are marked
invalid for v24/Triton production comparison because nonzero-W D1 is not
correct.

| T | D1 chunk_gdr ms | us per BT64 chunk |
|--:|--:|--:|
| 512 | `0.580543` | `72.568` |
| 1024 | `1.180955` | `73.810` |
| 2048 | `2.327338` | `72.729` |
| 4096 | `4.600916` | `71.889` |
| 8192 | `9.186611` | `71.770` |
| 16384 | `18.416265` | `71.939` |

The least-squares fit is:

```text
latency_ms = 0.014551 + 0.001122437 * T
R^2 = 0.999994
```

D1's `1.122 us/token` slope is not competitive with the existing v24
chunk_gdr context and cannot produce a Triton crossover. A full-forward
latency is intentionally not reported: this repository has no correct
BT64 full-forward composition using D1, and reporting one would be misleading.

## D2 Decision

D2 was not run. The requested D2 gate is for residual spill/scratch,
`AccVGPR=384`, or MFMA32 accumulator pressure after D1. D1 has zero spill,
zero scratch, `AccVGPR=164`, and no high AGPR range. Its dominant cost is not
residual MFMA32 accumulator pressure; it is the work amplification introduced
to force MFMA32 into V16/token16 microtiles. Changing to MFMA16 pred would
simultaneously change that work shape and would not isolate the stated gate.

## Final Recommendation

Do not create a compiler-owned block op for D1 and do not promote v30. The
experiment proves the macro-BT64 plus small active working-set concept can
exit the spill cliff, but this direct padded-MFMA32 realization is both
nonzero-W incorrect and throughput-negative.

The single next minimal architectural change is a **correct active-V16 pred
primitive without duplicated 32-row MFMA32 operands**. It must be designed as
a separately validated math/mapping primitive before any full recurrence
integration. It is not a K-load helper, a lifetime marker, or an AMDGPU RA
change.

## Reproduction and Files

- Commands: `codex_v30_hierarchical_audit/commands.sh`
- Benchmark CSV: `codex_v30_hierarchical_audit/benchmark_matrix.csv`
- Long slope: `codex_v30_hierarchical_audit/long_sequence_slope.json`
- rocprof: `codex_v30_hierarchical_audit/rocprof_t2048.json` and
  `codex_v30_hierarchical_audit/rocprof_t8192.json`
- Exact LTO: `rocprof_outputs/qwen_v30_hierarchical/exact_lto_d1/`

No production baseline was modified and no commit was created.

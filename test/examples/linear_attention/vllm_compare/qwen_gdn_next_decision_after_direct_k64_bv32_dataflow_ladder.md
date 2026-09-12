# Qwen Direct-K64 BV32 Dataflow Ladder Next Decision

## Decision

Keep the experimental direct-K64 line on the **BV32 typed-vector operand
producer**. Do not change production dispatch, RA, MFMA geometry, CTA
ownership, broad-K, compact-K or full-v29 pred correctness.

The completed controls give a clean priority order:

| control | T=2048 result | decision |
|:--|--:|:--|
| precomputed BF16 V-decay | `0.431541 -> 0.427275 ms`, `-0.99%` | close this micro-path |
| typed K/V BF16x8 operand local-load | `0.430039 -> 0.258745 ms`, `1.662x` | retain as experimental baseline |

Typed staging reduces the native W=0 diagnostic ratio from `3.59x` to
`2.16x` at T=2048. Its scalar-to-native gap slope falls from `9.001` to
`3.761 us/chunk`. It has zero scratch and zero MIR spills, retains exactly
the same 32768 total MFMA and 17 static barriers, and passes the frozen
direct-update correctness contract.

## Next Single Action

Implement one **pipeline/fusion control** on top of typed-vector BV32:

```text
typed K/V BF16x8 producer
  -> optional prefetch of the next token-half
  -> same A/B LDS consumer and same MFMA32 pair
```

It must not also alter BV32, 32-CTA ownership, WG128, K32 accumulation order,
LDS shape, persistent H1/H2 layout, state/output ABI, or block-dot math.
The success gate is a stable benefit at T=2048 *and* a lower 512-to-2048 body
slope. If either gate fails, close the pipeline line rather than reopening
old compact-K/broad-K or RA tuning.

The detailed evidence and raw session values are in
`compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_bv32_dataflow_cost_ladder_report.md`.

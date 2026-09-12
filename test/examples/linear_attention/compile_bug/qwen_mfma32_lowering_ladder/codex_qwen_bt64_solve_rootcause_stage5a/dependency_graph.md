# Dependency and Parallelism Comparison

## v18 recurrence

```text
load all A -> LDS
row 1 -> row 2 -> row 3 -> ... -> row 63
  each row: copy row -> barrier -> two partial reductions -> barrier
            -> group-0 final reduction/write -> barrier
store X
```

Each row has up to 63 columns in parallel, but the next row waits for the
previous row.  With BT64 this produces 63 serial recurrence stages.

## vLLM BT64 block graph

```text
D1^-1  D2^-1  D3^-1  D4^-1        (four 16x16 local inverse chains)
     \   |       |       /
  X21     X32     X43              (independent block-product level)
     \   |       |       /
        X31      X42               (second block-product level)
              \ /
              X41                 (third block-product level)
store ten lower/diagonal 16x16 blocks
```

The local inverse loops are still sequential within a 16x16 tile, but the
long 64-wide recurrence is replaced by a shallow block DAG.  Off-diagonal
work is represented as tiled matrix products.  The selected compiled ISA
contains `v_mfma_f32_16x16x4_f32`; the v18 ISA contains no MFMA mnemonic.

## What is and is not directly measured

- **Measured:** v18 has 63 source-level row stages and three explicit
  `al.syncthreads()` inside each; vLLM source has the block DAG shown above;
  dynamic LDS/SALU/VALU counts differ substantially.
- **Not available:** a reliable dynamic `s_barrier` counter.  Static ISA
  `s_barrier` counts cannot be equated with runtime executions in loops.
- **Inference supported by source plus counters:** restructuring the
  dependency graph, not merely changing the number of threads, is the main
  opportunity.

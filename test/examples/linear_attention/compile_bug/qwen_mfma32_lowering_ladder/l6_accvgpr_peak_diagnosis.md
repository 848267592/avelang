# L6 AccVGPR Peak Diagnosis

## Summary

The L6 baseline AccVGPR peak is not an update-MFMA-only problem and not a generic sequential-MFMA lifetime problem.  The evidence points to a Qwen-shaped interaction:

```text
pred/v_decay dataflow + broad k_all_t/kall_vec shared-view lowering + update MFMA path
```

The most concrete symptom is that the baseline ISA allocates many non-MFMA address/staging temporaries into AGPRs (`v_accvgpr_write_b32 a100..a131`).  The subtile variant removes most of that pressure while preserving the same dynamic MFMA count.

## Evidence Table

| variant | trace_us | AccVGPR | VGPR | LDS block | MFMA | VALU | VMEM | LDS inst | explicit acc write/read |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| baseline | 34.331 | 264 | 128 | 45056 | 5120 | 182144 | 22528 | 28672 | 155 / 155 |
| subtile | 19.109 | 168 | 96 | 32768 | 5120 | 120128 | 16384 | 21504 | 32 / 32 |
| no_pred_dependency | 9.774 | 80 | 96 | 20480 | 2048 | 69504 | 8192 | 7168 | 32 / 32 |
| minimal_frag | 3.205 | 4 | 12 | 1024 | 256 | 4096 | 2432 | 512 | 4 / 4 |

## Question Answers

1. **Is AccVGPR=264 caused mainly by pred MFMA32 accumulator region?**

Partially, but not by MFMA32 alone.  Previous independent `pred32_only_sink` evidence had AccVGPR around `184`, so pred-side MFMA32 is a large base cost.  However, baseline and subtile both contain 8 static `v_mfma_f32_32x32x8_bf16` instructions, while AccVGPR changes from `264` to `168`.  That delta means the extra peak is not inherent to the pred MFMA32 instruction sequence alone.

2. **Is it caused mainly by update MFMA16 accumulator region?**

No.  `L6_update_mfma_minimal_frag` uses MFMA16 update fragments and has AccVGPR `4`.  `L6_update_mfma_no_pred_dependency` has a fuller update-like path with 32 static MFMA16 instructions and AccVGPR `80`.  The update MFMA itself is not sufficient to explain AccVGPR `264`.

3. **Is it caused by multiple update accumulators/fragments being live together?**

Not as the primary cause.  The full no-pred update path still compiles to static max ACC index `3` and dynamic AccVGPR `80`.  Baseline reaches static max ACC index `131`, and those high AGPR indices occur before the update MFMA block.

4. **Is it caused by pred/v_decay values being live into update?**

Yes, as part of the interaction.  Removing pred/v_decay dependency drops AccVGPR from `264` to `80` and trace from `34.331 us` to `9.774 us`.  The failed `end_lifetime` marker experiments show that a high-level marker alone does not shorten the backend live intervals, but the no-pred variant proves the dataflow dependency is important.

5. **Is it caused by view/cast/subview address lowering producing many address temporaries?**

Strongly yes.  Baseline has much more address-generation ISA than subtile:

- `v_lshl`: `722 -> 410`
- `v_lshl_add`: `401 -> 242`
- `v_or`: `166 -> 116`
- `s_waitcnt`: `180 -> 131`
- explicit AGPR write/read pairs: `155/155 -> 32/32`

The high AGPR indices in baseline appear in address/staging code using `v_lshl*`, `v_sub`, and `v_accvgpr_write_b32`, not only in the MFMA issue lines.

6. **Is it caused by broad k_all_t/kall_vec staging?**

Yes, this is the cleanest isolated source difference.  Baseline materializes a broad transposed K shared buffer/view.  Subtile avoids that broad view and keeps the same dynamic MFMA count.  The result is a large reduction in trace, AccVGPR, VALU, VMEM, LDS inst, and static address-generation ops.

7. **Is it code-object max from one local peak, or true cross-region live overlap?**

It is code-object resource max with evidence of a local AGPR-heavy address/staging region plus pred/update dataflow overlap.  The global static max ACC index `131` is not inside a single MFMA-local snippet; it appears earlier where address/staging values are written into AGPRs.  This weakens the idea that a simple source lifetime marker after `v_decay_t` can fix it.

## Why end_lifetime Did Not Help

Both marker-only and late-survive `al.end_lifetime` experiments left counters unchanged:

| variant | trace_us | AccVGPR |
|:---|---:|---:|
| baseline no marker | 34.371 | 264 |
| baseline with marker | 34.291 | 264 |
| subtile no marker | 19.349 | 168 |
| subtile with marker | 19.189 | 168 |

The marker does not affect the backend-visible allocation pattern that puts many address/staging values into AGPRs.  The marker also does not change the broad shared-view lowering that creates the extra address/LDS work.

## Diagnosis

The L6 baseline peak is best described as:

```text
Broad shared K view/address lowering creates a high-pressure staging region,
and pred/v_decay-to-update dataflow keeps enough state live that the allocator
uses many AGPRs for non-MFMA temporaries.
```

This is narrower than a generic compiler lifetime bug and broader than a single MFMA intrinsic bug.

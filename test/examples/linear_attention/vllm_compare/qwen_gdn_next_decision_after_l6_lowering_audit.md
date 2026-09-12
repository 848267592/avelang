# Qwen GDN Next Decision After L6 Lowering Audit

## Summary

The L6 lowering audit is complete.  It did not create new kernels or modify v23/v24/v26/v27/v28.

Main conclusion:

```text
The high L6 baseline AccVGPR is caused by Qwen-shaped broad K shared-view/address
lowering interacting with pred/v_decay -> update dataflow.  It is not caused by
the MFMA16 update intrinsic alone, and the current end_lifetime marker does not
touch the backend-visible allocation problem.
```

Audit artifacts:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_lowering_artifacts.py`
- `test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_isa_register_audit.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_accvgpr_peak_diagnosis.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_compiler_culprit_classification.md`

## Key Data

| variant | trace_us | VGPR | AccVGPR | scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 34.331 | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| subtile | 19.109 | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| no_pred_dependency | 9.774 | 96 | 80 | 0 | 20480 | 2048 | 69504 | 8192 | 7168 |
| minimal_frag | 3.205 | 12 | 4 | 0 | 1024 | 256 | 4096 | 2432 | 512 |

Static ISA confirms the same direction:

| variant | mfma32 | mfma16 | `v_lshl` | `v_lshl_add` | ds_read/write | max static ACC idx | explicit acc write/read |
|:---|---:|---:|---:|---:|:---|---:|:---|
| baseline | 8 | 32 | 722 | 401 | 88 / 152 | 131 | 155 / 155 |
| subtile | 8 | 32 | 410 | 242 | 80 / 104 | 15 | 32 / 32 |
| no_pred_dependency | 0 | 32 | 406 | 194 | 32 / 80 | 3 | 32 / 32 |
| minimal_frag | 0 | 4 | 14 | 4 | 4 / 4 | 3 | 4 / 4 |

## What Exactly Causes High AccVGPR?

The clearest concrete cause is not an MFMA instruction itself.  The baseline ISA contains a pre-MFMA address/staging region that writes ordinary address/data temporaries into AGPRs:

```asm
v_lshl_add_u64 ...
v_sub_u32_e32 ...
v_accvgpr_write_b32 a131, v13
v_accvgpr_write_b32 a130, v12
```

That region is much smaller or absent in the subtile variant.  Baseline has 155 explicit AGPR write/read pairs; subtile has 32.

So the high peak is best described as:

```text
broad k_all_t/kall_vec shared-view lowering creates many address/staging
temporaries; pred/v_decay dataflow into update keeps enough pressure alive
that the allocator spills ordinary temporaries into AGPRs.
```

## Why end_lifetime Did Not Help

`al.end_lifetime` did not change L6 counters:

| variant | trace_us | AccVGPR |
|:---|---:|---:|
| baseline no marker | 34.371 | 264 |
| baseline marker | 34.291 | 264 |
| subtile no marker | 19.349 | 168 |
| subtile marker | 19.189 | 168 |

Reason: the marker does not alter the backend-visible broad shared-view/address lowering, and it does not prevent non-MFMA address/staging temporaries from being allocated in AGPRs.  The no-pred-dependency variant proves dataflow matters, but the current high-level marker is not a usable backend allocation boundary.

## Why K-Subtile Helped Isolated But Failed Full Qwen

In isolated L6, subtile removes the broad K view and cuts trace by about `44%` while keeping the same dynamic MFMA count.

In full Qwen, the direct source rewrite changed the larger recurrence context and regressed badly:

| kernel | trace_us | VGPR | AccVGPR | scratch | LDS block |
|:---|---:|---:|---:|---:|---:|
| original full v29 | 817.775 | 128 | 264 | 0 | 61440 |
| k_subtile full exp | 2239.567 | 128 | 384 | 84 | 49152 |

Conclusion: do not continue source-only K-subtile tuning.  The isolated result should be used as a compiler-lowering clue, not as a production source rewrite.

## Next Compiler Fix

Recommended minimal fix:

```text
Implement or prototype a fixed-layout lowering/helper for the Qwen update K fragment.
```

Target the pattern represented today as:

```text
k_all_t[128,BT] + kall_vec packed view -> mfma_16x16x16 update B operand
```

The lowering should:

- generate compact per-update-tile DS reads;
- avoid generic full-view address arithmetic;
- reduce `v_lshl*`, LDS traffic, and explicit AGPR write/read pressure;
- avoid assigning non-MFMA address temporaries into high-index AGPRs where possible.

This is a smaller and better-supported compiler direction than broad lifetime markers or another full Qwen source copy.

## Implement Now Or Collect One More Artifact?

Collect one more backend artifact before implementing:

```text
post-ROCDL or LLVM/MIR register-allocation/liveness dump for baseline vs subtile,
focused on the region that emits v_accvgpr_write_b32 a100..a131.
```

The current audit identifies the culprit class, but not the exact pass where ordinary address temporaries become AGPRs.  That artifact will decide whether the fix belongs in MLIR shared-view lowering, LLVM address simplification, or AMDGPU register-allocation constraints.

## Next Single Action

Add a compiler dump hook for the L6 baseline/subtile JIT path that captures late LLVM/MIR or equivalent register-allocation diagnostics around the high-AGPR address/staging region.

Do not continue:

- source-level K-subtile tuning;
- `end_lifetime` marker experiments;
- full Qwen v29 copies;
- v23/v24 local tuning;
- old FullOp or generic sequential-MFMA investigations.

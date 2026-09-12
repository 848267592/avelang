# Compiler Late Lifetime Lowering Report

## Summary

This pass continued the lifetime-boundary investigation beyond the initial marker-only implementation.

Result:

- `ave.end_lifetime` now survives past `lower_ave_lang_to_memref_pass.cc`.
- A new late erase pass removes `ave.end_lifetime` after GPU outlining and before backend AMDGPU lowering.
- No LLVM `lifetime.end` is emitted in this implementation.
- Correctness-equivalent L6 variants still show no material VGPR/AccVGPR/trace improvement.
- Therefore the current late-survive/late-erase marker does not solve the Qwen-shaped lifetime/resource issue.

## Operand Audit

Detailed operand classification is in:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/lifetime_operand_audit.md`

Key points:

| operand | classification | can lower to real LLVM lifetime.end? | expected AccVGPR effect |
|:---|:---|:---:|:---|
| `pred_acc` | post-MFMA SSA/vector accumulator value | no useful memref base | low; SSA last use is already visible |
| `pred_partial` | workgroup/shared memref | not safely as `memref.dealloc` or stack lifetime | only useful for phase marker/shared reuse |
| `state_bf16` | workgroup/shared memref | not safely as stack lifetime | only useful for shared reuse/diagnostics |
| `w_bf16` | workgroup/shared memref | not safely as stack lifetime | only useful for shared reuse/diagnostics |
| `state_vec` | view/cast of `state_bf16` | only by recovering base | same as base shared memref |
| `w_vec` | view/cast of `w_bf16` | only by recovering base | same as base shared memref |

The marker intentionally does not include `v_decay_t`, K staging buffers, or state values still needed by update.

## Compiler Changes

Changed files:

- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`
- `lib/Target/GPU/lower_to_llvm.cc`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/lifetime_operand_audit.md`

Python/JIT integration from the previous marker patch remains:

- `python/avelang/language/core.py`
- `python/avelang/language/__init__.py`
- `python/avelang/runtime/jit.py`

## Marker Lifetime In The Pipeline

Previous behavior:

- `EndLifetimeLoweringPattern` erased `ave.end_lifetime` inside `lower_ave_lang_to_memref_pass.cc`.

New behavior:

- `EndLifetimeLoweringPattern` was removed from the memref lowering rewrite set.
- `ave.end_lifetime` is left as a side-effecting AveLang op after AveLang memref conversion.
- `EraseAveLangEndLifetimePass` was added in `lower_to_llvm.cc` after GPU outlining and before backend lowering.

This proves the marker can survive beyond AveLang-to-memref lowering.  It is still not a real backend-visible lifetime primitive, because the new late pass erases it before AMDGPU lowering.

## LLVM Lifetime.End Status

LLVM `lifetime.end` was not emitted.

Reason:

- The current marker operands are mostly shared/workgroup memrefs and views of shared memrefs.
- Emitting `memref.dealloc` or stack lifetime intrinsics for workgroup memory is not semantically safe.
- The post-MFMA accumulator operand does not have a stable memref base allocation suitable for `llvm.lifetime.end`.
- A real LLVM lifetime lowering would need a later pass after pointer materialization and should apply only to private/local allocations proven dead after the marker.

## Shared/Workgroup Handling

Shared/workgroup lifetime is diagnostic-only in this pass.

No shared allocator reuse was implemented.  No shared deallocation was emitted.

## L6 Variants

Correctness-equivalent variants:

- `L6_baseline_no_lifetime`
- `L6_with_end_lifetime_after_vdecay`
- `L6_subtile_no_lifetime`
- `L6_subtile_with_end_lifetime_after_vdecay`

Additional memref-only variants were added:

- `L6_memref_only_end_lifetime`
- `L6_subtile_memref_only_end_lifetime`

The memref-only variants are invalid/profiling-only in the current source form: their checksum is `0` and their MFMA count is `0`, so they are not evidence of a valid optimization.

## L6 Counters

Command:

```bash
PYTHONPATH=/workspace/project/avelang/python:/opt/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py \
  --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```

Correctness-equivalent counters:

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_no_lifetime` | `34.371` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_with_end_lifetime_after_vdecay` | `34.291` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile_no_lifetime` | `19.349` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_subtile_with_end_lifetime_after_vdecay` | `19.189` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |

Deltas:

- broad-K marker: trace `-0.080 us`, AccVGPR `0`, VGPR `0`, Scratch `0`.
- K-subtile marker: trace `-0.160 us`, AccVGPR `0`, VGPR `0`, Scratch `0`.

Invalid/profiling-only memref variants:

| variant | checksum | trace_us | VGPR | AccVGPR | MFMA |
|:---|---:|---:|---:|---:|---:|
| `L6_memref_only_end_lifetime` | 0 | `2.163` | 8 | 0 | 0 |
| `L6_subtile_memref_only_end_lifetime` | 0 | `2.284` | 8 | 0 | 0 |

These are not valid speedups.

## IR Evidence

Code-path evidence:

- `lower_ave_lang_to_memref_pass.cc` no longer contains an `EndLifetimeLoweringPattern` in the greedy lowering pattern list.
- `lower_to_llvm.cc` contains `EraseAveLangEndLifetimePass`, inserted after GPU outlining and before backend lowering.

This proves the marker survives later than the previous implementation.  It does not prove LLVM lifetime metadata is present; no LLVM lifetime intrinsic is emitted by this patch.

## Interpretation

The unchanged counters on correctness-equivalent variants show that simply keeping `ave.end_lifetime` alive until after GPU outlining is still not enough.  Because it is erased before AMDGPU lowering, the AMDGPU register allocator cannot use it.

The operand audit also weakens the expectation that ordinary LLVM lifetime intrinsics alone would fix this L6 case: the important operands are shared memrefs and MFMA accumulator/dataflow values, not simple private allocas with obvious stack lifetime ends.

## Conclusion

Late-survive/late-erase lifetime markers do not improve L6 AccVGPR/VGPR/trace.

Do not create the full Qwen lifetime-boundary copy from this result.

Next compiler action should be one of:

1. implement a real backend-visible phase boundary that AMDGPU lowering/register allocation can use;
2. add conservative shared allocation reuse/diagnostics for non-overlapping workgroup buffers;
3. investigate Qwen-shaped accumulator/dataflow lowering directly rather than relying on lifetime markers.

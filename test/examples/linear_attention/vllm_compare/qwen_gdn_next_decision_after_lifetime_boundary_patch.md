# Qwen GDN Next Decision After Lifetime Boundary Patch

## Summary

A minimal `al.end_lifetime(...)` / `al.discard(...)` mechanism has been implemented, fixed at the Python/JIT export layer, rebuilt/synchronized into the Docker Python package, and profiled on the isolated L6 repro.

No full Qwen v29 lifetime-boundary experiment was created, because the gating condition was not met:

```text
Only if isolated L6 improves, create full Qwen experimental copy.
```

## What Was Implemented

Compiler/language hook:

```python
al.end_lifetime(x, ...)
al.discard(x, ...)
```

Experiment files:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py`

Design/report files:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/compiler_lifetime_patch_design.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/compiler_lifetime_boundary_patch_report.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_lifetime_boundary_profile_report.md`

Integration fixes:

- exported `end_lifetime` / `discard` from `python/avelang/language`;
- restored JIT dependency handling for `__avelang_builtin__` stubs;
- synchronized the freshly built `build-vllm-rocm722/python/_avelang_bindings...so` into `python/_avelang_bindings...so` inside the Docker workspace.

## Did We Locate A Real Compiler Issue?

Partially.

The previous evidence suggested a pred/v_decay to update lifetime issue.  The marker-only `al.end_lifetime(...)` hook compiles and runs, but does not change AccVGPR/VGPR/instruction counters.  This supports the narrower conclusion that a frontend marker erased at AveLang-to-memref lowering is too early to affect AMDGPU register allocation.

## Did The Patch Reduce AccVGPR/Scratch/Trace?

No material reduction.

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_no_lifetime` | `34.491` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_with_end_lifetime_after_vdecay` | `34.291` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile_no_lifetime` | `19.229` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_subtile_with_end_lifetime_after_vdecay` | `19.269` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |

Deltas:

- broad-K marker: trace `-0.200 us`, AccVGPR `0`, Scratch `0`;
- K-subtile marker: trace `+0.040 us`, AccVGPR `0`, Scratch `0`.

## Is The Issue Fixed By Lifetime Boundary?

No, not by the current marker-only implementation.

The unchanged counters indicate the marker is erased too early to affect LLVM/AMDGPU register allocation.  A real fix needs a late lifetime-boundary pass or LLVM `lifetime.end`/scheduling primitive after memref/private value materialization.

## Next Single Action

Implement the next compiler step: make `ave.end_lifetime` survive beyond AveLang-to-memref lowering, or lower it to a real late-stage lifetime/scheduling primitive that can influence AMDGPU register allocation.

Do not create a full Qwen v29 lifetime-boundary copy from the current marker-only patch.  Do not continue K-subtile source-only tuning and do not modify v23/v24/v26/v27/v28.

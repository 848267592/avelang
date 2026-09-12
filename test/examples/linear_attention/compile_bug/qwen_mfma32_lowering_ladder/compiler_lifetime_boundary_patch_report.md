# Compiler Lifetime Boundary Patch Report

## Summary

Implemented a minimal Avelang lifetime-boundary marker:

```python
al.end_lifetime(x, ...)
al.discard(x, ...)
```

The marker is visible in AveLang IR and protected from early DCE by a conservative memory effect.  The current lowering is marker-only: it is erased at AveLang-to-memref lowering and does not yet become LLVM `lifetime.end` or a backend-visible AMDGPU liveness boundary.

## Compiler Files Changed

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.h`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/IR/builtin_module.h`
- `lib/IR/builtin_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`
- `python/avelang/language/core.py`
- `python/avelang/language/__init__.py`
- `python/avelang/runtime/jit.py`

## Op Design

`ave.end_lifetime`:

- variadic operands;
- no results;
- verifier requires at least one operand;
- has `MemoryEffects::Write` on the default resource;
- no numerical semantics.

Python DSL entry points:

- `al.end_lifetime(...)`
- `al.discard(...)`

The alias exists so future source can use either wording without another compiler patch.

Python frontend fix:

- `end_lifetime` and `discard` are now exported from `avelang.language`.
- Both stubs are marked `__avelang_builtin__ = True`.
- The JIT dependency collector again ignores functions marked `__avelang_builtin__`, so these DSL stubs are not mistaken for ordinary Python functions that must be decorated with `@avelang.jit`.

## Lowering Behavior

Current behavior:

1. Python frontend emits `cf::EndLifetimeOp`.
2. The op survives early AveLang IR cleanup because it is not pure and advertises a side effect.
3. `EndLifetimeLoweringPattern` erases it in `lower_ave_lang_to_memref_pass.cc`.

This is not yet a real lifetime end for LLVM or AMDGPU register allocation.

Why no stronger lowering was added in this patch:

- The relevant shared/private allocas are hoisted intentionally before/around memref lowering.
- No existing explicit lifetime/dealloc infrastructure was found in the Avelang pass path.
- Lowering to `memref.dealloc` for workgroup/private allocas would be semantically risky.
- A real private lifetime path needs a later pass after memref pointers are materialized.
- A real shared-memory reuse path needs a conservative non-overlap allocator or local rewrite pass.

## Isolated L6 Test Files

Added:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py`

The marker is inserted after `v_decay_t` staging and synchronization:

```python
al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
al.syncthreads()
```

It does not mark `v_decay_t`, K staging buffers, or state values needed by update.

Variants:

- `L6_baseline_no_lifetime`
- `L6_with_end_lifetime_after_vdecay`
- `L6_subtile_no_lifetime`
- `L6_subtile_with_end_lifetime_after_vdecay`

## Build Result

Docker ROCm build command:

```bash
docker exec ac739c57a0bf sh -lc \
  'cd /workspace/project/avelang && cmake --build build-vllm-rocm722 --target _avelang_bindings -j 16'
```

Result:

```text
_avelang_bindings.cpython-312-x86_64-linux-gnu.so linked successfully
```

## Syntax Checks

Local syntax checks passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_lifetime python3 -m py_compile \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py

PYTHONPYCACHEPREFIX=/tmp/pycache_lifetime python3 -m py_compile \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py
```

## Profiling Status

Completed in the Docker ROCm environment after fixing two integration issues:

1. `al.end_lifetime` was missing from the Python DSL export surface.
2. The Docker workspace loaded `python/_avelang_bindings...so`, which was stale.  The freshly built `build-vllm-rocm722/python/_avelang_bindings...so` was copied into `python/`.

Smoke:

```bash
PYTHONPATH=/workspace/project/avelang/python:/opt/avelang/python \
PYTHONDONTWRITEBYTECODE=1 HIP_LAUNCH_BLOCKING=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py \
  --variant all --seed 20260621 --warmup 1 --repeat 2 --json
```

Result: all four variants ran and produced finite sink checksums.

Profile:

```bash
PYTHONPATH=/workspace/project/avelang/python:/opt/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py \
  --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```

Generated:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_lifetime_boundary_profile_report.md`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary/`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary_hsaco/`

Key counters:

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_no_lifetime` | `34.491` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_with_end_lifetime_after_vdecay` | `34.291` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile_no_lifetime` | `19.229` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_subtile_with_end_lifetime_after_vdecay` | `19.269` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |

Deltas:

- Broad-K marker: trace `-0.200 us`, AccVGPR `0`, Scratch `0`.
- K-subtile marker: trace `+0.040 us`, AccVGPR `0`, Scratch `0`.

Because isolated L6 did not improve materially, the full Qwen lifetime-boundary experiment was not created or run.

## Interpretation

At this point the patch proves the frontend/dialect hook can be built and used from source, but it does not produce a compiler-resource improvement.

The unchanged AccVGPR/VGPR/instruction counters confirm the marker is erased too early to affect LLVM/AMDGPU register allocation.  The next compiler step should make the marker survive later or lower it to a real lifetime/scheduling primitive.  Source-level `al.end_lifetime(...)` alone should not be promoted to full Qwen v29.

## Next Required Commands

Do not create the full Qwen lifetime-boundary copy from this marker-only patch.

Next compiler action:

- implement a late lifetime-boundary lowering that survives beyond AveLang-to-memref lowering, or
- lower private values to a real LLVM lifetime/scheduling primitive after memref pointer materialization, then rerun the same L6 profile.

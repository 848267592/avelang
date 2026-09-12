# Qwen MFMA32 K-Staging Helper Opportunity Report

## Scope

This report inspects whether the `L4 -> L5` K staging cost has an obvious tiny compiler/source helper opportunity.

No compiler patch was implemented in this pass.  The reason is not that a helper is impossible; it is that the current static inspection does not reveal a clearly safe, narrow canonicalization that should be applied without the new L5 variant measurements.

## Commands/Areas Inspected

The requested broad search was performed with `rg` over `include`, `lib`, and `python` equivalents:

```bash
rg -n "make_shared|ViewOp|view\\(|MemoryEffects|bf16|i32|lds|shared|workgroup|mfma_16x16|mfma_32x32" avelang/include avelang/lib avelang/python
rg -n "CreateMakeSharedFunction|CreateViewFunction|AveLangMemRefViewOp|withResolvedMemorySpace|MemoryEffects|mfma_32x32|raw_buffer|make_shared" avelang/lib avelang/include avelang/python
```

Key files inspected:

- `lib/IR/builtin_module.cc`
- `lib/IR/layout_operation.cc`
- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`
- `lib/Dialect/AveLang/IR/AveLangMemRefOps.td`
- `lib/IR/Intrinsics/amdgpu_module.cc`
- `lib/IR/Intrinsics/amdgpu_mfma_signatures.h`
- `python/avelang_kernels/amdgpu_gemm.py`

## Findings

### `make_shared`

`make_shared` is registered in `builtin_module.cc` and lowered by `CreateMakeSharedFunction`, which delegates to `CreateMakeTensorWithMemorySpace(..., gpu::AddressSpace::Workgroup, "make_shared")`.

Important behavior:

- Shapes must be static.
- If only shape is provided, row-major strides are generated.
- A byte-addressed base alloca is created, then cast to the requested element/layout.
- There is no visible special-case for BF16 packed K staging or transposed Qwen layouts.

Implication:

`k_all_t = al.make_shared((128, BT), al.bf16)` creates a normal row-major workgroup memref.  The source transposition is expressed by indexing `[kk, tok]`; no obvious Qwen-specific helper exists here.

### `view(memref, dtype, layout)`

The three-argument `view` path is implemented in `layout_operation.cc` and eventually lowered through `AveLangMemRefViewOp` to `memref.view` in `lower_ave_lang_to_memref_pass.cc`.

Observed behavior:

- `view(memref, dtype, layout)` requires a `make_layout` layout.
- The lowering uses memref view/reinterpret mechanics and resolved memory space.
- The static audit did not reveal a special canonicalization that recognizes a constant packed-BF16 view like:

```python
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
```

Implication:

The expensive static ISA growth around `v_lshl`, `v_lshl_add`, `v_or`, `v_bfe`, `ds_read`, and `ds_write` plausibly comes from general layout/index lowering rather than a missed obvious one-line lowering rule.

### Packed BF16/i32 Support

Packed vector patterns exist in source examples:

- `python/avelang_kernels/amdgpu_gemm.py` uses `raw_buffer_load_x4`, shared vector views, and BF16 fragment views.
- `lib/IR/mlir_generator_test.cc` includes `view_packed_subscript` and MFMA BF16 tests.
- The standalone source-level probes previously confirmed `mfma_32x32x8_bf16_f32` and `raw_buffer_store_x4` can be generated.

But the failed first smoke of `L5_packed_i32_contiguous_load` also showed a limitation:

```text
Failed to convert from type vector<4xbf16> to type f32
```

The source had attempted to convert a vector fragment directly to f32.  The file was corrected to index one scalar element (`frag[0, 0, 0]`).  This still needs docker JIT rerun.

Implication:

Packed BF16/i32 is supported, but source must be careful about vector/scalar fragment shape.  This is not enough evidence for a compiler patch.

### MemoryEffects / aliasing

`AveLangMemRefOps.td` declares `MemoryEffectsOpInterface` on memref ops.  The static pass audit did not uncover a clear local aliasing bug in view lowering.  A conservative alias/lifetime behavior remains possible, but it should be demonstrated by the focused variants first.

## Possible Tiny Helpers

These are possible, but not yet justified by data:

1. A source-level helper for Qwen update K staging that stages only the update tile/subtile instead of full `k_all_t[128,BT]`.
2. A source-level helper that provides a packed BF16/i32 shared layout with scalar-safe accessors.
3. A very narrow canonicalization for constant-layout `view(memref,bf16->i32)` if isolated variants prove `kall_vec` is the culprit.

Not recommended yet:

- Triton-like block-dot lowering.
- A Qwen-only backend pass.
- A broad memref/view aliasing redesign.

## Variant Evidence Added

The focused L5/L6 profilers were run after the initial docker blockage cleared.

Key L5 results:

| variant | trace_us | VGPR | AccVGPR | VALU | VMEM | LDS inst | interpretation |
|:---|---:|---:|---:|---:|---:|---:|:---|
| `baseline_L5` | `36.614` | 120 | 144 | 201728 | 25088 | 19456 | full transposed K staging |
| `L5_khalf_stage_64` | `23.635` | 112 | 152 | 114304 | 16896 | 14592 | promising finite K-half staging |
| `L5_subtile_k_stage_16token` | `32.769` | 124 | 148 | 111552 | 12800 | 11520 | modest finite subtile win |
| `L5_direct_global_k_update_probe` | `13.740` | 96 | 168 | 101824 | 10752 | 11392 | diagnostic only, fastest |

Key L6 results:

| variant | trace_us | AccVGPR | MFMA | interpretation |
|:---|---:|---:|---:|:---|
| `baseline_L5_no_update` | `36.454` | 144 | 1024 | no update MFMA |
| `L6_one_update_mfma_only` | `31.727` | 320 | 1152 | one update branch already high |
| `L6_one_ktile_update` | `40.140` | 336 | 1536 | one full Ktile highest AccVGPR |
| `L6_full_update_like_current` | `33.971` | 264 | 5120 | larger loop lowers AccVGPR but stays high |
| `L6_update_acc_scope_split_source` | `34.131` | 264 | 5120 | source barrier does not help |

## Decision

No compiler/helper patch was implemented.

The evidence now supports a source/helper direction rather than a broad compiler change:

- Avoid full `k_all_t[128,BT]` staging when possible.
- Prefer staged K-half or smaller update subtile staging.
- Treat direct-global as an upper-bound diagnostic, not a production replacement.

A tiny source helper may be worthwhile if it expresses update K-half/subtile staging with less indexing overhead and can feed the existing MFMA16 update pattern.  A compiler patch is still not justified because the static audit did not identify a narrow canonicalization, and the strongest win is achievable by changing source staging granularity.

# K-Subtile Helper Design

## Search Summary

Relevant extension points found in the repo:

- MFMA wrappers:
  - `python/avelang_kernels/amdgpu_gemm.py`
  - `lib/IR/Intrinsics/amdgpu_mfma_signatures.h`
  - `lib/IR/Intrinsics/amdgpu_module.cc`
  - `lib/Dialect/AveLang/Transforms/lower_gpuop_to_intrinsics_pass.cc`
- Raw buffer helpers:
  - `al.amdgpu.raw_buffer_load_x{1,2,4}`
  - `al.amdgpu.raw_buffer_store_x{1,2,4}`
  - definitions/lowering in `lib/IR/Intrinsics/amdgpu_module.cc` and `lib/IR/expr_generator.cc`
- Shared/view lowering:
  - `make_shared` creation in `lib/IR/builtin_module.cc`
  - `AveLangMemRefViewOp` lowering in `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`
  - `AveLangMemRefSubViewOp` lowering in the same pass

## Decision

Use Option A: source-level helper/pattern inside the experimental examples.

Reason:

- The validated `L6_subtile16_stage_full_update_like` already produces a much better lowered shape without compiler changes.
- Existing Avelang primitives are sufficient: `make_shared`, `view`, scalar shared staging, and existing MFMA16 update calls.
- A new backend intrinsic or compiler rewrite would be larger than needed and harder to justify from current evidence.

## Helper Pattern

The bad broad pattern stages all `K[128,BT]`:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
```

The replacement stages only the current 16-token subtile:

```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))
```

The update reads:

```python
b_words = ksub_vec[tile * 16 + lane_col, 0]
b_words = ksub_vec[tile * 16 + lane_col, 1]
```

instead of:

```python
b_words = kall_vec[base_k + lane_col, pack_base]
b_words = kall_vec[base_k + lane_col, pack_base + 1]
```

## Why This Is Minimal

- No compiler changes.
- No new language API.
- No Triton-like block-dot lowering.
- No change to MFMA shape or update math.
- The pattern can be copied directly into a full Qwen experimental kernel while leaving all baselines untouched.

## Risk

The isolated helper improves resource counters, but full Qwen recurrence may still fail or regress due to:

- pred/v_decay live range remaining high;
- recurrence amplification;
- extra barriers from per-subtile staging;
- interaction with full chunk loop/state update.

So the next step is an experimental full Qwen copy only, not productionization.

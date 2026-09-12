# Compiler Lifetime Boundary Patch Design

## Objective

Add the smallest Avelang language/compiler hook for marking a phase boundary after Qwen-shaped pred/v_decay production and before update MFMA consumption:

```python
al.end_lifetime(pred_side_values)
```

The immediate purpose is diagnostic: determine whether an explicit source-level lifetime marker can reduce the excessive resource pressure observed when pred/v_decay temporaries remain live into the update path.

## Existing Representation

`al.full(...)` is emitted as `ave.full` (`FullOp`) in `lib/Dialect/AveLang/IR/AveLangOps.td`.  It lowers in `lower_ave_lang_to_memref_pass.cc` to a private `memref.alloca`, a flattened reinterpret-cast view, and vector/scalar stores that initialize the buffer.

`al.make_shared(...)` and `al.make_local(...)` both use `CreateMakeTensorWithMemorySpace(...)` in `lib/IR/builtin_module.cc`.  They emit AveLang memref allocas with GPU workgroup or private memory space.  The existing allocation interface explicitly allows static AveLang allocas to be hoisted to loop/block scope.

`al.view(...)` is emitted as either vector bitcast/shape-cast for vector operands or `ave.memref.cast` / `ave.memref.view` for memrefs.  AveLang memref view/cast/subview ops are lowered to builtin memref operations in `lower_ave_lang_to_memref_pass.cc`.

## Existing Lifetime Support

No existing explicit `lifetime.end`, `dealloc`, or `alloca_scope` mechanism was found in the Avelang pass path.  The relevant alloca behavior is the opposite of scoped lifetime: `allocation_op_interface_impl.cc` and `hoist_alloca_pass.cc` intentionally hoist static allocas to function entry/block scope so LLVM SROA can scalarize them.

This means shared/private allocations are effectively kernel/function-wide unless later LLVM/AMDGPU lowering can prove a shorter SSA lifetime.  `gpu.barrier` / `al.syncthreads()` is a thread synchronization point, not a register/shared allocation lifetime boundary.

## Implemented Op

New op:

```mlir
ave.end_lifetime %x, %y : type(%x), type(%y)
```

Source API:

```python
al.end_lifetime(x, y)
al.discard(x, y)
```

Files changed:

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.h`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/IR/builtin_module.h`
- `lib/IR/builtin_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`

The op has variadic operands, no results, a verifier requiring at least one operand, and a conservative `MemoryEffects::Write` effect so early DCE/CSE should not silently erase it.

## Current Lowering Behavior

The first patch is marker-only:

- The marker is visible in AveLang IR.
- It is side-effecting before AveLang-to-memref lowering.
- It is erased by `EndLifetimeLoweringPattern` in `lower_ave_lang_to_memref_pass.cc`.
- It does not yet lower to LLVM `lifetime.end`.
- It does not yet alter shared-memory allocation reuse.
- It does not yet create a late AMDGPU scheduling/register-allocation boundary.

This is intentionally minimal.  It lets the L6 repro test whether an early side-effect marker changes CSE/hoist/lowering behavior.  If counters do not change, the reason is expected: the marker is removed too early for LLVM/AMDGPU register allocation to consume.

## Realistic Effects

Private lifetime:

- A real implementation would need to lower memref-like operands to LLVM `llvm.lifetime.end` after memref-to-LLVM pointer materialization.
- That requires a late pass where the address/pointer of the relevant alloca or stack slot is available.

Shared-memory lifetime:

- Current shared buffers are static workgroup allocas.
- Reusing shared allocation for non-overlapping lifetimes would need a conservative allocator or a local rewrite pass that can prove non-overlap.
- The current marker does not do this.

Scheduling/liveness:

- The desired Qwen-specific effect is a phase boundary between pred/v_decay and update.
- A high-level marker erased before GPU outlining cannot directly constrain AMDGPU register allocation.
- A future pass would need to preserve the boundary until after MFMA lowering or convert it into a backend-visible no-code lifetime/scheduling marker.

## Next Validation

Use:

- `repro_qwen_mfma32_l6_lifetime_boundary.py`
- `profile_qwen_mfma32_l6_lifetime_boundary.py`

Required variants:

- `L6_baseline_no_lifetime`
- `L6_with_end_lifetime_after_vdecay`
- `L6_subtile_no_lifetime`
- `L6_subtile_with_end_lifetime_after_vdecay`

If AccVGPR/trace do not change, the next compiler action is not more source tuning.  It is a late lifetime pass or LLVM lifetime lowering that survives to the stage where register/shared allocation decisions are made.

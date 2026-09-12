# L6 Lifetime Operand Audit

## Marker Site

Current original-operand marker in `repro_qwen_mfma32_l6_lifetime_boundary.py`:

```python
al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
```

The marker is placed after `v_decay_t` has been staged and synchronized, and before the update MFMA path starts.

## Operand Classification

| operand | source role | classification | base allocation | survives memref lowering | used after marker | real lifetime.end possible | expected VGPR/AccVGPR impact |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| `pred_acc` | MFMA32 accumulator for `W @ state.T` partial | starts as `al.full`, then becomes MFMA result value | no stable memref base after MFMA assignment | as SSA/vector value, not as a memref allocation | no | no useful memref lifetime.end | Low. SSA last-use is already explicit; keeping marker as a real operand can become an extra use and extend lifetime. |
| `pred_partial` | shared `[2,32,BV]` partial-pred reduction buffer | workgroup/shared memref | yes, shared allocation | yes | no | not as `memref.dealloc`; shared memory dealloc is unsafe | Could help only with a backend-visible phase marker or shared allocation reuse. No direct AccVGPR guarantee. |
| `state_bf16` | staged old state for pred MFMA | workgroup/shared memref | yes, shared allocation | yes | no | not as `memref.dealloc`; shared memory dealloc is unsafe | Could help LDS allocation reuse if a conservative shared allocator used the boundary. Not expected to reduce AccVGPR by itself. |
| `w_bf16` | staged W tile for pred MFMA | workgroup/shared memref | yes, shared allocation | yes | no | not as `memref.dealloc`; shared memory dealloc is unsafe | Same as `state_bf16`: useful mainly for shared allocation reuse or phase diagnostics. |
| `state_vec` | packed view of `state_bf16` | view/cast of shared memref | base is `state_bf16` | yes, as view/cast until lowered | no | only by recovering base allocation | Same as base `state_bf16`; marker should trace through view/cast to the base if doing shared diagnostics. |
| `w_vec` | packed view of `w_bf16` | view/cast of shared memref | base is `w_bf16` | yes, as view/cast until lowered | no | only by recovering base allocation | Same as base `w_bf16`; marker should trace through view/cast to the base if doing shared diagnostics. |

## Values Deliberately Not Marked

| value | reason |
|:---|:---|
| `v_decay_t` / `vdecay_vec` | Still needed by the update MFMA path. Ending its lifetime here would be wrong. |
| K staging buffers | Created after the marker and needed by update. |
| `state_in`, `k`, `w`, `u`, `decay`, `sink` | Global inputs/outputs or externally visible buffers. Do not lifetime-end. |

## Lowering Implications

- Pure SSA/vector values such as the post-MFMA `pred_acc` do not have a memref base allocation.  A late marker operand on such a value is not a safe route to lower to `llvm.lifetime.end`; the backend already has SSA last-use information.
- Shared/workgroup memrefs (`pred_partial`, `state_bf16`, `w_bf16`) must not be lowered to `memref.dealloc` or LLVM stack lifetime intrinsics.  Their useful interpretations are:
  - backend-visible phase marker;
  - diagnostic/non-overlap reporting;
  - conservative shared allocation reuse in a future allocator.
- View operands (`state_vec`, `w_vec`) should be traced back to `state_bf16` and `w_bf16` for any real shared-memory analysis.
- A private/local memref lifetime lowering would only be applicable to values that remain memref-backed private allocations at the marker.  The current L6 marker operands are dominated by shared buffers and a post-MFMA accumulator SSA value, so LLVM `lifetime.end` is not expected to reduce AccVGPR here.

## Expected Outcome

If `ave.end_lifetime` is merely preserved later and then erased before AMDGPU lowering, no resource improvement is expected.  If it is lowered to LLVM lifetime intrinsics only for private memrefs, this L6 marker still may not improve AccVGPR because its important live pressure is accumulator/dataflow related and its memref operands are shared.  A future useful implementation would likely need either:

- a backend-visible phase boundary that helps scheduling/register allocation around pred-to-update, or
- shared allocation reuse/diagnostics for non-overlapping workgroup buffers.

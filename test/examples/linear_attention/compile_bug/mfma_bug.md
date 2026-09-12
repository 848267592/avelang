# Avelang MFMA Sequential-Regions Bug: Root-Cause Archive

This document records the bug, the failing repro, the temporary return to the
broken compiler state, and the final fix.

The canonical final report is still:

```text
test/examples/linear_attention/vllm_compare/avelang_mfma_final_compiler_bug_report.md
```

## Short version

The bug was not a Qwen math issue and not a source-level shared-memory layout
mistake. The root cause was that `FullOp` was incorrectly marked `Pure`:

```td
def FullOp : AveLang_Op<"full", [Pure]>
```

`al.full(...)` materializes a fresh temporary tensor/private memref, so two
independent zero initializers are not interchangeable. Because `FullOp` was
marked pure, CSE was allowed to merge those initializers. In a kernel that runs
an MFMA pred region followed by an MFMA update region, the update accumulator
was reused from the pred accumulator instead of starting from zero.

The fix was:

```td
def FullOp : AveLang_Op<"full", []>
```

## What the original code looked like

The important pattern was two independent MFMA regions in the same kernel:

```python
pred_acc = al.full((4,), 0.0, al.f32)
pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(..., pred_acc)

acc = al.full((4,), 0.0, al.f32)
acc = al.amdgpu.mfma_16x16x16_bf16_f32(..., acc)
```

That second `al.full((4,), 0.0, al.f32)` must create a fresh zero accumulator.
Before the fix, the compiler was allowed to treat the two `al.full` ops as
freely mergeable pure values, which is what made the update MFMA inherit the
pred accumulator.

The standalone repro now also writes the pred MFMA result into a small global
`pred_sink` so the pred MFMA is not merely a dead value in source.

## How to reproduce the broken compiler state

The actual broken state was reproduced by temporarily changing `FullOp` back to
`[Pure]`, rebuilding, and forcing Python to load the rebuilt bindings:

```bash
cd /workspace/project/avelang

# temporary debug-only change
# def FullOp : AveLang_Op<"full", [Pure]> {

ninja -C build-vllm-rocm722 -j 16

PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python \
HIP_LAUNCH_BLOCKING=1 \
PYTHONDONTWRITEBYTECODE=1 \
python test/examples/linear_attention/compile_bug/repro_mfma_fullop_pure_cse.py
```

Important detail: if you do not set `PYTHONPATH` to prefer
`build-vllm-rocm722/python`, Python can keep loading the already-fixed
`python/_avelang_bindings*.so` and the bug will not appear.

## Repro logs from the broken state

Standalone compiler-team repro:

```text
torch=2.10.0+rocm7.2.2.git40d237bf
hip=7.2.53211
device=AMD Instinct MI300X
expected_before_fix=pred_one_mfma_then_update fails when FullOp has Pure
expected_after_fix=all cases pass when FullOp has no Pure trait
case=update_only,ok=True,max_abs=4.76837158e-07,max_rel=2.57184802e-05
case=no_op_pred_no_mfma,ok=True,max_abs=4.76837158e-07,max_rel=2.57184802e-05
case=pred_one_mfma_then_update,ok=False,max_abs=34.7282333,max_rel=12222.3633
```

Regression test:

```text
FAILED test_mfma_sequential_regions_regression.py::TestMFMASequentialRegionsRegression::test_pred_one_mfma_then_update
AssertionError: False is not true : pred_one_mfma_then_update max_abs=36.7038459777832
```

Full isolation suite:

```text
update_only_baseline         pass
dummy_footprint              pass
pred_restaged_existing       fail, max_abs=36.7384834
all_shared_declared_at_top   fail, max_abs=36.7384834
no_dead_store_before_pred    fail, max_abs=36.7384834
no_op_pred_no_mfma           pass
pred_mfma_consume_acc        fail, max_abs=36.7384834
pred_one_mfma_only           fail, max_abs=36.703846
pred_two_mfma_only           fail, max_abs=36.6977043
extra_barriers_dummy_lds_reads fail, max_abs=36.7384834
padded_unique_shared_buffers_canary fail, max_abs=36.7384834
```

These results are the important part:

- `update_only_baseline` passes.
- `no_op_pred_no_mfma` passes.
- One real pred MFMA is enough to trigger the later update corruption.
- Extra barriers, dummy LDS reads, different shared-buffer layouts, and dummy
stores do not fix it.

## What changed to make it pass

The fix was only the `FullOp` trait change:

```diff
-def FullOp : AveLang_Op<"full", [Pure]> {
+def FullOp : AveLang_Op<"full", []> {
```

After rebuilding the compiler with that change, the same repro passes:

```text
case=update_only,ok=True,max_abs=4.76837158e-07,max_rel=2.57184802e-05
case=no_op_pred_no_mfma,ok=True,max_abs=4.76837158e-07,max_rel=2.57184802e-05
case=pred_one_mfma_then_update,ok=True,max_abs=4.76837158e-07,max_rel=2.57184802e-05
```

The regression test also passes again:

```text
3 passed in 6.66s
```

## Why this is not a pred-dead-code bug

There are two separate ideas here:

1. If a pred MFMA result is truly unused, removing it is legal dead-code
   elimination.
2. The actual bug was not that pred MFMA was removed. The bug was that two
   independent `al.full` initializers were CSE-merged because `FullOp` was marked
   pure, so the later update MFMA inherited the earlier pred accumulator.

That is why `no_op_pred_no_mfma` passes: pred staging and control flow are fine
when no pred accumulator exists. But once a real pred MFMA executes, the later
update should still start from a fresh zero accumulator, and that was the piece
that broke.

So the correct answer is:

- `pred_mfma` itself was not the root bug.
- The source was not “shared accumulator” code.
- The compiler incorrectly treated `al.full` as a pure value and merged the two
  zero-initializer ops.

## Which version had the problem

The problem showed up on the `v11` integrated MFMA update path in Qwen GDN.
Earlier `v9/v10` scalar update fallbacks did not expose the issue.

In compiler terms, the bug exists in any AveLang build where `FullOp` still has
the `Pure` trait.

## Aftermath

The canonical report now lives in:

```text
test/examples/linear_attention/vllm_compare/avelang_mfma_final_compiler_bug_report.md
```

This file is kept as the concise archive for the compile-bug incident, the
broken-state repro, and the reasoning that ruled out the older LDS/dead-store
hypothesis.

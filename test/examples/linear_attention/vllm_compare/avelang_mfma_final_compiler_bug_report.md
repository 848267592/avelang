# Avelang MFMA sequential-regions compiler bug report

## Resolution Status

Resolved in this workspace.

The root cause was **not LDS allocator reuse** and not a waitcnt/barrier issue. The root cause was that `FullOp` was incorrectly marked `Pure` even though it materializes a fresh temporary tensor/private memref. CSE was therefore allowed to merge independent `al.full((4,), 0.0, al.f32)` initializers. In sequential MFMA regions, the update accumulator initializer was merged with the pred accumulator initializer and, after promotion/lowering, the first update MFMA used the pred MFMA result as its accumulator instead of zero.

Fix:

```diff
-def FullOp : AveLang_Op<"full", [Pure]> {
+def FullOp : AveLang_Op<"full", []> {
```

Fixed file:

```text
lib/Dialect/AveLang/IR/AveLangOps.td
```

Regression test added:

```text
test/examples/linear_attention/vllm_compare/test_mfma_sequential_regions_regression.py
```

The regression covers:

```text
update_only_baseline         pass
no_op_pred_no_mfma           pass
pred_one_mfma_then_update    failed before fix, passes after fix
```

Post-fix validation in Docker `ac739c57a0bf` on MI300X:

```bash
cd /workspace/project/avelang
HIP_LAUNCH_BLOCKING=1 python -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_mfma_sequential_regions_regression.py -s
# ...
# 3 passed in 6.85s

python -m pytest -q test/examples/gemm/amdgpu/test_amdgpu_gemm.py -s
# .
# 1 passed in 3.25s

cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 python repro_v11_bf16_correctness.py --both
# both no-initial-state and initial-state cases passed
```

Full isolation suite after the fix:

```text
update_only_baseline                 pass, max_abs=9.53674316e-07
dummy_footprint                      pass, max_abs=9.53674316e-07
pred_restaged_existing               pass, max_abs=9.53674316e-07
all_shared_declared_at_top           pass, max_abs=9.53674316e-07
no_dead_store_before_pred            pass, max_abs=9.53674316e-07
no_op_pred_no_mfma                   pass, max_abs=9.53674316e-07
pred_mfma_consume_acc                pass, max_abs=9.53674316e-07
pred_one_mfma_only                   pass, max_abs=9.53674316e-07
pred_two_mfma_only                   pass, max_abs=9.53674316e-07
extra_barriers_dummy_lds_reads       pass, max_abs=9.53674316e-07
padded_unique_shared_buffers_canary  pass, max_abs=9.53674316e-07
```

LLVM IR evidence:

Before the fix, the first update MFMA used the pred MFMA result as its accumulator:

```llvm
%839 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
%854 = call <4 x float> @llvm.amdgcn.mfma...( ..., %839, ... )
```

After the fix, each new update tile starts from `zeroinitializer` again:

```llvm
%605 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
...
%652 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
```

Generated evidence files:

```text
test/examples/linear_attention/vllm_compare/issue_evidence/pred_one_then_update.ll
test/examples/linear_attention/vllm_compare/issue_evidence/pred_one_then_update_after_fix.ll
test/examples/linear_attention/vllm_compare/issue_evidence/update_only.s
test/examples/linear_attention/vllm_compare/issue_evidence/pred_one_then_update.s
```

## Summary

This was not a Qwen GDN math bug and not a normal source-level shared-memory layout bug. The minimal repro originally looked like an Avelang backend/LDS/MFMA-liveness bug, but the final root cause is higher-level: `FullOp` was marked pure even though it allocates/materializes a fresh temporary tensor.

The critical trigger was: an unrelated pred MFMA region executes before an update MFMA region in the same kernel. The update inputs are fully restaged from global memory and use distinct shared buffers, but the update output was silently corrupted because the update accumulator zero initializer had been CSE-merged with the pred accumulator initializer.

## Environment

| Field | Value |
|---|---|
| GPU | AMD Instinct MI300X |
| PyTorch | 2.10.0+rocm7.2.2.git40d237bf |
| HIP | 7.2.53211 |
| ROCm runtime | 1.18 |
| Docker container | `ac739c57a0bf` |
| Source file | `test/examples/linear_attention/vllm_compare/prototype_qwen_gdn_mfma_delta_staged.py` |

`git rev-parse HEAD` did not return a commit in the Docker checkout used for this run.

## Command

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 python prototype_qwen_gdn_mfma_delta_staged.py --isolation-experiments
```

## Expected computation

All kernels compute or should preserve the same update result:

```python
expected = init * scale + v_decay.float().T @ k_chunk.float()
```

Shape:

```text
v_decay: [16,16] BF16
k_chunk: [16,128] BF16
init:    [16,128] FP32
out:     [16,128] FP32
```

## Pre-fix isolation results

These are the isolation results from before the `FullOp` fix. They were useful for
ruling out source-level math/shared-memory mistakes, but they are not the final
root-cause explanation by themselves.

| test | expected | status | max_abs | max_rel | conclusion |
|---|---:|---:|---:|---:|---|
| update_only_baseline | pass | pass | 9.53674316e-07 | 7.51819925e-07 | update MFMA alone is correct |
| dummy_footprint | pass | pass | 9.53674316e-07 | 7.51819925e-07 | static shared footprint alone is not the bug |
| pred_restaged_existing | fail | fail | 36.7384834 | 902.786987 | existing pred MFMA + full restage + update reproduces corruption |
| all_shared_declared_at_top | pass-if-shared-lifetime-bug | fail | 36.7384834 | 902.786987 | not caused by `make_shared` after pred MFMA |
| no_dead_store_before_pred | pass-if-dead-store-bug | fail | 36.7384834 | 902.786987 | not caused by dead stores into update shared buffers before pred |
| no_op_pred_no_mfma | pass-if-MFMA-trigger | pass | 9.53674316e-07 | 7.51819925e-07 | pred staging/loop without pred MFMA is safe; MFMA instruction is the trigger |
| pred_mfma_consume_acc | pass-if-dead-acc-bug | fail | 36.7384834 | 902.786987 | consuming pred accumulator does not fix it; not dead-acc elimination |
| pred_one_mfma_only | diagnostic | fail | 36.703846 | 906.002258 | one pred MFMA is enough to corrupt following update MFMA |
| pred_two_mfma_only | diagnostic | fail | 36.6977043 | 905.885193 | two pred MFMAs also corrupt |
| extra_barriers_dummy_lds_reads | pass-if-waitcnt-barrier-bug | fail | 36.7384834 | 902.786987 | extra barriers and dummy LDS reads do not fix it |
| padded_unique_shared_buffers_canary | pass-if-layout-overlap-bug | fail | 36.7384834 | 902.786987 | padding does not fix; canary shows no source-level overwrite |

## Per-tile max_abs

| test | 0:16 | 16:32 | 32:48 | 48:64 | 64:80 | 80:96 | 96:112 | 112:128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| update_only_baseline | 9.53674316e-07 | 4.76837158e-07 | 9.53674316e-07 | 4.76837158e-07 | 4.76837158e-07 | 9.53674316e-07 | 9.53674316e-07 | 2.38418579e-07 |
| dummy_footprint | 9.53674316e-07 | 4.76837158e-07 | 9.53674316e-07 | 4.76837158e-07 | 4.76837158e-07 | 9.53674316e-07 | 9.53674316e-07 | 2.38418579e-07 |
| pred_restaged_existing | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |
| all_shared_declared_at_top | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |
| no_dead_store_before_pred | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |
| no_op_pred_no_mfma | 9.53674316e-07 | 4.76837158e-07 | 9.53674316e-07 | 4.76837158e-07 | 4.76837158e-07 | 9.53674316e-07 | 9.53674316e-07 | 2.38418579e-07 |
| pred_mfma_consume_acc | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |
| pred_one_mfma_only | 0.13134861 | 12.4624643 | 20.969101 | 23.1918125 | 26.6854115 | 33.5090218 | 32.7329216 | 36.703846 |
| pred_two_mfma_only | 0.165035725 | 12.4995155 | 20.9335785 | 23.1562901 | 26.679266 | 33.5598602 | 32.7837639 | 36.6977043 |
| extra_barriers_dummy_lds_reads | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |
| padded_unique_shared_buffers_canary | 0.345523 | 12.4388371 | 20.8295784 | 23.05229 | 26.7200489 | 33.3836937 | 32.6075935 | 36.7384834 |

## Canary result

The padded shared-buffer canary used padding-only locations in each shared buffer:

```text
h0/h1/w0/w1 padding column 79
v_decay_t2 padding column 31
k_all_t2 padding rows 128:143
```

Result:

```text
canary_before_max_abs = 0
canary_after_max_abs  = 0
```

The canary did not observe source-level shared-memory overwrite. This does not prove compiler LDS offsets are correct in generated code, but it rules out an obvious user-level write into the canary slots.

## Final root-cause interpretation

The old working theory was that this might be an LDS allocator/liveness issue, a
dead-store issue, or a waitcnt/barrier issue. That theory is now superseded.

The actual root cause was `FullOp` being incorrectly marked `Pure`:

```td
def FullOp : AveLang_Op<"full", [Pure]>
```

This was wrong because `al.full(...)` materializes a fresh temporary tensor/private
memref. Two separate calls like these are not interchangeable values:

```python
pred_acc = al.full((4,), 0.0, al.f32)
acc = al.full((4,), 0.0, al.f32)
```

Because `FullOp` was marked pure, CSE was allowed to merge the independent zero
initializers. After lowering, the first update MFMA used the pred MFMA result as
its accumulator instead of starting from zero.

Before the fix, the LLVM IR showed this bad dataflow:

```llvm
%839 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
%854 = call <4 x float> @llvm.amdgcn.mfma...( ..., %839, ... )
```

After removing `Pure` from `FullOp`, the update MFMA starts from a fresh zero
initializer again:

```llvm
%605 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
...
%652 = call <4 x float> @llvm.amdgcn.mfma...( ..., zeroinitializer, ... )
```

## Clarification on dead code

The no-op pred case passing does not mean the bug was caused by dead code.
It only showed that pred staging and the surrounding control flow were harmless
when no pred MFMA wrote an accumulator value.

Similarly, the fact that earlier pred results were unused is not itself a bug:
a compiler is allowed to remove genuinely dead pred computation. The corruption
appeared when an actual pred MFMA remained in the IR and its accumulator value was
incorrectly reused by a later, logically independent update MFMA because the two
`al.full` zero initializers had been CSE-merged.

So the correct distinction is:

- Normal and legal: removing an unused pred MFMA result if it is truly dead.
- Buggy behavior: merging two separate `al.full` temporary tensor initializers so
  the later update MFMA starts from the earlier pred accumulator.

## Ruled-out causes

These were useful intermediate checks, but they are not the final root cause:

- Not Qwen GDN math.
- Not source-level shared-memory layout misuse.
- Not ordinary dead stores into update shared buffers.
- Not fixed by declaring all shared buffers at top.
- Not fixed by extra barriers or dummy LDS reads.
- Not fixed by consuming `pred_acc`.
- Not explained by canary-detected source-level shared overwrite.

## Current v11 implication

Before the compiler fix, the safe Qwen v11 path was pred MFMA plus scalar BF16
update. After the `FullOp` fix and regression validation, integrated update MFMA
can be re-tried, but it still needs its own Qwen v11 correctness and performance
validation before becoming the default path.

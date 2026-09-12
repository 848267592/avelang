# Qwen Late B-Fragment Lowering Report

## Summary

This was one narrow compiler attempt to keep the Qwen update MFMA16 B-fragment
as a dedicated operation beyond producer-consumer rewriting, then lower it to
a direct 8-byte workgroup-memory load. It compiled and ran, but it did **not**
pass the pressure-reduction gate on the real-MFMA32 reduced repro. Stop here;
do not apply it to full v29.

The earlier root cause remains: full v29 combines the MFMA32 pred accumulator,
`pred_partial/state/v_decay`, the 32-window loop, and the update B-load at a
register-allocation threshold. The first direct LDS lowering does not remove
enough live state to cross that threshold.

## Changed Compiler Files

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.h`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_kfrag_lds_pass.{h,cc}`
- `lib/Dialect/AveLang/Transforms/CMakeLists.txt`
- `lib/Target/GPU/lower_to_llvm.cc`

`AVELANG_QWEN_KFRAG_LATE_BLOAD=1` selects the experiment. The default remains
the prior generic `vector.load` rewrite, so no existing baseline changes.

## Lowering Design

The old rewrite replaced each persistent fragment consumer immediately with:

```mlir
vector.load %compact[%k_column, %local_token] : memref<128x64xbf16, 3>, vector<4xbf16>
```

The experimental path instead creates
`ave.gpu.amdgpu_qwen_update_kfrag_lds_load(base, k_column, local_token)`.
After GPU outlining, `lower-qwen-kfrag-lds` computes the fixed row-major byte
address `(k_column * 64 + local_token) * 2` and emits one aligned
`llvm.load <4 x bf16>` from address space 3. It creates no temporary alloca,
no generic `vector.load`, and no dynamic memref/vector GEP chain for the B
fragment itself. The intended AMDGPU selection is `ds_read_b64`.

Debug evidence from the late run:

```text
persistent_ops_seen=4
rewritten=1 ... broad_producer_stores_erased=1 ... late_bfrag=1
```

Thus the producer match fired, the broad producer was erased, and the
persistent operation survived through the rewrite boundary to the late pass.

## Reduced Repro

`repro_qwen_kfrag_full_loop_regression.py` was extended for R2-R4 to carry the
real v29 `mfma_32x32x8_bf16_f32` pred schedule, a 16-element pred accumulator,
unpacked `pred_partial[2,32,32]`, workgroup state, BF16 v-decay staging, and
the four update MFMA16 B-fragment consumers over 32 windows. Stable small
inputs are used because this is a lowering sink rather than a normalized GDN
recurrence reference.

New launch wrappers:

- `repro_qwen_late_bfrag_with_real_pred.py`
- `profile_qwen_late_bfrag_with_real_pred.py`

## Measurements

At the reduced full-loop R4 gate (128-thread workgroup, grid 4096):

| path | normal median ms | VGPR | AccVGPR | Scratch | LDS block | MFMA | LDS inst | Occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| A generic `vector.load` | 0.5292 | 128 | 144 | 0 B | 61440 B | collected | 644288 | 0.6413% |
| B direct late LDS load | 0.5634 | 124 | 148 | 0 B | 61440 B | collected | 644288 | 0.6438% |

The profiler CSVs were produced under Docker at `/tmp/qwen_late_generic` and
`/tmp/qwen_late_direct`; the reusable command is in the decision note below.
Both variants produced finite diagnostic sinks. This reduced sink is not a
reference GDN calculation, so its checksum is only a finite-execution check;
it is not evidence of full-v29 correctness.

## Result And Exact Blocker

The dedicated operation works architecturally: it survives the early rewrite
and is not lowered by the generic vector-load path. However, the direct LLVM
load path changes only a small part of the live region. It saves four ordinary
VGPRs but raises AccVGPR by four and slows the reduced kernel slightly. The
generic control already has Scratch=0, so the required scratch-removal signal
cannot be demonstrated here.

Consequently this attempt does **not** establish that direct packed LDS reads
will fix full v29's `AccVGPR=384` / 736-B scratch failure. The likely blocker
is the full MFMA32 pred/live-region composition rather than the final generic
B-fragment vector-load instruction alone. A second lowering rewrite would be
speculation and is intentionally not attempted.

## Reproduction

```bash
cd /workspace/project/avelang
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
AVELANG_QWEN_KFRAG_LATE_BLOAD=0 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_kfrag_full_loop_regression.py \
  --variant R4_full_loop_skeleton_rewrite --warmup 5 --repeat 20 --json

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
AVELANG_QWEN_KFRAG_LATE_BLOAD=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_kfrag_full_loop_regression.py \
  --variant R4_full_loop_skeleton_rewrite --warmup 5 --repeat 20 --json
```

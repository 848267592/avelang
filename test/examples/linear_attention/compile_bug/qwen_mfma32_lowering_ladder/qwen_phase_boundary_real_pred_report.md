# Qwen Pred/Update Phase-Boundary Report

## Exact Question

The isolated L6 producer-consumer rewrite improves from `34.412 us` to
`18.628 us` and `AccVGPR 264` to `180`, while full v29 regresses to
`AccVGPR 384` with `736 B` scratch. This experiment asks whether a *real*
pred-to-update dataflow boundary, rather than an erased lifetime marker, can
remove the full-composition pressure.

Previously excluded primary causes are repeated global K load, broad K-stage
residue, scalar reload cloning, a final B-fragment vector load alone,
end-lifetime markers, and source K-subtiles.

## Reduced Repro And Boundary

The existing `repro_qwen_kfrag_full_loop_regression.py` now contains the real
v29 MFMA32 pred schedule for the pred-bearing variants:

- two K-half `mfma_32x32x8_bf16_f32` accumulator regions;
- `pred_partial[2,32,32]` workgroup materialization;
- BF16 `v_decay_t[32,64]` staging;
- shared state and the four update MFMA16 B-fragment consumers;
- 32 windows of loop pressure.

Variants:

| variant | structure |
|:--|:--|
| A `A_current_fused` | Existing fused pred/update source shape. |
| B `B_hard_shared_phase_boundary` | Pred accumulator only writes `pred_partial`; an explicit workgroup barrier follows, and all downstream pred consumers reload `pred_partial`. No `pred_acc` or pre-boundary `pred_live` SSA value feeds the update region. |
| C `C_no_pred_accumulator_control` | Keeps the update path but removes the MFMA32 pred accumulator schedule. |

The barrier establishes workgroup data visibility; unlike `end_lifetime`, the
B source shape also removes the pred-accumulator SSA use chain after the
materialization point. The update still consumes shared `v_decay_t`, not a
pred accumulator vector.

## Finite Benchmark Gate

MI300 Docker measurements, 128 threads, 32-window reduced loop, generic
B-load control (`AVELANG_QWEN_KFRAG_LATE_BLOAD=0`):

| variant | median ms | finite |
|:--|--:|:--|
| A current fused | `0.532050` | yes |
| B hard shared phase boundary | `0.546571` | yes |
| C no pred accumulator control | `0.334236` | yes |

B is `2.73%` slower than A. C is `37.18%` faster than A. This is strong
evidence that the real MFMA32 pred phase is a pressure amplifier, but the
specific phase-boundary materialization does not relieve it.

The diagnostic sink is intentionally not a full GDN numeric reference: it
uses stable scaled random inputs to keep the synthetic 32-window recurrence
finite. A and B execute the same pred/update MFMA counts by construction, but
their sinks represent different allowed phase-boundary observations, so this
is a pressure gate rather than a full-forward equivalence test.

## IR/Dataflow Interpretation

In A, the MFMA32 accumulator writes `pred_partial` and immediately derives
`pred_live`; update-adjacent v-decay preparation can therefore retain the
pred-side region under the same fused control flow. In B, `pred_acc` has only
the workgroup-store use. After `al.syncthreads()`, correction/v-decay reloads
`pred_partial`; the update phase has no direct SSA/vector operand from
`pred_acc`.

Thus B is a real source-level dataflow cut, not a marker. Its loss shows that
the remaining pressure is not explained by one direct pred-accumulator SSA
edge. The shared materialization/barrier adds coordination and LDS lifetime
without enough allocator benefit in this reduced composition.

## Rocprof Results

Twelve/nearby repeated dispatches were collected using the existing Docker
MI300 ROCm environment. Counter values are invariant per dispatch; trace is
the median of the matching kernel rows.

| variant | trace us | VGPR | AccVGPR | SGPR | Scratch | LDS block | MFMA | VALU | SALU | VMEM | LDS inst | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A current fused | `513.423` | 128 | 144 | 112 | 0 B | 61440 B | 147456 | 1736576 | 406272 | 206848 | 644288 | 0.6431% |
| B hard shared boundary | `539.181` | 128 | 144 | 112 | 0 B | 61440 B | 147456 | 1736576 | 406272 | 206848 | 644288 | 0.6385% |
| C no pred accumulator | `314.126` | 124 | 132 | 112 | 0 B | 20480 B | 131072 | 554560 | 272640 | 172032 | 425984 | 0.6333% |

A and B are identical in every collected dynamic instruction/resource metric.
The added barrier does not shorten the allocation seen by the backend; it only
adds synchronization cost, increasing trace by `25.758 us` (`5.02%`). C's
lower AccVGPR and trace confirm that the MFMA32 pred region is the dominant
amplifier, not the update MFMA16 instruction by itself.

The rocprof artifacts are under:

```text
test/examples/linear_attention/rocprof_outputs/qwen_phase_boundary_real_pred/
```

Static HSACO/MIR counts were deliberately not regenerated after this negative
counter gate: the dynamic A/B identity establishes that this one permitted
boundary attempt made no allocation change. The source-level IR evidence is
the explicit B-only `pred_partial` workgroup store -> `syncthreads` -> reload
chain described above; A keeps the immediate `pred_live` SSA path.

Reproduce:

```bash
cd /workspace/project/avelang
PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
PYTHONDONTWRITEBYTECODE=1 AVELANG_QWEN_KFRAG_LATE_BLOAD=0 \
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_phase_boundary_real_pred.py
```
cd /home/jiandongliu/project/avelang

cat > /tmp/qwen_phase_boundary_compiler_fix_prompt.txt <<'PROMPT'
You are working in:

/home/jiandongliu/project/avelang

Main objective:
Continue the exact isolated-K-to-full failure investigation.

Do NOT switch to Triton comparison.
Do NOT pursue split-kernel pred/update.
Do NOT tune v24.
Do NOT continue v29 blind patching.
Do NOT continue K-subtile, end_lifetime marker, read-side helper, or late B-load variants.

The precise question is:

Why does the K-fragment producer-consumer rewrite succeed in isolated L6 but fail in full v29, and can we fix the full composition by introducing a real phase/dataflow boundary between the MFMA32 pred phase and the MFMA16 update phase inside the same kernel?

Known facts:
1. Isolated L6 producer-consumer rewrite succeeded:
   trace 34.412 us -> 18.628 us
   AccVGPR 264 -> 180
   Scratch 0

2. Full v29 kfrag rewrite preserved semantics but regressed:
   0.837143 ms -> 1.338669 ms
   AccVGPR 264 -> 384
   Scratch 0 -> 736 B
   VMEM 399360 -> 601984

3. Root-cause audit classified the failure as Category E:
   full-v29 MFMA32 pred accumulator/live region
   + pred_partial/state/v_decay
   + 32-window loop
   + update MFMA16 path
   cross the register allocation threshold.

4. The following are already excluded as primary causes:
   - repeated global K load;
   - old broad K staging residue;
   - scalar reload cloning;
   - final B-fragment vector.load alone;
   - lifetime/end_lifetime marker;
   - source-level K-subtile.

Current hypothesis:
The true remaining problem is that pred-phase SSA/vector/accumulator live ranges are not structurally cut before update-phase lowering. A marker is insufficient. We need a real dataflow/IR phase boundary so that update MFMA16 does not overlap with pred MFMA32 accumulator and pred_partial/state/v_decay temporaries more than necessary.

Do NOT:
- Do not implement two separate production kernels.
- Do not create per-chunk kernel launches.
- Do not compare against Triton.
- Do not modify LLVM/AMDGPU register allocator.
- Do not hard-ban AGPR.
- Do not touch v23/v24/v26/v27/v28.
- Do not create a new production full Qwen variant.
- Do not make broad compiler architecture rewrites.
- Do not try more K-load local variants.

Allowed:
- Use or extend existing reduced repros.
- Add one new reduced repro if needed.
- Add compiler diagnostics to show live/dataflow crossing the pred/update boundary.
- Add at most one narrow phase-boundary compiler/source-lowering experiment.
- Test only reduced repro first.
- Test full v29 experiment only if reduced gate succeeds.

================================================================================
TASK 1: Build the right reduced repro
================================================================================

Use the existing root-cause repros if possible:

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_kfrag_full_loop_regression.py
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_late_bfrag_with_real_pred.py

If needed, create:

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_phase_boundary_real_pred.py

and profiling script:

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_phase_boundary_real_pred.py

The repro must contain:
- real MFMA32 pred accumulator schedule;
- pred_partial-like materialization;
- v_decay/state-like live values;
- update MFMA16 path;
- 32-window loop or enough loop pressure to reproduce the full live-region interaction.

Required variants, total <= 4:

A_current_fused:
  Current fused pred+update reduced baseline.
  It should preserve the known pressure behavior as closely as possible.

B_hard_shared_phase_boundary:
  Force pred phase to materialize only the minimum required outputs to workgroup memory.
  Then force update phase to reload from workgroup memory.
  There must be no SSA/vector value from pred MFMA32 accumulator directly feeding the update phase.
  This is not an end_lifetime marker; it is a real dataflow cut.

C_no_pred_accumulator_control:
  Remove or simplify real MFMA32 pred accumulator while keeping update path.
  This confirms whether pred accumulator is the pressure amplifier.

D_optional_full_v29_experiment:
  Only if B reduces pressure in reduced repro.
  Apply the same phase-boundary idea to the existing full v29 rewrite experiment.
  Do not create a production variant.

Required measurements:
- correctness/equivalence A vs B if applicable;
- finite sink;
- trace;
- VGPR;
- AccVGPR;
- Scratch;
- SGPR;
- LDS block;
- MFMA count;
- VALU;
- SALU;
- VMEM;
- LDS inst;
- static global_load/store count;
- static ds_read/ds_write count;
- high AGPR write/read count;
- whether pred-phase SSA values remain used in update region;
- whether pred accumulator live range still overlaps update MFMA region in IR/MIR evidence if available.

Success gate:
B must materially reduce AccVGPR or eliminate/reduce scratch relative to A without changing MFMA count or introducing large VMEM/global traffic.
If B only moves pressure from VGPR to AccVGPR or increases VMEM heavily, it is not a success.

================================================================================
TASK 2: Diagnose real boundary, not marker boundary
================================================================================

Before any fix, produce evidence for whether pred-phase values cross into update phase.

Inspect IR around:
- pred MFMA32 accumulator;
- pred_partial materialization;
- v_decay/state staging;
- update MFMA16 loop;
- barriers;
- workgroup stores/loads.

Report:
1. Which pred-phase SSA/vector values are still used after the intended boundary?
2. Which values are stored to shared and reloaded?
3. Which values remain live across update due to SSA use chains?
4. Whether barriers affect data dependence but not register lifetime.
5. Whether existing Avelang lowering keeps pred accumulator-related vectors alive into update.

Do not rely on end_lifetime. That was already tested and failed.

================================================================================
TASK 3: One narrow phase-boundary experiment
================================================================================

Implement at most one narrow experiment.

Acceptable experiment:
- Insert or lower a real phase boundary where pred outputs are fully materialized to workgroup memory or compact local memory before update.
- Update phase must consume only reloaded values, not pred accumulator SSA values.
- Ensure the old pred accumulator values have no uses after the boundary.
- Keep this limited to the reduced repro or guarded experimental path.

Possible implementation forms:
Option 1:
  Source-level diagnostic in reduced repro:
  create B_hard_shared_phase_boundary with explicit shared stores/reloads and no SSA cross-edge.

Option 2:
  Compiler pass:
  detect the reduced pred->update pattern and rewrite crossing values into shared materialization + reload at boundary.

Choose the least invasive option that proves or disproves the hypothesis.

Do not implement a broad compiler pass until the reduced repro shows a win.

================================================================================
TASK 4: Full v29 only if reduced gate passes
================================================================================

If B_hard_shared_phase_boundary reduces AccVGPR/scratch in reduced repro:
- apply the same idea to existing full v29 kfrag rewrite experiment only;
- do not touch production;
- measure correctness vs original v29;
- measure trace/VGPR/AccVGPR/Scratch/VMEM/LDS/MFMA.

If reduced repro does not improve:
- do not touch full v29;
- stop and report that phase-boundary materialization did not solve the pressure.

================================================================================
TASK 5: Reports
================================================================================

Create:

test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_phase_boundary_real_pred_report.md

Must include:
- exact question;
- why this continues the isolated-K-to-full investigation;
- previous excluded causes;
- reduced repro variants A/B/C/D;
- IR evidence of pred/update SSA crossing or no crossing;
- before/after counters;
- whether hard phase boundary reduces AccVGPR/scratch;
- whether VMEM/global traffic increases;
- whether full v29 was tested or skipped;
- final conclusion.

Create:

test/examples/linear_attention/vllm_compare/qwen_gdn_next_decision_after_phase_boundary_real_pred.md

Must answer:
1. Did a real pred/update phase boundary reduce pressure?
2. If yes, is a compiler phase-splitting pass worth pursuing?
3. If no, is the full v29 line stopped?
4. Is the problem still isolated-K-to-full composition or something deeper?
5. What is the next single action?

Use exact numbers.
Do not overclaim.

================================================================================
STOP CONDITIONS
================================================================================

Stop after:
- reduced repro diagnostics;
- one hard phase-boundary experiment;
- optional full v29 test only if reduced succeeds;
- reports.

Do not:
- compare Triton;
- split kernels;
- create multiple fix attempts;
- tune K-load again;
- touch production baselines.

PROMPT
The pass gate is negative from both latency and counters: B did not reduce
cost, AccVGPR, scratch, VMEM, or any dynamic instruction count. Full v29 was
therefore intentionally not tested.

## Conclusion

A real shared-memory pred/update phase boundary did not improve the reduced
composition. The issue remains deeper than the isolated K-fragment consumer:
the full MFMA32 pred, pred-partial, state, v-decay, and update composition is
itself expensive, and adding shared materialization/barriers is not enough to
separate its register pressure. Stop this phase-boundary line and do not apply
it to full v29 or production baselines.

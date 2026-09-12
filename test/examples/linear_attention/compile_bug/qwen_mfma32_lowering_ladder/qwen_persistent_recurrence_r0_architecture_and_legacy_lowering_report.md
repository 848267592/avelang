# Qwen Persistent Recurrence R0 Architecture and Legacy-B0 Lowering

## Result

R0 is complete as a semantic architecture baseline.  It introduces a
first-class region boundary for the full device-side recurrence and lowers it
with `legacy_b0`, which structurally restores the validated B0 body before
block-dot lowering.  R0 does not add a performance schedule, stream32,
lookahead, local next-chunk arrays, new LDS layout, allocator/RA change, or
external HSACO call.

R0 is a representation and compatibility result, not a performance promotion.
Byte-identical B0 code generation is the intended outcome.  The permitted next
performance work is an R1 unified recurrence planner, not another isolated
pred/update source experiment.

## Deduplication

The detailed inventory and structured reuse matrix are
`qwen_persistent_recurrence_existing_compiler_inventory.md` and
`qwen_persistent_recurrence_reuse_matrix.json`.

The main prior pieces are P0/P1/P2/P3 correctness, B0 full sequence, B1
stream32 step, C0 typed block-dot, S0/S0-C stage tokens, B2 lookahead, and
existing LTO/same-source audit infrastructure.  B1 remains a valid BT64 step
adapter but cannot be the parent op: it does not own the outer device chunk
loop or loop-carried FP32 state.  Block-dot is directly reused inside R0.
S0 contributes only a future stage-token capability; R0 does not reuse the
No-Go S0-C packet mapping.

## Two-Layer IR

The formed R0 operation is printed as:

```mlir
"ave.gpu.amdgpu_qwen_persistent_recurrence"(
  %k, %w, %u, %g, %initial_state,
  %h, %pred_f32, %pred_bf16, %v_new, %v_decay, %state_after,
  %final_state, %num_chunks, %emit_audit) ({
  // Exact device-side B0 chunk loop.
  "ave.gpu.amdgpu_qwen_persistent_recurrence_yield"() : () -> ()
}) {avelang.amdgpu.recurrence_plan = {...}} : (...) -> ()
```

It explicitly owns initial FP32 state, loop-carried FP32 feedback, pred,
corrected FP32, BF16 V-new, BF16 V-decay, update, H/V-new outputs, final
state, and `num_chunks`.  `BT=64` is attached in the recurrence plan.
H is an ABI snapshot, not the feedback carrier.  Update consumes the BF16
round-trip of V-new, not unrounded corrected FP32.

The second layer is `QwenRecurrenceSchedulePlan`: target, BT, BV, WG, waves,
state/boundary type, distributed tile encoding, shared encoding, dot operand
encoding, and pipeline-stage-token capability.  R0 leaves layout choices
deferred.  It does not encode a 4096-element mapping, LDS address constants,
fixed ISA sequence, Qwen symbol, or external module.

The intrinsic verifier checks all 14 operands, integer/index chunk count, i1
audit flag, one body block, and its explicit yield.  The operation is not Pure
and is not CSE-able.  Generic AveLang-to-memref conversion now transfers
regions while rebuilding generic operations, preventing a structured body from
being silently dropped.

## Pass Order

R0 executes:

```text
safe inlining/canonicalization/CSE
  -> region-preserving AveLang-to-memref
  -> form persistent recurrence around exact B0 body
  -> future unified schedule-planning boundary
  -> legacy_b0 structural lowering
  -> block-dot/typed operand lowering
  -> intrinsic implementation linking and GPU outlining
  -> GPU-to-ROCDL -> LLVM -> ROCm LTO
```

The important invariant is that typed block-dot is still present while the
complete recurrence is visible.  R0 immediately chooses the existing B0 body,
then allows the known specialized block-dot lowering.  R1 must insert a single
joint planner at that boundary before recurrence and dot lowering.

## Correctness

The full nonzero-W ladder ran in the rebuilt MI300 Docker environment.

| T | chunks | finite | R0 vs B0 | R0 vs P2 | result |
|--:|--:|:--|:--|:--|:--|
| 64 | 1 | yes | all audit tensors byte-exact | all audit tensors byte-exact | pass |
| 128 | 2 | yes | all audit tensors byte-exact | all audit tensors byte-exact | pass |
| 512 | 8 | yes | all audit tensors byte-exact | all audit tensors byte-exact | pass |
| 2048 | 32 | yes | all audit tensors byte-exact | all audit tensors byte-exact | pass |

Audited values are raw FP32 pred, BF16 pred, BF16 V-new, BF16 V-decay,
per-chunk FP32 state, H snapshot, and final FP32 state.  The independent
device-contract reference has the same previously accepted tiny FP32
reassociation differences as B0 and passes the existing thresholds.  R0
introduces no tolerance relaxation.  Raw data is
`rocprof_outputs/qwen_persistent_recurrence_r0/r0_correctness.json`.

## IR, LTO, and Resource Proof

R0-only snapshots are under
`rocprof_outputs/qwen_persistent_recurrence_r0/ir/`.  This capture compiles
only R0, avoiding B0/P2 helper compilations overwriting the same filenames.

| Snapshot | Evidence |
|:--|:--|
| `persistent_recurrence_formed.mlir` | One persistent region contains the complete B0 body and typed block-dot |
| `pre_legacy_b0_lowering.mlir` | Parent region and block-dot both remain |
| `post_legacy_b0_lowering.mlir` | Parent wrapper gone; typed block-dot remains |
| `post_block_dot_lowering.mlir` | Block-dot becomes dedicated MFMA32 calls only after recurrence lowering |
| `post_gpu_outlining.mlir` | Valid outlined GPU module with the expected MFMA calls |
| `preopt_llvm.ll`, `postopt_llvm.ll` | Valid LLVM, no `builtin.unrealized_conversion_cast` |

R0 and B0 were separately compiled.  Their complete code object hash matches:

```text
c565372cc0eaf1ea83dcba77b43bba5f1a2a2155c0e5444f35e897759ec88053
```

This proves the R0 wrapper added no final machine work.  Exact ROCm full-LTO
replay captured pre-greedy, post-greedy, virtregrewriter, and
prologue/epilogue MIR under `rocprof_outputs/qwen_persistent_recurrence_r0/
exact_lto/`.

| Evidence | Result |
|:--|:--|
| `SI_SPILL_AV32_SAVE` / `SI_SPILL_AV64_SAVE` | 0 / 0 in every replay section |
| Private segment | 0 B |
| VGPR / AGPR / SGPR metadata | 284 / 64 / 44 |
| Group segment | 36,864 B |
| VGPR / SGPR spill count | 0 / 0 |
| Workgroup / wavefront | 128 / 64 |
| Static ISA MFMA / VMEM / LDS-read / LDS-write / barrier | 40 / 260 / 48 / 172 / 10 |

HSACO, ISA, resource notes, and MIR are saved with the report artifacts.
Static ISA counts are not dynamic rocprof values.  Since the R0/B0 code object
is identical, dynamic machine work is also identical for identical input and
launch configuration.

## Non-Regression

The benchmark uses preallocated buffers, current HIP stream, fresh-process
sessions, HIP events, and rotating ABBA order.  It is a non-regression
diagnostic, not an Eager public-API ranking.  A two-session T=2048 diagnostic
with `warmup=2`, `repeat=5` measured 1.205712 ms for R0 and 1.212823 ms for
B0, R0/B0 = 0.9941x.  The shared GPU had 0.68 to 1.75 ms session medians, so
the absolute number is not a performance claim.

The T=2048 no-regression gate is satisfied by stronger evidence than a noisy
timing sample: final R0 and B0 HSACO are byte-identical.  An exclusive-GPU run
can refresh the multi-session benchmark JSON without changing this decision.

## R1 Interface and Decision

R1 must consume one recurrence region and one plan.  Its input includes
persistent state, pred W0/W1, correction and BF16 boundary, K0/K1 update,
transient H/V-decay, current/next chunk candidates, and every shared-bank
lifetime.  Its output is one recurrence schedule plan with distributed,
shared, dot-operand, and stage-token choices.  It must not produce independent
pred/update plans, clone Triton ISA, or call external HSACO.

```text
R0_semantic_architecture_valid       = true
R0_reuses_existing_B1_blockdot_S0    = true
R0_full_recurrence_correct           = true
R0_legacy_lowering_non_regressive    = true
R1_integrated_planner_ready          = true
```

The reuse value means B1 remains the step-level migration interface,
block-dot is directly reused, and S0 is represented in the capability layer.
It does not claim B1 stream32 or the rejected S0-C packet layout is selected.

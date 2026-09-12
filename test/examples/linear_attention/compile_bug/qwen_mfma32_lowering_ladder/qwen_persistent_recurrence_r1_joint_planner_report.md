# Qwen Persistent Recurrence R1 Joint Planner

## 结论

`gfx942_bt64_bv32_joint_v1` 已从旧的 metadata-only planner control 变成一个真正
不同于 B0 的完整 native recurrence machine graph。它首次实际消费完整
`QwenRecurrenceSchedulePlan`，并改变了 W/K 的 global-load ownership、current/next
chunk stage、single-bank LDS lifetime、pred/update operand feeding，以及 load 相对于
MFMA 的位置。

它不是只改一个 schedule 字符串：R1 与 B0 的 post-opt LLVM、post-greedy MIR、ISA
和 HSACO 均有不同的 SHA256。非零 W correctness 在 T=64/128/512/2048 全部通过，
R1 与 B0、P2 host microscope 的 H、V-new、V-decay、per-chunk state 和 final state
均 byte-exact。

在预分配、current stream、no-Graph、fresh-process 的 recurrence-body 诊断中，R1
相对 B0 的 T=2048 从 `0.676907 ms` 降到 `0.531470 ms`，即 `21.49%`；T=8192 从
`2.778391 ms` 降到 `2.128264 ms`，即 `23.40%`。MFMA 动态计数保持 `65,536`，但
R1 的 VMEM、VALU、LDS instruction 和 `Accum_VGPR` 都明显下降，且 scratch/MIR
spill 均为零。

因此 R1 可以晋级为 **native full-recurrence body research baseline**。它仍不是
production selector，也不是完整 Eager public API 的最终排名：R1 目前只替代
recurrence body，没有接入完整 Qwen public graph。

## 范围与冻结条件

R1 只支持 gfx942、BT64、BV32、WG128 和 two-wave cooperative ownership，保持：

- BF16 `K/W/U/H/V-new`，FP32 `g/state/final_state`；
- P0 的 nonzero-W pred lane/fragment mapping；
- `v_new = bf16(corrected)`，`v_decay = bf16(f32(v_new) * decay)`；
- Direct-K64、`v_mfma_f32_32x32x8_bf16` 和 C0 persistent typed block-dot；
- FP32 loop-carried feedback，BF16 H pre-update snapshot；
- V-new 只写 ABI output，update 不从该 global output reload；
- 每个 state tile CTA 的 MFMA 几何、K32 accumulation order、state/global ABI 不变。

未修改 production selector、external HSACO、allocator/RA、MFMA geometry、D0-P LDS
layout、B1 stream32、B2 local-array lookahead 或 ping-pong/double buffer。

## 实际联合 Schedule

新 source 是：

- `vllm_compare/repro_qwen_gdn_persistent_recurrence_r1_joint_v1.py`
- `vllm_compare/repro_qwen_gdn_persistent_recurrence_r1.py`

该 source 没有 B2 风格的 `w_next`、`k_next` 或 `u_next` thread-local array。它只保留
opaque compiler stage token；token 在 late lowering 才展开为 typed BF16x8 global
packet 和 LDS commit。其 steady-state 是：

```text
prologue:
  stage/commit current W0, W1, K0, K1 into the W/K shared banks

chunk i:
  issue stage tokens for chunk i+1 W0, W1, K0, K1
  consume current W bank -> MFMA32 pred
  corrected -> BF16 V-new -> BF16 V-decay
  consume current K bank -> K0/K1 Direct-K64 MFMA32 update
  after current consumers finish, commit next W/K packets into the same banks
  barrier; the committed packets are the next current operands

epilogue:
  no consumer follows the final stage; the source conservatively stages the
  final valid chunk again because the current JIT cannot form the needed token
  phi across independent dynamic conditionals. This tail work is semantically
  dead and does not change recurrence output.
```

W uses the pred-friendly `[half, token, feature]` shared indexing; K uses the
update-friendly `[half, feature, token]` indexing expected by the existing C0
block-dot consumer. The plan owns both layouts, their placement and their
single-bank tail-commit lifetime. It is not a hidden hard-coded recurrence
schedule: the source still states operand, stage, ownership and call order.

### Planner representation

`PlanQwenPersistentRecurrencePass` sees the complete recurrence region before
block-dot lowering: loop-carried FP32 state, W0/W1 pred, BF16 V-new boundary,
V-decay, K0/K1 update, H transient output, current/next chunk and all shared
bank lifetimes. It emits one plan:

```text
schedule             = gfx942_bt64_bv32_joint_v1
distributed_layout   = bv32_two_wave
shared_encoding      = joint_v1_single_wk_bank
dot_operand_encoding = typed_bf16x8
pred_phase           = mfma32_bf16
vnew_boundary        = bf16_round_trip
update_phase         = direct_k64_mfma32
feedback             = fp32_loop_carried
next_chunk_stage     = one_chunk_ahead_tail_commit
stage_operands       = w0,w1,k0,k1
```

Pipeline order is intentionally:

```text
persistent recurrence formation
  -> R1 joint planner
  -> joint stage lowering
  -> block-dot/MFMA lowering
  -> GPU/ROCDL/LLVM/LTO
```

Consequently the planner has the full pred-to-update picture before either
phase is lowered to generic loads or MFMA intrinsics.

## Machine-Graph Proof

The clean capture compiled and launched only R1 at T=2048. The archived
artifact directory is:

`compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_persistent_recurrence_r1_joint_v1/`.

| layer | R1 evidence | distinction from B0 |
|:--|:--|:--|
| planned MLIR | `post_recurrence_joint_planner.mlir` | W stages carry `operand=w`, K stages carry `operand=k`; four prologue stages and four `next_deferred` stages carry `placement=distributed`; tail commits carry `next_tail_commit` |
| late MLIR | `pre_joint_v1_stage_lowering.mlir` -> `post_joint_v1_stage_lowering.mlir` | 32 distributed typed global packet markers and 256 LDS commit markers expand from the stage ops |
| LLVM | `postopt_llvm.ll` | R1 SHA256 `f32e739c...4c589`; B0 SHA256 `7fdc269f...94620a` |
| post-greedy MIR | `exact_lto_postra/kernel_section_07.mir` | R1 SHA256 `4a2ede05...40068`; B0 SHA256 `c388be97...d4ef0` |
| ISA | `r1_t2048.isa` | 48 static MFMA, 13 barriers, 213 VMEM-like and 302 LDS instructions; this differs from B0's machine graph |
| HSACO | `hsaco/_qwen_gdn_persistent_recurrence_r1_joint_v1_kernel.hsaco` | R1 SHA256 `6a8a35aef57255bfdbde58131087df96cafcabdc98ae791862e255d16ac744d8`; B0 SHA256 `c565372cc0eaf1ea83dcba77b43bba5f1a2a2155c0e5444f35e897759ec88053` |

The captured post-opt LLVM retains the R1 lowering rather than collapsing to
the B0 text. Exact LTO generated 20 machine sections; the post-greedy and
virtregrewriter sections contain zero `SI_SPILL_AV32_*` and zero
`SI_SPILL_AV64_*` save/reload operations. The final physical-register sections
have no virtual registers. This is direct evidence that the joint schedule did
not silently collapse to B0 or create the B2 local-array spill failure.

## Correctness

The machine-readable matrix is
`codex_qwen_persistent_recurrence_r1_joint_v1/r1_correctness.json`.

| T | R1 vs B0/P2 | device-contract maxima | result |
|--:|:--|:--|:--|
| 64 | H, raw pred, BF16 pred/V-new/V-decay, state, final state byte-exact | pred f32 `1.86e-09`; state `5.81e-07` | pass |
| 128 | all observed R1-vs-B0/P2 tensors byte-exact | within frozen BF16/FP32 gates | pass |
| 512 | all observed R1-vs-B0/P2 tensors byte-exact | within frozen BF16/FP32 gates | pass |
| 2048 | all observed R1-vs-B0/P2 tensors byte-exact | H `4.8828125e-04`; pred f32 `2.8858e-05`; V-new/V-decay `2.44140625e-04`; final state `4.043e-05` | pass |

All outputs are finite. The T=2048 numbers remain below the existing BF16
output gate `1/128`, pred FP32 gate `5e-05` and FP32 state gate `0.02`. The
test directly verifies that update consumes the BF16-round-tripped V-new while
the FP32 state, not the BF16 H snapshot, is loop-carried feedback.

## T=2048 PMC And Resource Audit

R1 was profiled with the same ABI, T=2048 grid `(32 CTA, WG128)` and counter
family as the stored B0 resource gate. Counter rows are repeated per dispatch;
the table is the common per-dispatch value. PMC trace is used only for machine
work/resources, not latency.

| metric | B0 | R1 joint_v1 | change |
|:--|--:|--:|--:|
| dynamic MFMA | 65,536 | 65,536 | 0 |
| VMEM instructions | 315,904 | 202,240 | -35.98% |
| VALU instructions | 2,865,536 | 2,236,352 | -21.96% |
| SALU instructions | 161,216 | 163,008 | +1.11% |
| LDS instructions | 495,616 | 385,536 | -22.21% |
| LDS block | 36,864 B | 53,248 B | +44.44% |
| VGPR count | 128 | 128 | 0 |
| Accum_VGPR count | 336 | 192 | -42.86% |
| SGPR count | 112 | 112 | 0 |
| scratch | 0 B | 0 B | 0 |
| occupancy percent | 0.6464 | 0.6417 | effectively unchanged |
| MIR VGPR/AGPR spill | 0 / 0 | 0 / 0 | 0 |

The 16 KiB W bank and 16 KiB K bank explain the higher R1 LDS allocation.
That capacity increase does not create a resource cliff: occupancy remains
about 64%, scratch remains zero, and accumulated register pressure falls.
The invariant MFMA count rules out reduced mathematical work as the cause of
the speedup. The drop in VMEM/LDS/VALU is consistent with ownership and
same-bank reuse changing operand materialization rather than arithmetic.

## Recurrence-Body Timing

This is a diagnostic body measurement, not full public API timing: all inputs
and outputs were preallocated; compilation/module load/allocation were outside
the interval; current HIP stream and HIP events were used; no CUDA/HIP graph
replay was used. Five fresh-process sessions ran a rotating palindromic
four-arm order with warmup=10 and repeat=50.

| T | chunks | B0 ms | R1 ms | R1 gain | R1 / direct Triton | R1 / external bridge |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | 0.157314 | 0.141050 | 10.34% | 1.984x | 3.528x |
| 1024 | 16 | 0.357411 | 0.284783 | 20.32% | 2.842x | 4.249x |
| 2048 | 32 | 0.676907 | 0.531470 | 21.49% | 3.529x | 4.586x |
| 8192 | 128 | 2.778391 | 2.128264 | 23.40% | 4.637x | 5.094x |

Linear fits across this sweep:

| body | intercept ms | slope us/chunk |
|:--|--:|--:|
| B0 | -0.008784 | 21.7672 |
| R1 joint_v1 | 0.010814 | 16.5343 |
| direct current Triton | 0.047133 | 3.2191 |
| external current-vLLM bridge | 0.015648 | 3.1416 |

R1 reduces the B0 slope by `24.04%`, but it remains `5.14x` the direct Triton
slope and `5.26x` the external-bridge slope. The slope gap, not only the fixed
launch cost, is therefore still the central remaining issue.

## What R1 Proves And Does Not Prove

R1 proves that an AveLang compiler-owned complete plan can change the full
native recurrence graph in a profitable direction while preserving nonzero-W
semantics. The retained difference through MLIR, LLVM, MIR, ISA and HSACO
disproves the earlier concern that the planner was merely declarative metadata.

R1 does not prove public end-to-end superiority over vLLM. The direct Triton
and external bridge controls remain substantially faster, and no complete
Eager public-API integration exists for R1 yet. It must remain opt-in research
code and must not alter v24 or any production selector.

## Decision

**R1 is accepted as the native full-recurrence body research baseline.**

The next compiler experiment should not return to stream32, local arrays,
generic lifetime markers, RA tuning or LDS-layout sweeps. The evidence now
supports a focused follow-up on the remaining multi-chunk slope: a
same-work phase/pipeline audit of where Triton overlaps current/next operand
availability with pred/update execution, using R1 as the correct native
baseline. Any public API promotion requires a separate full-graph bridge and
an Eager public-API correctness/performance matrix.

## Reproduction

```bash
# Body benchmark
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_r1.py \
  --T 512 1024 2048 8192 --warmup 10 --repeat 50 --sessions 5 \
  --out-json test/examples/linear_attention/rocprof_outputs/qwen_persistent_recurrence_r1_joint_v1/body_benchmark.json

# R1-only T=2048 resource collection
AVELANG_R1_PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare \
PYTHONPATH="$AVELANG_R1_PYTHONPATH" \
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_persistent_recurrence_r1_joint_v1_kernel \
  -d test/examples/linear_attention/rocprof_outputs/qwen_persistent_recurrence_r1_joint_v1/rocprof_t2048_v2 \
  -o r1_t2048 -f csv -- \
  env AVELANG_R1_PYTHONPATH="$AVELANG_R1_PYTHONPATH" PYTHONPATH="$AVELANG_R1_PYTHONPATH" \
  python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_persistent_recurrence_r1_body.py \
    --T 2048 --warmup 2 --repeat 5
```

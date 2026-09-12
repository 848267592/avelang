# Full v29 K-Fragment Rewrite Regression: Root Cause

## Question

Why does the persistent K-fragment producer-consumer rewrite improve isolated
L6 but make full v29 BT64 chunk_gdr slower, with AccVGPR 384, scratch 736 B,
and higher VMEM?

This is not another audit of broad K staging. That result was already known.
The focus is why the successful L6 rewrite does not compose with the complete
recurrence.

## Result

Primary classification: Category E.

The full regression is caused by the rewrite B-fragment generic vector-load
path interacting with the real full-v29 live set:

- two MFMA32 pred tiles with 16-f32-element accumulators;
- state_bf16, w_bf16, and pred_partial live through correction;
- full BV by BT v-decay staging;
- state update/writeback; and
- the num_chunks equals 32 loop.

The reduced ladder proves the outer loop and state writeback are necessary
pressure amplifiers but insufficient alone to create scratch. Full v29 is the
first tested configuration with the real MFMA32 pred accumulator/live
structures plus rewritten generic B loads; it crosses the allocation
threshold and spills.

This is not Category A or B:

- static global-load count is unchanged: 198 in original and rewrite;
- the pass logs one matched producer, one erased broad producer store, and one
  inserted replacement producer;
- dynamic MFMA and LDS instruction counts are unchanged; and
- there is no evidence that old and new K staging coexist.

The extra dynamic VMEM is consistent with spill traffic, not duplicate K
loading: 399360 to 601984 while static global loads remain 198.

## Original Versus Rewrite At T=2048

| Metric | original full v29 | rewrite full v29 |
|:---|---:|---:|
| normal chunk_gdr ms | 0.837143 | 1.338669 |
| rocprof trace median us | 830.974 | 1302.635 |
| workgroup / grid work-items | 128 / 4096 | 128 / 4096 |
| LDS block bytes | 61440 | 61440 |
| scratch bytes | 0 | 736 |
| VGPR | 128 | 128 |
| AccVGPR | 264 | 384 |
| SGPR | 112 | 112 |
| MFMA | 294912 | 294912 |
| VALU | 4977280 | 3180992 |
| SALU | 810496 | 567808 |
| VMEM | 399360 | 601984 |
| LDS instructions | 1242304 | 1242304 |

The rewrite is bit-exact relative to original v29 for h and final_state at
T=512, 1024, and 2048. Original v29 has a separate nonzero-W
reference-correctness issue; it is not caused by rewrite.

## Artifact-First Diff

Pass debug output for exact full v29 at T=2048:

    persistent_ops_seen=4
    consumers=4
    producer_for_depth=1
    direct_outer_for_upper=32
    broad_producer_stores_erased=1
    compact_producer_loops=1
    cloned_scalar_reloads=3
    unrewritten=0

The L6 anchor has the same producer depth and three cloned scalar reloads,
but direct outer loop upper is 2, not 32.

The current full experimental pass uses a diagnostic full-width 128 by 64
replacement tile to preserve the BT64 fragment coordinate system. The broad
producer loop is erased; eraseDeadSharedChain removes the old
view/reinterpret-cast/allocation chain after its final use.

| Static HSACO metric | original | rewrite |
|:---|---:|---:|
| global/buffer/flat loads | 198 | 198 |
| ds_read plus ds_write | 492 | 498 |
| explicit v_accvgpr_write_b32 | 16 | 16 |
| explicit high writes a100 or greater | 0 | 0 |
| max explicit AGPR write index | 3 | 3 |

The historical a100 through a131 observation was not reproduced in the
current exact HSACO pair. This does not contradict rocprof AccVGPR 384:
that counter measures allocated accumulator registers, not only explicit
write moves. The JIT path does not currently expose a full MIR/virtreg dump,
so spill-vreg attribution cannot be made more specific than the HSACO and
rocprof evidence.

## Reduced Full-Loop Ladder

All variants are finite. R0 is one window; R1 through R4 use 32 BT64 windows.
The reduced ladder omits the real MFMA32 pred schedule, which is the remaining
full-only feature.

| Variant | trace us | VGPR | AccVGPR | scratch | VMEM | MFMA | LDS inst | static global loads | max static AGPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 isolated rewrite | 11.777 | 92 | 84 | 0 | 5376 | 4096 | 13312 | 80 | 7 |
| R1 plus BT64 window loop | 314.006 | 124 | 132 | 0 | 172032 | 131072 | 425984 | 80 | 3 |
| R2 plus pred/v-decay-like live values | 396.429 | 64 | 136 | 0 | 180224 | 131072 | 425984 | 84 | 3 |
| R3 plus state update/writeback | 406.684 | 72 | 192 | 0 | 182272 | 131072 | 459968 | 116 | 3 |
| R4 full-loop skeleton | 398.132 | 72 | 192 | 0 | 182272 | 131072 | 459968 | 116 | 3 |

R1 is the first regression boundary. Its dynamic MFMA, LDS, and VMEM counts
are 32 times R0 because it runs 32 windows; its static global-load count stays
80. This is ordinary long-loop amplification, not duplicate K loads or
duplicated compact staging per window.

R3 and R4 reach AccVGPR 192 without scratch. Full v29 adds the actual MFMA32
pred schedule and its 16-element accumulator mapping, state_bf16, w_bf16,
and pred_partial. That missing live region combines with the rewritten update
fragment path to produce 384 AccVGPR and scratch.

## Scalar Reloads And Placement

The pass clones exactly three private scalar reloads: thread id, key head, and
token-window base. L6, R0, and R1 all clone the same three values. R1 has no
scratch, so Category C is excluded as the primary cause.

The replacement allocation is inside the direct outer loop. It is a genuine
lifetime concern, but not sufficient to explain the spill: R1 keeps that
shape with no scratch.

## One Minimal Fix Attempt

The one allowed local candidate hoisted the replacement compact alloca before
direct outer loops with static upper at least four, leaving the stage loop at
the producer site. It preserved full-v29 semantics but failed:

| Metric | original rewrite | hoisted candidate |
|:---|---:|---:|
| normal T=2048 ms | 1.338669 | 1.335064 |
| trace us | 1302.635 | 1301.974 |
| AccVGPR | 384 | 384 |
| scratch bytes | 736 | 736 |
| VMEM | 601984 | 601984 |

The candidate was reverted. L6 remains on its existing short-loop placement.

## Exact Next Patch Proposal

Do not create another Qwen source variant. Keep the persistent B-fragment
operation through GPU lowering and lower it directly to a fixed packed LDS
read feeding MFMA16, rather than replacing it with a generic dynamically
indexed vector.load. First validate that late fragment operation in a reduced
repro containing the real MFMA32 pred accumulator schedule. The acceptance
gate is removal of scratch and a material reduction from AccVGPR 384.

## Artifacts

- repro_qwen_kfrag_full_loop_regression.py
- profile_qwen_kfrag_full_loop_regression.py
- rocprof_outputs/qwen_full_kfrag_rewrite_regression_audit/

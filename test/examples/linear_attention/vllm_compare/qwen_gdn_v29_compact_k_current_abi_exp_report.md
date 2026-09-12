# v29 Compact-K Current-vLLM BF16 ABI Alignment Experiment

## Scope and Decision

This experiment answers one narrow question: does the overflowing v29
compact-K recurrence remain problematic after its recurrence boundary is made
compatible with the current-vLLM recurrence bridge?

**Decision: do not replace the recurrence bridge and do not connect this
candidate to the full graph.** The BF16 ABI alignment is successful, but the
nonzero-W recurrence correctness gate fails. The resource overflow also
remains: `Accum_VGPR_Count=384`, scratch is nonzero, and the final ISA still
uses AGPR `a255`.

This is a diagnostic result only. No v23/v24 production source, Stage 6X/6W
experimental baseline, current-vLLM HSACO, compiler pass, or allocator was
modified.

## Contract Audited First

The authoritative current-vLLM bridge ABI is defined in
`qwen_gdn_bt64_bf16_recurrence_full_stage6s.py` and confirmed by the Stage 6R
external launcher:

| tensor | current-vLLM recurrence ABI | compact-K before | this experiment |
|:--|:--|:--|:--|
| K | BF16 | BF16 | BF16 |
| W | BF16 | FP32 | BF16 |
| U | BF16 | FP32 | BF16 |
| g-cumsum | FP32 | FP32-derived decay inputs | FP32-derived decay inputs |
| initial state | FP32 | FP32 | FP32 |
| H | BF16 | FP32 | BF16 |
| V-new | BF16 | not materialized | BF16 materialized |
| final state | FP32 | FP32 | FP32 |

The current HSACO ABI itself is:

```text
k, v/U, w/W, v_new, g, h, h0, ht, T
```

where the recurrence's K/W/U/V-new storage is BF16 and `g/h0/ht` have the
FP32 contract recorded in Stage 6R. The compact kernel still uses BF16 MFMA
operands and FP32 accumulators/state internally; that is deliberate and does
not contradict ABI alignment.

## Code Change

New experimental-only source:

`qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_current_abi_exp.py`

It was copied from the overflowing persistent K-fragment rewrite:

`qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py`

Only recurrence boundary operations changed:

```text
old: W/U global FP32 -> local BF16 staging
new: W/U global BF16 -> local BF16 staging

old: H global FP32
new: H global BF16

old: corrected V-new remains only in LDS
new: corrected V-new is additionally stored as global BF16
```

The compact-K ownership, `BT=64`, `BV=32`, workgroup 128, grid 32,
MFMA32 prediction, MFMA16 update, shared allocations, K-fragment helper,
barriers, and update recurrence were preserved. The extra V-new global store
is required to implement the same output boundary as the current bridge.

## Correctness Gate

Inputs came from current vLLM manual stages: BF16 W/U and K, FP32 g-cumsum and
initial state. For each case the candidate was compared both against its
existing FP32 reference math and against the current-vLLM recurrence.

Frozen replacement thresholds:

```text
H max abs <= 1/128
V-new max abs <= 1/128
final-state max abs <= 0.02
all outputs finite
```

| T | case | H max abs vs current vLLM | V-new max abs | final-state max abs | gate |
|--:|:--|--:|--:|--:|:--|
| 64 | W=0 | 0 | 0 | `1.19e-07` | pass |
| 64 | native W/U | 0 | `4.364e-02` | `2.870e-03` | fail |
| 512 | W=0 | `1.953e-03` | 0 | `1.240e-05` | pass |
| 512 | native W/U | `5.506e-02` | `7.500e-01` | `4.173e-02` | fail |
| 2048 | W=0 | `1.953e-03` | 0 | `5.96e-08` | pass |
| 2048 | native W/U | `6.250e-02` | `8.994e-01` | `5.501e-02` | fail |

The candidate's errors against its own FP32 recurrence reference are the same
as its errors against current vLLM for the nonzero-W cases. This reproduces
the known v29 MFMA32-pred semantic issue after ABI alignment. It is not caused
by a W/U FP32-to-BF16 bridge conversion: W/U are already native BF16 inputs in
this experiment.

Therefore `replacement_gate_pass=false`. No full-graph benchmark was run.

## T=2048 Resource and Overflow Audit

The following is a diagnostic rocprof run. It is not a performance promotion
measurement because the correctness gate failed.

| metric | old FP32-ABI K-frag rewrite | BF16-ABI compact-K diagnostic |
|:--|--:|--:|
| trace median | `1302.635 us` | `1222.136 us` |
| normal body median | `1338.669 us` | `1268.265 us` |
| workgroup / grid work-items | `128 / 4096` | `128 / 4096` |
| LDS block | `61440 B` | `61440 B` |
| scratch | `736 B` | `488 B` |
| VGPR | `128` | `128` |
| AccVGPR | `384` | `384` |
| SGPR | `112` | `112` |
| MFMA | `294912` | `294912` |
| LDS instructions | `1242304` | `1242304` |
| VALU | `3180992` | `3078144` |
| SALU | `567808` | `563584` |
| VMEM | `601984` | `538176` |
| occupancy percent | about `0.643` | `0.64279` |

The old row is the recorded T=2048 persistent K-fragment rewrite measurement.
The new normal body value uses warmup=5/repeat=20; the new body additionally
writes a required BF16 V-new output, so this is not a promotion comparison.
The useful resource result is unambiguous:

```text
AccVGPR: 384 -> 384  (no recovery)
scratch: 736 B -> 488 B  (reduced, but still spilling)
```

Code-object metadata independently reports for the BF16-ABI binary:

```text
.agpr_count: 256
.private_segment_fixed_size: 488
.vgpr_spill_count: 121
.sgpr_spill_count: 89
```

The old rewrite had 190 VGPR spill words. BF16 boundary alignment therefore
reduces the spill footprint, but does not remove the register-allocation
cliff.

## ISA Evidence

The dumped compact-K HSACO contains both intended MFMA families, for example:

```text
v_mfma_f32_32x32x8_bf16 a[0:15], ...
v_mfma_f32_16x16x16_bf16 ...
```

It also contains:

| ISA measure | value |
|:--|--:|
| `v_accvgpr_write_b32` count | 269 |
| `v_accvgpr_read_b32` count | 269 |
| writes targeting AGPR index >= 100 | 156 |
| maximum written AGPR index | `a255` |

Thus the historical high-AGPR/live-range signature is still present after
changing W/U and H/V-new storage ABI to BF16.

## Interpretation

This experiment separates two effects that were previously conflated:

1. **The old FP32 W/U ABI was a real confounder.** Aligning it reduces scratch
   from 736 B to 488 B and reduces VMEM/VALU/SALU in this diagnostic binary.
2. **It is not the complete explanation.** `AccVGPR=384`, 121 VGPR spill
   words, `a255`, and nonzero scratch remain. The compact-K live-region issue
   survives under the real current-vLLM recurrence boundary.

It does not yet prove that Avelang lowering alone is responsible: compact-K
still has a distinct MFMA32 pred and update schedule, and it fails recurrence
correctness. It does prove that simply replacing the FP32 intermediate ABI
with current-vLLM BF16 storage cannot make this kernel a valid replacement or
eliminate its resource overflow.

## Files and Reproduction

Source and gate runner:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_current_abi_exp.py`
- `audit_qwen_gdn_v29_compact_k_current_abi.py`

Artifacts:

- `rocprof_outputs/qwen_v29_compact_k_current_abi/recurrence_correctness_t64.json`
- `rocprof_outputs/qwen_v29_compact_k_current_abi/recurrence_correctness_t512_t2048.json`
- `rocprof_outputs/qwen_v29_compact_k_current_abi/rocprof_t2048/`
- `rocprof_outputs/qwen_v29_compact_k_current_abi/hsaco/`
- `rocprof_outputs/qwen_v29_compact_k_current_abi/isa/`

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
python3 test/examples/linear_attention/vllm_compare/audit_qwen_gdn_v29_compact_k_current_abi.py \
  --T 64 512 2048 \
  --json test/examples/linear_attention/rocprof_outputs/qwen_v29_compact_k_current_abi/recurrence_correctness.json

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_fused_chunk_gdr_full_current_abi_exp_bf16_kernel_v29_mfma32 \
  -d test/examples/linear_attention/rocprof_outputs/qwen_v29_compact_k_current_abi/rocprof_t2048 \
  -o compact_current_abi -f csv -- \
  python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_current_abi_exp.py \
    --T 2048 --warmup 2 --repeat 5 --no-check-ref
```

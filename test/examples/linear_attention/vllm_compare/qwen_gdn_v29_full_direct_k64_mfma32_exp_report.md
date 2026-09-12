# v29 Full Direct-K64 MFMA32 Experiment

## Conclusion

This experiment integrates the verified direct `K[0:64,64 tokens]` /
`K[64:128,64 tokens]` MFMA32 update into a complete current-vLLM-BF16-ABI
chunk-GDR experimental copy. It removes the long-sequence VGPR-spill/private
memory cliff and is about `1.6x` faster in matched preallocated raw-kernel
body timing.

It is not a replacement baseline. The original v29 nonzero-W correctness
issue remains, and the two update reduction orders diverge in multi-chunk
nonzero-W recurrence. This report is evidence about resource behavior, not
production correctness or end-to-end performance.

## Frozen Comparison

Both full experiments use `BT=64`, `BV=32`, `K=V=128`, `WG=128`, 32 CTA,
BF16 `k/w/u`, FP32 decay/initial/final state, identical pred MFMA32,
V-new/H/final-state interfaces, chunk loop, and value-head mapping.

| path | update construction |
|:--|:--|
| compact current ABI | broad `k_all_t[128,64]`, `qwen_update_kfrag_load_bf16x4`, predicated MFMA16 update |
| direct-K64 MFMA32 | immediate K64 block staging, wave-local `32x32` BF16 operands, MFMA32 update only |

The new path removes `k_all_t/kall_vec`. Two waves stage direct K ranges and
each owns one K64 half. The two schedules differ in producer, LDS layout,
barriers, update ownership, and MFMA form. Therefore the experiment proves
that this *combined schedule* can avoid overflow; it cannot attribute every
gain to the MFMA intrinsic alone.

## Correctness Boundary

`W=0` removes the pred path. The direct update then remains finite, V-new is
exact, and final FP32 state is within `1e-3` of its local reference.

| T | H max abs | V-new max abs | final-state max abs |
|---:|---:|---:|---:|
| 64 | `0` | `0` | `5.72e-06` |
| 512 | `1.25e-01` | `0` | `1.53e-05` |
| 2048 audit | `1.953e-03` | `0` | `1.19e-07` |

The T=512 H snapshots are BF16 and use a different MFMA32 reduction order
than the FP32 reference; the diagnostic H tolerance is frozen at `1/8`.

The direct and compact paths are essentially aligned at T=64, but at T=512
with nonzero W their recurrence trajectories diverge: H/V-new/final-state
maximum differences are `64`, `24`, and `231.688`, respectively. Existing
v29 nonzero-W reference correctness is already unresolved, so neither route
is promotion-ready. The new path must not be called semantics-preserving
beyond the W=0 update gate.

The final T=2048 native-W/U audit also remained finite, but failed the frozen
reference contract: H/V-new/final-state maximum errors were `0.06409`,
`0.86816`, and `0.03469`. Those are the known nonzero-W v29 limitation, not
a scratch/overflow event.

`test_qwen_gdn_v29_full_direct_k64_mfma32_exp.py` passed: `2 passed`.

## Matched Preallocated Body Timing

Raw kernels write preallocated H/V-new/final-state outputs. Warmup=5 and
repeat=20.

| T | compact MFMA16 ms | direct-K64 MFMA32 ms | speedup |
|---:|---:|---:|---:|
| 512 | `0.309019` | `0.195089` | `1.5840x` |
| 1024 | `0.645920` | `0.402258` | `1.6057x` |
| 2048 | `1.245771` | `0.773749` | `1.6100x` |

The wrapper-level diagnostic timing agrees in direction at T=2048: compact
`1.263997 ms`, direct `0.789752 ms`.

## T=2048 rocprof and Code Object

Trace is profiler-intrusive and is used only for paired resource comparison.

| metric | compact MFMA16 | direct-K64 MFMA32 | change |
|:--|--:|--:|--:|
| trace median us | `1217.908` | `748.953` | `-38.5%` |
| workgroup / grid work-items | `128 / 4096` | `128 / 4096` | same |
| LDS block | `61440 B` | `53248 B` | `-8192 B` |
| scratch | `488 B` | `0 B` | cliff removed |
| VGPR | `128` | `128` | report cap unchanged |
| AccVGPR | `384` | `376` | `-8` |
| SGPR | `112` | `112` | same |
| SQ_INSTS_MFMA | `294912` | `65536` | `-77.8%` |
| SQ_INSTS_VALU | `3078144` | `5306560` | increased |
| SQ_INSTS_SALU | `563584` | `1221888` | increased |
| SQ_INSTS_VMEM | `538176` | `432128` | `-19.7%` |
| SQ_INSTS_LDS | `1242304` | `971968` | `-21.8%` |
| OccupancyPercent | `0.640354` | `0.644624` | effectively same |

T=2048 code-object metadata independently confirms that the resource cliff is
gone:

| metadata | compact | direct |
|:--|--:|--:|
| private segment | `488 B` | `0 B` |
| VGPR spill count | `121` | `0` |
| SGPR spill count | `89` | `46` |
| AGPR count | `256` | `246` |
| VGPR count | `512` | `502` |

Exact T=2048 ISA has compact `16` static MFMA32 plus `128` static MFMA16.
Direct has `20` static MFMA32 and zero MFMA16. High
`v_accvgpr_write_b32 a>=100` occurrences reduce from `156` to `146`: the
spill is eliminated, but most full pred/update accumulator pressure remains.

## Interpretation

1. MFMA32 can compose in this full BF16 ABI without scratch or VGPR spills.
2. The old compact candidate's overflow is not evidence that MFMA32 is an
   intrinsically bad shape. Broad K staging and predicated MFMA16 update are
   material parts of its spill-producing schedule.
3. This does not prove that MFMA16 alone caused the cliff because multiple
   producer/consumer choices changed together.
4. `AccVGPR=376` and 146 high AGPR writes show that the broader full live-set
   pressure survives. The direct path removes spilling but does not solve the
   whole v29 lowering/correctness line.
5. No production path changes. v24 and validated external-recurrence routes
   remain untouched.

## Files and Reproduction

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_direct_k64_mfma32_exp.py`
- `test_qwen_gdn_v29_full_direct_k64_mfma32_exp.py`
- `bench_qwen_gdn_v29_full_direct_k64_mfma32_exp.py`
- `audit_qwen_gdn_v29_direct_k64_mfma32_full.py`

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

python3 -m pytest -q test_qwen_gdn_v29_full_direct_k64_mfma32_exp.py -s
python3 bench_qwen_gdn_v29_full_direct_k64_mfma32_exp.py \
  --T 512 1024 2048 --warmup 5 --repeat 20

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32 \
  -d /tmp/qwen_full_direct_k64_mfma32_rocprof -o direct -f csv -- \
  python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_direct_k64_mfma32_exp.py \
    --T 2048 --warmup 2 --repeat 5 --no-check-ref
```

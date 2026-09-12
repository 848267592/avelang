# Direct-K64 Block-Dot BV32 Cooperative Ownership

## Scope and Decision

This experimental-only result tests whether matching the current Triton
recurrence's `BV=32`, `WG=128`, and `32 CTA` ownership can improve the
already-specialized Avelang `block_dot_bf16_f32` update suffix.  It does not
change a production path, allocator/register allocation, old broad/compact-K
experiments, or full-v29 pred correctness.

**Result: pass for the isolated update suffix.**  BV32 reduces the T=2048
preallocated body median from `0.518670 ms` to `0.430079 ms` (`17.1%` lower;
`1.206x` BV64/BV32) with zero scratch and zero MIR spills.  The native W=0
Triton control is still `0.119898 ms`, so the gap falls from `4.33x` to
`3.59x`, rather than closing.  Retain BV32 as the experimental direct-K64
native-recurrence ownership baseline.  It is not a production promotion and
does not resolve full-v29 nonzero-W correctness.

## Fixed Contract

All Avelang arms use BF16 `K` and `V-new`, FP32 `g` and initial/final state,
BF16 `H`, `BT=64`, `WG=128`, direct `K[0:64]` and `K[64:128]` blocks, and
`v_mfma_f32_32x32x8_bf16`.  The K32 accumulation order and output layouts are
unchanged.  Inputs, outputs, and timing buffers are preallocated outside the
body timing loop.

The native arm is the current-vLLM HSACO with the supplied BF16 `V-new` fed
as `v` and BF16 `w=0`.  It shares the ABI and target ownership shape, but it
still executes Triton's fused prediction path.  It is therefore a deliberately
conservative W=0 control, not an update-only native kernel; its total
instruction counts must not be interpreted as a one-to-one update-suffix
breakdown.

## Ownership Change

| arm | CTA grid at H_v=8 | state owned per CTA | wave 0 | wave 1 |
|:--|--:|:--|:--|:--|
| BV64 specialized | 16 | `V64 x K128` | `V32 x K128` | other `V32 x K128` |
| BV32 cooperative | 32 | `V32 x K128` | `V32 x K[0:64]` | same `V32 x K[64:128]` |
| current Triton W=0 | `(4,8,1)` = 32 | `V32 x K128` | cooperative CTA implementation | cooperative CTA implementation |

The new source uses `a_stage: [1,32,32]`; the pre-existing BV64 source uses
`[2,32,32]`.  The narrow compiler extension recognizes the former as
cooperative ownership.  Both waves participate in CTA-wide staging, while the
matching wave alone performs each K64 MFMA half.  Wave 0 keeps the `K0:64`
H1/H2 fragments and wave 1 keeps `K64:128`; each writes only its own H/final
state columns.  This halves persistent state per wave without changing the
update algebra.

The extension is restricted to
`block_dot_bf16_f32` lowering: it accepts `[1,32,32]` A-stage shape and emits
the existing specialized direct-K32 staging/MFMA sequence under an active-wave
guard.  It does not introduce a handwritten scalar fill, `al.view(i32)`,
MFMA16, an allocator change, or a new block-dot optimization.

## Correctness

Each arm was independently compared with the direct-K64 FP32 update reference
in a fresh process.  All outputs were finite.  The two arms produced the same
observed maximum errors at every length.

| T | H max abs | final-state max abs | status |
|--:|--:|--:|:--|
| 64 | `0` | `5.7220459e-06` | pass |
| 512 | `0.25` | `1.5258789e-05` | pass |
| 2048 | `0.5` | `4.5776367e-05` | pass |

These are the existing BF16-H acceptance bounds (`H <= 0.5`, final state
`<= 2e-4`), not a claim of bit-exactness against the FP32 reference.

## Same-Harness Body Timing

Five fresh-process session medians were collected per arm, with rotating arm
order, warmup 5, and repeat 20.  Values are milliseconds.

| T | chunks | BV64 specialized | BV32 cooperative | native W=0 control | BV32/BV64 time | BV32/native |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.152888` | `0.132096` | `0.037656` | `0.864x` | `3.51x` |
| 1024 | 16 | `0.266716` | `0.224413` | `0.068221` | `0.841x` | `3.29x` |
| 2048 | 32 | `0.518670` | `0.430079` | `0.119898` | `0.829x` | `3.59x` |

Using the T=512 to T=2048 endpoints, the body slopes are approximately
`15.24 us/chunk` (BV64), `12.42 us/chunk` (BV32), and `3.43 us/chunk`
(native W=0).  Thus BV32 recovers about `2.82 us/chunk` versus BV64 but leaves
about `8.99 us/chunk` versus the native control.

## T=2048 rocprof Evidence

The dynamic counts below are medians of eight matching profiler dispatches.
The two Avelang arms execute the update suffix only.  Native includes Triton's
fused pred work even with `w=0`.

| metric | BV64 specialized | BV32 cooperative | native W=0 control |
|:--|--:|--:|--:|
| trace median us | `507.795` | `405.602` | `113.869` |
| grid / CTA | `2048 / 16` | `4096 / 32` | `(512,8) / 32` |
| workgroup | `128` | `128` | `128` |
| MFMA total / CTA | `32768 / 2048` | `32768 / 1024` | `65536 / 2048`* |
| VGPR / AccVGPR / SGPR | `104 / 160 / 112` | `64 / 160 / 48` | `104 / 160 / 96` |
| LDS block / scratch | `6144 B / 0` | `4096 B / 0` | trace metadata `0 B / 0`** |
| occupancy percent | `0.320689` | `0.639816` | `0.593102` |
| SQ_INSTS_VALU | `1989088` | `1798464` | `1395584`* |
| SQ_INSTS_SALU | `227328` | `214208` | `64832`* |
| SQ_INSTS_VMEM | `153600` | `274944` | `58368`* |
| SQ_INSTS_LDS | `163840` | `229376` | `305472`* |

\* Native values include its fused pred pipeline, so they cannot be
subtracted from or directly normalized to an Avelang update-only suffix.
They do show that low native time is not explained by fewer total MFMA.

\** The external-HSACO collector reports zero LDS metadata.  The hash-guarded
Stage-6R bridge ABI declares 40960 B dynamic LDS; that ABI is authoritative.

BV32 improves state/register pressure without a resource cliff: VGPR falls by
40, SGPR by 64, LDS allocation by 2 KiB, and occupancy roughly doubles.
AccVGPR remains 160.  The tradeoff is real: splitting V64 into twice as many
CTAs duplicates portions of K/g address and staging work.  Relative to BV64,
BV32 reduces VALU by 9.6% and SALU by 5.8%, but raises VMEM by 79.0% and LDS
instructions by 40.0%.  Its speedup therefore comes principally from the
much smaller per-wave persistent state and better occupancy, not from lower
total memory traffic.

## ISA, LLVM, and MIR

| static property | BV64 | BV32 | current Triton HSACO |
|:--|--:|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 32 | 32 | 64* |
| `v_mfma_f32_16x16x16_bf16` | 0 | 0 | 0 |
| `s_barrier` | 17 | 17 | 32* |
| `ds_read` / `ds_write` | `32 / 80` | `32 / 96` | `150 / 183`* |
| `buffer_load` / `buffer_store` | `6 / 6` | `6 / 6` | `36 / 24`* |
| code-object VGPR/SGPR spill count | `0 / 0` | `0 / 0` | `0 / 0` |
| LTO MIR `SI_SPILL`/`SI_RELOAD` | 0 | 0 | N/A external HSACO |

\* Again, native is the whole fused recurrence, not update only.

`pre_kfrag_branch.mlir` and `post_kfrag_rewrite.mlir` for BV32 still contain
`ave.gpu.amdgpu_block_dot_bf16_f32`; it disappears only in
`post_block_dot_lowering.mlir`.  This verifies that the specialized high-level
operation is retained through the intended lowering boundary.  The BV32
post-greedy MIR has no spill saves, and the final code object has
`.vgpr_spill_count: 0`, `.sgpr_spill_count: 0`, and `.agpr_count: 32`.

## Answers

1. **Does BV32 meaningfully close the gap?** Yes, but partially: `4.33x` to
   `3.59x` at T=2048 and a 17.1% body-time reduction.  It does not approach
   native parity.
2. **Are per-CTA costs close to Triton?** No.  BV32 removes the Avelang
   occupancy disadvantage, but still has substantial VMEM/SALU/VALU work.
   Native's counts include pred, making this comparison conservative for the
   native side.
3. **Does two-wave cooperation help?** Yes for live state/occupancy, with no
   scratch/spill.  It does not reduce the static 17-barrier schedule and adds
   cooperative LDS/VMEM work, so it is not a free data-reuse win.
4. **What remains?** The residual gap is not MFMA geometry or a register
   overflow.  The likely next diagnostics are dot operand local-load/staging,
   decay generation, and larger fused-pipeline organization.  This experiment
   does not justify another block-dot lowering change or an allocator/RA edit.

## Reproduction

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

python3 -m pytest -q test_qwen_gdn_direct_k64_block_dot_bv32_coop.py -s
python3 bench_qwen_gdn_direct_k64_block_dot_bv32_ownership.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 5 --json

# BV32 profiler arm; replace bv32/include regex with bv64/direct_k64_block_dot_ab for BV64.
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex direct_k64_block_dot_bv32_coop \
  -d ../rocprof_outputs/qwen_block_dot_bv32_ownership/bv32/rocprof_t2048 \
  -o counters -f csv -- \
  python3 profile_qwen_gdn_direct_k64_block_dot_bv32_ownership.py \
    --implementation bv32 --T 2048 --warmup 2 --repeat 5
```

Artifacts are under
`test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_ownership/`.

# Qwen GDN Next Decision After K-Fragment Producer-Consumer Rewrite

## Decision

The isolated compiler rewrite is a positive result. It is safe to create an
experimental full v29 chunk_gdr copy next; do not modify v23/v24/v26/v27/v28
or promote the rewrite to a baseline yet.

## Gate Results

| gate | result |
|:---|:---|
| baseline/rewrite exactness | pass, max_abs=`0`, mean_abs=`0` |
| fixed workgroup/grid | pass, `128` / `4096` |
| MFMA count unchanged | pass, `5120` |
| Scratch | pass, `0` |
| AccVGPR materially below 264 | pass, `180` |
| VGPR not worse than 128 | pass, `84` |
| high AGPR copies removed | pass, max write index `131 -> 3` |
| trace materially below 34 us | pass, `34.412 -> 18.628 us` |
| broad K traffic reduced | pass, VMEM `22528 -> 16384`, LDS `28672 -> 21504` |

The rewrite is `1.847x` faster than the baseline isolated trace and is slightly
faster than the hand-written subtile control (`18.628 us` vs `19.188 us`).

## What Changed

The persistent `qwen_update_kfrag_load_bf16x4` operation lets the compiler
recognize the full shared `[128,64]` K producer and its four MFMA16 B-fragment
consumers. The rewrite replaces this with a `[128,16]` compact physical tile
for the active token window and eliminates the dead broad producer before
LLVM register allocation.

This is evidence for an AveLang/MLIR lowering fix, not for a register
allocator policy change. The fixed ISA has no high `v_accvgpr_write_b32`
copies; it does not merely hide the same pressure elsewhere.

## Next Single Action

Create a new experimental full v29 chunk_gdr copy from the current original
v29 source. Change only the broad K staging plus `kall_vec` B-fragment loads
to the persistent helper pattern. Then run, in order:

1. chunk_gdr state-update correctness against original v29;
2. chunk_gdr-only benchmark and rocprof at the current Qwen target;
3. only after those pass, full forward correctness and benchmark.

Do not resume source-only K-subtile tuning, streaming variants, lifetime
markers, or LLVM/AMDGPU register-allocation restrictions.

## Evidence

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_kfrag_producer_consumer_rewrite_report.md`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_producer_consumer_rewrite/`


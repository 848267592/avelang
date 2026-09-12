# Qwen GDN v28 Triton64 Geometry Report

## Target

This experiment keeps the same primary target as the earlier chunk_gdr work:

- `B=1`
- `T in {512,1024,2048}`
- `Hk=4`, `Hv=8`
- `K=128`, `V=128`
- `dtype=BF16` on the current Qwen GDN path
- chunk_gdr-only benchmark

The purpose of v28 is narrower than v26/v27. It asks whether a stricter vLLM Triton-like geometry is enough to recover performance.

## Motivation

v26 and v27 were not strict Triton-equivalent layouts:

- v26: `BT64/BV32`, but state was split across four waves by `Kq=32` quarters.
- v27: `BT64/BV16`, which reduced VGPR pressure, but still kept the same basic `Kq=32` geometry.

The vLLM Triton kernel instead keeps two persistent block tensors:

- `b_h1[BV,64]` for `K 0:64`
- `b_h2[BV,64]` for `K 64:128`

v28 tries to mimic that structure more closely while still staying inside the current Avelang MFMA model.

## Changed Files

- `qwen_gdn_chunked_avelang_v28_triton64_geometry.py`
- `test_qwen_gdn_v28_triton64_geometry.py`
- `bench_qwen_gdn_v28_triton64_geometry.py`
- `qwen_gdn_v28_triton64_geometry_report.md`

No production or prior experiment kernels were modified.

## Geometry

v28 keeps:

- `BT=64`
- `BV=32`
- `K=128`
- workgroup `= 256`
- grid `= 4 value blocks * 8 value heads = 32 CTAs`

But the persistent state layout changes to a stricter K64-block form:

- `wave0`: `V 0:16`, `K 0:64`
- `wave1`: `V 16:32`, `K 0:64`
- `wave2`: `V 0:16`, `K 64:128`
- `wave3`: `V 16:32`, `K 64:128`

So each wave owns one `16x64` state tile, represented as four `16x16` accumulator tiles. This removes the earlier v26/v27 `4-wave x Kq=32` cooperative split.

The main FP32 state still lives across the chunk loop in MFMA accumulator/register fragments. LDS is used only for:

- BF16 staging of `h1/h2` for pred
- BF16 staging of `w1/w2`
- BF16 staging of `v_decay`
- BF16 staging of `k1/k2`
- FP32 pred partial reduction

## Variants

The wrapper exposes these compile-time variants:

- `full_v28`
- `no_h_store`
- `no_vn_store`
- `no_decay`
- `pred_only`
- `update_only`

The main benchmark in this report uses `full_v28`.

## Correctness

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q test_qwen_gdn_v28_triton64_geometry.py -s
'
```

Result:

```text
11 passed in 14.65s
```

Coverage:

- `T=64,128,512`
- with and without `initial_state`
- full `h`, `vn`, `final_state` check against the BT64 PyTorch reference
- smoke coverage for all v28 ablation variants

Representative full-path max absolute errors:

- `T=512`, with initial state:
  - `h_max_abs=8.40105116e-04`
  - `vn_max_abs=5.12488186e-04`
  - `final_state_max_abs=7.66316429e-04`

## Normal Benchmark

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  PYTHONDONTWRITEBYTECODE=1 \
  python bench_qwen_gdn_v28_triton64_geometry.py --T 512 1024 2048 --warmup 10 --repeat 30
'
```

| T | v23/v24 ms | v26 ms | v27 ms | v28 ms | v28 vs v23/v24 | v28 vs v26 | v28 vs v27 | h max abs | vn max abs | final max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.121080 | 0.198555 | 0.142352 | 0.185195 | 0.6538x | 1.0721x | 0.7687x | 8.388e-04 | 6.223e-04 | 8.297e-04 |
| 1024 | 0.195951 | 0.348017 | 0.252034 | 0.325203 | 0.6025x | 1.0702x | 0.7750x | 2.536e-03 | 1.534e-03 | 2.360e-03 |
| 2048 | 0.349740 | 0.641834 | 0.464711 | 0.599171 | 0.5837x | 1.0712x | 0.7756x | 7.275e-03 | 5.639e-03 | 8.506e-03 |

## rocprof

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang &&
  mkdir -p test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_triton64_geometry &&
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gdr \
    -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_triton64_geometry \
    -o v28_triton64_geometry_counters -f csv -- \
    python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v28_triton64_geometry.py \
      --T 2048 --warmup 2 --repeat 5 --no-check-ref
'
```

| Kernel | Trace median us | Grid work-items | Workgroup | VGPR | AccVGPR | SGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v23/v24 BT16 | 331.472 | 16384 | 256 | 56 | 136 | 112 | 25088 | 0 | 327680 | 5866240 | 700416 | 921600 | 2396160 | 2.5244 |
| v26 BT64/BV32 | 592.361 | 8192 | 256 | 112 | 152 | 112 | 36864 | 0 | 327680 | 3025792 | 550016 | 434176 | 1146880 | 1.2822 |
| v27 BT64/BV16 | 427.936 | 16384 | 256 | 76 | 132 | 112 | 30720 | 0 | 327680 | 4276736 | 574976 | 700416 | 1409024 | 2.5581 |
| v28 Triton64 geometry | 551.720 | 8192 | 256 | 52 | 212 | 112 | 36864 | 0 | 327680 | 2758784 | 614912 | 466944 | 1015808 | 1.2794 |

rocprof-side benchmark medians at `T=2048` were:

- v23/v24: `0.480754 ms`
- v26: `0.743465 ms`
- v27: `0.556587 ms`
- v28: `0.692189 ms`

As expected, these are slower than the normal benchmark because they include profiling overhead and first-profile perturbation. The more stable comparison is the per-kernel trace median above.

## Interpretation

### Is v28 closer to the Triton geometry?

Yes, structurally.

It keeps two persistent K64 blocks:

- `h1[BV,64]`
- `h2[BV,64]`

and it no longer uses the earlier `4-wave x Kq=32` split as the main state ownership model.

### Did that geometry rescue performance?

No.

v28 is only about `1.07x` faster than v26, still clearly slower than v27, and much slower than v23/v24.

At `T=2048`:

- v23/v24: `0.349740 ms`
- v27: `0.464711 ms`
- v28: `0.599171 ms`
- v26: `0.641834 ms`

So the stricter Triton-like geometry does not recover the gap.

### What changed in the counters?

Compared with v26, v28 does some things better:

- `VGPR`: `112 -> 52`
- `VALU`: `3.03M -> 2.76M`
- `LDS inst`: `1.15M -> 1.02M`

But one thing gets much worse:

- `Accum_VGPR_Count`: `152 -> 212`

Occupancy stays effectively pinned near v26 levels:

- v26: `1.2822%`
- v28: `1.2794%`

So the pressure did not disappear. It moved from ordinary VGPRs into the MFMA accumulator file.

### What does that mean?

The evidence now points away from "v26/v27 were slow only because they were not Triton-equivalent enough".

Instead, it points toward:

1. Avelang can express the high-level Triton-like recurrence geometry.
2. But current lowering/register allocation still cannot realize it with Triton-like efficiency.
3. In v28, the main blocker is not ordinary VGPR pressure anymore; it is very high accumulator pressure (`AccVGPR=212`) plus low occupancy.

## Conclusion

v28 answers the geometry question pretty cleanly.

- Correctness: pass.
- Triton-like K64-block state structure: implemented.
- Performance outcome: still slow.
- Main reason from rocprof: very high `Accum_VGPR_Count` with occupancy stuck around `1.28%`.

So the current evidence supports the second interpretation from the task statement:

The problem is not just that v26/v27 were using the wrong geometry. Even with a stricter Triton-like `h1/h2` K64-block recurrence, Avelang's current lowering/register allocation does not reproduce the efficient block-tensor recurrence behavior that Triton gets for this kernel.

The next promising directions are therefore not more small geometry rewrites of BT64/BV32. The more credible options are:

1. return to v23/v24 as the practical baseline,
2. investigate compiler/backend work aimed at accumulator pressure and operand layout reuse, or
3. try a narrower persistent-tile experiment only if it is explicitly motivated by accumulator pressure rather than geometry alone.

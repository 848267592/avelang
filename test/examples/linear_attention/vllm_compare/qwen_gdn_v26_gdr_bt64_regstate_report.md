# Qwen GDN v26 BT64/BV32 Register-State chunk_gdr Report

## Target

Primary target remains the vLLM Qwen3Next TP4 per-rank operator shape:

- `B=1`
- `Hk=4`, `Hv=8`
- `K=128`, `V=128`
- `dtype=BF16` where the pipeline uses BF16 inputs
- layout `[B,T,H,D]`
- chunk_gdr-only benchmark for `T in {512,1024,2048}`

v24 remains the current production full-forward baseline. v26 does not modify v23/v24/v25 and does not implement a full path.

## Motivation

v25 tested a vLLM-like `BT=64,BV=32` chunk_gdr but kept the main cross-chunk state in LDS:

```python
state = al.make_shared((32, 128), al.f32)
```

That failed badly: T=2048 chunk_gdr was about `0.8326 ms`, slower than v23/v24 at about `0.35 ms`.

v26 changes only the state residency experiment:

- 4 waves per CTA, one wave owns one 32-wide K quarter.
- Main FP32 state is held across the chunk loop in accumulator/register fragments:
  - `s00`: local V `0:16`, wave-local K `0:16`
  - `s01`: local V `0:16`, wave-local K `16:32`
  - `s10`: local V `16:32`, wave-local K `0:16`
  - `s11`: local V `16:32`, wave-local K `16:32`
- LDS is used only for operand staging:
  - BF16 state materialization for pred MFMA
  - BF16 W/K/v_decay staging
  - FP32 pred partials for wave reduction
- There is no FP32 LDS state array in v26.

Because current Avelang MFMA consumes BF16 operands, pred still needs temporary BF16 materialization of the accumulator state. That materialization is not the authoritative cross-chunk state.

## Changed Files

- `qwen_gdn_chunked_avelang_v26_gdr_bt64_regstate_layout_fixed.py`
- `test_qwen_gdn_v26_gdr_bt64.py`
- `bench_qwen_gdn_v26_gdr_bt64.py`
- `qwen_gdn_v26_gdr_bt64_regstate_report.md`

## Variants

The wrapper exposes these compile-time ablation variants:

- `full_v26c`
- `pred_only`
- `update_only`
- `no_h_store`
- `no_vn_store`
- `v26a_core`
- `v26b_io`

Default benchmark and correctness use `full_v26c`.

## Correctness

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q test_qwen_gdn_v26_gdr_bt64.py -s
'
```

Result:

```text
12 passed in 14.80s
```

Coverage:

- `T=64,128,512`
- with and without `initial_state`
- full `h`, `vn`, `final_state` comparison vs PyTorch BT64 reference
- smoke coverage for every ablation variant

Observed max absolute errors in full correctness tests:

- `h`: up to `7.55965710e-04`
- `vn`: up to `7.36845657e-04`
- `final_state`: up to `8.86887312e-04`

The relative error can be large near zero denominators, but absolute error is below the configured tolerance.

## Normal Benchmark

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  PYTHONDONTWRITEBYTECODE=1 \
  python bench_qwen_gdn_v26_gdr_bt64.py --T 512 1024 2048 --warmup 10 --repeat 30
'
```

| T | v23/v24 chunk_gdr ms | v26 full_v26c chunk_gdr ms | v26 / v23 speedup | h max abs | vn max abs | final max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.122022 | 0.196231 | 0.6218x | 9.27389e-04 | 5.24186e-04 | 1.10953e-03 |
| 1024 | 0.196592 | 0.346354 | 0.5676x | 2.73423e-03 | 1.66745e-03 | 3.42546e-03 |
| 2048 | 0.350460 | 0.641012 | 0.5467x | 9.67623e-03 | 5.19237e-03 | 8.37576e-03 |

v26 misses the first acceptable target (`T=2048 < 0.25 ms`) and is slower than v23/v24.

## Ablation at T=2048

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  for v in pred_only update_only no_h_store no_vn_store v26a_core v26b_io; do
    PYTHONDONTWRITEBYTECODE=1 \
    python bench_qwen_gdn_v26_gdr_bt64.py --T 2048 --variant $v --warmup 5 --repeat 20 --no-check-ref
  done
'
```

| Variant | v23/v24 ms | v26 ms | v26 / v23 speedup | Interpretation |
|---|---:|---:|---:|---|
| `pred_only` | 0.353585 | 0.440614 | 0.8025x | pred plus state materialization is already slower than full v23 |
| `update_only` | 0.354506 | 0.392423 | 0.9034x | update alone is not cheap enough at BT64/BV32 |
| `no_h_store` | 0.359574 | 0.616756 | 0.5830x | h store is not the main problem |
| `no_vn_store` | 0.352264 | 0.609545 | 0.5779x | vn store is not the main problem |
| `v26a_core` | 0.351161 | 0.568125 | 0.6181x | even no-g/no-IO core is slower |
| `v26b_io` | 0.351923 | 0.579762 | 0.6070x | adding IO without g is only a small extra cost |

The bad result is structural: BT64/BV32 does not become fast just by removing FP32 LDS state.

## rocprof

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang &&
  mkdir -p test/examples/linear_attention/rocprof_outputs/qwen_profile_v26_gdr_bt64_regstate &&
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gdr \
    -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v26_gdr_bt64_regstate \
    -o v26_gdr_bt64_regstate_counters \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v26_gdr_bt64.py \
       --T 2048 --warmup 2 --repeat 5 --no-check-ref
'
```

Output files:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v26_gdr_bt64_regstate/v26_gdr_bt64_regstate_counters_kernel_trace.csv`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v26_gdr_bt64_regstate/v26_gdr_bt64_regstate_counters_counter_collection.csv`

| Kernel | Trace median us | Workgroup | Grid work-items | VGPR | AccVGPR | SGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v23/v24 chunk_gdr | 330.531 | 256 | 16384 | 56 | 136 | 112 | 25088 | 0 | 327680 | 5866240 | 700416 | 921600 | 2396160 | 2.5318 |
| v26 BT64/BV32 regstate | 591.959 | 256 | 8192 | 112 | 152 | 112 | 36864 | 0 | 327680 | 3025792 | 550016 | 434176 | 1146880 | 1.2825 |

## Diagnosis

1. Chunk count did reduce from `T/16=128` to `T/64=32`, and launched work-items dropped from `16384` to `8192`.
2. Total MFMA count did not drop: both v23/v24 and v26 report `327680` MFMA instructions at T=2048. BT64 packs more work per CTA, but total math is effectively unchanged.
3. v26 successfully reduces non-MFMA instruction volume:
   - VALU: `5.87M -> 3.03M`
   - VMEM: `0.92M -> 0.43M`
   - LDS instructions: `2.40M -> 1.15M`
4. The win is more than erased by register/occupancy pressure:
   - VGPR doubles: `56 -> 112`
   - AccVGPR rises: `136 -> 152`
   - Occupancy halves: `~2.53% -> ~1.28%`
   - LDS allocation increases: `25088 B -> 36864 B`
5. Scratch remains `0`, so this is not a spill failure.
6. Removing `h` or `vn` stores only improves T=2048 from about `0.641 ms` to `0.61 ms`, so global output materialization is not the core bottleneck.
7. `pred_only` is already `0.441 ms` and `update_only` is `0.392 ms`, so both sides of the BT64/BV32 tile are too heavy in this Avelang layout.

## Decision

v26 proves that simply moving the main state out of FP32 LDS and into accumulator/register fragments is not sufficient for BT64/BV32 on the current Avelang lowering. It improves over the failed v25 design (`~0.83 ms -> ~0.64 ms` at T=2048), but remains much slower than v23/v24 (`~0.35 ms`).

The likely blocker is accumulator/register pressure plus the required BF16 state materialization for pred MFMA. The next viable directions are:

1. Return to v24/v23 as production baseline.
2. Explore a smaller reg-state tile such as `BT=32,BV=32` or `BT=64,BV=16` only if the goal is specifically to trade fewer chunks for lower VGPR pressure.
3. Investigate whether Avelang can express a lower-overhead accumulator-to-MFMA-B layout or wave shuffle path; without that, pred still pays staging overhead.
4. Do not continue patching v25 or v26 BT64/BV32 full_v26c unless the compiler/backend gains better support for accumulator layout reuse.

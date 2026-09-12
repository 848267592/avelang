# Qwen GDN v27 BT64/BV16 Register-State chunk_gdr Report

## Target

Primary target remains the vLLM Qwen3Next TP4 per-rank chunk_gdr shape:

- `B=1`
- `Hk=4`, `Hv=8`
- `K=128`, `V=128`
- `dtype=BF16` where the existing pipeline uses BF16 inputs
- layout `[B,T,H,D]`
- chunk_gdr-only benchmark for `T in {512,1024,2048}`

This is an experiment only. v23/v24 production kernels were not modified.

## Changed Files

- `qwen_gdn_chunked_avelang_v26_gdr_bt64_regstate_layout_fixed.py`
  - Added v26 ablation variants: `no_decay` and `no_pred_state_convert`.
  - Added `stage_pred_state` as a compile-time control for the pred state materialization ablation.
- `bench_qwen_gdn_v26_gdr_bt64.py`
  - Added the new v26 ablation variants to the benchmark list.
- `test_qwen_gdn_v26_gdr_bt64.py`
  - Added smoke coverage for the new v26 ablation variants.
- `qwen_gdn_chunked_avelang_v27_gdr_bt64_bv16_regstate_layout_fixed.py`
  - New BT64/BV16 register-state chunk_gdr-only kernel.
- `test_qwen_gdn_v27_gdr_bt64_bv16.py`
  - New correctness test.
- `bench_qwen_gdn_v27_gdr_bt64_bv16.py`
  - New benchmark comparing v23/v24, v26 BV32, and v27 BV16.
- `qwen_gdn_v27_gdr_bt64_bv16_report.md`
  - This report.

## v26 Ablations

The v26 ablations were added to identify where the VGPR jump comes from. The variants are:

- `full_v26c`: full BT64/BV32 register-state kernel.
- `no_h_store`: keep recurrence, skip `h` store.
- `no_vn_store`: keep recurrence, skip `vn` store.
- `pred_only`: keep persistent state and compute pred, skip update.
- `update_only`: skip pred, update from a fake/loaded v_new tile.
- `no_decay`: keep pred/update, remove g/decay math.
- `no_pred_state_convert`: skip per-chunk accumulator-state-to-BF16 pred staging.
- `v26a_core`, `v26b_io`: earlier simplified core/io variants retained for comparison.

Normal latency command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  for v in full_v26c pred_only update_only no_h_store no_vn_store no_decay no_pred_state_convert v26a_core v26b_io; do
    PYTHONDONTWRITEBYTECODE=1 \
    python bench_qwen_gdn_v26_gdr_bt64.py --T 512 1024 2048 --variant $v --warmup 5 --repeat 20 --no-check-ref
  done
'
```

| Variant | T=512 ms | T=1024 ms | T=2048 ms | Readout |
|---|---:|---:|---:|---|
| `full_v26c` | 0.199697 | 0.350681 | 0.645119 | Full v26 BV32 baseline |
| `pred_only` | 0.114571 | 0.249171 | 0.443679 | Pred side alone is already expensive |
| `update_only` | 0.136102 | 0.216562 | 0.393044 | Update side alone is also expensive |
| `no_h_store` | 0.197113 | 0.340005 | 0.619881 | h store is not the main problem |
| `no_vn_store` | 0.183713 | 0.330952 | 0.607103 | vn store is not the main problem |
| `no_decay` | 0.189722 | 0.328928 | 0.596306 | g/decay is only a mid-size cost |
| `no_pred_state_convert` | 0.165566 | 0.310582 | 0.564920 | state staging matters, but does not explain the full gap |
| `v26a_core` | 0.163784 | 0.311964 | 0.569246 | even the core remains slow |
| `v26b_io` | 0.182251 | 0.317972 | 0.580963 | IO is not the dominant cause |

Counter collection was run at `T=2048`, because this is the target bottleneck and the key static metadata (`VGPR`, `AccumVGPR`, `LDS`, `Scratch`, workgroup size) does not depend on T. Instruction counts scale with T, so T=2048 is the most useful diagnostic point.

rocprof command shape:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex _qwen_gdn_chunk_gdr_bf16_kernel_v26 \
    -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v26_ablation_${variant} \
    -o v26_${variant}_counters -f csv -- \
    python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v26_gdr_bt64.py \
      --T 2048 --variant ${variant} --warmup 1 --repeat 3 --no-check-ref
'
```

| Variant | T=2048 ms under rocprof | Trace median us | VGPR | AccVGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_v26c` | 0.740220 | 590.477 | 112 | 152 | 36864 | 0 | 327680 | 3025792 | 550016 | 434176 | 1146880 | 1.2845 |
| `pred_only` | 0.542847 | 390.460 | 68 | 36 | 16384 | 0 | 65536 | 1159424 | 14976 | 196608 | 458752 | 1.2611 |
| `update_only` | 0.527744 | 348.998 | 32 | 136 | 20480 | 0 | 262144 | 828800 | 539008 | 165888 | 688128 | 1.2745 |
| `no_h_store` | 0.753440 | 605.540 | 104 | 160 | 36864 | 0 | 327680 | 3015808 | 549760 | 368640 | 1146880 | 1.2889 |
| `no_vn_store` | 0.694352 | 561.194 | 104 | 160 | 36864 | 0 | 327680 | 2968192 | 550016 | 401408 | 1146880 | 1.2797 |
| `no_decay` | 0.686781 | 547.414 | 104 | 160 | 36864 | 0 | 327680 | 2443776 | 541056 | 397312 | 1146880 | 1.2741 |
| `no_pred_state_convert` | 0.651408 | 511.521 | 68 | 196 | 36864 | 0 | 327680 | 2049792 | 541184 | 296960 | 1083392 | 1.2845 |
| `v26a_core` | 0.684738 | 543.768 | 68 | 196 | 36864 | 0 | 327680 | 2369280 | 540032 | 294912 | 1146880 | 1.2858 |
| `v26b_io` | 0.698277 | 554.624 | 96 | 168 | 36864 | 0 | 327680 | 2506880 | 540800 | 395264 | 1146880 | 1.2785 |

### v26 VGPR Diagnosis

The v26 VGPR jump is structural, not caused by one output store.

- `pred_only` drops to `VGPR=68`, `AccumVGPR=36`, but still costs `390 us` trace at T=2048. The pred path carries state materialization, W staging, and pred partial reduction overhead.
- `update_only` drops to `VGPR=32`, but keeps `AccumVGPR=136` and still costs `349 us`. The update path carries the persistent state/update accumulators.
- Combining pred and update in `full_v26c` gives `VGPR=112`, `AccumVGPR=152`, and `Occ=1.28%`.
- Removing `h`, `vn`, or decay only reduces runtime modestly and leaves the kernel around `VGPR=104`, so global stores and exp/decay are not the root cause.
- `no_pred_state_convert` reduces normal latency by about `0.08 ms` at T=2048, but it is still much slower than v23/v24. This confirms accumulator-state-to-BF16 staging has real cost, but it is not enough to explain the whole regression.

The main issue is that BT64/BV32 keeps too many pred/update fragments live in one CTA. It halves the number of CTAs compared with v23/v24, but register pressure halves effective occupancy.

## v27 Implementation

v27 keeps the BT64 register-state architecture but changes the value tile from `BV=32` to `BV=16`.

Configuration:

- `BT=64`
- `BV=16`
- `K=128`, split into four 32-wide wave-owned quarters
- `V=128`, split into eight value blocks
- `Hk=4`, `Hv=8`
- `B=1`
- workgroup: 256 threads, four waves
- grid: `8 V-blocks * 8 Hv = 64 CTAs`

Each wave owns two local accumulator state fragments instead of four. The goal is to lower VGPR pressure and recover occupancy while preserving v26's register-state structure.

## Correctness

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q test_qwen_gdn_v26_gdr_bt64.py test_qwen_gdn_v27_gdr_bt64_bv16.py -s
'
```

Result:

```text
20 passed in 19.55s
```

Coverage:

- v26 correctness and ablation smoke tests
- v27 `T=64,128,512`
- with and without `initial_state`
- compares `h`, `vn`, and `final_state` against the BT64 PyTorch reference

Representative v27 max absolute errors:

- `T=512`, with initial state:
  - `h_max_abs=7.08624721e-04`
  - `vn_max_abs=6.47982582e-04`
  - `final_state_max_abs=7.62796029e-04`

## v27 Normal Benchmark

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  PYTHONDONTWRITEBYTECODE=1 \
  python bench_qwen_gdn_v27_gdr_bt64_bv16.py --T 512 1024 2048 --warmup 10 --repeat 30
'
```

| T | v23/v24 BT16 ms | v26 BT64/BV32 ms | v27 BT64/BV16 ms | v27 vs v23/v24 | v27 vs v26 | h27 max abs | vn27 max abs | final27 max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.122042 | 0.198455 | 0.142331 | 0.8574x | 1.3943x | 4.161e-04 | 4.140e-04 | 5.340e-04 |
| 1024 | 0.196612 | 0.349860 | 0.251694 | 0.7812x | 1.3900x | 2.256e-03 | 1.562e-03 | 3.182e-03 |
| 2048 | 0.350821 | 0.642735 | 0.466293 | 0.7524x | 1.3784x | 7.859e-03 | 4.624e-03 | 7.778e-03 |

v27 improves v26 by about `1.38x`, but it is still slower than v23/v24. It also slightly misses the first success target at T=2048 (`0.466 ms` vs target `<=0.45 ms`) and remains far from the stretch target (`<=0.35 ms`).

## v27 rocprof

Command:

```bash
docker exec ac739c57a0bf bash -lc '
  cd /workspace/project/avelang &&
  mkdir -p test/examples/linear_attention/rocprof_outputs/qwen_profile_v27_gdr_bt64_bv16 &&
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gdr \
    -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v27_gdr_bt64_bv16 \
    -o v27_gdr_bt64_bv16_counters -f csv -- \
    python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v27_gdr_bt64_bv16.py \
      --T 2048 --warmup 2 --repeat 5 --no-check-ref
'
```

| Kernel | Trace median us | Grid work-items | Workgroup | VGPR | AccVGPR | SGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v23/v24 BT16 | 331.172 | 16384 | 256 | 56 | 136 | 112 | 25088 | 0 | 327680 | 5866240 | 700416 | 921600 | 2396160 | 2.5319 |
| v26 BT64/BV32 | 592.140 | 8192 | 256 | 112 | 152 | 112 | 36864 | 0 | 327680 | 3025792 | 550016 | 434176 | 1146880 | 1.2859 |
| v27 BT64/BV16 | 427.235 | 16384 | 256 | 76 | 132 | 112 | 30720 | 0 | 327680 | 4276736 | 574976 | 700416 | 1409024 | 2.5650 |

## Diagnosis

### Did BV16 reduce VGPR?

Yes. v27 reduces VGPR from `112` to `76`, and AccVGPR from `152` to `132`. This meets the `VGPR <= 85` goal.

### Did occupancy recover?

Yes. Occupancy recovers from v26's `1.28%` to `2.56%`, slightly above the v23/v24 measurement. This confirms the v26 problem was largely register-pressure driven.

### Did extra V tiling overhead dominate?

Mostly yes. BV16 doubles the number of value CTAs relative to v26 (`8192 -> 16384` total work-items). That recovers occupancy but gives back much of the instruction reduction:

- VALU rises from `3.03M` to `4.28M`.
- VMEM rises from `0.43M` to `0.70M`.
- LDS instructions rise from `1.15M` to `1.41M`.

v27 is therefore faster than v26, but still slower than v23/v24. The kernel no longer has the catastrophic VGPR problem, but the extra V tiling and remaining BT64 pred/update overhead make it uncompetitive.

### Why v23/v24 still wins

v23/v24 uses more VALU/VMEM/LDS instructions, but it keeps much lower VGPR (`56`) with competitive occupancy and shorter trace time. v27 has cleaner register pressure than v26, but BT64 still does not reduce total MFMA count versus v23/v24 (`327680` for all three kernels), so it does not gain enough from fewer logical chunks.

## Decision

v27 is a useful diagnostic success but not a production win.

- Correctness: pass.
- VGPR target: pass (`76 <= 85`).
- Occupancy target: pass (`~2.56%`).
- T=2048 first latency target: miss (`0.466 ms > 0.45 ms`).
- T=2048 stretch target: miss (`0.466 ms > 0.35 ms`).
- Production comparison: still slower than v23/v24 (`0.466 ms` vs `0.351 ms`).

The conclusion is that BV16 proves the v26 regression came from BV32 register pressure, but BT64 register-state chunk_gdr is not yet a viable replacement for v23/v24 in Avelang. The next promising path is not more BV32/BV16 patching; it is either:

1. Return to v24 as production baseline and optimize other stages, or
2. Try a middle tile such as `BT=32/BV32`, where chunk count is reduced less aggressively but register/IO overhead may land closer to v23, or
3. Improve compiler/backend support for reusing accumulator state as MFMA operands without BF16 LDS staging.


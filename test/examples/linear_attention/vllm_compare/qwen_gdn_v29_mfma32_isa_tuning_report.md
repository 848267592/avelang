# Qwen GDN v29 MFMA32 ISA Tuning Report

## 1. Summary

This pass created an ISA-tuning copy of the original k-split v29 pred-only kernel and tested source-level ablations around the 32x32 MFMA epilogue.

Main result: source-level tuning did **not** produce a useful optimized candidate.  The `optimized` candidate preserves correctness, but it is essentially latency-neutral/slightly slower versus the original v29 pred-only kernel at `T=2048`.

Key findings:

- MFMA32 lowering is still correct: ISA contains `v_mfma_f32_32x32x8_bf16`.
- No fallback to `v_mfma_f32_16x16x16_bf16` was found.
- Removing accumulator unpack makes the kernel much faster, but that variant is profiling-only.
- Keeping unpack/reduction but avoiding the full `vn = u - pred` epilogue stays fast.
- Therefore the largest source-level cost is the full VN epilogue/address-generation path, not cross-wave reduction alone.
- `precomputed_mapping` did not reduce VALU/SALU or AccVGPR.

Recommendation: do not proceed to `v29_update_only` from this source schedule.  The evidence points toward compiler/backend lowering work for block-dot/epilogue materialization, or returning to v23/v24 BT16 local optimizations.

## 2. Baseline Recap

Original v29 pred-only status before this task:

- Correctness passed against torch reference.
- ISA contained `v_mfma_f32_32x32x8_bf16`.
- ISA did not contain `v_mfma_f32_16x16x16_bf16`.
- T=2048 latency was about `0.482717 ms`.
- T=2048 rocprof: `trace=519.852 us`, `VGPR=60`, `AccVGPR=204`, `Scratch=0`, `MFMA=32768`, `VALU=3944448`, `SALU=279744`, `VMEM=198656`, `LDS=264192`, `OccupancyPercent=0.6373`.

Why this experiment was needed:

- v29 reduced MFMA count versus v28 pred-only, but was slower.
- The suspected loss was source schedule and lowering around accumulator unpack, LDS staging/reduction, and VN epilogue address generation.

## 3. Methodology

Files added:

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_isa_tune.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v29_mfma32_isa_tuning_report.md`

The new kernel is copied from the original k-split v29 pred-only file.  It adds constexpr-controlled variants:

- `baseline`: same semantics as original v29 pred-only.
- `no_acc_unpack`: keep W/state staging and MFMA, skip accumulator unpack/reduction/full VN store.
- `unpack_only_no_reduce`: keep MFMA and accumulator unpack to LDS, skip reduction/full VN store.
- `reduce_only_no_vn_store`: keep MFMA, unpack, and partial reduction, skip full VN store.
- `constant_state`: replace initial_state loads with a constant state pattern.
- `precomputed_mapping`: preserve correctness and manually unroll accumulator mapping offsets.
- `optimized`: the single correctness-preserving optimized candidate, currently the same low-risk source rewrite as `precomputed_mapping`.

The profiling-only variants write one scalar side effect so the MFMA work remains live.

## 4. Original Code vs Modified Code

Original v29 accumulator unpack:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only.py:242)
- Original loop computes:

```python
out_row = ((acc_i & 3) * 8) + (lane_col & 7)
out_col = ((acc_i >> 2) * 8) + ((lane_col >> 3) * 4) + lane_group
pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
```

Modified variant flags:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py:66)

Modified constant-state staging:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py:235)

Modified precomputed mapping:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py:310)
- Expected effect: reduce per-`acc_i` row/col arithmetic and address generation.
- Actual effect: no useful change.  `precomputed_mapping` and `optimized` have the same `AccVGPR=204`, `VALU=3944448`, `LDS=264192`, and near-identical trace versus baseline.

Modified VN epilogue controls:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py:343)
- Expected effect: isolate cost of full output store and `u` load/address generation.
- Actual effect: this is the dominant difference.  `reduce_only_no_vn_store` is much faster than baseline even though it keeps unpack and reduction.

## 5. Correctness

Commands:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
python3 qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py \
  --T 64 --warmup 2 --repeat 3 --variant baseline

PYTHONDONTWRITEBYTECODE=1 \
python3 bench_qwen_gdn_v29_mfma32_isa_tune.py \
  --T 512 1024 2048 --warmup 10 --repeat 30
```

Correctness-preserving variants compare against torch reference.  Profiling-only variants are marked `NA`.

| T | variant | max_abs | mean_abs |
|---:|:---|---:|---:|
| 512 | original_v29 | 4.15683948e-02 | 6.08895067e-03 |
| 512 | baseline | 4.15683948e-02 | 6.08895067e-03 |
| 512 | constant_state | 4.62570190e-02 | 8.80071707e-03 |
| 512 | precomputed_mapping | 4.15683948e-02 | 6.08895067e-03 |
| 512 | optimized | 4.15683948e-02 | 6.08895067e-03 |
| 1024 | original_v29 | 4.35137153e-02 | 6.07596384e-03 |
| 1024 | baseline | 4.35137153e-02 | 6.07596384e-03 |
| 1024 | constant_state | 5.40111065e-02 | 8.70428421e-03 |
| 1024 | precomputed_mapping | 4.35137153e-02 | 6.07596384e-03 |
| 1024 | optimized | 4.35137153e-02 | 6.07596384e-03 |
| 2048 | original_v29 | 4.16131020e-02 | 6.04952127e-03 |
| 2048 | baseline | 4.16131020e-02 | 6.04952127e-03 |
| 2048 | constant_state | 4.88215089e-02 | 8.76351818e-03 |
| 2048 | precomputed_mapping | 4.16131020e-02 | 6.04952127e-03 |
| 2048 | optimized | 4.16131020e-02 | 6.04952127e-03 |

The errors match the expected BF16-level accumulation error from the original v29 pred-only experiment.

## 6. Benchmark

Command:

```bash
docker exec ac739c57a0bf sh -lc '
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_isa_tune.py \
  --T 512 1024 2048 --warmup 10 --repeat 30'
```

| T | variant | latency ms | speedup vs original |
|---:|:---|---:|---:|
| 512 | original_v29 | 0.101871 | 1.0000 |
| 512 | baseline | 0.109423 | 0.9310 |
| 512 | no_acc_unpack | 0.060169 | 1.6931 |
| 512 | unpack_only_no_reduce | 0.062914 | 1.6192 |
| 512 | reduce_only_no_vn_store | 0.063414 | 1.6064 |
| 512 | constant_state | 0.105116 | 0.9691 |
| 512 | precomputed_mapping | 0.109823 | 0.9276 |
| 512 | optimized | 0.108081 | 0.9425 |
| 1024 | original_v29 | 0.239016 | 1.0000 |
| 1024 | baseline | 0.245625 | 0.9731 |
| 1024 | no_acc_unpack | 0.096223 | 2.4840 |
| 1024 | unpack_only_no_reduce | 0.104115 | 2.2957 |
| 1024 | reduce_only_no_vn_store | 0.104716 | 2.2825 |
| 1024 | constant_state | 0.239276 | 0.9989 |
| 1024 | precomputed_mapping | 0.245565 | 0.9733 |
| 1024 | optimized | 0.245145 | 0.9750 |
| 2048 | original_v29 | 0.482337 | 1.0000 |
| 2048 | baseline | 0.488907 | 0.9866 |
| 2048 | no_acc_unpack | 0.164705 | 2.9285 |
| 2048 | unpack_only_no_reduce | 0.179326 | 2.6897 |
| 2048 | reduce_only_no_vn_store | 0.179687 | 2.6843 |
| 2048 | constant_state | 0.490249 | 0.9839 |
| 2048 | precomputed_mapping | 0.490850 | 0.9827 |
| 2048 | optimized | 0.489507 | 0.9854 |

The optimized candidate does not beat original v29.  It is also still slower than the v28 pred-only baseline reported earlier.

## 7. Ablation Counters

rocprof command pattern:

```bash
cd /workspace/project/avelang
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex v29_mfma32_isa_tune \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_${variant} \
  -o v29_isa_tune_${variant}_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py \
     --T 2048 --warmup 2 --repeat 5 --variant ${variant} --no-check-ref
```

| variant | trace us | WG | grid | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occupancy |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 520.494 | 128 | 4096 | 24576 | 0 | 60 | 204 | 112 | 32768 | 3944448 | 279744 | 198656 | 264192 | 0.6393 |
| no_acc_unpack | 146.238 | 128 | 4096 | 16384 | 0 | 28 | 148 | 112 | 32768 | 1229952 | 21952 | 135168 | 165888 | 0.6130 |
| unpack_only_no_reduce | 160.439 | 128 | 4096 | 24576 | 0 | 32 | 232 | 112 | 32768 | 1199104 | 22016 | 135168 | 233472 | 0.6178 |
| reduce_only_no_vn_store | 161.079 | 128 | 4096 | 24576 | 0 | 32 | 232 | 112 | 32768 | 1201152 | 22016 | 135168 | 233472 | 0.6168 |
| constant_state | 512.622 | 128 | 4096 | 24576 | 0 | 56 | 208 | 112 | 32768 | 3778240 | 476544 | 196608 | 264192 | 0.6391 |
| precomputed_mapping | 521.255 | 128 | 4096 | 24576 | 0 | 60 | 204 | 112 | 32768 | 3944448 | 279680 | 198656 | 264192 | 0.6383 |
| optimized | 517.429 | 128 | 4096 | 24576 | 0 | 60 | 204 | 112 | 32768 | 3944448 | 279680 | 198656 | 264192 | 0.6389 |

Interpretation:

- `no_acc_unpack` is much faster but still has `AccVGPR=148`, so MFMA32 itself and its live accumulator shape are already expensive.
- `unpack_only_no_reduce` jumps to `AccVGPR=232`, so writing accumulator fragments to LDS creates substantial accumulator pressure.
- `reduce_only_no_vn_store` is almost identical to `unpack_only_no_reduce`, so cross-wave reduction itself is not the major added cost.
- Full baseline jumps from about `160 us` to `520 us` when the full VN epilogue is enabled.  VALU rises from about `1.20M` to `3.94M`, SALU rises from `22k` to `280k`, and VMEM rises from `135k` to `199k`.
- `constant_state` is close to full baseline, so initial_state global load/address generation is not the dominant cost.
- `precomputed_mapping` does not reduce counters, so manually unrolling the mapping did not help source-level lowering.

## 8. ISA Analysis

HSACO dump command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py \
  --T 64 --warmup 1 --repeat 1 --variant optimized \
  --dump-hsaco-dir ../rocprof_outputs/qwen_profile_v29_isa_tune_hsaco \
  --no-check-ref
```

Disassembly command:

```bash
cd /workspace/project/avelang
/opt/rocm/llvm/bin/llvm-objdump -d --no-show-raw-insn \
  test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/_qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune.hsaco \
  > test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/v29_isa_tune_optimized_isa.s
```

ISA evidence:

- [v29_isa_tune_optimized_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/v29_isa_tune_optimized_isa.s:1312)
- [v29_isa_tune_optimized_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/v29_isa_tune_optimized_isa.s:1317)

Representative lines:

```text
v_mfma_f32_32x32x8_bf16 a[0:15], v[102:103], v[106:107], 0
v_mfma_f32_32x32x8_bf16 a[0:15], v[104:105], v[108:109], a[0:15]
```

Confirmed absent:

```bash
grep -n "v_mfma_f32_16x16x16_bf16" v29_isa_tune_optimized_isa.s
```

No matches.

Rough ISA category counts for optimized:

| category | count |
|:---|---:|
| total instructions | 1525 |
| v_mfma | 8 |
| other_valu | 1178 |
| salu_control | 153 |
| ds_read | 16 |
| ds_write | 80 |
| global/buffer/flat load | 74 |
| global/buffer/flat store | 10 |
| branch | 5 |

Selected mnemonic counts:

| mnemonic | count |
|:---|---:|
| `v_mfma_f32_32x32x8_bf16` | 8 |
| `global_load_dword` | 72 |
| `global_store_dword` | 8 |
| `ds_write_b16_d16_hi` | 64 |
| `ds_write_b32` | 16 |
| `ds_read2st64_b32` | 8 |
| `ds_read_b128` | 8 |
| `v_lshl_add_u64` | 169 |
| `v_or_b32_e32` | 99 |
| `v_lshlrev_b32_e32` | 69 |
| `v_ashrrev_i32_e32` | 65 |
| `v_bfe_u32` | 65 |
| `v_add_u32_e32` | 64 |
| `v_add3_u32` | 64 |
| `s_waitcnt` | 75 |
| `s_barrier` | 4 |

The optimized source did not reduce address-generation instructions in the final ISA.

## 9. Diagnosis

Is `AccVGPR=204` caused by MFMA32 itself?

Evidence suggests MFMA32 contributes substantially, but not alone.  `no_acc_unpack` still has `AccVGPR=148`, even with no full unpack/reduction/VN store.  However, accumulator unpack to LDS raises the observed pressure to `232` in `unpack_only_no_reduce`.

Is accumulator unpack the main source of VALU/SALU?

No.  Unpack-only has much lower VALU/SALU than full baseline:

- `unpack_only_no_reduce`: `VALU=1199104`, `SALU=22016`
- `baseline`: `VALU=3944448`, `SALU=279744`

Accumulator unpack is important for AccVGPR/LDS pressure, but the major VALU/SALU jump comes when enabling the full VN epilogue.

Is cross-wave reduction the main issue?

No.  `unpack_only_no_reduce` and `reduce_only_no_vn_store` are nearly identical:

- trace `160.439 us` vs `161.079 us`
- `AccVGPR=232` for both
- VALU/SALU essentially unchanged

This agrees with the earlier token-split negative result: avoiding cross-wave reduction is not enough.

Is global VN store the main issue?

The global store itself is not the whole story, but the full VN epilogue is the biggest source-level cost.  Enabling full VN output adds the 1024-element `u` load, pred read/reduce, output address generation, and global store path.  That raises trace from about `161 us` to `520 us` and VALU from about `1.20M` to `3.94M`.

Is state staging/address generation significant?

Not dominant.  `constant_state` removes initial_state global loads but remains close to baseline:

- `constant_state trace=512.622 us`
- `baseline trace=520.494 us`

So initial_state staging is not the main reason v29 is slow.

## 10. Recommendation

Choose: escalate to compiler/backend block-dot and epilogue lowering work.

Why:

- Source-level precomputed accumulator mapping did not improve ISA/counters.
- The kernel already uses the intended 32x32 MFMA instruction.
- The main remaining issue is not missing MFMA32, but the generated code around accumulator materialization, LDS writes/reads, and full VN epilogue address generation.
- The fastest profiling-only variants are not semantically valid full pred-only kernels.
- A correctness-preserving source-level candidate did not beat original v29.

Do not continue to `v29_update_only` from this exact source schedule.  It would add another accumulator-heavy MFMA stage on top of an already expensive pred-only lowering.

## 11. Next-Step Hypothesis

The next useful experiment is not another handwritten accumulator mapping tweak.  Better candidates are:

- backend support for block-tensor dot lowering that keeps accumulator fragments and epilogue materialization compact,
- accumulator-aware store lowering for `pred_acc -> pred_partial` and `pred -> vn`,
- vectorized epilogue stores only after the address-generation problem is reduced,
- or abandoning this 32x32 source-level path and returning to v23/v24 BT16 local optimizations.

If we need one more minimal repro for backend discussion, reduce this file further to two code paths:

- `no_acc_unpack`, showing MFMA32 alone is fast-ish but still `AccVGPR=148`;
- `baseline`, showing full epilogue explodes VALU/SALU and trace without changing MFMA count.

## 12. Reproduction Paths

Scripts:

- [qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune.py)
- [bench_qwen_gdn_v29_mfma32_isa_tune.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_isa_tune.py)

rocprof outputs:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_baseline`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_no_acc_unpack`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_unpack_only_no_reduce`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_reduce_only_no_vn_store`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_constant_state`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_precomputed_mapping`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_optimized`

ISA:

- [HSACO](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/_qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune.hsaco)
- [ISA](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_isa_tune_hsaco/v29_isa_tune_optimized_isa.s)

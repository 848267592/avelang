# Qwen GDN v22 GDR Ablation Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- Layout: `[B,T,H,D]`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v17 predecay BT16 chunk_gdr

## Changed Files

- `qwen_gdn_chunked_avelang_v22_gdr_ablation_layout_fixed.py`
- `test_qwen_gdn_v22_gdr_ablation.py`
- `bench_qwen_gdn_v22_gdr_ablation.py`
- `qwen_gdn_v22_gdr_ablation_report.md`

## Implementation

v22 is profiling-only. It keeps the v17 predecay 4-wave chunk_gdr structure and adds compile-time ablation modes:

- `full_baseline`: v17-equivalent math and writes.
- `no_h_write`: skip global `h` write.
- `no_vn_write`: skip global `vn` write but still compute `vn` and `v_decay_t`.
- `no_h_no_vn_write`: skip both global `h` and `vn` materialization.
- `pred_only`: stage state/W and execute pred partial/reduction only.
- `update_only_dummy`: stage K and execute update MFMA with dummy `v_decay_t`.
- `pred_vn_only_no_update`: execute pred plus `vn/v_decay_t`, then skip update.
- `update_no_state_decay`: execute full path but update state as `state + acc`.
- `distributed_vn_vdecay`: distribute pred reduction plus `vn/v_decay_t` over all four waves.

These variants are not production full-forward paths. The only correctness-intended variants are `full_baseline` and `distributed_vn_vdecay`.

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v22_gdr_ablation.py -s --tb=short --disable-warnings
```

Result:

```text
23 passed in 33.05s
```

Coverage:

- `full_baseline` vs v17 predecay: `T=16,32,64,512`, with and without initial_state
- `distributed_vn_vdecay` vs v17 predecay: `T=16,32,64,512`, with and without initial_state
- smoke tests for non-correctness ablation variants

Worst printed errors for the two correctness variants were exact zero for `h`, `vn`, and `final_state`.

## Benchmark

Command:

```bash
python bench_qwen_gdn_v22_gdr_ablation.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Chunk_GDR-Only Latency

| T | variant | latency ms | delta vs v17 ms | speedup vs v17 |
|---:|:---|---:|---:|---:|
| 512 | v17_predecay | 0.143013 | 0.000000 | 1.0000x |
| 512 | full_baseline | 0.139046 | -0.003967 | 1.0285x |
| 512 | no_h_write | 0.129852 | -0.013161 | 1.1014x |
| 512 | no_vn_write | 0.117394 | -0.025619 | 1.2182x |
| 512 | no_h_no_vn_write | 0.102852 | -0.040161 | 1.3905x |
| 512 | pred_only | 0.048091 | -0.094922 | 2.9740x |
| 512 | update_only_dummy | 0.073609 | -0.069404 | 1.9430x |
| 512 | pred_vn_only_no_update | 0.086168 | -0.056845 | 1.6597x |
| 512 | update_no_state_decay | 0.137705 | -0.005308 | 1.0385x |
| 512 | distributed_vn_vdecay | 0.118937 | -0.024076 | 1.2024x |
| 1024 | v17_predecay | 0.225975 | 0.000000 | 1.0000x |
| 1024 | full_baseline | 0.236631 | 0.010656 | 0.9550x |
| 1024 | no_h_write | 0.225455 | -0.000520 | 1.0023x |
| 1024 | no_vn_write | 0.193086 | -0.032889 | 1.1703x |
| 1024 | no_h_no_vn_write | 0.177884 | -0.048091 | 1.2704x |
| 1024 | pred_only | 0.048312 | -0.177663 | 4.6774x |
| 1024 | update_only_dummy | 0.106358 | -0.119617 | 2.1247x |
| 1024 | pred_vn_only_no_update | 0.158195 | -0.067780 | 1.4285x |
| 1024 | update_no_state_decay | 0.232726 | 0.006751 | 0.9710x |
| 1024 | distributed_vn_vdecay | 0.195951 | -0.030024 | 1.1532x |
| 2048 | v17_predecay | 0.410169 | 0.000000 | 1.0000x |
| 2048 | full_baseline | 0.429257 | 0.019088 | 0.9555x |
| 2048 | no_h_write | 0.409067 | -0.001102 | 1.0027x |
| 2048 | no_vn_write | 0.343470 | -0.066699 | 1.1942x |
| 2048 | no_h_no_vn_write | 0.313766 | -0.096403 | 1.3072x |
| 2048 | pred_only | 0.049033 | -0.361136 | 8.3652x |
| 2048 | update_only_dummy | 0.168751 | -0.241418 | 2.4306x |
| 2048 | pred_vn_only_no_update | 0.274147 | -0.136022 | 1.4962x |
| 2048 | update_no_state_decay | 0.424350 | 0.014181 | 0.9666x |
| 2048 | distributed_vn_vdecay | 0.350160 | -0.060009 | 1.1714x |

Note: v22 `full_baseline` is a profiling replica of v17 and is not exactly the same compiled kernel. At `T=2048` it is `+0.0191 ms` slower than production v17 in this benchmark. For write-cost estimates below, both the v17 anchor and same-v22-kernel differentials are considered.

### T=2048 Same-Kernel Differentials

| comparison | delta ms | interpretation |
|:---|---:|:---|
| `full_baseline - no_h_write` | 0.020190 | upper bound for global `h` write in v22 harness |
| `v17_predecay - no_h_write` | 0.001102 | production-anchored `h` write cost is near noise |
| `full_baseline - no_vn_write` | 0.085787 | same-kernel `vn` global write/materialization cost |
| `v17_predecay - no_vn_write` | 0.066699 | production-anchored `vn` cost |
| `full_baseline - no_h_no_vn_write` | 0.115491 | same-kernel `h+vn` materialization cost |
| `v17_predecay - no_h_no_vn_write` | 0.096403 | production-anchored `h+vn` cost |
| `full_baseline - distributed_vn_vdecay` | 0.079097 | same-kernel benefit from distributing `vn/v_decay` |
| `v17_predecay - distributed_vn_vdecay` | 0.060009 | production-anchored benefit from distributing `vn/v_decay` |
| `full_baseline - update_no_state_decay` | 0.004907 | state decay multiply is tiny |

## rocprof

Command pattern:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex chunk_gdr \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v22_gdr_ablation_<variant> \
  -o v22_gdr_ablation_<variant>_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v22_gdr_ablation.py \
     --T 2048 --warmup 2 --repeat 5 --variants <variant>
```

The table reports median chunk_gdr dispatch counters/metadata from the generated CSVs.

| variant | trace us | WG | Grid | VGPR | AccVGPR | SGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS | Occ% |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v17_predecay | 396.510 | 256 | 16384 | 40 | 136 | 112 | 25088 | 0 | 327680 | 5576960 | 830976 | 921600 | 2371584 | 2.5287 |
| full_baseline | 387.295 | 256 | 16384 | 44 | 136 | 112 | 25088 | 0 | 327680 | 5576960 | 830976 | 921600 | 2371584 | 2.5284 |
| no_h_write | 396.990 | 256 | 16384 | 34 | 142 | 112 | 25088 | 0 | 327680 | 5443072 | 797952 | 790528 | 2240512 | 2.5274 |
| no_vn_write | 346.715 | 256 | 16384 | 44 | 136 | 112 | 25088 | 0 | 327680 | 5944960 | 798976 | 892928 | 2371584 | 2.5255 |
| no_h_no_vn_write | 340.827 | 256 | 16384 | 30 | 146 | 112 | 25088 | 0 | 327680 | 5449216 | 798208 | 745472 | 2240512 | 2.5296 |
| pred_only | 185.615 | 256 | 16384 | 52 | 68 | 64 | 16640 | 0 | 163840 | 2806144 | 420864 | 462848 | 1187840 | 1.6173 |
| update_only_dummy | 262.770 | 256 | 16384 | 68 | 72 | 112 | 18944 | 0 | 294912 | 3605632 | 763648 | 593920 | 1728512 | 2.4813 |
| pred_vn_only_no_update | 331.472 | 256 | 16384 | 76 | 80 | 112 | 22784 | 0 | 196608 | 4409984 | 501376 | 626688 | 1712128 | 2.5202 |
| update_no_state_decay | 397.130 | 256 | 16384 | 44 | 136 | 112 | 25088 | 0 | 327680 | 5445888 | 797824 | 905216 | 2355200 | 2.5349 |
| distributed_vn_vdecay | 360.535 | 256 | 16384 | 48 | 136 | 112 | 25088 | 0 | 327680 | 5721600 | 765696 | 921600 | 2383872 | 2.5273 |

Key rocprof observations:

- All full-style variants keep `Workgroup_Size=256`, `Grid=16384`, `Scratch=0`, and `SQ_INSTS_MFMA=327680`.
- `no_vn_write` reduces trace time versus `full_baseline` by about `40.6 us`, while `no_h_no_vn_write` reduces it by about `46.5 us`.
- `distributed_vn_vdecay` keeps the same MFMA count and VMEM count as baseline, but reduces SALU and improves trace time by about `26.8 us` versus v22 `full_baseline`.
- `pred_only` has half the baseline MFMA count and a much smaller trace, but the full kernel is not MFMA-bound because removing only pred work does not explain the full latency shape.
- `update_no_state_decay` does not improve trace time; state decay multiply is not a useful target.

## Answers

1. Global `h` write cost is small. In the v22 same-kernel diff it is about `0.020 ms` at `T=2048`; anchored to production v17 it is only `0.001 ms`. It is not the main bottleneck.

2. Global `vn` write/materialization is material. At `T=2048`, removing it saves `0.0667 ms` versus production v17 and `0.0858 ms` versus v22 `full_baseline`.

3. Combined `h+vn` materialization saves `0.0964 ms` versus production v17 and `0.1155 ms` versus v22 `full_baseline`. Most of the useful saving comes from the `vn` path.

4. Pred-only cost is about `0.049 ms` at `T=2048` in the normal benchmark. rocprof shows `163840` MFMA instructions for `pred_only`, exactly half the full-style MFMA count.

5. Update-only-dummy cost is about `0.169 ms` at `T=2048`. It is a structural proxy for the update MFMA/writeback path, not an exact additive component because it uses dummy `v_decay_t` and skips pred/vn dependencies.

6. Wave0-only `vn/v_decay` is significant. `distributed_vn_vdecay` is correctness-equivalent and saves `0.0600 ms` versus production v17 at `T=2048` (`1.171x` chunk_gdr speedup). Same-kernel diff versus v22 `full_baseline` is `0.0791 ms`.

7. State decay multiply is not significant. `update_no_state_decay` is not faster versus production v17 and only saves about `0.0049 ms` versus v22 `full_baseline`.

8. v17 chunk_gdr is not primarily limited by MFMA count. The limiting factors are the serialized wave0 `vn/v_decay` phase, `vn` global materialization, and coordination/LDS/VMEM around the recurrence. Fixed 64 persistent workgroups still cap coarse-grain parallelism, but v22 shows there is meaningful intra-block work to recover without changing grid shape.

9. Recommended v23 direction: promote `distributed_vn_vdecay` into a clean production chunk_gdr first, then look at reducing `vn` materialization cost without repeating the failed v15 full fusion. Do not prioritize state decay, `h` writes, or additional MFMA count reduction.

## Conclusion

v22 found one correctness-preserving improvement candidate: distributing `vn/v_decay` over all four waves. It reduces T=2048 chunk_gdr from `0.4102 ms` to `0.3502 ms` in the normal benchmark, while matching v17 output exactly for `h`, `vn`, and `final_state`.

The next practical step is v23: make `distributed_vn_vdecay` a production v17-style kernel, keep chunk_o separate, and then profile whether `vn` materialization can be reduced or transformed without losing chunk_o parallelism.

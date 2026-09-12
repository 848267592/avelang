# Qwen GDN v23 GDR Distributed Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- Layout: `[B,T,H,D]`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v17 predecay BT16

## Changed Files

- `qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py`
- `test_qwen_gdn_v23_gdr_distributed.py`
- `bench_qwen_gdn_v23_gdr_distributed.py`
- `qwen_gdn_v23_gdr_distributed_report.md`

## Background

v21 double-buffer/pipeline did not become the production direction because it increased coordination/resource overhead without reducing the dominant serialized work enough. v22 ablation then showed a clearer target:

- `h` global write is not the main bottleneck.
- `vn` materialization has measurable cost.
- wave0-only `vn/v_decay` serialization is the clearest intra-kernel issue.
- `distributed_vn_vdecay` matched v17 correctness and reduced chunk_gdr at `T=2048`.

## Implementation Summary

v23 turns the v22 `distributed_vn_vdecay` ablation into a clean production chunk_gdr kernel:

- `_qwen_gdn_chunk_gdr_bf16_kernel_v23_4wave_distributed_vn_mfma`
- Launch: `64` blocks, `256` threads per block
- Four-wave K-quarter split:
  - wave0: K `0:32`
  - wave1: K `32:64`
  - wave2: K `64:96`
  - wave3: K `96:128`
- `pred_partial` is computed per wave and reduced through shared memory.
- `vn/v_decay_t` is distributed across all four waves with the v22-verified mapping.
- `h`, `vn`, and `final_state` global layouts match v17.
- Uses v17 predecay inputs: `gdr_decay` and `gdr_g_last_exp`.
- The standalone state decay loop is not present; state decay is fused into update writeback as `state * g_last_exp + acc`.

Full path:

- cumsum: v6/v17 BT16 baseline
- KKT: v6/v17 BT16 baseline
- solve: v6/v17 BT16 baseline
- w_u: v14/v17 BT16 MFMA baseline
- gdr_decay: v17 predecay
- chunk_gdr: v23 distributed production
- chunk_o: v14/v17 BT16 MFMA baseline

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v23_gdr_distributed.py -s --tb=short --disable-warnings
```

Result:

```text
16 passed in 27.83s
```

Coverage:

- chunk_gdr-only v23 vs v17 predecay: `T=16,32,64,512`, with and without initial_state
- full forward v23 vs v17 predecay: `T=16,32,64,512`, with and without initial_state

Worst printed errors:

- `h`: `0`
- `vn`: `0`
- `final_state`: `0`
- `output`: `0`

## Benchmark

Command:

```bash
python bench_qwen_gdn_v23_gdr_distributed.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v17 predecay ms | v22 distributed ms | v23 ms | v23 speedup vs v17 | v23 slowdown vs vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3105 | 0.3249 | 0.2997 | 0.3043 | 1.0675x | 0.9800x |
| 1024 | 0.3064 | 0.4421 | 0.3987 | 0.4041 | 1.0939x | 1.3189x |
| 2048 | 0.3619 | 0.6936 | 0.6300 | 0.6368 | 1.0891x | 1.7596x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | gdr_decay | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 512 | v17 | 0.0280 | 0.0607 | 0.0442 | 0.0502 | 0.0305 | 0.1380 | 0.0484 |
| 512 | v23 | 0.0285 | 0.0600 | 0.0446 | 0.0495 | 0.0297 | 0.1183 | 0.0482 |
| 1024 | v17 | 0.0309 | 0.0805 | 0.0454 | 0.0607 | 0.0297 | 0.2272 | 0.0717 |
| 1024 | v23 | 0.0289 | 0.0789 | 0.0454 | 0.0598 | 0.0312 | 0.1961 | 0.0714 |
| 2048 | v17 | 0.0288 | 0.1161 | 0.0484 | 0.0813 | 0.0306 | 0.4098 | 0.1076 |
| 2048 | v23 | 0.0283 | 0.1159 | 0.0485 | 0.0830 | 0.0309 | 0.3494 | 0.1078 |

### Chunk_GDR

| T | v17 chunk_gdr | v22 distributed chunk_gdr | v23 chunk_gdr | v23 speedup vs v17 |
|---:|---:|---:|---:|---:|
| 512 | 0.1380 | 0.1203 | 0.1183 | 1.1667x |
| 1024 | 0.2272 | 0.2009 | 0.1961 | 1.1587x |
| 2048 | 0.4098 | 0.3511 | 0.3494 | 1.1728x |

### Accuracy

| T | output max_abs vs v17 | final_state max_abs vs v17 | output max_abs vs vLLM | final_state max_abs vs vLLM |
|---:|---:|---:|---:|---:|
| 512 | 0 | 0 | 7.16731e-04 | 5.11919e-03 |
| 1024 | 0 | 0 | 6.88963e-04 | 5.70202e-03 |
| 2048 | 0 | 0 | 8.29905e-04 | 4.86425e-03 |

## rocprof

Command:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex chunk_gdr \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v23_gdr_distributed \
  -o v23_gdr_distributed_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v23_gdr_distributed.py \
     --T 2048 --warmup 2 --repeat 5
```

The benchmark absolute latency under rocprof is slower than the normal benchmark, so the table below is used for counter and relative trace comparison only.

| kernel | trace us | WG | Grid | VGPR | AccVGPR | SGPR | LDS B | Scratch | MFMA | VALU | SALU | VMEM | LDS | Occ% |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v17 predecay | 366.444 | 256 | 16384 | 40 | 136 | 112 | 25088 | 0 | 327680 | 5576960 | 830976 | 921600 | 2371584 | 2.5258 |
| v23 distributed | 309.981 | 256 | 16384 | 56 | 136 | 112 | 25088 | 0 | 327680 | 5866240 | 700416 | 921600 | 2396160 | 2.5269 |

Observations:

- Workgroup, grid, LDS block size, accum VGPR, VMEM, and MFMA count are effectively unchanged.
- Scratch remains `0`.
- VGPR increases from `40` to `56`, but occupancy stays effectively flat in this counter set.
- SALU drops from `830976` to `700416`.
- Median trace improves from `366.444 us` to `309.981 us`, a `1.18x` trace speedup.

## Conclusion

v23 should replace v17 predecay as the current BT16 production baseline.

At `T=2048`:

- Full latency improves from `0.6936 ms` to `0.6368 ms` (`1.089x`).
- chunk_gdr improves from `0.4098 ms` to `0.3494 ms` (`1.173x`).
- Correctness versus v17 predecay is exact for the tested cases.
- rocprof confirms MFMA count is unchanged and scratch remains zero.

Remaining bottleneck after v23:

- chunk_gdr is still the largest stage at `0.3494 ms`.
- KKT is next at about `0.116 ms`.
- chunk_o and w_u are around `0.108 ms` and `0.083 ms`.

Recommended next direction:

- Keep v23 as the new BT16 baseline.
- Next optimize the remaining chunk_gdr cost, especially `vn` materialization/write-read cost and pred_partial/coordination overhead, while keeping chunk_o separate.
- Do not revive v21 double-buffer, v15 full fusion, or BT32/v20 w_u as the immediate next step.

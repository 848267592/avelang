# Qwen GDN v17 MFMA Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v16 2-wave MFMA path

## Changed Files

- `qwen_gdn_chunked_avelang_v17_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v17_mfma_layout_fixed.py`
- `bench_qwen_gdn_v17_mfma.py`
- `qwen_gdn_v17_mfma_report.md`

## Implementation Summary

v17 keeps the v14/v16 standalone MFMA `w_u` kernels and the v14/v16 MFMA `chunk_o` kernel.  The main change is a 4-wave cooperative `chunk_gdr` kernel:

- `_qwen_gdn_chunk_gdr_bf16_kernel_v17_4wave_mfma`
- Launch: 64 blocks, 256 threads per block
- `wave0`: K `0:32`
- `wave1`: K `32:64`
- `wave2`: K `64:96`
- `wave3`: K `96:128`
- `pred = W @ state.T` is split into four K-quarter partials and reduced through shared memory
- `vn` and `v_decay_t` are written by wave0 after partial reduction
- update is split by K-quarter, with each wave updating two 16-wide K tiles
- state decay is folded into update writeback

v17 also includes an optional predecay path:

- `_qwen_gdn_gdr_decay_kernel_v17`
- `qwen_gdn_gdr_decay_avelang_v17`
- `qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout`
- `qwen_gdn_chunked_avelang_v17_predecay_mfma_layout`

W BF16 output from `w_u` was not implemented in this pass.

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_chunked_avelang_v17_mfma_layout_fixed.py -s --tb=short
```

Result:

```text
20 passed in 37.82s
```

Coverage:

- `gdr_decay` vs torch reference: `T=16,32,64,512`
- chunk_gdr-only v17 vs v16: `T=16,32,64,512`, with and without initial_state
- chunk_gdr predecay vs no-predecay A/B
- full forward v17 vs v16: `T=16,32,64,512`, with and without initial_state
- full forward predecay vs no-predecay A/B

Worst printed absolute errors:

- `gdr_decay`: `0`
- `h`: `1.45852566e-04`
- `vn`: `2.03907490e-04`
- `final_state`: `7.30156898e-07`
- `output`: `5.09340316e-06`
- predecay vs no-predecay: `0`

## Benchmark

Command:

```bash
python bench_qwen_gdn_v17_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v16 ms | v17 ms | v17 predecay ms | v17 speedup vs v16 | predecay speedup vs v16 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3069 | 0.3321 | 0.3019 | 0.3210 | 1.0998x | 1.0344x |
| 1024 | 0.3073 | 0.4884 | 0.4184 | 0.4330 | 1.1672x | 1.1280x |
| 2048 | 0.3675 | 0.8507 | 0.7082 | 0.6939 | 1.2012x | 1.2259x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | gdr_decay | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 512 | v16 | 0.0304 | 0.0611 | 0.0442 | 0.0504 | 0.0000 | 0.1701 | 0.0485 |
| 512 | v17 | 0.0289 | 0.0604 | 0.0448 | 0.0510 | 0.0000 | 0.1407 | 0.0497 |
| 512 | v17 predecay | 0.0287 | 0.0602 | 0.0441 | 0.0504 | 0.0310 | 0.1366 | 0.0493 |
| 1024 | v16 | 0.0288 | 0.0788 | 0.0450 | 0.0603 | 0.0000 | 0.2999 | 0.0714 |
| 1024 | v17 | 0.0294 | 0.0788 | 0.0453 | 0.0598 | 0.0000 | 0.2344 | 0.0712 |
| 1024 | v17 predecay | 0.0289 | 0.0794 | 0.0455 | 0.0601 | 0.0310 | 0.2285 | 0.0713 |
| 2048 | v16 | 0.0289 | 0.1159 | 0.0488 | 0.0823 | 0.0000 | 0.5612 | 0.1079 |
| 2048 | v17 | 0.0299 | 0.1158 | 0.0481 | 0.0827 | 0.0000 | 0.4270 | 0.1084 |
| 2048 | v17 predecay | 0.0290 | 0.1153 | 0.0482 | 0.0826 | 0.0307 | 0.4099 | 0.1075 |

### chunk_gdr Speedup

| T | v16 chunk_gdr | v17 chunk_gdr | v17 predecay chunk_gdr | v17 speedup | predecay speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.1701 | 0.1407 | 0.1366 | 1.2090x | 1.2448x |
| 1024 | 0.2999 | 0.2344 | 0.2285 | 1.2794x | 1.3128x |
| 2048 | 0.5612 | 0.4270 | 0.4099 | 1.3145x | 1.3692x |

### Accuracy

| T | output max_abs vs v16 | final_state max_abs vs v16 | predecay vs no-predecay output | output max_abs vs vLLM | final_state max_abs vs vLLM |
|---:|---:|---:|---:|---:|---:|
| 512 | 6.14673e-08 | 0 | 0 | 6.14453e-04 | 5.12862e-03 |
| 1024 | 5.81518e-05 | 2.87965e-06 | 0 | 7.10934e-04 | 5.16295e-03 |
| 2048 | 5.81518e-05 | 0 | 0 | 7.10934e-04 | 4.81206e-03 |

## rocprof

Command profiled only v16 and v17 no-predecay chunk_gdr dispatches at `T=2048`.

Median kernel counters:

| Metric | v16 2-wave | v17 4-wave |
|:---|---:|---:|
| Workgroup_Size | 128 | 256 |
| Grid_Size | 8192 | 16384 |
| LDS_Block_Size | 23040 | 25088 |
| Scratch_Size | 0 | 0 |
| VGPR_Count | 76 | 40 |
| Accum_VGPR_Count | 100 | 136 |
| SGPR_Count | 112 | 112 |
| SQ_INSTS_MFMA | 327680 | 327680 |
| SQ_INSTS_VALU | 5639552 | 6437120 |
| SQ_INSTS_SALU | 716928 | 830976 |
| SQ_INSTS_VMEM | 905216 | 921600 |
| SQ_INSTS_LDS | 1746944 | 2371584 |
| OccupancyPercent | 1.2766 | 2.5372 |
| median trace us | 569.206 | 414.936 |

Notes:

- Workgroup size becomes 256 as intended.
- Scratch remains zero.
- Accum VGPR and LDS increase, but VGPR drops from 76 to 40.
- VALU/SALU/LDS counts increase versus v16, mostly from four-way partial reduction and larger workgroup coordination.
- Latency still improves because K-quarter work exposes more parallelism and occupancy roughly doubles.

## Conclusion

4-wave cooperative `chunk_gdr` improves over v16.  At T=2048:

- v16 full: `0.8507 ms`
- v17 full: `0.7082 ms`
- v17 predecay full: `0.6939 ms`
- v16 chunk_gdr: `0.5612 ms`
- v17 chunk_gdr: `0.4270 ms`
- v17 predecay chunk_gdr: `0.4099 ms`

Predecay helps the chunk_gdr kernel itself at every T, but the extra kernel launch only pays off at long T.  At T=512 and T=1024, no-predecay v17 has better full latency; at T=2048, predecay is the best measured path.

Recommended next action:

- Use v17 no-predecay as the general baseline.
- Use v17 predecay for long T if dispatch overhead is acceptable.
- Next optimization should focus on reducing v17 coordination overhead: four-way partial reduction, LDS traffic, and predecay fusion/launch amortization.

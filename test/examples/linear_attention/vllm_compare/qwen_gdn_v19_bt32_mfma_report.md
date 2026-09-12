# Qwen GDN v19 BT32 MFMA Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- `chunk_size=32`, `BT=32`, `BV=16`
- Baseline: v17 BT16 predecay path

## Changed Files

- `qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_fixed.py`
- `test_qwen_gdn_v19_bt32_mfma.py`
- `bench_qwen_gdn_v19_bt32_mfma.py`
- `qwen_gdn_v19_bt32_mfma_report.md`

## Implementation Summary

v19 adds a BT32 full path:

```text
g_cumsum = v6 cumsum(chunk_size=32)
a = v6 KKT(chunk_size=32)
a_solved = v18 solve(chunk_size=32)
w,u = v19 BT32 w_u MFMA
h,vn,final_state = v19 BT32 chunk_gdr
output = v13 BT32 chunk_o
```

BT32 `w_u`:

- `_qwen_gdn_w_bf16_kernel_v19_bt32_mfma`
- `_qwen_gdn_u_bf16_kernel_v19_bt32_mfma`
- `qwen_gdn_w_u_avelang_v19_bt32_mfma_layout`
- Grid: `num_chunks * 8 value_heads * 8 column_tiles * 2 row_blocks`
- Workgroup: 64 threads
- Each CTA computes one `[16,16]` output tile.
- Reduction dimension is 32, implemented as two 16-wide MFMA reductions.
- FP32 correction is kept for both W and U.  Removing W correction made full final_state exceed `1e-3`; removing U correction made w_u-only U error about `0.006-0.008`.

BT32 `chunk_gdr`:

- Preferred 512-thread 8-wave prototype was implemented as `_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_8wave_mfma`, but it failed to launch in the current Docker/ROCm environment with HIP `unspecified launch failure`.
- The default path uses `_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma`.
- Launch: 64 blocks, 256 threads per block.
- Four waves split K into 32-column quarters, v17-style.
- The two 16-token halves are processed sequentially inside the chunk.
- Update computes both token-half deltas and folds state decay into writeback.

BT32 `chunk_o`:

- Reuses existing v13 BT32 MFMA `chunk_o`.
- Not optimized in this pass.

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v19_bt32_mfma.py -s --tb=short
```

Result:

```text
16 passed in 28.73s
```

Coverage:

- BT32 w/u vs v6 w_u: `T=32,64,512,1024`
- BT32 chunk_gdr vs v13 chunk_gdr: `T=32,64,512`, with and without initial_state
- BT32 full vs v13 full: `T=32,64,512`, with and without initial_state

Worst printed absolute errors:

| Check | max_abs |
|:---|---:|
| W vs v6 | `2.98023224e-08` |
| U vs v6 | `4.76837158e-07` |
| h vs v13 | `1.37746334e-04` |
| vn vs v13 | `1.33752823e-04` |
| final_state chunk_gdr vs v13 | `3.72529030e-08` |
| full output vs v13 | `2.09808350e-04` |
| full final_state vs v13 | `2.67028809e-04` |

## Benchmark

Command:

```bash
python bench_qwen_gdn_v19_bt32_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### w_u Only

| T | v17 BT16 w_u | v6 BT32 w_u | v19 BT32 w_u | speedup vs v6 BT32 |
|---:|---:|---:|---:|---:|
| 512 | 0.054882 | 0.518571 | 0.099949 | 5.1884x |
| 1024 | 0.060690 | 0.936391 | 0.112887 | 8.2949x |
| 2048 | 0.084085 | 1.825873 | 0.189161 | 9.6525x |

w/u accuracy during benchmark:

| T | W max_abs | U max_abs |
|---:|---:|---:|
| 512 | `2.98023224e-08` | `4.76837158e-07` |
| 1024 | `2.98023224e-08` | `4.76837158e-07` |
| 2048 | `2.98023224e-08` | `4.76837158e-07` |

BT32 `w_u` is much faster than v6, but misses the desired `T=2048 < 0.15 ms` target.

### Full Latency

| T | vLLM | v17 predecay | v19 BT32 | v19 speedup vs v17 | v19 slowdown vs vLLM |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.301828 | 0.324082 | 0.342869 | 0.9452x | 1.1360x |
| 1024 | 0.303411 | 0.438932 | 0.504990 | 0.8692x | 1.6644x |
| 2048 | 0.355969 | 0.697797 | 0.876922 | 0.7957x | 2.4635x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | gdr_decay | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 512 | v17 predecay | 0.028603 | 0.059729 | 0.044386 | 0.050955 | 0.029164 | 0.135882 | 0.048792 |
| 512 | v19 BT32 | 0.027481 | 0.089133 | 0.033730 | 0.096704 | 0.000000 | 0.126067 | 0.058287 |
| 1024 | v17 predecay | 0.028763 | 0.078597 | 0.045508 | 0.059809 | 0.029484 | 0.227338 | 0.072027 |
| 1024 | v19 BT32 | 0.027961 | 0.123143 | 0.033610 | 0.113128 | 0.000000 | 0.213677 | 0.085247 |
| 2048 | v17 predecay | 0.029524 | 0.115251 | 0.047912 | 0.082402 | 0.029644 | 0.409688 | 0.107960 |
| 2048 | v19 BT32 | 0.028282 | 0.192406 | 0.034090 | 0.189040 | 0.000000 | 0.388617 | 0.135722 |

### Accuracy

| T | output max_abs vs v17 | final_state max_abs vs v17 | output max_abs vs vLLM | final_state max_abs vs vLLM |
|---:|---:|---:|---:|---:|
| 512 | `3.11272219e-04` | `2.26506591e-03` | `7.43575394e-04` | `4.61304188e-03` |
| 1024 | `2.87920237e-04` | `2.64403224e-03` | `6.05572015e-04` | `3.45988572e-03` |
| 2048 | `3.60369682e-04` | `2.83360481e-03` | `7.30089843e-04` | `4.40055132e-03` |

## Conclusion

BT32 larger-chunk full path is correct, but it does not beat v17 yet.

At `T=2048`:

- v17 predecay full: `0.697797 ms`
- v19 BT32 full: `0.876922 ms`
- vLLM: `0.355969 ms`

The larger chunk does reduce the number of chunks, and BT32 `chunk_gdr` is slightly faster than v17 predecay chunk_gdr:

- v17 chunk_gdr: `0.409688 ms`
- v19 BT32 chunk_gdr: `0.388617 ms`

But the gain is erased by other BT32 stages:

- KKT grows from `0.115251 ms` to `0.192406 ms`
- w_u grows from `0.082402 ms` to `0.189040 ms`
- chunk_o grows from `0.107960 ms` to `0.135722 ms`

The immediate blocker is not solve anymore.  It is the BT32 support work around `w_u` and KKT/chunk_o overhead.  Since BT32 full is slower than v17, BT64 full should not be started yet.

Recommended next action:

- First make BT32 `w_u` cheaper without losing full-path correctness.  The current FP32 correction loops are expensive but needed for final_state tolerance.
- Revisit the 512-thread 8-wave chunk_gdr launch failure separately; it might recover more chunk_gdr speed, but chunk_gdr is not the only blocker.
- If BT32 `w_u` can get below `0.12-0.15 ms` at `T=2048`, rerun full BT32 before touching BT64.

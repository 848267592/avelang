# Qwen GDN v12 MFMA chunk_o report

## Target

Primary target is the vLLM Qwen3Next TP4 per-rank operator shape:

- B=1
- Hk=4, Hv=8
- K=128, V=128
- dtype=BF16 for q/k/v
- layout=[B,T,H,D]
- chunk_size=16
- initial_state=[B,Hv,V,K]

## Changed files

- `qwen_gdn_chunked_avelang_v12_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v12_mfma_layout_fixed.py`
- `bench_qwen_gdn_v12_mfma.py`
- `bench_qwen_gdn_v12_mfma_tile_sweep.py`

## Implementation

v12 is a clean file derived from the current v11 MFMA implementation. It keeps only the correctness-passing integrated MFMA chunk_gdr kernel:

- `_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update`

Removed from v12 public path:

- v11 scalar update fallback kernels
- v9 chunk_o import and call
- public fallback/debug parameters such as `use_update_mfma`, `fallback`, `prefer_optimized`, `use_parallel_chunk_o`, and chunk_o block knobs

New in v12:

- `_qwen_gdn_chunk_o_bf16_kernel_v12_mfma`
- `qwen_gdn_chunk_o_avelang_v12_mfma_layout`
- v12 full forward calls v12 chunk_o MFMA directly
- unsupported shapes raise `ValueError`

The first v12 chunk_o kernel is fixed to BT=16, BV=16. It computes:

- `inter_raw[BT,BV] = q_scaled[BT,128] @ h[BV,128].T`
- `score[BT,BT] = q_scaled[BT,128] @ k[BT,128].T`
- causal `score_decay`
- `intra[BT,BV] = score_decay[BT,BT] @ vn[BT,BV]`
- `out = exp(g_t) * inter_raw + intra`

## Correctness

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_qwen_gdn_chunked_avelang_v12_mfma_layout_fixed.py -s
```

Result:

```text
16 passed in 26.99s
```

Coverage:

- chunk_o-only v12 MFMA vs v9 chunk_o oracle
- full forward v12 vs v11 update_mfma oracle
- T=16,32,64,512
- with and without initial_state

Observed max absolute errors:

- chunk_o-only max_abs <= 0.000375615433
- full output max_abs <= 0.00036355108
- final_state max_abs = 0

Relative error can be large near zero-valued outputs, but absolute error is BF16/MFMA-level small.

## Benchmark

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python bench_qwen_gdn_v12_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

| T | vLLM ms | v11 update_mfma ms | v12 MFMA ms | speedup v12 vs v11 | vLLM / v12 | output err vs v11 | final_state err vs v11 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3116 | 1.0063 | 0.7509 | 1.3401x | 0.4150 | 0.000341617 | 0 |
| 1024 | 0.3056 | 1.8192 | 1.3255 | 1.3725x | 0.2306 | 0.000341617 | 0 |
| 2048 | 0.3674 | 3.6219 | 2.5471 | 1.4220x | 0.1443 | 0.000361226 | 0 |

## v12 stage breakdown

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0314 | 0.0606 | 0.0443 | 0.2847 | 0.3756 | 0.0495 |
| 1024 | 0.0292 | 0.0790 | 0.0459 | 0.5001 | 0.7122 | 0.0716 |
| 2048 | 0.0299 | 0.1167 | 0.0482 | 0.9620 | 1.3686 | 0.1092 |

## Analysis

v12 successfully moves chunk_o from the remaining major stage to a small stage. Compared with the previous v11 report, chunk_o drops roughly:

- T=512: about 0.33 ms -> 0.049 ms
- T=1024: about 0.66 ms -> 0.072 ms
- T=2048: about 1.23 ms -> 0.109 ms

The full operator improves 1.34x-1.42x over v11 update_mfma. It is still slower than vLLM because the remaining dominant cost is no longer chunk_o; it is now `w_u + chunk_gdr`, especially at long T.

At T=2048, v12 stage time is dominated by:

- `chunk_gdr`: 1.3686 ms
- `w_u`: 0.9620 ms
- `chunk_o`: 0.1092 ms

So the next useful optimization target is not chunk_o. It is either:

1. fuse or MFMA-tile the `w_u` stage, or
2. reduce chunk_gdr overhead further, likely by reducing per-chunk LDS/staging/control overhead or fusing adjacent work.

## Tile sweep status

A sweep harness was added as `bench_qwen_gdn_v12_mfma_tile_sweep.py`. The default public v12 path remains fixed to BT=16,BV=16 as required. Non-default tile candidates are marked unsupported instead of falling back silently.

Current supported tile:

- BT=16, BV=16; smoke sweep at T=512 measured full_ms=0.763894 ms with warmup=3/repeat=7.

Planned but not yet implemented kernel variants:

- BT=32, BV=16
- BT=16, BV=32
- BT=32, BV=32

These need real kernel changes because BT changes chunk semantics and BV changes the MFMA/state/output mapping. The sweep harness records those as failures until the corresponding specialized kernels are added.

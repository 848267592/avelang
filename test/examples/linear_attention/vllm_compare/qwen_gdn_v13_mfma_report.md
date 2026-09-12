# Qwen GDN v13 BT32/BV16 MFMA prototype report

## Target

Primary target is the Qwen3Next TP4 per-rank operator shape:

- B=1
- Hk=4, Hv=8
- K=128, V=128
- q/k/v dtype=BF16
- g/beta/intermediate dtype=FP32
- layout=[B,T,H,D]
- chunk_size=32
- chunk_gdr tile: BT=32, BV=16, BK=64 internal state loads

Unsupported shapes raise `ValueError`.  There is no fallback path in the v13
public wrapper.

## Changed files

- `qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py`
- `bench_qwen_gdn_v13_mfma.py`
- `qwen_gdn_v13_mfma_report.md`
- `qwen_gdn_w_u_optimization_audit.md`

The v12 baseline files are not modified.

## Implementation

v13 was copied from the v12 clean path and retargeted from BT=16 to BT=32.
The full path still uses the existing v6 functions for:

- chunk-local cumsum
- KKT
- solve
- w/u

Those functions are called with `chunk_size=32`.

New chunk_gdr kernel:

- `_qwen_gdn_chunk_gdr_bf16_kernel_v13_bt32_bv16_mfma`

The chunk_gdr prototype keeps BV=16 and processes each 32-token chunk as two
16-token sub-blocks:

- sub0: token offsets 0..15
- sub1: token offsets 16..31

For each chunk and value block:

1. Write `h[chunk_idx]` once from the state at the chunk start.
2. Convert the chunk-start state to BF16 shared memory for MFMA pred.
3. Compute `pred_sub0[16,16] = W_sub0[16,128] @ H[16,128].T`.
4. Write `vn` and `v_decay_sub0` using `exp(g_last - g_t)`.
5. Compute `pred_sub1[16,16] = W_sub1[16,128] @ H[16,128].T`.
6. Write `vn` and `v_decay_sub1` using the same chunk `g_last`.
7. Apply `state *= exp(g_last)` once.
8. Accumulate `delta_sub0 = v_decay_sub0.T @ K_sub0`.
9. Accumulate `delta_sub1 = v_decay_sub1.T @ K_sub1`.
10. Write final_state after the last chunk.

`g_last` is `g[chunk_start + 31]`.

New chunk_o kernel:

- `_qwen_gdn_chunk_o_bf16_kernel_v13_bt32_bv16_mfma`

The chunk_o prototype uses two 16-token output sub-blocks per 32-token chunk.
Each output sub-block computes:

- inter: `q_sub[16,128] @ h[16,128].T`
- intra from source sub0: `score0[16,16] @ vn0[16,16]`
- intra from source sub1: `score1[16,16] @ vn1[16,16]`

The causal mask is applied with absolute chunk offsets:

- source sub0 offset: `0 + source_offset`
- source sub1 offset: `16 + source_offset`
- output offset: `out_base + token_offset`

This preserves full chunk_size=32 semantics while keeping the MFMA tile shape
close to the v12 chunk_o implementation.

## Correctness result

Test file:

```bash
docker exec ac739c57a0bf bash -lc 'cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare && env HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py -s --tb=short'
```

Coverage:

- T=32,64,512,1024
- with and without initial_state
- chunk_gdr v13 BT32 vs v6 chunk_gdr with chunk_size=32
- full forward v13 BT32 vs v12 BT16
- output and final_state max_abs/max_rel printed for every case

Result:

```text
16 passed in 43.40s
```

Observed max absolute errors:

- chunk_gdr h: <= 0.00285739
- chunk_gdr vn: <= 0.00186467
- chunk_gdr final_state: <= 0.00255853
- full output vs v12: <= 0.000441321
- full final_state vs v12: <= 0.00278249

Relative error can be large near zero-valued elements; absolute error is small.

## Syntax check

```bash
python3 -B -c "import ast, pathlib; files=('qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py','test_qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py','bench_qwen_gdn_v13_mfma.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]; print('syntax_ok')"
```

Result:

```text
syntax_ok
```

## Benchmark

Benchmark file:

```bash
docker exec ac739c57a0bf bash -lc 'cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare && env PYTHONDONTWRITEBYTECODE=1 python bench_qwen_gdn_v13_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30'
```

The benchmark compares:

- vLLM
- v12 BT=16/BV=16
- v13 BT=32/BV=16

It prints:

- full latency
- v12 stage breakdown
- v13 stage breakdown
- chunk_gdr
- w_u
- chunk_o
- speedup vs v12
- slowdown vs vLLM
- output/final_state max_abs vs v12 and vLLM

Full latency:

| T | vLLM ms | v12 BT16 ms | v13 BT32 ms | speedup v13 vs v12 | slowdown v13 vs vLLM | out err vs v12 | final_state err vs v12 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3104 | 0.7494 | 2.5580 | 0.2930x | 8.2420x | 0.00031868 | 0.00253338 |
| 1024 | 0.3046 | 1.3255 | 3.7174 | 0.3566x | 12.2053x | 0.00031868 | 0.00248396 |
| 2048 | 0.3640 | 2.5475 | 6.5339 | 0.3899x | 17.9503x | 0.000318855 | 0.00229996 |

v12 stage breakdown:

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0323 | 0.0622 | 0.0440 | 0.2841 | 0.3727 | 0.0492 |
| 1024 | 0.0286 | 0.0778 | 0.0471 | 0.4993 | 0.7093 | 0.0716 |
| 2048 | 0.0285 | 0.1148 | 0.0476 | 0.9644 | 1.3623 | 0.1075 |

v13 stage breakdown:

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0284 | 0.0880 | 1.4094 | 0.5194 | 0.4836 | 0.0575 |
| 1024 | 0.0280 | 0.1224 | 1.6059 | 0.9351 | 1.0154 | 0.0849 |
| 2048 | 0.0289 | 0.1928 | 2.4065 | 1.8264 | 1.9871 | 0.1363 |

## Analysis

v13 is correctness-passing, but not performance-viable in this form.

The intended BT=32 chunk_gdr experiment does not improve the bottleneck:

- T=2048 chunk_gdr: v12 1.3623 ms -> v13 1.9871 ms
- T=2048 w_u: v12 0.9644 ms -> v13 1.8264 ms
- T=2048 solve: v12 0.0476 ms -> v13 2.4065 ms
- T=2048 chunk_o remains small: v13 0.1363 ms

The largest regression is `solve`, because v6 solve is a scalar small-matrix
recurrence and `chunk_size=32` greatly increases its per-chunk work.  `w_u`
also regresses almost 2x, matching the audit expectation that the current w_u
scalar loop scales directly with chunk_size.  chunk_gdr itself is slower too:
the prototype halves the number of chunks but each program performs two
sub-block pred paths and two update paths serially, reducing effective
parallelism and increasing per-program LDS/control work.

Current bottleneck for v13 at T=2048:

1. solve: 2.4065 ms
2. chunk_gdr: 1.9871 ms
3. w_u: 1.8264 ms
4. chunk_o: 0.1363 ms

## Next action

- v13 doubles the w_u chunk size from 16 to 32.  The current v6 w_u kernel is
  scalar over chunk_size, so v13 may improve chunk_gdr while increasing w_u.
- chunk_o is intentionally not the optimization target.  The v13 chunk_o path
  preserves correctness with two 16-token output sub-blocks and includes the
  cross-sub-block intra contribution.
- Do not pursue this BT=32 full-path direction until solve and w_u have tiled
  implementations.
- The next practical optimization should be standalone w_u MFMA tiling at
  BT=16 first.  Keep v12 chunk_size=16 as the baseline until solve/w_u are no
  longer scalar bottlenecks.
- If revisiting BT=32 chunk_gdr, use a kernel design with more parallelism
  across the two 16-token sub-blocks rather than serializing both sub-blocks
  inside one program.

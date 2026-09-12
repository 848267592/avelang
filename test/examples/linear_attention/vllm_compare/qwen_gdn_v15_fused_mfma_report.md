# Qwen GDN v15 fused MFMA report

## Target

Primary target is the vLLM Qwen3Next TP4 per-rank operator shape:

- B=1
- Hk=4, Hv=8
- K=128, V=128
- dtype=BF16 for q/k/v
- dtype=FP32 for w/u/g/initial_state/output/final_state
- layout=[B,T,H,D]
- chunk_size=16
- initial_state=[B,Hv,V,K]

Unsupported shapes raise `ValueError`; v15 does not silently fall back.

## Changed files

- `qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed.py`
- `bench_qwen_gdn_v15_fused_mfma.py`
- `qwen_gdn_v15_fused_mfma_report.md`

v14 baseline files were not modified.

## Implementation

v15 was copied from v14 and keeps:

- v14 standalone MFMA tiled `w_u`
- v6 cumsum/KKT/solve
- v14 non-fused wrappers for reference/debug use inside the v15 file

New fused path:

- `_qwen_gdn_fused_chunk_gdr_o_bf16_kernel_v15_mfma`
- `qwen_gdn_fused_chunk_gdr_o_avelang_v15_mfma_layout`
- `qwen_gdn_chunked_avelang_v15_fused_mfma_layout_full`
- `qwen_gdn_chunked_avelang_v15_fused_mfma_layout`

The fused kernel uses one 64-thread workgroup per `(value_block, value_head)` and loops over all chunks, matching the state ownership model of v12/v14 `chunk_gdr`.

It does not allocate or write global `h`.

It does not allocate or write global `vn`.

Per chunk:

1. Stage old state, W, Q, K into shared memory.
2. Compute `pred = W @ state.T` with MFMA.
3. Compute local `vn = u - pred` in shared BF16 form.
4. Compute chunk_o `inter = q @ old_state.T`.
5. Compute chunk_o causal score and `intra = score_decay @ vn`.
6. Write output directly.
7. Compute update `delta_H = v_decay.T @ K`.
8. Write `state = state * exp(g_last) + delta_H`, folding state decay into update writeback.

The output is computed before state update, so inter uses the old chunk-start state.

## Correctness

Command, run inside Docker container `ac739c57a0bf`:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed.py -s --tb=short
```

Result:

```text
9 passed in 30.51s
```

Coverage:

- v15 fused full forward vs v14 full forward
- T=16,32,64,512
- with and without initial_state
- monkeypatch test that fails if v15 fused full calls separate v14 chunk_gdr/chunk_o wrappers

Observed max_abs against v14:

- output: 0
- final_state: 0

## Benchmark

Command, run inside Docker container `ac739c57a0bf`:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python bench_qwen_gdn_v15_fused_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

Full latency:

| T | vLLM ms | v14 MFMA ms | v15 fused ms | speedup v15 vs v14 | slowdown v15 vs vLLM |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.3083 | 0.5398 | 0.5302 | 1.0181x | 1.7198x |
| 1024 | 0.3018 | 0.8990 | 0.8971 | 1.0021x | 2.9723x |
| 2048 | 0.3604 | 1.6529 | 1.6766 | 0.9859x | 4.6521x |

v14 stage breakdown:

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o | chunk_gdr+chunk_o |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0279 | 0.0598 | 0.0437 | 0.0502 | 0.3735 | 0.0487 | 0.4222 |
| 1024 | 0.0287 | 0.0792 | 0.0452 | 0.0598 | 0.7105 | 0.0715 | 0.7821 |
| 2048 | 0.0284 | 0.1155 | 0.0479 | 0.0816 | 1.3622 | 0.1076 | 1.4698 |

v15 stage breakdown:

| T | cumsum | KKT | solve | w_u | fused_chunk_gdr_o | fused stage speedup vs v14 sum |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0278 | 0.0600 | 0.0430 | 0.0513 | 0.3927 | 1.0751x |
| 1024 | 0.0280 | 0.0790 | 0.0455 | 0.0601 | 0.7591 | 1.0303x |
| 2048 | 0.0277 | 0.1164 | 0.0479 | 0.0821 | 1.4784 | 0.9942x |

Benchmark max_abs:

| T | output vs v14 | final_state vs v14 | output vs vLLM | final_state vs vLLM |
|---:|---:|---:|---:|---:|
| 512 | 0 | 0 | 0.000614453 | 0.00512862 |
| 1024 | 0 | 0 | 0.000710934 | 0.00516295 |
| 2048 | 0 | 0 | 0.000710934 | 0.00481206 |

## Rocprof

Command:

```bash
cd /workspace/project/avelang
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex "qwen_gdn_fused_chunk_gdr_o" \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v15_fused \
  -o v15_fused_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v15_fused_mfma.py --T 2048 --warmup 2 --repeat 5
```

Fused kernel profile:

| kernel | workgroup | grid | LDS bytes | scratch | VGPR | acc VGPR | SGPR | median trace us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `_qwen_gdn_fused_chunk_gdr_o_bf16_kernel_v15_mfma` | 64 | 4096 | 30208 | 0 | 24 | 152 | 112 | 1433.65 |

Counter collection median:

| SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|---:|---:|---:|---:|---:|---:|
| 491520 | 13131904 | 1378880 | 995328 | 2295808 | 0.644743 |

For context, the existing v12/v14 split kernels in the available `qwen_profile_v12_wu` trace show:

| kernel | workgroup | grid | LDS bytes | scratch | VGPR | acc VGPR | median trace us |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk_gdr integrated MFMA | 64 | 4096 | 20992 | 0 | 104 | 32 | 1322.25 |
| chunk_o MFMA | 64 | 524288 | 13312 | 0 | 36 | 68 | 83.44 |

The fused kernel reduces VMEM versus the sum of separate chunk_gdr/chunk_o counters, and it removes global h/vn materialization. However, it also creates a longer-lived 64-block kernel with larger LDS and much higher accumulator pressure than either separate kernel alone.

## Analysis

v15 achieves the structural goal: the fused path writes only output and final_state, and correctness matches v14 exactly for the tested shapes.

Performance does not improve at the main long-T target. At T=2048:

- v14 full: 1.6529 ms
- v15 fused full: 1.6766 ms
- v14 chunk_gdr+chunk_o: 1.4698 ms
- v15 fused_chunk_gdr_o: 1.4784 ms

The likely reason is that v14 `chunk_o` was already cheap and highly parallel across `num_chunks * value_heads * value_tiles`. The fused kernel moves that work into the 64 persistent chunk_gdr-style blocks that must process chunks sequentially to preserve state order. That saves h/vn global traffic but loses chunk_o's parallel dispatch shape and increases the fused kernel resource footprint.

So v15 should not replace v14 as the default long-T path in its current form. It is a correct prototype and a useful negative result: h/vn materialization is not the dominant remaining cost once v14 w_u is optimized.

## Current bottleneck

For the best measured path, v14 remains the better baseline at T=2048:

- `chunk_gdr`: about 1.36 ms
- `chunk_o`: about 0.108 ms
- `w_u`: about 0.082 ms

The remaining bottleneck is still the stateful `chunk_gdr` work itself, not the global h/vn handoff to chunk_o.

## Next action

Do not continue optimizing `chunk_o` alone. For a useful v16, focus on reducing the resource footprint and serial work inside `chunk_gdr`:

- split pred/update lifetimes so the compiler does not keep excessive accumulator state live
- investigate a chunk_gdr kernel shape that increases parallelism without changing chunk_size
- reduce shared staging duplication, especially K staging for score/update-style paths
- consider fusion only if it preserves chunk_o parallelism or avoids the fused kernel's 64-block long-lived scheduling limit

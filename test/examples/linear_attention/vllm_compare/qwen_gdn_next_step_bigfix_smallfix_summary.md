# Qwen GDN Next Step Bigfix/Smallfix Summary

## Summary Conclusion

Small fix helped only modestly and only in v29 pred-only. It does not transfer cleanly to v23/v24 production because v23/v24 already use distributed VN/V-decay, not the repeated scalar contiguous-V epilogue loop that made v29 slow.

Big fix looks promising. The fused pred-update skeleton avoids full global VN materialization and stages corrected values into update-friendly shared memory. At `T=2048`, it is much closer to the no-VN-store upper bound than to original v29:

- original v29 pred-only: `0.482537 ms`
- grouped_v4 best: `0.462788 ms`
- fused_no_vn_store upper bound: `0.209131 ms`
- fused_pred_update_skeleton: `0.238193 ms`

Next single recommended action: implement `fused_pred_update_partial` that consumes staged `u_corr_t_bf16` for one or two update K tiles and checks state-update correctness.

## 1. Did Small Fix Help?

Yes, but only modestly at long T.

| T | original_v29 | grouped_v4_best | speedup |
|---:|---:|---:|---:|
| 512 | `0.101892` | `0.118937` | `0.8567x` |
| 1024 | `0.265454` | `0.244584` | `1.0853x` |
| 2048 | `0.484118` | `0.462367` | `1.0470x` |

Correctness matched original v29 BF16-level error:

- T=2048 max_abs `4.37299609e-02`
- T=2048 mean_abs `6.07558433e-03`

The clean file is:

- `qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best.py`

## 2. Did Grouped-V4 Transfer To v23/v24?

No.

v23/v24 production chunk_gdr uses distributed VN/V-decay:

```python
linear_vn = wave_id * 64 + lane
token_offset = linear_vn // 16
value_offset = linear_vn - token_offset * 16
vn[0, token_idx, value_head_idx, global_v] = v_new
v_decay_t[value_offset, token_offset] = ...
```

There is no v29-like `128 threads * 8 reps` full-VN scalar epilogue loop. Grouping four V values per thread would change the thread/data ownership and update staging, so it is not a low-risk local micro-optimization.

No `qwen_gdn_chunked_avelang_v24_local_grouped_v4_opt.py` was created.

## 3. Did Big Fused Pred-Update Look Viable?

Yes, as a skeleton/profiling result.

The prototype:

- uses original v29 k-split MFMA32 pred schedule;
- computes `u_corr = u - pred`;
- stages full corrected values into shared BF16 `[V, token]`;
- does not write full global `vn`;
- emits only a tiny side effect.

It is not a full correctness path yet, but it directly tests whether avoiding global VN materialization removes the large source-level cost.

Benchmark:

| T | original_v29 | grouped_v4_best | fused_no_vn_store | fused_pred_update_skeleton |
|---:|---:|---:|---:|---:|
| 512 | `0.105657` | `0.118676` | `0.068702` | `0.074771` |
| 1024 | `0.239937` | `0.240056` | `0.115932` | `0.135301` |
| 2048 | `0.482537` | `0.462788` | `0.209131` | `0.238193` |

Counters at T=2048:

| variant | trace_us | VGPR | AccVGPR | MFMA | VALU | SALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped_v4_best | `468.837` | 48 | 216 | 32768 | 3796160 | 279808 | 198656 | 247808 |
| fused_pred_update_skeleton | `205.846` | 72 | 192 | 32768 | 1548992 | 32128 | 167936 | 299008 |

This strongly supports continuing fusion, with the caveat that full update correctness is still unproven.

## 4. Should We Continue v29 Source-Level Fusion?

Yes, but only on the fused path.

Do not continue:

- flat epilogue;
- precomputed acc_i mapping;
- token_split;
- update_only reading global VN;
- more BT/BV/MFMA shape sweep.

Continue with:

- `fused_pred_update_partial`;
- then `fused_pred_update_full` only after partial state update correctness passes.

## 5. Should We Return To v23/v24 Production Optimization?

Not as the immediate next action. v24 is currently the production baseline and still relevant, but this experiment gives a clear larger next step in v29 fusion.

v24 baseline context:

| T | vLLM | v23 | v24 | v24 KKT | v24 chunk_gdr | v24 chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | `0.3075` | `0.3043` | `0.2916` | `0.0300` | `0.1187` | `0.0490` |
| 1024 | `0.3069` | `0.4071` | `0.3964` | `0.0289` | `0.1966` | `0.0725` |
| 2048 | `0.3643` | `0.6403` | `0.5873` | `0.0305` | `0.3490` | `0.1083` |

v24 still has a real gap to vLLM at long T, but grouped-v4 does not offer a clean local fix for it.

## 6. Is This Evidence For Compiler/Backend Block-Dot + Epilogue Lowering?

Partially.

Evidence so far:

- MFMA32 source generation works.
- The full scalar VN epilogue is very expensive.
- grouped-v4 reduces some VALU/LDS but does not generate vectorized stores.
- raw `buffer_store_dwordx4` is available in standalone source probes, but safe FP32 x4 packing inside this VN epilogue is not yet integrated.
- removing full global VN materialization gives a very large gain.

This supports two parallel interpretations:

1. Algorithm/source fusion is the biggest immediate win.
2. Backend/block-dot epilogue lowering could still matter, especially for vectorized store generation and reducing scalar address-generation overhead.

Do not overclaim a compiler bug from this alone; the fused skeleton already shows a source-level way to remove most of the cost by avoiding the epilogue.

## 7. Next Single Recommended Action

Implement `fused_pred_update_partial` in:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto.py`

Scope:

- consume staged `u_corr_t_bf16[V, token]`;
- run update MFMA for one or two K tiles;
- compare the updated state tile against a small torch/reference calculation;
- keep no global full-VN materialization;
- only then expand to full update.

## Exact Commands Used

Syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_v29_next python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best.py
PYTHONPYCACHEPREFIX=/tmp/pycache_v29_next python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto.py
PYTHONPYCACHEPREFIX=/tmp/pycache_v29_next python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_smallfix_grouped_v4.py \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_pred_update.py
```

Smallfix benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_smallfix_grouped_v4.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

Fused prototype benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_fused_pred_update.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

v23/v24 context:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v24_kkt_mfma.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

rocprof pattern:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex fused_pred_update_skeleton \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_fused_skeleton \
  -o v29_fused_skeleton_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto.py \
       --T 2048 --warmup 2 --repeat 5 --variant fused_pred_update_skeleton --no-check-ref
```

## File Paths

Track A:

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_smallfix_grouped_v4.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_smallfix_grouped_v4_report.md`

Track B:

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_pred_update.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v29_mfma32_fused_pred_update_report.md`

Summary:

- `test/examples/linear_attention/vllm_compare/qwen_gdn_next_step_bigfix_smallfix_summary.md`

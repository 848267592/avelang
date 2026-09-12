# Qwen GDN v29 MFMA32 Epilogue Optimization Report

## Summary

Best correctness-preserving source variant: `grouped_v4_epilogue`.

At `T=2048`, median benchmark latency:

- `original_v29`: `0.482197 ms`
- `baseline` copy: `0.492092 ms`
- `flat_epilogue`: `0.496097 ms`
- `grouped_v4_epilogue`: `0.460946 ms`
- `fused_no_vn_store` profiling-only: `0.208450 ms`

`grouped_v4_epilogue` improves the copied baseline by about `6.3%` and improves original v29 by about `4.6%`, but it is still not enough to rescue the MFMA32 pred-only direction versus the current BT16 production path. The profiling-only no-full-VN-store path remains much faster, so the main diagnosis is unchanged: full VN materialization/epilogue dominates this v29 pred-only source shape.

Go/no-go: do not proceed to `v29_update_only` from this source shape yet. The useful next direction is fused pred-update or backend/block-dot epilogue lowering, not more source-level acc-index polish.

## Why This Experiment Was Needed

Previous v29 ISA tuning showed:

- MFMA32 itself lowers correctly.
- `no_acc_unpack` at `T=2048` was about `0.164705 ms`.
- unpack/reduce without full VN store was around `160 us`.
- enabling the full VN epilogue raised runtime to about `0.49-0.52 ms`.
- precomputed accumulator mapping did not improve performance.

That made the VN epilogue the main remaining suspect:

```python
for rep_out in al.range(8):
    linear_o = tid + rep_out * WORKGROUP
    token_off_o = linear_o // BV
    local_v_o = linear_o - token_off_o * BV

    token_idx_o = chunk_start + token_base + token_off_o
    global_v_o = value_base + local_v_o

    pred_o = pred_partial[0, token_off_o, local_v_o] + pred_partial[1, token_off_o, local_v_o]
    vn[0, token_idx_o, value_head_idx, global_v_o] = (
        u[0, token_idx_o, value_head_idx, global_v_o] - pred_o
    )
```

It reduces two K-half partials, loads `u`, and writes full `vn` for `32 x 32` elements per token tile.

## Changed Files

- `qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_epilogue_opt.py`
- `bench_qwen_gdn_v29_mfma32_epilogue_opt.py`
- `qwen_gdn_v29_mfma32_epilogue_opt_report.md`

No v23/v24/v26/v27/v28 files were modified.

## Variants

### baseline

Equivalent to current full-correct v29 k-split MFMA32 pred-only behavior. Uses 4D tensor indexing for `u` and `vn`.

### flat_epilogue

Adds flat 1D views:

```python
out_offset = token_idx_o * (8 * 128) + value_head_idx * 128 + value_base + local_v_o
vn_flat[out_offset] = u_flat[out_offset] - pred_o
```

Purpose: remove 4D tensor indexing in the hot epilogue. Shifts were not used; arithmetic forms were kept for Avelang compatibility.

Result: correctness preserved, but no speedup.

### grouped_v4_epilogue

Processes four contiguous V elements per thread iteration:

```python
for rep4 in al.range(2):
    linear4 = tid + rep4 * WORKGROUP
    token_off4 = linear4 // 8
    group_v4 = linear4 - token_off4 * 8
    local_v4 = group_v4 * 4
    base_offset = token_idx4 * (8 * 128) + value_head_idx * 128 + value_base + local_v4
```

Then it computes/stores `pred0..pred3` using scalar stores. This reduces loop/address overhead but does not generate vectorized stores.

Result: correctness preserved, small but real improvement at long T.

### grouped_v4_raw_store

Not implemented. The previous standalone probe proved `raw_buffer_store_x4` can lower to `buffer_store_dwordx4` for an `i32` copy. In this VN epilogue, the kernel needs to pack four independent FP32 arithmetic results into the exact x4 raw-buffer operand type. I did not force this because the source-level packing semantics were not yet verified for this kernel.

### fused_no_vn_store

Profiling-only upper bound. It keeps MFMA32, accumulator unpack, partial reduction, and one dependent `u` load side effect, but does not store full VN.

Result: much faster, confirming full VN materialization remains the large cost.

## Correctness

Command examples:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
python3 qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_epilogue_opt.py \
  --T 64 --warmup 2 --repeat 3 --variant grouped_v4_epilogue
```

Benchmark correctness used the torch reference:

```python
pred = torch.einsum("bthk,bhvk->bthv", w.float(), initial_state.float())
vn = u - pred
```

| T | variant | max_abs | mean_abs | status |
|---:|:---|---:|---:|:---|
| 64 | baseline | `3.70277166e-02` | `6.05241023e-03` | pass, BF16-level |
| 64 | flat_epilogue | `3.70277166e-02` | `6.05241023e-03` | pass |
| 64 | grouped_v4_epilogue | `3.70277166e-02` | `6.05241023e-03` | pass |
| 512 | baseline | `3.64267826e-02` | `6.07679598e-03` | pass |
| 512 | flat_epilogue | `3.64267826e-02` | `6.07679598e-03` | pass |
| 512 | grouped_v4_epilogue | `3.64267826e-02` | `6.07679598e-03` | pass |
| 2048 | baseline | `4.14036512e-02` | `6.08357973e-03` | pass |
| 2048 | flat_epilogue | `4.14036512e-02` | `6.08357973e-03` | pass |
| 2048 | grouped_v4_epilogue | `4.14036512e-02` | `6.08357973e-03` | pass |
| 2048 | fused_no_vn_store | NA | NA | profiling-only |

## Benchmark

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_epilogue_opt.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

| T | original_v29 | baseline | flat_epilogue | grouped_v4_epilogue | fused_no_vn_store |
|---:|---:|---:|---:|---:|---:|
| 512 | `0.102292` | `0.111166` | `0.111846` | `0.117395` | `0.067821` |
| 1024 | `0.261629` | `0.248870` | `0.267137` | `0.238795` | `0.114971` |
| 2048 | `0.482197` | `0.492092` | `0.496097` | `0.460946` | `0.208450` |

`grouped_v4_epilogue` helps at T=1024/2048 but regresses at T=512.

## rocprof T=2048

Command pattern:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex v29_mfma32_epilogue_opt \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_epilogue_opt_${variant} \
  -o v29_epilogue_opt_${variant}_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_epilogue_opt.py \
       --T 2048 --warmup 2 --repeat 5 --variant ${variant} --no-check-ref
```

| variant | trace_us | WG | Grid | LDS bytes | Scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occupancy |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `494.956` | 128 | 4096 | 24576 | 0 | 60 | 204 | 112 | 32768 | 3944448 | 279744 | 198656 | 264192 | 0.6383 |
| flat_epilogue | `495.477` | 128 | 4096 | 24576 | 0 | 56 | 208 | 112 | 32768 | 3977024 | 279808 | 198656 | 264192 | 0.6383 |
| grouped_v4_epilogue | `468.376` | 128 | 4096 | 24576 | 0 | 48 | 216 | 112 | 32768 | 3796160 | 279808 | 198656 | 247808 | 0.6381 |
| fused_no_vn_store | `175.782` | 128 | 4096 | 24576 | 0 | 28 | 236 | 112 | 32768 | 1203136 | 33024 | 137216 | 233472 | 0.6185 |

## ISA Analysis

HSACO/ISA paths:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_epilogue_opt_hsaco*/`

MFMA evidence:

- `v_mfma_f32_32x32x8_bf16` is present.
- `v_mfma_f32_16x16x16_bf16` is absent.

Key static mnemonic counts:

| mnemonic | baseline | flat | grouped_v4 | fused_no_store |
|:---|---:|---:|---:|---:|
| `v_mfma_f32_32x32x8_bf16` | 8 | 8 | 8 | 16 |
| `v_mfma_f32_16x16x16_bf16` | 0 | 0 | 0 | 0 |
| `global_load_dword` | 72 | 72 | 72 | 97 |
| `global_store_dword` | 8 | 8 | 8 | 2 |
| `buffer_store_dwordx4` | 0 | 0 | 0 | 0 |
| `ds_read_b128` | 8 | 8 | 12 | 16 |
| `ds_write_b16_d16_hi` | 64 | 64 | 64 | 96 |
| `ds_write_b32` | 16 | 16 | 16 | 32 |
| `v_lshl_add_u64` | 169 | 160 | 148 | 160 |
| `v_lshlrev_b32_e32` | 69 | 68 | 75 | 99 |
| `v_add_u32_e32` | 64 | 71 | 68 | 80 |
| `v_add3_u32` | 64 | 64 | 64 | 96 |
| `v_or_b32_e32` | 99 | 106 | 97 | 129 |
| `v_bfe_u32` | 64 | 64 | 64 | 96 |
| `s_waitcnt` | 75 | 75 | 69 | 80 |
| `s_barrier` | 4 | 4 | 4 | 7 |

Static ISA supports the dynamic counters:

- flat 1D indexing lowers `v_lshl_add_u64` slightly but adds other address ops, so total VALU does not improve.
- grouped_v4 reduces total static instruction count and dynamic VALU/LDS somewhat.
- no vectorized store appears; stores remain scalar/global or format stores.

## Diagnosis

Did flat 1D indexing reduce VALU/SALU?

- No. Dynamic VALU increased slightly: `3,944,448 -> 3,977,024`; SALU was effectively unchanged. It reduced one kind of address op but did not reduce total address-generation work.

Did grouped_v4 reduce VALU/SALU or VMEM?

- It reduced dynamic VALU: `3,944,448 -> 3,796,160`.
- It reduced LDS instructions: `264,192 -> 247,808`.
- SALU and VMEM were basically unchanged.
- Trace improved from `494.956 us` to `468.376 us`.

Was raw_buffer_store_x4 feasible?

- Not safely in this pass. The existing standalone probe proves the intrinsic can generate `buffer_store_dwordx4`, but packing four independent FP32 epilogue results into the required x4 raw-buffer operand inside this kernel needs a separate correctness probe. I did not add a risky raw store path to this experiment.

Is full VN materialization still the dominant cost?

- Yes. `fused_no_vn_store` drops trace to `175.782 us` and dynamic VALU to `1,203,136`, even though it is only profiling-equivalent. This strongly suggests the full VN epilogue/materialization is still the large cost.

Is source-level optimization enough?

- Not for this v29 shape. Grouping four scalar stores gives a modest improvement, but not enough. The remaining gap points to either backend/block-dot epilogue lowering or algorithmic fusion that avoids global VN materialization.

## Recommendation

Do not keep spending time on `acc_i` mapping or flat indexing. Also do not start `v29_update_only` from this source form yet.

Recommended next action:

1. Pursue fused pred-update so `vn` is consumed directly instead of materialized globally.
2. In parallel, escalate backend/block-dot epilogue lowering evidence: Avelang is still scalarizing the VN writeback, while raw x4 needs a cleaner source API/path for four FP32 results.
3. Keep `grouped_v4_epilogue` as a useful source-level data point, not a production replacement.

## Reproduction Commands

Syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_v29_epilogue python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_epilogue_opt.py
PYTHONPYCACHEPREFIX=/tmp/pycache_v29_epilogue python3 -m py_compile \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_epilogue_opt.py
```

Smoke:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 \
python3 qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_epilogue_opt.py \
  --T 64 --warmup 2 --repeat 3 --variant baseline
```

Benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_epilogue_opt.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

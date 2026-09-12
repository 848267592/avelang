# Qwen GDN v20 BT32 Native MFMA Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/a/a_solved/output/final_state`
- `chunk_size=32`, `BT=32`, `BV=16`
- Baselines: v17 predecay BT16, v19 BT32

## Changed Files

- `qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_fixed.py`
- `test_qwen_gdn_v20_bt32_native_mfma.py`
- `bench_qwen_gdn_v20_bt32_native_mfma.py`
- `qwen_gdn_v20_bt32_native_mfma_report.md`

No v17/v18/v19 baseline files were modified.

## Implementation Summary

v20 adds native 32x32x8 BF16 MFMA kernels for the BT32 stages that regressed in v19:

- `_qwen_gdn_w_bf16_kernel_v20_bt32_32x32_mfma`
- `_qwen_gdn_u_bf16_kernel_v20_bt32_32x32_mfma`
- `_qwen_gdn_kkt_bf16_kernel_v20_bt32_32x32_mfma`

It also adds a minimal standalone MFMA smoke path:

- `_qwen_gdn_mfma_32x32_smoke_kernel_v20`
- `qwen_gdn_mfma_32x32_smoke_v20`

The full v20 path is:

1. v6 cumsum with `chunk_size=32`
2. v20 native BT32 KKT
3. v18 BT32 solve
4. v20 native BT32 w_u
5. v19 BT32 chunk_gdr
6. v19 BT32 chunk_o

## 32x32 MFMA Availability

`mfma_32x32x8_bf16_f32` is available in Avelang:

- `docs/content/language-reference/hardware-intrinsics.md`
- `docs/content/tutorials/03-intrinsics.md`
- `lib/IR/Intrinsics/amdgpu_mfma_signatures.h`

The implementation follows the tutorial layout:

- one 64-thread wave computes one `32 x 32` output tile
- each lane holds 16 FP32 accumulator values
- `lane_col = lane & 31`
- `lane_group = lane >> 5`
- output column mapping:
  `((r >> 2) << 3) + lane_group * 4 + (r & 3)`

## Correctness Status

Executed in Docker container `ac739c57a0bf`.

### 32x32 MFMA Smoke

Command:

```bash
python -m pytest -q test_qwen_gdn_v20_bt32_native_mfma.py::test_v20_mfma_32x32_smoke_matches_torch -s --tb=short
```

Result:

```text
smoke_max_abs=1.90734863e-06,smoke_max_rel=5.33124839e-05
1 passed
```

### v20 KKT vs v6 BT32 KKT

Command:

```bash
python -m pytest -q test_qwen_gdn_v20_bt32_native_mfma.py::test_v20_bt32_kkt_matches_v6 -s --tb=short
```

Result:

```text
4 passed
```

Shapes covered:

- `T=32,64,512,1024`

Worst printed error:

- `KKT max_abs <= 7.4505806e-08`

### v20 w_u and Full Path

Initial w_u test exposed a B-tile staging transpose bug in the new 32x32 W/U kernels. The staging was corrected so that the B tile is:

```text
B[row=output_col, col=source_token] = k_or_v[source_token, output_col]
```

instead of crossing source/output-column indices.

After that fix, Docker access was blocked by the platform usage limit, so the corrected w_u and full-path tests have not yet been rerun. They must be rerun before using v20 benchmark numbers.

Pending commands:

```bash
python -m pytest -q test_qwen_gdn_v20_bt32_native_mfma.py::test_v20_bt32_w_u_matches_v6 -s --tb=short
python -m pytest -q test_qwen_gdn_v20_bt32_native_mfma.py::test_v20_bt32_full_matches_v19 -s --tb=short
python -m pytest -q test_qwen_gdn_v20_bt32_native_mfma.py -s --tb=short
```

## Benchmark Status

Benchmark script was added but not executed after the w_u staging fix because Docker access was blocked:

```bash
python bench_qwen_gdn_v20_bt32_native_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

The script compares:

- vLLM
- v17 predecay BT16
- v19 BT32
- v20 BT32 native KKT/w_u

It prints:

- full latency
- output/final_state accuracy
- stage breakdown: `cumsum`, `KKT`, `solve`, `w_u`, `gdr_decay`, `chunk_gdr`, `chunk_o`

## rocprof Status

rocprof was not run for v20 because w_u/full correctness is still pending.

Once correctness passes, profile:

- `_qwen_gdn_w_bf16_kernel_v20_bt32_32x32_mfma`
- `_qwen_gdn_u_bf16_kernel_v20_bt32_32x32_mfma`
- `_qwen_gdn_kkt_bf16_kernel_v20_bt32_32x32_mfma`

Compare against v19:

- `_qwen_gdn_w_bf16_kernel_v19_bt32_mfma`
- `_qwen_gdn_u_bf16_kernel_v19_bt32_mfma`
- `_qwen_gdn_kkt_bf16_kernel_v6_standalone`

Counters:

- `SQ_INSTS_MFMA`
- `SQ_INSTS_VALU`
- `SQ_INSTS_SALU`
- `SQ_INSTS_VMEM`
- `SQ_INSTS_LDS`
- `OccupancyPercent`
- `VGPR_Count`
- `Accum_VGPR_Count`
- `LDS_Block_Size`
- `Scratch_Size`
- median trace us

## Current Conclusion

The 32x32 MFMA API and lane mapping are confirmed usable. Native BT32 KKT correctness is confirmed and removes the scalar workgroup=1 implementation from that stage.

v20 is not yet a validated full-path result because corrected w_u and full correctness could not be rerun after Docker access was blocked. Next action is to rerun w_u/full correctness, then benchmark and rocprof. If w_u passes and KKT/w_u stage latency drops as intended, the next BT32 blocker will likely be v19 chunk_gdr/chunk_o.

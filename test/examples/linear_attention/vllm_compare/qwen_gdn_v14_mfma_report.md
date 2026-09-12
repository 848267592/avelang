# Qwen GDN v14 MFMA w_u report

## Target

Primary target is the vLLM Qwen3Next TP4 per-rank operator shape:

- B=1
- Hk=4, Hv=8
- K=128, V=128
- dtype=BF16 for q/k/v
- dtype=FP32 for g/beta/a_solved and intermediate outputs
- layout=[B,T,H,D]
- chunk_size=16
- initial_state=[B,Hv,V,K]

Unsupported shapes raise `ValueError`; v14 does not silently fall back to v6.

## Changed files

- `qwen_gdn_chunked_avelang_v14_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v14_mfma_layout_fixed.py`
- `bench_qwen_gdn_v14_mfma.py`
- `qwen_gdn_v14_mfma_report.md`

The v14 path was copied from the stable v12 file. It keeps v12 `chunk_gdr` and `chunk_o`, keeps v6 cumsum/KKT/solve, and replaces only the `w_u` stage.

## Implementation

New standalone w/u path:

- `_qwen_gdn_w_bf16_kernel_v14_mfma`
- `_qwen_gdn_u_bf16_kernel_v14_mfma`
- `qwen_gdn_w_u_avelang_v14_mfma_layout`

Each kernel computes one 16x16 output tile per program:

- W grid: `num_chunks * 8 value_heads * 8 k_tiles`
- U grid: `num_chunks * 8 value_heads * 8 v_tiles`
- workgroup: `(64,1,1)`
- tile shape: 16 tokens x 16 columns

Mathematics:

```text
A_w[t,s] = a_solved[t,s] * beta[s] * exp(g_cumsum[s])
A_u[t,s] = a_solved[t,s] * beta[s]

W[16,128] = A_w[16,16] @ K_chunk[16,128]
U[16,128] = A_u[16,16] @ V_chunk[16,128]
```

The main product uses the same MFMA pattern as v12:

- shared `A_bf16[16,16]`
- shared `B_bf16[16,16]`
- packed `i32` BF16 views
- `al.amdgpu.mfma_16x16x16_bf16_f32`
- FP32 accumulator

Accuracy note: pure BF16 staging of `A` was too loose for the v6 FP32 oracle, especially for `U`. v14 keeps the MFMA main path and adds a small FP32 residual correction:

```text
out = bf16(A) @ B + (A_fp32 - bf16(A)) @ B
```

This preserves the tiled/MFMA implementation while keeping full-forward errors under the requested 1e-3 tolerance.

## Correctness

Command, run inside Docker container `ac739c57a0bf`:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_qwen_gdn_chunked_avelang_v14_mfma_layout_fixed.py -s --tb=short
```

Result:

```text
12 passed in 32.69s
```

Coverage:

- w_u-only v14 vs v6 standalone oracle
- full forward v14 vs v12 full forward
- T=16,32,64,512
- with and without initial_state for full forward

Observed errors were below tolerance:

- w/u-only max_abs <= 4.77e-7
- full output max_abs <= 1e-3
- final_state max_abs <= 1e-3

## Benchmark

Command, run inside Docker container `ac739c57a0bf`:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python bench_qwen_gdn_v14_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

Full latency:

| T | vLLM ms | v12 MFMA ms | v14 MFMA ms | full speedup v14 vs v12 | w_u speedup v14 vs v12 | v14 slowdown vs vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3049 | 0.7520 | 0.5376 | 1.3987x | 5.7799x | 1.7635x |
| 1024 | 0.3003 | 1.3242 | 0.9000 | 1.4714x | 8.3411x | 2.9971x |
| 2048 | 0.3635 | 2.5467 | 1.6534 | 1.5403x | 11.7483x | 4.5491x |

v12 stage breakdown:

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0281 | 0.0605 | 0.0439 | 0.2829 | 0.3722 | 0.0489 |
| 1024 | 0.0279 | 0.0787 | 0.0451 | 0.4995 | 0.7123 | 0.0718 |
| 2048 | 0.0288 | 0.1152 | 0.0481 | 0.9624 | 1.3609 | 0.1088 |

v14 stage breakdown:

| T | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0283 | 0.0597 | 0.0438 | 0.0490 | 0.3738 | 0.0490 |
| 1024 | 0.0282 | 0.0791 | 0.0445 | 0.0599 | 0.7092 | 0.0708 |
| 2048 | 0.0290 | 0.1153 | 0.0490 | 0.0819 | 1.3602 | 0.1081 |

Benchmark max_abs against v12:

| T | output max_abs | final_state max_abs |
|---:|---:|---:|
| 512 | 0.000118732 | 0.000329003 |
| 1024 | 0.000118732 | 0.000071943 |
| 2048 | 0.000118732 | 0.000145525 |

## Rocprof

Command:

```bash
cd /workspace/project/avelang
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex "qwen_gdn_[wu]_bf16_kernel_v14" \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v14_wu \
  -o v14_wu_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v14_mfma.py --T 2048 --warmup 2 --repeat 5
```

Counter collection confirms the new kernels use MFMA and 64-thread workgroups:

| kernel | workgroup | grid | LDS bytes | scratch | VGPR | acc VGPR | SGPR | SQ_INSTS_MFMA | OccupancyPercent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_qwen_gdn_w_bf16_kernel_v14_mfma` | 64 | 524288 | 1024 | 0 | 68 | 4 | 112 | 32768 | 34.82 |
| `_qwen_gdn_u_bf16_kernel_v14_mfma` | 64 | 524288 | 1024 | 0 | 68 | 4 | 80 | 32768 | 40.93 |

Median trace duration in the profiled T=2048 run:

| kernel | median us |
|---|---:|
| `_qwen_gdn_w_bf16_kernel_v14_mfma` | 33.71 |
| `_qwen_gdn_u_bf16_kernel_v14_mfma` | 24.38 |
| `_qwen_gdn_w_u_bf16_kernel_v6_standalone` from v12 baseline section | 947.67 |

The trace contains `_qwen_gdn_w_u_bf16_kernel_v6_standalone` because the benchmark also runs the v12 baseline. The v14 path dispatches the two new W/U kernels above.

## Analysis

v14 removes the old single-thread-per-token/value-head w_u bottleneck. At T=2048, w_u drops from 0.9624 ms to 0.0819 ms, a 11.75x stage speedup. Full latency improves from 2.5467 ms to 1.6534 ms, a 1.54x full-operator speedup.

After this change, w_u is no longer the main bottleneck. At T=2048, the dominant stage is:

- `chunk_gdr`: 1.3602 ms
- `chunk_o`: 0.1081 ms
- `w_u`: 0.0819 ms

The v14 operator is still 4.55x slower than vLLM at T=2048, so the remaining gap is mostly in `chunk_gdr` and shared overhead around the chunk pipeline, not in `w_u`.

## Next action

Do not continue optimizing `chunk_o`; it is still small. The next useful work is a focused `chunk_gdr` pass:

- rocprof `chunk_gdr` counters on v14 at T=2048
- inspect MFMA/VALU/VMEM/LDS balance and occupancy
- evaluate whether `chunk_gdr` can reduce per-chunk staging/control overhead or use a better value-head/state tile
- revisit fusion only if it removes meaningful global traffic without reintroducing the old w_u scalar bottleneck

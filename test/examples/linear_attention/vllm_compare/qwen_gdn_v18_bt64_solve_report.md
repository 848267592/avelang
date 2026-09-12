# Qwen GDN v18 BT32/BT64 Solve Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Solve input dtype: FP32
- Solve target chunk sizes: `BT=32` and `BT=64`
- Baseline: v6 solve as called by the old v13 BT32 path

## Changed Files

- `qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py`
- `test_qwen_gdn_v18_bt64_solve.py`
- `bench_qwen_gdn_v18_bt64_solve.py`
- `qwen_gdn_v18_bt64_solve_report.md`

## Solve Audit

The existing v6 entry point is:

- `qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=...)`
- JIT kernel: `_qwen_gdn_solve_kernel_v6_standalone`

Input/output:

- `a`: FP32 `[B, T, Hv, chunk_size]`
- `a_solved`: FP32 `[B, T, Hv, chunk_size]`
- For the current target: `[1, T, 8, BT]`

Layout:

```text
shape  = (batch_size, num_tokens, num_heads, chunk_size)
stride = (num_tokens * num_heads * chunk_size,
          num_heads * chunk_size,
          chunk_size,
          1)
```

For each `(batch, chunk, value_head)`, v6 materializes one chunk-local matrix:

```text
M0[row, col] = -a[batch, chunk_start + row, head, col]
```

Then it runs a lower-triangular row recurrence:

```text
M[row, col] = M0[row, col]                                      if col >= row
M[row, col] = M0[row, col] + sum_i<row M0[row, i] * M[i, col]   if col < row
```

Finally:

```text
a_solved[row, col] = M[row, col] + 1(row == col)
```

This is a small lower-triangular recurrence, not a dense independent matrix multiply.

v13 BT32 used this same v6 solve path:

```text
a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=32)
```

v6 launch shape:

```text
grid  = B * num_chunks * Hv
block = (1, 1, 1)
```

At `T=2048`:

| chunk_size | num_chunks | v6 grid blocks | workgroup |
|---:|---:|---:|---:|
| 16 | 128 | 1024 | 1 thread |
| 32 | 64 | 512 | 1 thread |
| 64 | 32 | 256 | 1 thread |

The explosion comes from doing all row/col/inner loops on one thread.  Per chunk, work scales roughly as `O(BT^3)` for the recurrence, and the private local matrix grows as `BT^2`.  Reducing the number of chunks by 2x does not compensate for the much larger per-chunk serial loop and local memory pressure.

## v18 Implementation

v18 adds:

- `_qwen_gdn_solve_kernel_v18_parallel`
- `qwen_gdn_solve_avelang_v18_layout(a, chunk_size=32|64)`
- `qwen_gdn_solve_avelang_v18_bt32_layout`
- `qwen_gdn_solve_avelang_v18_bt64_layout`

Launch:

```text
grid  = num_chunks * 8
block = (128, 1, 1)
```

The row dependency remains sequential, but each row update is parallelized:

- Matrix is staged in shared memory.
- Columns are processed in parallel.
- Inner reduction is split across groups per column:
  - BT32: 4 groups per column
  - BT64: 2 groups per column
- A short shared-memory reduction combines the partials.

This avoids the v6 single-thread scan over the full chunk matrix.

The file also includes a minimal BT32 full smoke path:

- `qwen_gdn_chunked_avelang_v18_bt32_layout`
- It uses v18 solve.
- It still uses v6 `w_u` and v13 BT32 `chunk_gdr/chunk_o`.
- Therefore it is correctness-only; full-path performance is not treated as valid.

W BF16 output was not implemented.

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v18_bt64_solve.py -s --tb=short
```

Result:

```text
12 passed in 26.61s
```

Coverage:

- BT32 solve vs v6: `T=32,64,512,1024`
- BT64 solve vs v6: `T=64,128,512,1024`
- BT32 full smoke vs v13: `T=32,64`, with and without initial_state

Worst printed errors:

| Check | max_abs |
|:---|---:|
| BT32 solve | `2.98023224e-08` |
| BT64 solve | `4.47034836e-08` |
| BT32 smoke output | `6.37478661e-06` |
| BT32 smoke final_state | `1.13248825e-05` |

## Solve Benchmark

Command:

```bash
python bench_qwen_gdn_v18_bt64_solve.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Latency

| T | v6 BT16 | v6 BT32 | v18 BT32 | v6 BT64 | v18 BT64 |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.047170 | 1.420009 | 0.034010 | 12.283989 | 0.120880 |
| 1024 | 0.051036 | 1.719895 | 0.033490 | 18.503570 | 0.122963 |
| 2048 | 0.053740 | 2.383721 | 0.034231 | 21.117110 | 0.126688 |

### Speedup

| T | BT32 speedup vs v6 | BT64 speedup vs v6 |
|---:|---:|---:|
| 512 | 41.7521x | 101.6218x |
| 1024 | 51.3555x | 150.4808x |
| 2048 | 69.6363x | 166.6853x |

Accuracy during benchmark:

| T | BT32 max_abs | BT64 max_abs |
|---:|---:|---:|
| 512 | `2.98023224e-08` | `2.98023224e-08` |
| 1024 | `2.98023224e-08` | `2.98023224e-08` |
| 2048 | `2.98023224e-08` | `2.98023224e-08` |

## Decision

BT32 solve is now viable.  At `T=2048`, it drops from the old v13/v6 `2.3837 ms` to `0.0342 ms`, well under the `0.15 ms` target.

BT64 solve is also usable as a solve-only stage.  At `T=2048`, it is `0.1267 ms`, under `0.5 ms`.  It is still slower than BT32 solve, but no longer catastrophic.

The larger-chunk blocker has moved past solve.  The next full BT32/BT64 blockers are:

- `w_u`: current performant MFMA path is BT16-only; the BT32 smoke path uses old v6 `w_u`, so full latency would be invalid.
- `chunk_gdr`: v13 has a BT32 prototype, but it is not the v17 4-wave style and should be revisited after BT32 `w_u`.
- `chunk_o`: existing BT32 path is available for correctness, but it is not the current bottleneck and should not be the next focus.

## Next Action

Recommended next step:

1. Extend `w_u` MFMA to BT32 using 16-token subtiles or a native BT32 tiled path.
2. Then port the v17 cooperative chunk_gdr idea to BT32.
3. Benchmark full BT32 only after those two stages are valid.

BT64 full path should wait until BT32 proves a net full-latency win.

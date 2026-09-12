# Qwen GDN w_u optimization audit

## Scope

This audit covers the current `qwen_gdn_w_u_avelang_v6_standalone`
implementation used by the v12 and v13 clean paths.  It is analysis-only and
does not rewrite w_u.

## 1. Math and inputs/outputs

Inputs:

- `k`: `[B, T, Hk, K]`, BF16 or FP32
- `v`: `[B, T, Hv, V]`, BF16 or FP32
- `g_cumsum`: `[B, T, Hv]`, FP32 chunk-local cumulative gate
- `beta`: `[B, T, Hv]`, FP32
- `a_solved`: `[B, T, Hv, chunk_size]`, FP32 solved triangular weights

Outputs:

- `w`: `[B, T, Hv, K]`, FP32
- `u`: `[B, T, Hv, V]`, FP32

For a target token `t`, value head `hv`, chunk start `c`, source offset `d`,
source token `s = c + d`, and key head `hk = hv // (Hv / Hk)`:

```text
w[b,t,hv,kk] =
  sum_d a_solved[b,t,hv,d]
        * k[b,s,hk,kk]
        * beta[b,s,hv]
        * exp(g_cumsum[b,s,hv])

u[b,t,hv,vv] =
  sum_d a_solved[b,t,hv,d]
        * v[b,s,hv,vv]
        * beta[b,s,hv]
```

The sum scans all `d in [0, chunk_size)`, guarded by `s < T`.

## 2. Kernel count and global memory paths

Optimized BF16 path:

- wrapper branch: `qwen_gdn_w_u_avelang_v6_standalone(..., prefer_optimized=True)`
- kernel: `_qwen_gdn_w_u_bf16_kernel_v6_standalone`
- kernel launches: 1
- grid: `B * T * Hv`
- block size: 1 thread

Optimized FP32 path:

- kernel: `_qwen_gdn_w_u_fp32_opt_kernel_v6_standalone`
- kernel launches: 1
- grid: `B * T * Hv`
- block size: 1 thread

Fallback path:

- used when `prefer_optimized=False`
- launches separate w and u kernels
- kernel launches: 2

For the BF16 optimized path, each program handles one `(b, t, hv)`.

Global reads per program:

- `a_solved[b,t,hv,d]`: `chunk_size` FP32 reads
- `beta[b,s,hv]`: `chunk_size` FP32 reads
- `g_cumsum[b,s,hv]`: `chunk_size` FP32 reads for w
- `k[b,s,hk,kk]`: `chunk_size * K` BF16 reads
- `v[b,s,hv,vv]`: `chunk_size * V` BF16 reads

Global writes per program:

- `w[b,t,hv,kk]`: `K` FP32 writes
- `u[b,t,hv,vv]`: `V` FP32 writes

For the current target with `B=1`, `Hv=8`, `K=128`, `V=128`,
`T=2048`, and v12 `chunk_size=16`:

- programs: `2048 * 8 = 16384`
- scalar inner work per program: `16 * (128 + 128) = 4096` vector updates
- total scalar vector updates: about 67 million
- k/v BF16 traffic read repeatedly by target token/head: about 134 MB
- w/u FP32 output traffic: about 16 MB

With v13 `chunk_size=32`, the k/v and scalar accumulation work roughly double.

Empirical confirmation from the v13 BT32 benchmark:

| T | v12 chunk=16 w_u | v13 chunk=32 w_u | ratio |
|---:|---:|---:|---:|
| 512 | 0.2841 ms | 0.5194 ms | 1.83x |
| 1024 | 0.4993 ms | 0.9351 ms | 1.87x |
| 2048 | 0.9644 ms | 1.8264 ms | 1.89x |

This confirms that the current w_u implementation scales close to linearly
with chunk_size for the target shape.

## 3. Why w_u is about 0.96 ms at T=2048

The v12 report measured w_u at 0.9620 ms for T=2048.  That time is consistent
with the current mapping:

- one GPU thread performs the full chunk reduction for one `(token, value_head)`
- no MFMA is used
- no warp-level or block-level cooperation is used inside the K/V reductions
- k/v are reread for each target token instead of staged per chunk/head
- `w` and `u` are fully materialized to global memory and then read by chunk_gdr
- each source contributes scalar `exp(g_cumsum)` work for w

The result is enough parallel programs to occupy the GPU at large T, but each
program is a long scalar loop with poor arithmetic intensity and repeated
global memory traffic.

## 4. Main cost classification

Primary costs:

- scalar dot/reduction: yes, this is a dominant issue
- global memory traffic: yes, especially repeated k/v reads and materialized
  w/u writes

Secondary costs:

- exp/convert/index overhead: meaningful, because every program evaluates
  `chunk_size` exponentials and repeatedly converts BF16 k/v to FP32
- triangular solve related: indirect only.  w_u consumes `a_solved`, but the
  solve itself is a separate stage

The current bottleneck is not the triangular solve.  It is the scalar mapping
of a small dense per-chunk matrix multiply:

```text
w = A_solved @ (beta * exp(g) * K)
u = A_solved @ (beta * V)
```

## 5. Optimization suitability

MFMA tiling:

- Suitable.
- `A_solved` is `[BT, BT]`.
- `K` and `V` operands are `[BT, 128]`.
- For BT=16 or BT=32, the operation maps naturally to 16x16 BF16/FP32 MFMA
  tiles after staging/casting the small A matrix and source K/V tiles.
- Expected benefit is high because it replaces single-thread scalar loops with
  cooperative tile math.

Fusion with chunk_gdr:

- Potentially suitable, but higher risk.
- Fusion could avoid materializing and rereading `w` and `u`.
- It may also compute `u - w @ H` closer to chunk_gdr, reducing global memory
  pressure.
- The risk is high because chunk_gdr has recurrent state, per-value-block
  state layout, and chunk boundary semantics.  A fused kernel would be harder
  to debug than a standalone w_u tile kernel.

Vectorization/layout-only optimization:

- Suitable as a smaller step, but likely limited.
- Splitting K/V dimensions across lanes or programs could improve occupancy and
  coalescing.
- It still leaves the operation as scalar reductions and still materializes
  w/u.

## 6. Recommendation

Recommended next step:

1. Optimize w_u standalone first with a BT=16/32 MFMA-tiled kernel.
2. Keep the public contract identical: inputs `k/v/g_cumsum/beta/a_solved`,
   outputs `w/u`.
3. Benchmark standalone w_u and full forward against v12/v13.
4. Only after that, consider w_u + chunk_gdr fusion.

Expected benefit:

- Standalone MFMA w_u could plausibly reduce the T=2048 v12 w_u stage from
  about 0.96 ms to the few-tenths-of-a-millisecond range if staging and global
  writes are efficient.
- For v13 chunk_size=32, the standalone optimization is even more important
  because the current scalar w_u work roughly doubles.

Risk:

- Standalone MFMA w_u: medium.  The math is local to each chunk/head and can be
  tested directly against the existing v6 w_u output.
- w_u + chunk_gdr fusion: high.  It has larger correctness surface area and
  makes failures harder to localize.

Conclusion:

Do not optimize chunk_o next.  First make w_u standalone tiled/MFMA and measure
whether v13 BT=32 moves the bottleneck balance.  Fusion should be a second
phase after a fast standalone w_u exists as an oracle.

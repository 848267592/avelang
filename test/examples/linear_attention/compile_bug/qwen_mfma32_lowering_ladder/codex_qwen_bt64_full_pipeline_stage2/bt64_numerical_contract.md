# BT64 Numerical Contract

- Fixed input layout: `q/k [1,T,4,128]`, `v [1,T,8,128]`, `g/beta [1,T,8]`.
- Input dtypes: BF16 `q/k/v`; FP32 `g/beta`; FP32 optional `initial_state`.
- `T >= 64` and `T % 64 == 0`; no tail path is claimed.
- `g_cumsum[t] = sum(g[chunk_start:t+1])` independently per 64-token chunk.
- KKT is lower triangular: `A[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t]-g[s])` only for `s < t`.
- `A_solved` is the inverse unit-lower-triangular transform represented by
  vLLM `solve_tril`; the candidate uses the v18 BT64 Avelang solve.
- `w = A_solved @ (beta * exp(g_cumsum) * k)` and
  `u = A_solved @ (beta * v)` within each chunk.
- The recurrence is immutable Stage 1 CASE C: FP32 `w/u/g/h0`, XF32 pred,
  BF16 `h`, BF16 update input, FP32 `v_new` and final state.
- `initial_state=None` is explicitly adapted to an all-zero FP32 state before
  calling asm; this is mathematically equivalent to vLLM's null-state mode.
- Chunk-o computes `q @ h^T * exp(g)` plus causal
  `(q @ k^T) * exp(g_t-g_s) @ v_new`, then applies `scale=1/sqrt(128)`.
- Public output is BF16 `[1,T,8,128]`; final state is FP32 `[1,8,128,128]`
  when requested.

The frozen acceptance thresholds are `1/128` absolute for public BF16 output
and `2e-2` for FP32 final state. They were selected before the full matrix run
and are not adjusted per case.


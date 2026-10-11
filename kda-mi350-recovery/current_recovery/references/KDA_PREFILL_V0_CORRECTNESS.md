# Avelang KDA Prefill v0 correctness

Status: **PASS**. This document records a correctness-only implementation; it makes no performance claim and must not be used as a benchmark result.

## Provenance and fixed contract

| Item | Value |
|---|---|
| AITER reference | `ROCm/aiter` commit `7bb44998274fa679ece0a37bf8426ab780b1837d` |
| Avelang KDA v0 source commit | `c06c11c00e07ab3532e83bbe0b27e3968f3c4b99` |
| GPU | AMD Instinct MI300X, gfx942 (`torch.cuda` capability `[9, 4]`) |
| ROCm/HIP | HIP `7.2.26015` / ROCm 7.2 |
| PyTorch | `2.11.0+rocm7.2` |
| Triton (AITER reference) | `3.7.0` |
| AITER mode | `AITER_TRITON_ONLY=1`, `CHUNK_DELTA_ATTN_USE_FLASH_KDA=1` |
| Shape | `B=1, H=HV=12, K=V=128, C=32, BC=16` |
| Inputs | `q/k/v/g=BF16`, `beta/A_log/dt_bias=FP32` |
| State | FP32 canonical `[N,H,K,V]`, `state_v_first=False` |
| Gate | `lower_bound=-5`, safe fused gate, in-kernel q/k L2 norm and beta sigmoid |
| T scope | `32..4096`, unsegmented path only |

The older Stage-5 input helper sampled raw beta in BF16. This v0 stage explicitly requires FP32 beta, so the test converts the same deterministic sample to FP32 before passing it to **both** AITER and Avelang. No candidate uses a PyTorch KDA operation or an external HSACO bridge.

## AITER FlashKDA dataflow

The AITER reference is `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py`.

```text
K1 _flash_kda_prepare_kernel, parallel over (chunk, head)
  raw q/k -> FP32 row L2 norm
  raw g + dt_bias + A_log -> lower-bound sigmoid gate -> local log2 cumsum
  raw beta -> sigmoid(beta)
  -> kd, qd, kr, gt
  -> midpoint-pivot K/Q fragments
  -> L = -tril(diag(beta) K_piv K_inv^T, -1)
  -> INV = (I - L)^-1 by BC=16 doubling then 16->32 merge
  -> Mqk = tril(Q_piv K_inv^T)

K2 _flash_kda_segment_kernel, sequential over chunks per (head, value tile)
  H is persistent FP32 [K,V] state
  pred = Kd @ H
  u = sigmoid(raw_beta) * (v - pred)
  U = INV @ u
  output = Qd @ H + Mqk @ U          # H before update
  H = H * gt + Kr^T @ U
```

The workspaces retain AITER's physical layouts and dtype boundaries:

| Workspace | Shape | dtype | Meaning |
|---|---|---|---|
| `kd`, `qd`, `kr` | `[H*NT, C, K]` | BF16 | decayed K, scaled decayed Q, restored K |
| `gt` | `[H*NT, K]` | FP32 | full chunk gate product |
| `inv_mqk[:,0]` | `[H*NT, C, C]` | FP16 | `INV` |
| `inv_mqk[:,1]` | `[H*NT, C, C]` | FP16 | `Mqk` |

`INV` and `Mqk` stay FP16 on purpose. AITER documents that FP16 has a better mantissa/range tradeoff than BF16 for the long recurrent walk; v0 keeps that boundary unchanged. The pivot is the chunk midpoint and `BC=16` constrains the intermediate Neumann powers before the lower 16-to-32 block merge.

## Avelang v0 structure

The implementation is in `test/examples/kda/avelang_impl/prefill/kda_prefill_v0.py`.

```text
avelang_kimi_delta_attn
  -> avelang_kda_prepare_v0 (conceptual K1)
       _kda_v0_prepare_vectors_kernel
       _kda_v0_build_l_mqk_kernel
       _kda_v0_init_diagonal_inverse_kernel
       _kda_v0_matrix_multiply_kernel / copy / add (K1 doubling)
       _kda_v0_make_merge_offdiag_kernel (K1 BC=16 -> 32 merge)
       _kda_v0_store_inverse_kernel
  -> avelang_kda_prefill_k2_from_workspace (conceptual K2)
       _kda_v0_segment_kernel
  -> (output [B,T,H,V], final_state [N,H,K,V])
```

K1 is intentionally decomposed into simple Avelang scalar kernels rather than written as an opaque fused kernel. Its temporary FP32 matrices expose the doubling/merge dependencies directly; only final `INV/Mqk` are narrowed to FP16. K2 owns one `(head, value-column)` FP32 state vector per program and matches AITER's rounding boundaries for BF16 `Kd/Qd/Kr`, FP16 `INV/Mqk` and the BF16 state/U operands of its dot products.

This organization is deliberately not a MI300X schedule. It has extra kernel launches and scalar work ownership solely to make each algebraic dependency testable.

## Correctness method

`test_kda_prefill_correctness.py` first monkeypatches AITER in-process only to count calls and verifies that every public reference invocation dispatches exactly once to `flash_kda_fwd`. The original function is restored before the candidate runs. A separate local debug wrapper calls only AITER K1 to expose golden workspace tensors; no file under `third_party/aiter` is modified.

The independent recurrence from `kda_baseline/bench_common.py` is the second oracle. It was written for the older V-first pool, so the test transposes only the *reference* state between `[N,H,K,V]` and `[N,H,V,K]`; the Avelang candidate does no state-layout conversion.

End-to-end tolerance is unchanged from Stage 5B: `atol=rtol=3e-2` for both output and final state. K1 standalone tolerances follow stored dtype: BF16 vectors `5e-3`, FP32 `gt` `1e-4`, FP16 `INV/Mqk` `3e-2`.

## Standalone checks

K1 workspace max-absolute error against a direct AITER K1 launch:

| T | kd | qd | kr | gt | INV/Mqk |
|---:|---:|---:|---:|---:|---:|
| 32 | 4.55e-13 | 2.27e-13 | 6.10e-05 | 4.72e-16 | 1.53e-05 |
| 127 | 1.19e-07 | 3.81e-06 | 4.88e-04 | 5.83e-16 | 3.05e-05 |

Using AITER's golden K1 workspace, Avelang K2 also passed at `T=32` and `T=127`, with both zero and random nonzero FP32 initial state. Its worst max-absolute error versus AITER was `7.63e-06` for output and `2.98e-08` for final state.

## Full prefill results

Each row reports the larger max-absolute error across zero and nonzero initial state. Every case is finite and passed the unchanged `3e-2` tolerance.

| T | output vs AITER | state vs AITER | output vs FP32 reference | state vs FP32 reference | result |
|---:|---:|---:|---:|---:|---|
| 32 | 6.10e-05 | 2.91e-05 | 1.22e-04 | 7.91e-04 | PASS |
| 64 | 6.10e-05 | 2.33e-06 | 1.22e-04 | 1.02e-03 | PASS |
| 128 | 6.10e-05 | 1.34e-05 | 1.22e-04 | 1.34e-03 | PASS |
| 256 | 6.10e-05 | 5.82e-05 | 1.22e-04 | 8.22e-04 | PASS |
| 512 | 1.22e-04 | 1.19e-04 | 1.22e-04 | 1.12e-03 | PASS |
| 1024 | 1.22e-04 | 2.24e-05 | 1.83e-04 | 1.28e-03 | PASS |
| 2048 | 1.22e-04 | 1.61e-05 | 2.44e-04 | 7.77e-04 | PASS |
| 4096 | 1.22e-04 | 2.84e-05 | 1.83e-04 | 8.36e-04 | PASS |
| 33 (tail) | 6.10e-05 | 9.88e-06 | 1.22e-04 | 7.33e-04 | PASS |
| 65 (tail) | 1.22e-04 | 1.81e-04 | 1.22e-04 | 6.94e-04 | PASS |
| 127 (tail) | 6.10e-05 | 1.62e-04 | 1.22e-04 | 1.03e-03 | PASS |

The machine-readable result is `test/examples/kda/reports/kda_prefill_v0_correctness_results.json`; the full stdout log is `test/examples/kda/reports/kda_prefill_v0_correctness.log`.

## Reused Qwen experience

No Qwen KDA/GDN math or schedule was reused. The only reused Avelang patterns are the correctness-oriented primitives from Qwen experiments: `al.make_tensor` with explicit contiguous layouts, `al.make_local` for persistent FP32 state, explicit BF16 conversion points, one-workgroup ownership, and host Python wrappers that validate shape/dtype before launching. The KDA equations, workspace contract, pivot, inverse schedule and state order all come from AITER FlashKDA.

## Deliberate limitations

- Decode is not implemented.
- Segmented affine scan for `T >= 8192` is not implemented.
- Only `B=1,H=12,K=V=128,C=32,BC=16`, FP32 beta and K-first state are accepted.
- `state_v_first=True`, grouped value heads and training/backward are rejected.
- K1 is intentionally decomposed; no autotuning, MFMA schedule, LDS staging, vectorization or performance benchmark was attempted.

## First ISA targets for a later performance-only stage

1. AITER `_flash_kda_prepare_kernel` versus the Avelang K1 vector/matrix stages (`_kda_v0_prepare_vectors_kernel`, `_kda_v0_build_l_mqk_kernel`) before they are fused.
2. AITER `_flash_kda_segment_kernel` versus `_kda_v0_segment_kernel`.

The first comparison must inspect MFMA selection, LDS `ds_read/ds_write`, buffer accesses, barriers/synchronization, wave ownership, permutations, memory offsets and register spills. It is a later stage; no ISA dumping or performance change was made here.

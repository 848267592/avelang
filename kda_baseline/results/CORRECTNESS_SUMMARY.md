# Unified KDA correctness — Stage 4 gate

Run date: 2026-09-10 UTC.  This is a correctness-only stage; no invocation
used `--mode bench`.

## Contract

- Input: BF16, `H=12`, `K=V=128`; recurrent state: FP32; `lower_bound=-5.0`.
- Prefill: raw `q/k/v`, raw per-K gate, raw beta, `A_log`, `dt_bias`, and
  initial state -> output plus final state.
- Decode: post-conv packed QKV, raw gate/beta, `A_log`, `dt_bias`, and state
  -> output plus updated state.  Causal convolution and RMSNorm are excluded.
- Reference: independent FP32 recurrence in `../bench_common.py`, not one
  framework used as the other's oracle.
- Threshold: `atol=rtol=3e-2` for both output and FP32 state.

## PASS matrix

| Phase | Backend / actual path | Shapes | Largest output abs error | Largest state abs error |
|---|---|---|---:|---:|
| Prefill | SGLang `fla.kda.chunk_kda` ordinary Triton | `T=1,16,64,127,128` | `1.2207e-4` | `1.3334e-3` |
| Prefill | vLLM AMD `chunk_kda_prefill(use_fused_chunk=False)` vendored Triton | `T=1,16,64,127,128` | `1.2207e-4` | `1.3334e-3` |
| Decode | SGLang direct `fused_recurrent_kda_packed_decode_kernel` ordinary Triton | `B=1,8,16,32,64,128` | `7.6294e-6` | `5.9605e-8` |
| Decode | vLLM AMD `fused_recurrent_kda_packed_decode` | `B=1,8,16,32,64,128` | `7.6294e-6` | `5.9605e-8` |

SGLang Decode launches the local source's Triton kernel directly so its
optional CUDA-JIT packed-decode branch is never selected on MI300X.  vLLM
Prefill explicitly keeps `use_fused_chunk=False`, so it never selects the
gfx950-oriented ROCm fused-chunk path.

## Artifacts

- `correctness_prefill_sglang.jsonl`
- `correctness_decode_sglang.jsonl`
- `correctness_prefill_vllm.jsonl`
- `correctness_decode_vllm.jsonl`
- Per-container command logs are retained alongside the JSONL results where
  available.

The local vLLM source import reports missing optional `vllm._C*` modules, but
the directly requested AMD Triton KDA operators executed successfully.  This
does not affect the Stage 4 result.

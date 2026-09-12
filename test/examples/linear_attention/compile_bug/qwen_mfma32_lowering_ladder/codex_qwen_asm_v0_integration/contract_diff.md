# Qwen BT64 ASM v0 Contract Diff

## Classification: CASE C

The frozen Triton kernel is a valid, exact raw FLA recurrence operator, but it
is not ABI-and-cast identical to Avelang's historical BT64 v29 recurrence or
to v24's BT16 production stage.  The decisive difference is visible in the
compiler-stage AMDGCN, not just in a Python wrapper:

```text
golden Triton pred: v_mfma_f32_32x32x4_xf32
Avelang v29 pred:  bf16(w) / bf16(state) + bf16 MFMA32 path
```

The direct project reference and historical v29 reference agree with the
latter BF16-pred definition.  The raw asm-v0 operator therefore preserves the
Triton semantics exactly; it is not substituted into v24/v29 under a false
claim of numerical equivalence.

## Sources Audited

| surface | source |
|:--|:--|
| current failed BT64 recurrence | `vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full.py` |
| current correct production recurrence boundary | `vllm_compare/qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py` and `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py` |
| v31 P16 implementation comparison | `vllm_compare/qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred.py` |
| authority raw recurrence | installed vLLM `chunk_delta_h.py`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| frozen compiler-stage source | `codex_triton_fullseq_asm_opt_audit/golden_fullseq/shared/original_from_triton.s` |
| Avelang forward caller | `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py` |

## Mathematical and Visible-Output Contract

| item | Avelang BT64 v29 / reference | frozen Triton / asm v0 | adapter result |
|:--|:--|:--|:--|
| pred | `bf16(w) @ bf16(old_state).T` | XF32 dot of FP32 `w` and FP32 resident state | no unsafe conversion |
| corrected value | `u - pred`, FP32 | `u - pred`, FP32 | `v_new` returned FP32 |
| decay input | precomputed `exp(g_last-g_t)` plus `exp(g_last)` | raw `g`, exponentiation in kernel | raw `g` is passed unchanged |
| update input | BF16 `v_decay` and BF16 `k` | decay then BF16 `b_v`, BF16 `k` | core schedule unchanged |
| h | FP32 chunk-start state | BF16 chunk-start state | raw output is BF16; optional adapter widens values to FP32 |
| final state | FP32 | FP32 | FP32 |
| initial state | optional FP32, zero when absent | optional in vLLM; asm v0 requires non-null FP32 | explicit guard/fallback |

`g`'s mathematical recurrence and the value-head-to-key-head mapping
(`key_head = value_head // 2`) agree.  The XF32 pred cast/order and the BF16
`h` materialization are real observable differences, not ABI details.

## Layout, Ownership, and Guards

| item | Avelang BT64 v29 | frozen Triton / asm v0 |
|:--|:--|:--|
| inputs | `k [1,T,4,128] bf16`, `w/u/g fp32` | same |
| raw outputs | `h [1,T/64,8,128,128] fp32`, no global `vn` | `h bf16`, `v_new fp32`, final state fp32 |
| full v24 outputs | `h/vn/final_state` fp32 at BT16 | not directly consumable: BT64 changes chunk count and h dtype |
| grid ownership | 32 blocks, 128 threads | `(V/BV=4, H=8)`, 256 threads |
| value tile | BV32 | BV32 |
| T tail | divisible by 64 | Triton supports a tail; asm v0 intentionally guards to `T % 64 == 0` |
| optional inputs | v29 has predecay inputs | raw vLLM supports `gk`, varlen, null initial/final; asm v0 deliberately supports only raw `g` + non-null initial/final state |

## Safe Integration Boundary

The implemented experimental API is:

```python
qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g, initial_state)
```

It is an opaque runtime dispatch using a dedicated new HSACO symbol.  It
checks gfx942, exact fixed layout/dtypes, `T % 64 == 0`, non-null initial
state, artifact SHA256, current stream, and the 88-byte kernarg ABI before
calling the module cache.  It does not lower the core operation into generic
AveLang memref/vector/MFMA operations.

`qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0(...)` exists only as an
explicit output-container adapter: it widens raw BF16 `h` to FP32 after the
external kernel.  It is not wired into v24 because no correct BT64 upstream
KKT/solve/w_u/chunk_o chain with the same numerical contract is available.

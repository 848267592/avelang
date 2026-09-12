# Stage 3 Source Audit

## Actual v24 Fast Path

`qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py` is not a file that
defines all of its fast stages.  It imports the W/U and output helpers from
`qwen_gdn_chunked_avelang_v17_mfma_layout_fixed.py` at lines 24-34 and calls
them at lines 174 and 186:

- W/U: `qwen_gdn_w_u_avelang_v14_mfma_layout`, defined at v17 line 1063.
- chunk-o: `qwen_gdn_chunk_o_avelang_v14_mfma_layout`, defined at v17 line
  1330.
- Both use a 64-thread workgroup, a 16x16x16 BF16 MFMA microkernel, packed
  `i32` shared views, and a FP32 output correction after BF16 staging.

The BT16 assumptions are real, not cosmetic: `BT = 16`, shared tiles are
16 by 16, the packed tensor views use `(16, 2, 4)`, and the launch maps one
program to a 16-token by 16-column tile.  Directly calling these wrappers
under `chunk_size=64` is rejected and would implement the wrong chunk-local
math even if the guard were removed.

## Stage 2 Fallbacks

The opt-in Stage 2 wrapper
`qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py` instead calls the generic
v6 wrappers:

- `qwen_gdn_w_u_avelang_v6_standalone` at line 136;
- `qwen_gdn_chunk_o_avelang_v6_standalone` through
  `qwen_gdn_chunk_o_bt64_avelang_experimental` at lines 90-113 and 140.

Those wrappers accept BT64 semantically but their kernel geometry is scalar:
one thread per program, no MFMA, no LDS work sharing.  That is why the Stage 2
T=2048 profile reported 3.612 ms W/U and 11.761 ms chunk-o.

The authoritative vLLM source-stage names recorded by the Stage 2 capture are
`ops/wy_fast.py:recompute_w_u_fwd` and `ops/chunk_o.py:chunk_fwd_o`; vLLM is
used as a golden/reference harness only.  The Stage 3 candidate has no vLLM
import and never calls the vLLM full wrapper.

## Stage 3 Port

`qwen_gdn_bt64_native_wu_chunko_mfma_v1.py` retains the validated v24/v14
64-lane mapping and `mfma_16x16x16_bf16_f32` primitive.  It represents a
BT64 chunk as four 16-token source subtiles:

- W/U use 4 row tiles x 8 output-column tiles, accumulating four source
  token-16 tiles into each 16x16 result tile.  Launch grid:
  `num_chunks * 8 heads * 4 row tiles * 8 columns`, workgroup 64.
- chunk-o uses 4 output-token tiles x 8 value tiles, visits all four source
  token tiles, and applies `source_abs <= output_abs` for the full 64-token
  causal relation.  Launch grid:
  `num_chunks * 8 heads * 8 value tiles * 4 output tiles`, workgroup 64.

The port preserves the BT64 asm ABI: W/U remain FP32; recurrence H is read as
BF16 directly; V-new and output accumulation are FP32; public output converts
to BF16 only at the public boundary.  No existing validated native-BT64 W/U or
chunk-o was found wired into the Stage 2 graph.

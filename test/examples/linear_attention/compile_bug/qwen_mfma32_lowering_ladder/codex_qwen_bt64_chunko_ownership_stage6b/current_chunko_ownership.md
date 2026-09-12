# Current Avelang BT64 Chunk-O Ownership

Source: `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`, kernel
`_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0`.

At `T=2048`, `NT=32` and there are `32 * 8 = 256` chunk/value-head pairs.
The current launch is `grid=NT * 8 * 8=2048`, `workgroup=256`; it assigns
eight CTAs to every chunk/head, one for every V16 range.

| item | ownership |
|---|---|
| CTA | one `[chunk=64 tokens, value-head, V=16]` output tile |
| four waves | wave `w` owns token rows `16*w : 16*w+16` |
| lane | `lane_col=lane&15` selects a V/K column; `lane_group=lane>>4` plus `r=0..3` selects four token rows |
| output | one CTA writes its unique `[token16,V16]` FP32 output; no partial global output or atomic |

Each V16 CTA stages Q, repeatedly stages the four K16 panels, builds QK score
tiles, stages H16 and V-new16, and writes its final FP32 result. The score
calculation is executed for all four source panels by every output wave,
including upper-triangular work that is subsequently zeroed by the causal
condition. The same Q/K score work is repeated across the eight V16 CTAs of a
chunk/head.

At the tensor-data level, each chunk/head reads the 64x128 BF16 Q and K tiles
eight times (128 KiB apiece before cache effects). H and V-new data are
V-specific, so their unique 128-wide ranges must still be covered once; they
are not the main repeated operand in this mapping.

Measured full-graph rocprof resources at T=2048: 2048 CTAs, 256 threads/CTA,
`MFMA=458752`, `VMEM=851968`, `LDS=1343488`, `VGPR=112`,
`Accum_VGPR=64`, `LDS block=27136 B`, `Scratch=0 B`.

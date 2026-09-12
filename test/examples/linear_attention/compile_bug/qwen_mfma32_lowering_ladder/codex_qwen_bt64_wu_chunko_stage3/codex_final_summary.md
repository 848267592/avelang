# Stage 3 Summary

The opt-in native BT64 W/U and chunk-o stages are correct under the frozen
full-output contract and replace the two Stage 2 scalar fallbacks only in
`qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1`.  At T=2048 the full graph
measures `1.0922 ms`, a `14.69x` improvement over Stage 2's `16.0413 ms`.

This is a successful recovery of GPU MFMA parallelism, not production
promotion.  The candidate remains slower than v24 BT16 (`0.5882 ms`) and the
two-session vLLM reference (`0.3613 ms`).  The next measured bottleneck is the
generic BT64 KKT stage (`0.3502 ms` at T=2048), so a native BT64 KKT is the
next isolated stage to gate.


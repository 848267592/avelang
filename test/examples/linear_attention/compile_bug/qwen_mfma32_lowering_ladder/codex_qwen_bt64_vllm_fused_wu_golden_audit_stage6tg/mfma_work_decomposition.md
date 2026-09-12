# MFMA Work Decomposition

At T=2048, `32 chunks * 8 value heads = 256` chunk-heads and each normal path has one CTA per chunk-head.

| implementation | W main | W residual | U main | U residual | MFMA/CTA | MFMA/dispatch |
|---|---|---|---|---|---|---|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

CTA reduction did not eliminate F1 math. F1 has a 16x normalized dynamic MFMA factor: 2x geometry, 2x residual pass, and 4x predicated lane-group execution. Native vLLM uses one-pass MFMA32 and no residual dot.

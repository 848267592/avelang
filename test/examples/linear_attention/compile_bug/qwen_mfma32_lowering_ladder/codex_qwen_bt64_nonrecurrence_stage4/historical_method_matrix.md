# Historical Method Matrix

| Stage | Successful evidence | Failed or limited evidence | Stage 4 choice |
|:--|:--|:--|:--|
| KKT | v24 BT16 native 16x16 MFMA cut scalar KKT from 0.1151 to 0.0293 ms; v20 proved token-matrix MFMA is viable | v6 BT64 is one-thread scalar work; v20's wider W/U path was not fully validated | Decompose BT64 into a 4x4 grid of proven 16x16 MFMA tiles |
| solve | v18 parallelizes the row recurrence and made BT64 usable (21.1 ms to 0.127 ms) | The recurrence remains sequential by row and is 2.36x slower than vLLM's hierarchical solve | Keep v18; audit and defer a dedicated hierarchical solve |
| W/U | v14/v24 validated 16x16 BF16 MFMA plus FP32 residual correction | Stage 3 mechanically repeated token16 ownership; v20 showed that merely changing MFMA geometry is not enough; v21 double buffering had no real async copy | One BT64 CTA owns four token16 waves; replace scalar correction with a second residual MFMA |
| chunk-o | v14/v24 and Stage 3 established correct MFMA16 inter/intra math | Stage 2 scalar fallback was catastrophic; a first Stage 4 combined-accumulator attempt exceeded numerical tolerance | One BT64 CTA owns four output-token tiles and reuses staged Q/K/H/V; retain separate inter/intra accumulators until final sum |
| recurrence | gfx942 asm-v0 is correct, scratch-free, and about 0.20 ms uninstrumented | v29/v30 compiler/source experiments exposed register cliffs without beating the frozen recurrence | Reuse asm-v0 byte-for-byte; no modification |

The reports audited directly were `qwen_gdn_v14_mfma_report.md`,
`qwen_gdn_v18_bt64_solve_report.md`, `qwen_gdn_v20_bt32_native_mfma_report.md`,
`qwen_gdn_v24_kkt_mfma_report.md`, `qwen_gfx942_bt64_full_pipeline_stage2_report.md`,
and `qwen_gfx942_bt64_native_wu_chunko_stage3_report.md`.

# Stage 2 Final Summary

The experimental BT64 full forward is implemented and correct under its
frozen public contract. It integrates the untouched gfx942 asm-v0 recurrence,
but it is not a performance candidate because its generic BT64 `chunk_o` and
W/U stages are scalar. The unique evidence-backed Stage 3 target is native
parallel BT64 `chunk_o`.

See `../qwen_gfx942_bt64_full_pipeline_stage2_report.md` for the complete
execution graph, numerical contract, correctness matrix, profiling data, and
next-step decision.

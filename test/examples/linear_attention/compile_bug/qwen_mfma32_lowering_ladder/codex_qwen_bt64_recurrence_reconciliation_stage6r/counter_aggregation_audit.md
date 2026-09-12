# Counter Aggregation Audit

Stage 6A requested three explicit graph replays. The raw trace contains extra recurrence symbols because the process includes Triton autotuning/capture activity: asm has 5 matching dispatches and vLLM has 1,315. The fresh Stage 6R trace reproduces this asymmetry (5 versus 1,211). It is not a counter multiplier.

For both implementations the final three matching dispatch IDs are the three explicit post-capture replays. Counter values are a median over those IDs, not a sum. Therefore Stage 6A's `196608` versus `65536` MFMA values are already per recurrence dispatch.

- asm-v0: `196608` MFMA/dispatch, `6144`/chunk, `768`/chunk-head.
- current vLLM: `65536` MFMA/dispatch, `2048`/chunk, `256`/chunk-head.

Conclusion: `counter_aggregation_error_found=false`; the 3x dynamic-MFMA difference is real.

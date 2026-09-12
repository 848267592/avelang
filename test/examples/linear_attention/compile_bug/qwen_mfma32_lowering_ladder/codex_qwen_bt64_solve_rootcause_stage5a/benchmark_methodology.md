# Benchmark Methodology

All standalone GPU results use the same gfx942 target, `B=1,H=8,BT=64`,
contiguous FP32 Stage 4 KKT output, current HIP stream, 20 warmups, 100 HIP
event repeats, and three independent sessions.  Input construction,
autotuning, JIT compilation, and output allocation are outside the body-only
interval.

For v18, the body measurement dispatches the existing JIT kernel with a
preallocated output.  For vLLM, the installed selected Triton kernel is
called at its wrapper-selected `num_warps=4,num_stages=5`; `out.zero_()` is
issued and synchronized **before** each event because the wrapper relies on
zeroed unwritten upper-triangular elements.  The timed interval therefore
contains the solve body but not allocation or preparation.

`public_wrapper_*` rows are retained separately.  They include `empty_like`
or `zeros_like` behavior and must not be mixed with the body-only fit.
Earlier raw data that accidentally included vLLM output zeroing in the direct
event are preserved as `*_v1_includes_vllm_zero.*` and are not used below.

The model is ordinary least squares over 1,2,8,16,32,64,128,256 chunks:

```text
latency_us = intercept_us + slope_us_per_chunk * chunks
```

The exact summary and raw samples are in `standalone_benchmark.csv`,
`chunk_count_sweep.csv`, `latency_fit.json`, and `raw_sessions/`.

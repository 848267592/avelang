# Baseline Methodology

All final full-path numbers are medians of three independent Python sessions,
each using HIP events with warmup 10 and repeat 50. Stage 4, Stage 3, and v24
use the same input generation and cached-allocation wrapper methodology. vLLM
runs in separate processes after Triton warmup. Raw session files are retained
under `full_pipeline/`.

The final T=2048 medians are Stage 3 `1.089079 ms`, Stage 4 `0.475867 ms`,
v24 `0.587173 ms`, and vLLM `0.365043 ms`. Unlike the older Stage 3 report,
this run obtained three vLLM sessions; no older two-session value was promoted
into the new table.

Stage timings are independently timed dispatch groups and are not expected to
sum exactly to cached full latency. rocprof trace values are counter-instrumented
device durations and are not substituted for normal HIP-event latency.

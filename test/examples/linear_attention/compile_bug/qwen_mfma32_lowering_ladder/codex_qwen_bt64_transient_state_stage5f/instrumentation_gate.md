# Stage 5F Instrumentation Gate

`rocprofv3 --kernel-trace` failed the predeclared T=2048 tail calibration. It changed the larger A/B latency by `28.402 us` (limit `16.508 us`) and changed the paired penalty by `24.457 us` (limit `6.277 us`). The mode is rejected; no heavier PMC, replay, timestamp, cache, PC-sampling, thread-trace, or clock-state run was performed.

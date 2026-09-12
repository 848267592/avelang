# Full-Graph Gain Analysis

Negative `stage6s_minus_current_us` in the raw CSV means Graph B is faster. This table reports the sign-normalized Graph-A minus Graph-B gain. The interval is a paired bootstrap interval across equal-position session medians; it is not a profiler-derived duration.

| T | chunks | Graph-A minus Graph-B gain (us) | 95% CI (us) | Graph-B minus vLLM (us) |
|--:|--:|--:|:--|--:|
| 512 | 8 | 1.370 | [1.122, 1.735] | 20.049 |
| 1024 | 16 | 10.319 | [10.295, 10.351] | 51.865 |
| 2048 | 32 | 26.647 | [26.595, 26.700] | 120.206 |
| 4096 | 64 | 58.960 | [58.787, 59.132] | 225.767 |
| 8192 | 128 | 126.288 | [126.171, 126.392] | 429.022 |
| 16384 | 256 | 278.926 | [278.277, 279.447] | 841.739 |

The fitted Graph-A minus vLLM gap slope is `4.396726 us/chunk`; Graph-B minus vLLM is `3.284716 us/chunk`. Stage 6S removes `1.112009 us/chunk` of the current full-graph slope. At T=2048 the direct recurrence-body gain is `40.099 us`, the three conversion graph costs together are `20.750 us`, and the diagnostic pre-full net is `19.349 us`. The observed full gain is `26.647 us`; it must not be described as a direct sum of body timings.

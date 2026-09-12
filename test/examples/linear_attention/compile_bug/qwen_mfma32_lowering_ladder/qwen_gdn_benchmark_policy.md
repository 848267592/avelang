# Qwen GDN Benchmark Policy

## Primary Performance Metric

From this point forward, the primary metric for any claim that an Avelang
Qwen GDN path is faster than, matches, or is closer to vLLM is **direct eager
public-API latency**. The measurement must invoke the public operator entry
for both implementations without CUDA/HIP Graph replay.

The eager contract includes all required logical dispatches, intermediate
global-memory traffic, casts, and host-to-device launch/queue gaps. It is the
only metric used for the production leaderboard and for comparisons with the
BT16 v24 baseline.

## Required Eager Contract

- Same `q/k/v/g/beta/initial_state` values, dtype, layout, device and current
  stream.
- If a public API exposes output/intermediate buffers, preallocate and reuse
  them before timing. Otherwise warm up the public wrapper and explicitly
  label the result `cached allocator`; do not claim allocation-free timing.
- JIT compilation, module load, autotune, allocation and capture excluded.
- Fixed warmup, repeat and session counts for both sides.
- Balanced ABBA ordering across Avelang and vLLM sessions.
- Report the complete `T` sweep, median, p10, p90, `Avelang / vLLM` ratio,
  and correctness versus the native vLLM result.
- Do not compare a BT16 or BT64 result to a vLLM number measured by a different
  timing boundary.

## Secondary CUDA/HIP Graph Metric

CUDA/HIP Graph replay remains allowed as a secondary diagnostic. It measures
the fixed-shape steady-state device schedule after capture and is useful for
separating GPU body work from eager dispatch overhead.

It must be labelled `graph replay` and must not be used as the production
leaderboard, a direct comparison with eager v24, or an unqualified claim of
end-to-end superiority over vLLM. A graph result is valid only when Avelang
and vLLM use the same replay protocol.

## Stage 6S Status

The existing Stage 6S A/B/C table is a graph-replay experiment. Its native
vLLM T=2048 value is `0.188800 ms`; the separately verified direct eager
native-vLLM value is `0.360936 ms`. Stage 6S has not yet been measured as an
eager full graph, so it is **not yet rankable against v24** on the primary
leaderboard.

The next eligible comparison is one eager harness containing v24, Stage 6S
and native vLLM under this exact contract, with correctness gates enabled.

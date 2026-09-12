# Eager Methodology

Every timed sample is one complete public API call. Allocations, casts, dispatches and wrapper glue are included. HIP events are authoritative; synchronized wall time is supplemental. No CUDA/HIP Graph is used.

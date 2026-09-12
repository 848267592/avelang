# Stage 6T-Golden Eager Public API Contract

All authoritative timing and correctness modes invoke only complete eager public APIs. Allocation, casts, dispatches, wrapper work, and returned output construction occur inside each timed call. Capture, IR/ISA, profiler counters, and intermediate W/U comparisons are diagnostic-only.

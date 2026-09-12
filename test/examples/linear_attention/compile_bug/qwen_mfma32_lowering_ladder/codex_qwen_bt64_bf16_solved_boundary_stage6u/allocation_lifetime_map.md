# Allocation and Lifetime Map

U0 materializes FP32 solved A, casts it to BF16, then allocates BF16 W/U. U1
allocates BF16 solved A directly and BF16 W/U directly; it has no FP32 solved
tensor, solved cast, FP32 W/U tensor, or W/U cast. Both retain the frozen BF16
recurrence, BF16-to-FP32 V-new cast, FP32 chunk-o staging and final BF16 cast.
Within C0, W accumulators are stored before U accumulators are created, so no
large W/U accumulator sets overlap.

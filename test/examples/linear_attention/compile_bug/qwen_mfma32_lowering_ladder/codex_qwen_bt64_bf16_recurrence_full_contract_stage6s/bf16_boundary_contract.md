# Stage 6S BF16 Boundary Contract

All three boundaries are numerical tensor casts. They allocate distinct, contiguous tensors during graph capture and preserve graph-owned output addresses during replay. None is a pointer reinterpretation or stride transform.

| boundary | producer tensor | bridge/consumer tensor | layout | measurement |
|:--|:--|:--|:--|:--|
| W | FP32 `[1,T,8,128]` | BF16 `[1,T,8,128]` | contiguous, same stride | `W FP32->BF16` |
| U | FP32 `[1,T,8,128]` | BF16 `[1,T,8,128]` | contiguous, same stride | `U FP32->BF16` |
| V-new | BF16 `[1,T,8,128]` | FP32 `[1,T,8,128]` | contiguous, same stride | `V-new BF16->FP32` |

The recurrence ABI is fixed: K/W/U are BF16; g and initial state are FP32; h and v-new are BF16; final state is FP32. Its launch remains grid `(4,8,1)`, workgroup `128`, and dynamic LDS `40960 B`.

The W/U conversion is BF16 rounding, so it can create expected finite rounding difference. Across the executed correctness matrix, widened-back W max abs was `0.0009517372`, widened-back U max abs was `0.0145950317`, and widened v-new max abs was exactly `0`. No executed case observed a threshold failure, NaN, Inf, saturation report, subnormal-specific failure, or layout change. The public output and final-state thresholds remain the authoritative contract.

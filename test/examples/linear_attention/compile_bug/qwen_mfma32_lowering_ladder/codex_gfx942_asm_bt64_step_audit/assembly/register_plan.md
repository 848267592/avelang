# Planned BT64/BV32 Assembly Register Plan

This is a plan, not a claim that a full Qwen assembly kernel exists yet.

| Class | Intended use | Lifetime discipline |
|:--|:--|:--|
| SGPR | kernarg pointers, descriptors, workgroup/head mapping, window constants | no per-fragment values |
| VGPR | lane/address arithmetic, loaded packed BF16 operands, FP32 epilogue values, state source | discard each K16 window after its MFMA consumes it |
| AGPR | one current P16 pred accumulator or one current update accumulator | pred AGPR must be serialized to LDS before update AGPR begins |
| LDS | compact two-K64 pred inputs, partial pred tile, V-decay `[64,32]`, one K16 window | no broad `[128,64]` transposed K tile |

The required P16 mapping is the proven v31 mapping:

```text
wave_id = tid / 64
lane = tid % 64
lane_col = lane & 15
lane_group = lane >> 4
K half = wave_id * 64
seg32 = 0, 1
vec_idx = lane_group + seg32 * 4
MFMA16 consumes packed BF16 fragments [0], then [1]
```

That map produced the verified direct `W[16,128] @ state[16,128].T` primitive.
The final full assembly implementation must preserve it rather than use the
permanently rejected padded MFMA32 construction.

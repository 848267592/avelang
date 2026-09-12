# Store Lowering Analysis

P0 ISA contains FP32 `v_mfma_f32_16x16x4_f32` compute followed by scalar
`global_store_short` and `global_store_short_d16_hi` BF16 stores. No scratch
instruction or spill is present. P1 was not implemented: the solve ownership
maps each lane to scattered diagonal/lower coordinates, and no local,
layout-preserving contiguous four-value store expression was demonstrated.
Forcing a packed write would mix ownership/layout changes into the boundary
experiment.

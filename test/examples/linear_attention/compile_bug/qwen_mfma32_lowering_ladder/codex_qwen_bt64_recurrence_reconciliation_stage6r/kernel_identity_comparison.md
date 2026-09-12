# Kernel Identity Comparison

- Current asm-v0 SHA256: `eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226`
- Current Stage 6A vLLM actual SHA256: `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`
- Historical Triton original/rebuilt SHA256: `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` / `98b503bc74ef233bf54bdf439d89c622fd9a573dd0b9c87e116550ccde751622`

**Classification: `different_specialization`.** The current vLLM artifact is not byte-identical to asm-v0 and cannot be called code-equivalent: it has BF16 W/U/v_new, `BV=32`, two waves, 40,960 B shared memory, and a different static MFMA schedule. Historical original/rebuilt/asm-v0 remain a separate FP32 ABI lineage.

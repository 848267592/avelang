# Source to IR/ISA MFMA Map

- Source: four W call groups at lines 133-140 and four U groups at 171-178,
  two MFMA calls per group.
- Pre-link LLVM: 64 C0 MFMA intrinsic call sites after compile-time column
  specialization (plus one declaration).
- ISA: 64 static `v_mfma_f32_16x16x16_bf16` instructions.
- PMC: 262144 MFMA at T=2048; divided by 256 CTAs gives 1024/CTA.
- F1 PMC: 524288, exactly 2x C0.
- Native vLLM PMC: 32768, or 128/CTA.

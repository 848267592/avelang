# vLLM Math Reconstruction

{
  "solved_dtype": "BF16",
  "W": "dot(A_bf16, BF16(K_bf16*beta_fp32*exp(g_fp32)))",
  "U": "dot(A_bf16, BF16(V_bf16*beta_fp32))",
  "mfma": "32x32x8 BF16 to FP32",
  "residual_correction": false,
  "source_phase_order": "two BV=64 U blocks/store, then two BK=64 W blocks/store"
}

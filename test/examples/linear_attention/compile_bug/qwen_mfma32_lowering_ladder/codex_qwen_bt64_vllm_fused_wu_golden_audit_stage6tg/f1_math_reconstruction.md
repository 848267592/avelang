# F1 Math Reconstruction

{
  "solved_dtype": "FP32",
  "W": "coeff=A_fp32[t,s]*beta_fp32[s]*exp(g_fp32[s]); BF16(coeff) MFMA16 plus BF16(coeff-FP32(BF16(coeff))) residual MFMA16 against K_bf16",
  "U": "coeff=A_fp32[t,s]*beta_fp32[s]; BF16(coeff) MFMA16 plus BF16(coeff-FP32(BF16(coeff))) residual MFMA16 against V_bf16",
  "mfma": "16x16x16 BF16 to FP32",
  "residual_correction": true
}

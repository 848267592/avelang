# C0 Consumer Contract

C0 is `_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u`. It accepts BF16
`A/K/V`, FP32 `g/beta`, and writes BF16 W/U. One CTA owns one `(chunk, value
head)`, with BT=64, WG=256 and 256 CTAs at T=2048. It rejects every unsupported
dtype, shape or chunk size and never falls back to F1.

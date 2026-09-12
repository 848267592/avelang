# ISA Difference

Current vLLM has `64` static `v_mfma_f32_32x32x8_bf16` and `0` XF32 MFMA instructions. asm-v0 has `48` BF16 plus `96` `v_mfma_f32_32x32x4_xf32`. The dynamic 3x MFMA ratio is consistent with the different pred/update schedule, not a replay sum.

Static ISA counts: asm/vLLM buffer-load `12/36`, buffer-store `22/24`, ds-read `268/150`, ds-write `215/183`, barriers `44/32`. See `instruction_diff.csv` for every counted mnemonic family.

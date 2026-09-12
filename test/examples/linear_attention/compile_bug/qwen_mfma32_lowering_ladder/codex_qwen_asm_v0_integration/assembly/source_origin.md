# Assembly v0 Source Origin

`qwen_gdn_bt64_gfx942_asm_v0.s` is a mechanically renamed copy of the
verified Triton compiler-stage AMDGCN source:

`../codex_triton_fullseq_asm_opt_audit/golden_fullseq/shared/original_from_triton.s`

The source is not derived from objdump output.  The only intentional source
change is the externally visible kernel symbol:

```text
chunk_gated_delta_rule_fwd_kernel_h_blockdim64
-> qwen_gdn_bt64_gfx942_asm_v0
```

That replacement includes ELF symbol declarations and AMDGPU metadata.  The
instruction body, register numbering, MFMA sequence, LDS offsets, waits, and
barriers are unchanged.  `source_patch.diff` is generated from these two
sources after the artifact build.

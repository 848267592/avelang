# Instruction and Store Decomposition

F0 and F1 have identical static MFMA16, LDS and barrier counts. The F1 output epilogue changes F0 `global_store_dword` into `global_store_short_d16_hi`. F1 ISA visibly executes `v_bfe_u32`, `v_add3_u32`, `v_or_b32`, `v_cmp_u_f32` and `v_cndmask_b32` to prepare each BF16 result before the short store. Native vLLM emits packed `buffer_store_dwordx2`. The F1-versus-F0 327680 dynamic VALU delta is therefore localized to BF16 conversion/pack/store preparation. PMC categories cannot allocate an exact count to each individual opcode, so that lower-level partition is N/A.

| implementation | MFMA_static | store_static | store_form | ds_read_static | ds_write_static | barrier_static |
|---|---|---|---|---|---|---|
| F0 static | 32 | 16 | global_store_dword | 48 | 20 | 10 |
| F1 static | 32 | 16 | global_store_short_d16_hi | 48 | 20 | 10 |
| native vLLM static T2048 | 32 | 16 | buffer_store_dwordx2 | 40 | 34 | 9 |
| F1 PMC T2048 | 524288 | 491520 | dynamic counter only | 1114112 | included | N/A dynamic |
| native vLLM PMC T2048 | 32768 | 59392 | dynamic counter only | 38912 | included | N/A dynamic |

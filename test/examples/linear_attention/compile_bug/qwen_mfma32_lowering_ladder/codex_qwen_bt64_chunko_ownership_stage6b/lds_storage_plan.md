# LDS Storage Plan

| region | bytes | lifetime |
|---|---:|---|
| `q_scaled_bf16[64,128]` | 16384 | entire CTA |
| `k_bf16[16,128]` | 4096 | one source16 iteration |
| `h_bf16[16,128]` | 4096 | one V16 subtile |
| `score_decay_bf16[4,4,16,16]` | 8192 | after QK build through all V16 subtiles |
| `vn_t_bf16[16,16]` | 512 | one source16 and V16 subtile |

O0/O1 use 33280 B reported LDS block. Stage 4 uses 27136 B. The score cache
is intentional reuse storage, but it raises LDS and AccVGPR pressure enough to
limit occupancy.

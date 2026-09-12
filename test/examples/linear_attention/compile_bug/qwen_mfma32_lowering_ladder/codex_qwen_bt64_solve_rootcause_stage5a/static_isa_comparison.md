# Static ISA and Code-Object Comparison

The actual HSACOs are preserved under `isa/`; they were disassembled with
`llvm-objdump`, not inferred from source.  Static instruction-text counts
are diagnostic only: loop bodies and compiler scheduling make them different
from rocprof dynamic counts.

| item | v18 HSACO | selected vLLM Triton HSACO |
|:--|--:|--:|
| ISA MFMA mnemonic | none | `v_mfma_f32_16x16x4_f32` present, 64 text matches |
| static `s_barrier` matches | 4 | 56 |
| static `s_waitcnt` matches | 111 | 91 |
| static `ds_read` / `ds_write` | 99 / 35 | 58 / 52 |
| static global/buffer loads | 32 global + 6 buffer | 18 global |
| static global/buffer stores | 32 global + 6 buffer | 10 buffer |
| static address `v_lshl_add_u64` | 128 | 26 |
| static conditional branches | 120 | 26 |
| captured HSACO file size | 13,416 B | 13,680 B |

Code-object metadata is likewise distinct from the profiler resource report:

| metadata field | v18 | vLLM |
|:--|--:|--:|
| VGPR count | 221 | 72 |
| AGPR count | 0 | 16 |
| SGPR count | 86 | 48 |
| private segment | 0 B | 0 B |
| VGPR spill count | 0 | 0 |
| LDS/group segment | 17,664 B | 0 B |

Triton cache metadata advertises 3,072 B shared memory while the rocprof
dispatch metadata reports `LDS_Block_Size=0`; those are different reporting
layers and are not collapsed into one number.  The unambiguous common fact is
zero private scratch and zero metadata spills for both final code objects.

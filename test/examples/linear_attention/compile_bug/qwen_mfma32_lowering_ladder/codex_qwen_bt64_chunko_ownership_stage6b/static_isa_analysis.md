# Static ISA and Code-Object Analysis

Captured HSACO, disassembly, pre-link bitcode, disassembled LLVM IR, and
replayable link argv are under `isa/{current,o0,o1}/` and `ir/{current,o0,o1}/`.
The capture uses the ordinary JIT plus its existing
`AVELANG_AMDGPU_LINK_DEBUG_DIR` hook; no compiler change was made.

| static ISA text count | current | O0 | O1 |
|---|---:|---:|---:|
| `v_mfma` | 56 | 56 | 80 |
| `ds_read` | 72 | 72 | 112 |
| `ds_write` | 92 | 92 | 104 |
| `buffer_load` | 6 | 6 | 6 |
| `buffer_store` | 6 | 6 | 6 |
| `s_barrier` | 13 | 19 | 29 |
| `s_waitcnt` | 144 | 155 | 196 |

Code-object metadata is spill-free in all three cases. Its physical register
numbers must not be conflated with rocprof's resource counters:

| metadata | current | O0 | O1 |
|---|---:|---:|---:|
| VGPR | 124 | 84 | 84 |
| AGPR | 12 | 8 | 8 |
| SGPR | 42 | 34 | 33 |
| LDS B | 27136 | 33280 | 33280 |
| private B / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

O0 has the same static MFMA count as current but is launched in one fourth as
many CTAs and skips upper score tiles dynamically. O1 contains more static
MFMA/DS/barrier code and its dynamic counts and latency regress. O0's static
barrier count grows because the score cache must remain coherent across the
four sequential V16 subtiles; that cost is visible in its higher LDS and
Accum_VGPR resource usage.

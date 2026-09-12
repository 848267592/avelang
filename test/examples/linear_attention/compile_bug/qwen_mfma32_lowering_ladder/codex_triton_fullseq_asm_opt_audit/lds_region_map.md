# LDS Region Map

The code object has zero static group segment in metadata, but the Triton
launcher passes `57,344 B` of dynamic LDS. The table uses compiler-stage
AMDGCN, not source variable names alone. Immediate DS offsets identify three
address bands; dynamic base-register accesses prevent a stronger per-byte
claim for the low band.

| region | byte range | source/assembly evidence | lifetime conclusion |
|:--|:--|:--|:--|
| low layout scratch | `0..32767` | dynamic DS bases; recurrence source lines 112--246 | loop-carried conversion/staging, no strict free interval proven |
| K repack | `32768..49151` | `ds_write_b32 ... offset:32768/40960`, then `ds_read_b64` around source 240/246 | produced and consumed by update path in every iteration |
| W/V fragments | `49152..57343` | writes/reads at 49152, 49408, 50176, 51200, 52608; source 154/177/235/240/247 | compiler already reuses this band between phase-local layouts; barriers surround each producer/consumer pair |

The repeated `s_barrier` and the recurrence loop make address adjacency
insufficient proof of non-overlap. No extra alias is safe to implement from
the available compiler-stage evidence.

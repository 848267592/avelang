# Code Object Comparison

F1 standalone HSACO reports private segment 0 B, VGPR spill 0 and SGPR spill 0. The captured normal native vLLM T2048 HSACO also reports private 0 B and zero spills. This is from code-object metadata, not inferred from scratch. Triton cache metadata reports 8192 B shared while code-object fixed group segment and profiler LDS-block report 0; the audit preserves the disagreement and does not invent an explanation.

| implementation | config | VGPR | AccVGPR | SGPR | LDS_fixed_B | private_B | VGPR_spill | SGPR_spill | source |
|---|---|---|---|---|---|---|---|---|---|
| F1 static HSACO | WG256 | 72 | 8 | 35 | 3072 | 0 | 0 | 0 | standalone readelf |
| native vLLM normal static HSACO | 4w/2s WG256 | 180 | 16 | 101 | 0 | 0 | 0 | 0 | captured normal readelf |
| F1 PMC | WG256 | 64 | 8 | 48 | 3072 | N/A PMC | N/A PMC | N/A PMC | stable first snapshot |
| native vLLM PMC | instrumented WG128 | 60 | 164 | 112 | 0 | N/A PMC | N/A PMC | N/A PMC | autotune-perturbed diagnostic |

# Counter Normalization

Static ISA count, dynamic SQ count, dispatch count, CTA count and profiler replay state are distinct. F1 uses the first complete matching PMC dispatch before later collector drift. Native vLLM is filtered to Grid_Size=32768, namely 256 CTAs; its profiler config differs from normal eager capture. Timestamps are diagnostic-only.

| implementation | CTA | WG | MFMA | MFMA/CTA | VALU | VMEM | LDS |
|---|---|---|---|---|---|---|---|
| F1 stable first PMC dispatch | 256 | 256 | 524288 | 2048.0 | 4846592 | 491520 | 1114112 |
| native vLLM PMC dispatch | 256 | 128 | 32768 | 128.0 | 1378816 | 59392 | 38912 |

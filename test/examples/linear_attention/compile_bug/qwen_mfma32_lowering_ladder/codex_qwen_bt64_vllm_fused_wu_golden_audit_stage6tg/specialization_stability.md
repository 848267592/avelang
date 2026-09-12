# Actual vLLM Specialization Stability

Each specialization was captured after a real eager `chunk_gated_delta_rule` call. T=512 and T=2048 select the same 4w/2s HSACO. T=8192 and T=16384 select the same 2w/3s HSACO. Thus it is not stable across the full T sweep. An independent T=2048 capture selected the same 4w/2s configuration and HSACO hash.

| T | config | grid | CTA | WG | Triton shared B | HSACO |
|---|---|---|---|---|---|---|
| 512 | 4w/2s | (8,8,1) | 64 | 256 | 8192 | d7cd8744d597dcc0 |
| 2048 | 4w/2s | (32,8,1) | 256 | 256 | 8192 | d7cd8744d597dcc0 |
| 8192 | 2w/3s | (128,8,1) | 1024 | 128 | 16384 | dee9be336292ad53 |
| 16384 | 2w/3s | (256,8,1) | 2048 | 128 | 16384 | dee9be336292ad53 |

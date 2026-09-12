# Eager Baseline Stability

All values are medians of five session medians under warmup=30 and repeat=200, with a HIP event around a complete public eager API call, balanced order, allocation included, and no graph. F1 regresses at T=2048 and wins against Stage6S at T=8192 and T=16384, reproducing Stage6T direction.

| T | Stage6S ms | F1 ms | vLLM ms | F1-S us | F1/vLLM |
|---|---|---|---|---|---|
| 512 | 0.272184 | 0.262711 | 0.367967 | -9.474 | 0.714x |
| 2048 | 0.381487 | 0.397290 | 0.415096 | 15.803 | 0.957x |
| 8192 | 1.088638 | 1.038984 | 0.810765 | -49.654 | 1.281x |
| 16384 | 2.078308 | 1.955686 | 1.332881 | -122.622 | 1.467x |

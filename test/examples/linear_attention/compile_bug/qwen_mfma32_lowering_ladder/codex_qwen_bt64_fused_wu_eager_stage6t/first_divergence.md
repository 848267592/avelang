# First Divergence

No accepted public API case crossed a frozen threshold. F0 and F1 public output/final state were bit-identical in every executed case. The diagnostic T=64 W/U checks found F0 FP32 versus current separate FP32 max_abs=0 and F1 BF16 versus `F0.to(BF16)` max_abs=0.

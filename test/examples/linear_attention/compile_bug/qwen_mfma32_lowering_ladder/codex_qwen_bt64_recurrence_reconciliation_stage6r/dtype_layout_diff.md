# Dtype and Layout Difference

Both paths receive contiguous `[1,T,4,128]` BF16 K, `[1,T,8]` FP32 g-cumsum, and `[1,8,128,128]` FP32 initial state. Current vLLM's Stage 6A `recompute_w_u_fwd` produces contiguous BF16 `[1,T,8,128]` W/U and its recurrence writes BF16 v_new. The Stage 6A asm body benchmark explicitly casts those W/U buffers to FP32 outside the event and asm-v0 writes FP32 v_new.

Therefore the original 40-us body comparison was a native-ABI comparison, not identical element-type execution. When both consume the same FP32 W/U values, current vLLM source and asm-v0 are output-bit-exact, but their code objects still differ.

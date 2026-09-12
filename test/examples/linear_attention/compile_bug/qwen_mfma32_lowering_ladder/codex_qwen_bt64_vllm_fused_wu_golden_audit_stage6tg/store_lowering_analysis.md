# Store Lowering Analysis

F1's source-level BF16 conversion lowers to scalar short stores and per-value preparation. Native vLLM's selected T2048 ISA uses packed `buffer_store_dwordx2`, not dwordx4. This is a measured secondary lowering difference, but it cannot explain F1's 16x MFMA excess by itself.

# BT64 Chunk-O Contract

The experimental consumer calls
`qwen_gdn_chunk_o_avelang_v6_standalone(..., chunk_size=64)`. It consumes
BF16 `q/k`, FP32 `g`, BT64 contiguous `v_new`, and a contiguous FP32 view of
the frozen recurrence's BF16 `h` tensor `[1,n_chunks,8,128,128]`.

This is deliberately distinct from the v24 BT16 chunk-o wrapper: it uses one
64-token causal tile and never partitions a BT64 state into four BT16 calls.
The FP32 view conversion is an explicit bridge for the generic v6 primitive,
not an assertion that its cast order equals v24. Full public output correctness
is checked against the vLLM wrapper under the frozen Stage 2 thresholds.

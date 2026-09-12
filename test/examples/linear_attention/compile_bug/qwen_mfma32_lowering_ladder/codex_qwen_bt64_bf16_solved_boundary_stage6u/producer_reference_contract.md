# Producer Reference Contract

P-REF calls the unchanged FP32 hierarchical solve and then performs a numeric
`torch.float32 -> torch.bfloat16` cast on the current stream. It preserves the
contiguous `[1,T,8,64]` layout and is a control, not the selected producer.

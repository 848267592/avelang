# Launch Difference

At T=2048 both paths use 32 CTAs with grid `(4,8,1)`. Current vLLM obtains this from `BV=32` and has workgroup 128 (two waves). asm-v0 has workgroup 256 (four waves). Current vLLM selected `num_warps=2`, `num_stages=2`; asm-v0 is the fixed historical WG256 specialization.

Rocprof reports static LDS block size zero for both code objects because the LDS amount is supplied by launch; the explicit launch arguments are 40,960 B vLLM and 57,344 B asm-v0.

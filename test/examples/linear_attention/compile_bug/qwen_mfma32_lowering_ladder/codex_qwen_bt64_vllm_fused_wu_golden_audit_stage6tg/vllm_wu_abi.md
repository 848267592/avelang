# Actual vLLM W/U ABI

{
  "bytes": 80,
  "pointer_offsets": {
    "k": 0,
    "v": 8,
    "beta": 16,
    "w": 24,
    "u": 32,
    "A": 40,
    "g": 48
  },
  "runtime_scalar": {
    "T": 56,
    "bytes": 4
  },
  "note": "The fixed-shape specialized TTIR removes source-level optional cu_seqlens and chunk_indices pointers."
}

| tensor | vLLM | F1 |
|---|---|---|
| k | BF16 [1,T,4,128], token-major | BF16 [1,T,4,128], token-major |
| v | BF16 [1,T,8,128], token-major | BF16 [1,T,8,128], token-major |
| beta | FP32 [1,T,8] | FP32 [1,T,8] |
| g | FP32 cumsum [1,T,8] | FP32 cumsum [1,T,8] |
| A/a_solved | BF16 [1,T,8,64] | FP32 [1,T,8,64] |
| W/U | BF16 [1,T,8,128] | BF16 [1,T,8,128] |

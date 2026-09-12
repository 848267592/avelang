# F1 W/U ABI

{
  "bytes": 56,
  "pointer_offsets": {
    "k": 0,
    "v": 8,
    "g": 16,
    "beta": 24,
    "a_solved": 32,
    "w": 40,
    "u": 48
  },
  "specialization": "num_tokens and num_chunks are Avelang constexpr values"
}

| tensor | vLLM | F1 |
|---|---|---|
| k | BF16 [1,T,4,128], token-major | BF16 [1,T,4,128], token-major |
| v | BF16 [1,T,8,128], token-major | BF16 [1,T,8,128], token-major |
| beta | FP32 [1,T,8] | FP32 [1,T,8] |
| g | FP32 cumsum [1,T,8] | FP32 cumsum [1,T,8] |
| A/a_solved | BF16 [1,T,8,64] | FP32 [1,T,8,64] |
| W/U | BF16 [1,T,8,128] | BF16 [1,T,8,128] |

# Shape and Layout Contract

All tensors are contiguous token-major `[B,T,H,D]` except the state tensors:

| name | shape | dtype | producer -> consumer |
|:--|:--|:--|:--|
| `g_cumsum` | `[1,T,8]` | FP32 | cumsum -> KKT/WU/asm/chunk_o |
| `A/A_solved` | `[1,T,8,64]` | FP32 candidate | KKT -> solve -> WU |
| `w` | `[1,T,8,128]` | FP32 | WU -> asm |
| `u` | `[1,T,8,128]` | FP32 | WU -> asm |
| `h_bf16` | `[1,T/64,8,128,128]` | BF16 | asm -> BT64 chunk_o |
| `v_new` | `[1,T,8,128]` | FP32 | asm -> BT64 chunk_o |
| `final_state` | `[1,8,128,128]` | FP32 | asm -> public return |
| `output` | `[1,T,8,128]` | BF16 | chunk_o -> public return |

The fixed head map is `value_head // 2` for the corresponding key head.


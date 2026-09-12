# Graph A vs Graph B

## Shared Prefix and Suffix

```text
cumsum -> KKT -> captured Stage-5B hierarchical FP32 solve -> W FP32 -> U FP32
```

Both graphs then finish with the unchanged Stage-4 FP32 chunk-o, FP32 output staging, and existing FP32-to-BF16 public-output cast.

## Only Delta

```text
Graph A: asm-v0 recurrence(k BF16, W FP32, U FP32, g FP32, h0 FP32)
         -> h BF16, v-new FP32

Graph B: W FP32 -> BF16
         U FP32 -> BF16
         current-vLLM recurrence(k BF16, W BF16, U BF16, g FP32, h0 FP32)
         -> h BF16, v-new BF16, final-state FP32
         v-new BF16 -> FP32
```

There is no pointer reinterpretation, dtype fallback, hidden change to chunk-o, output staging, or final cast. Graph B therefore has three added boundary dispatches and 11 dispatches in total, versus eight for Graph A.

## HSACO Identities

| component | SHA256 | symbol |
|:--|:--|:--|
| historical Graph-A recurrence | `eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226` | `qwen_gdn_bt64_gfx942_asm_v0` |
| Graph-B current-vLLM recurrence | `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e` | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| shared captured Stage-5B solve | `b8c2dee0b1b94269170ff333da8fa912a38ae6198abd71de80d0898645091536` | `_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1` |

Graph B checks the current-vLLM hash before loading its module. Any missing bridge, missing code object, hash mismatch, wrong dtype/layout, non-gfx942 environment, or `T % 64 != 0` raises; there is no fallback to asm-v0.

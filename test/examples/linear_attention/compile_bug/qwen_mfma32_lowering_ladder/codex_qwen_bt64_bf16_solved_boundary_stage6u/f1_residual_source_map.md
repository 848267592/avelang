# F1 Residual Source Map

| Phase | Main | Residual | Store |
|:--|:--|:--|:--|
| W | `qwen_gdn_bt64_fused_wu_eager_stage6t.py:339-362` | `:364-383` | `:385-390` |
| U | `qwen_gdn_bt64_fused_wu_eager_stage6t.py:398-421` | `:422-441` | `:442-447` |

The residual coefficient is explicitly computed as
`coeff - f32(bf16(coeff))`. C0 accepts BF16 solved A and removes both residual
regions; it does not silently route to F1.

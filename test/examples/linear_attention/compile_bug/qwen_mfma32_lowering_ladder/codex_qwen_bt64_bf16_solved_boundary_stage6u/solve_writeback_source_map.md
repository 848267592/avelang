# Solve Writeback Source Map

| Region | Source lines | Contract |
|:--|:--|:--|
| FP32 input/output tensor declarations | `qwen_gdn_solve_bt64_hierarchical_fp32_v1.py:28-43` | Current global boundary is FP32. |
| Whole-output clear | `:72-79` | Writes zero to diagonal, lower, and strict-upper positions before selective overwrite. |
| Strict-lower diagonal recurrence | `:81-106` | FP32 shared computation. |
| Diagonal identity | `:107-112` | Adds I after recurrence; must remain before quantized writeback. |
| Diagonal block writeback | `:114-123` | Final values can be converted only at this store. |
| X21/X32/X43 writeback | `:167-177` | Final strict-lower blocks. |
| X31 writeback | `:216-219` | Final strict-lower block. |
| X42 writeback | `:258-261` | Final strict-lower block. |
| X41 writeback | `:313-315` | Final strict-lower block. |
| Wrapper allocation/launch | `:333-344` | Current `empty_like(a)` is FP32; P0 must allocate BF16. |

P0 must alter only the output pointer/tensor dtype, these final stores, and
the wrapper allocation dtype. The FP32 `a`, `a_frag`, `x`, `work`, all
accumulators and all MFMA calls remain unchanged.

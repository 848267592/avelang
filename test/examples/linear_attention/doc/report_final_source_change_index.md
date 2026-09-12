# 关键源码修改索引

本索引不是 Git 提交历史。它根据每个阶段保存的源码、报告、测试和 ISA/JSON 产物记录“从什么结构改为什么结构”。未实现项明确标为 N/A。

| Stage | 文件 / 函数 | 优化前 | 优化后 | ownership / dtype / MFMA | 结果与处理 |
|---|---|---|---|---|---|
| 4 KKT | `../vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`，`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0` | BT64 scalar、WG1、无 MFMA | 16x16 K tile CTA | CTA `(chunk, head, row16, col16)`；BF16 input/FP32 acc；MFMA16 | 保留：T2048 KKT 约 0.046589 ms，scratch 0 |
| 4 W/U | 同文件，`_qwen_gdn_w/u_bf16_kernel_bt64_mfma_v2_s1` | token16 机械分块、scalar residual | BT64 四-wave ownership，residual 改为 MFMA | WG256；BF16 operand/FP32 accum；MFMA16 | 保留为历史 Stage4 组件，后由 Stage6T/U 取代 |
| 4 chunk-o | 同文件，`_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0` | scalar/分散小 tile | BT64 cooperative CTA | WG256，token16 x V16；BF16 MFMA/FP32 acc | 保留至 6W storage boundary 改造 |
| 4 failed | W/U no-correction | 试图删除 correction | 数值 gate 不通过 | 数学项被删 | 放弃：不以 tolerance 掩盖 |
| 4 failed | chunk-o accumulator merge | inter/intra 累加器合并 | 减 live state 的尝试 | 资源/性能不达标 | 放弃 |
| 5B | `../vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`，hierarchical solve | 63-row LDS recurrence | 四个 16x16 diagonal + 六 lower block DAG | 256 threads；FP32 `mfma_16x16x4_f32_f32`；8 KiB LDS | standalone 保留；新 intrinsic enablement 是前置小型功能补齐 |
| 5E | `../vllm_compare/qwen_gdn_bt64_solve_direct_out_stage5e_audit.py` | wrapper 分配/私有输出 | caller provided exact common output | 仅审计 wrapper | 不进入生产；用于排除 pointer 主因 |
| 6S | `../vllm_compare/qwen_gdn_bt64_bf16_recurrence_full_stage6s.py`，`qwen_gdn_bt64_stage6s_recurrence_bridge` | historical asm-v0 FP32 W/U/V-new ABI | guarded current-vLLM BF16 external HSACO ABI，三条显式 cast | external HSACO，WG128，BF16 W/U/V-new，FP32 state | opt-in bridge；不是 Avelang codegen，也非默认 selector |
| 6T F0/F1 | `../vllm_compare/qwen_gdn_bt64_fused_wu_eager_stage6t.py` | W、U 两个 dispatch | 单个 fused W/U，F0 FP32、F1 BF16 write | 256 CTA / WG256；MFMA16 | F0/F1 保留为证据；T2048 未稳定获益 |
| 6U P0 | `../vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`，`_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u` | FP32 compute + FP32 global `a_solved` + cast | **同样 FP32 compute**，最后 `al.convert(..., al.bf16)` store | 256 CTA；FP32 MFMA16x4；BF16 output | 与 FP32 solve 后 cast bit-exact；保留 |
| 6U C0 | `../vllm_compare/qwen_gdn_bt64_bf16_solved_boundary_stage6u.py`，`_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u` | F1 main + residual，FP32 solved | BF16 solved，main-only W/U，直接 BF16 W/U | 256 CTA/WG256；1024 MFMA/CTA | 保留，成为 U1 的核心 |
| 6V V0 | `../vllm_compare/qwen_gdn_bt64_predicate_collapse_stage6v.py` | 四个 `if lane_group` predicated MFMA region | nested conditional expression select 后一次 uniform MFMA | 动态 MFMA 1024 -> 256/CTA；VGPR 68 -> 100 | isolated gate 成功；V1 full 未晋级 |
| 6W W1 | `../vllm_compare/qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py`，`_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w` | BF16 V-new cast FP32，chunk-o FP32 V-new，FP32 out + cast BF16 | BF16 V-new direct load，FP32 accumulation，BF16 direct public store | WG256，MFMA16/LDS geometry unchanged；`global_load_ushort`/`global_store_short_d16_hi` | bit-exact vs U1；删除两 dispatch；当前 Avelang experimental baseline（paired shared environment） |
| next candidate | KKT -> solve `a` handoff | FP32 `a` global write then read | N/A，尚未实施 | 两端为 Avelang；T2048 roundtrip 8 MiB | 只登记可行性审计，不得写成已完成 |

## 真实代码片段定位

1. **P0 的边界收窄**：`qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py:28-42,114-124`。`a` 与 LDS `x/work` 都为 FP32，只有 `out` 的 pointer/dtype 及 writeback 是 BF16。
2. **hierarchical block DAG**：同文件 `127-323`。`X21/X32/X43 -> X31/X42 -> X41` 通过 FP32 `mfma_16x16x4_f32_f32` 计算。
3. **predicate collapse**：`qwen_gdn_bt64_predicate_collapse_stage6v.py:101-113,145-155`。条件表达式成为 `arith.select`；不要替换成没有 SSA result 的 statement-level `if`。
4. **6W direct BF16 boundary**：`qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py:75-96,156-160,202-206,256-281`。函数显式要求 BF16 `v_new/h/out`，内部 MFMA accumulator 仍为 FP32。
5. **6S external bridge**：`qwen_gdn_bt64_bf16_recurrence_full_stage6s.py:31-73,211-233`。包含 hash、symbol、grid/WG 与 BF16 ABI guard；这不是把 external HSACO 伪装成 AveLang lowering。

## 明确未做的修改

- 没有改 v24、生产 selector、v18 基线或 immutable recurrence HSACO；
- 没有在 LLVM/AMDGPU RA 层硬限制 AGPR；
- 没有把 v29 的 MFMA32、persistent-kfrag、lifetime-marker 负实验接回 BT64 生产风格图；
- P1 packed solve writeback、U2 和 KKT-to-solve handoff 都是 N/A 或未来候选。

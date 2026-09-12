# 独立 Qwen / KDA 备份分支：文件 allowlist（草案）

> 这是新的备份分支规划，不依赖此前已经推送到 `master` 或
> `qwen-gdn-optimization_ljd` 的内容。当前只整理，不创建分支、不 `git add`、不
> commit、不 push。

## 1. 分支目标

建议新分支名：

```text
backup/qwen-kda-repro-2026-09
```

该分支要能直接 clone 后看到本次迁移所需的 Qwen 代码，而不是让未来的自己再去
找旧分支。目标覆盖三条线：

1. Stage 6S BF16 recurrence full graph；
2. Stage 6X/X2 KKT-to-solve full graph，以及 X2+Z5B chunk-o；
3. T=8192 Q@H 指标/机器链对齐实验：
   - `repro_qwen_gdn_t8192_native_shaped_qh_wg128.py`
   - `repro_qwen_gdn_t8192_qh_parity_wg128.py`

官方 SGLang/vLLM/AITER 源码不复制进这个分支；另建环境清单记录 URL、commit、
Docker digest 和重建命令。

## 2. 分支内应直接保存的 Qwen Python 源码

### 2.1 Stage 6S / X2 / X2+Z5B full graph

以下文件组成 `vllm_compare/` 的最小本地模块闭包。不要只依赖旧分支中“已经
tracked”的文件；新备份分支按下面清单直接加入：

```text
test/examples/linear_attention/vllm_compare/
├── qwen_gdn_bt64_gfx942_asm_v0_experimental.py
├── qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py
├── qwen_gdn_bt64_bf16_recurrence_full_stage6s.py
├── qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py
├── qwen_gdn_bt64_bf16_solved_boundary_stage6u.py
├── qwen_gdn_bt64_nonrecurrence_mfma_v2.py
├── qwen_gdn_bt64_native_wu_chunko_mfma_v1.py
├── qwen_gdn_chunked_avelang_v6_vllm_layout_fixed.py
├── qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py
├── qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py
├── qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py
├── qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py
├── qwen_gdn_solve_bt64_hierarchical_fp32_v1.py
├── qwen_gdn_bt64_kkt_solve_handoff_stage6x.py
├── qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py
├── qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py
├── qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py
├── bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_eager.py
└── test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py
```

说明：`qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py` 虽然是旧目录中的
未跟踪文件，但 `codex_qwen_bt64_full_pipeline_stage2/bench_stage2.py` 会直接导入
它，因此必须纳入新分支，不能按“旧实验版本”漏掉。

### 2.2 两个 T=8192 Q@H 指标对齐 probe

这两个文件明确纳入新分支，不能再按“单点探针”排除：

```text
test/examples/linear_attention/vllm_compare/
├── repro_qwen_gdn_t8192_native_shaped_qh_wg128.py
└── repro_qwen_gdn_t8192_qh_parity_wg128.py
```

它们不是完整 KDA，但属于你的指标对齐尝试，是后续判断 Q@H logical parity、
WG128/two-wave contract 和 LDS-to-MFMA 差异的实验入口。

probe 的辅助脚本和最终小报告也纳入：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
├── replay_qwen_v29_lto_mir.py
└── qwen_t8192_native_shaped_qh_parity.md
```

## 3. Full graph 的 benchmark / contract 骨架

这部分不是自动生成结果，而是未来重新运行 Stage 6S/X2 所需的脚本和约束：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_full_pipeline_stage2/
    ├── stage2_runner.py
    ├── bench_stage2.py
    ├── profile_candidate_stage.py
    ├── summarize_stage2.py
    ├── commands.sh
    ├── benchmark_methodology.md
    ├── bt64_numerical_contract.md
    ├── bt64_numerical_contract.json
    ├── full_operator_contract.json
    ├── full_operator_execution_graph.md
    ├── shape_layout_contract.md
    ├── stage_source_map.md
    ├── codex_final_summary.md
    ├── final_decision.json
    └── avelang_bt64_stages/README.md
```

Stage 6R 的 capture 脚本还会导入 Stage 6A 的真实 producer-chain audit，因此一并
保存其源码入口和冻结 measurement contract：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_full_graph_gap_stage6a/
    ├── stage6a_full_graph_audit.py
    └── frozen_measurement_contract.json
```

`full_correctness_results.*`、`stage_correctness_results.*`、benchmark CSV、
`golden_capture/` 和 rocprof 不作为代码依赖；最终数值结论已经在第 6 节报告中
保存。

## 4. Stage 6S 的外部 bridge / 重建脚本

Stage 6S 不是单纯的 Python 文件。它需要 current-vLLM recurrence 的 gfx942
HSACO、Stage 5B solve HSACO，以及两个 HIP bridge。新分支保存“源码 + 捕获/构建
脚本 + contract”，不保存机器生成的 `.hsaco`、`.so`、`.o`：

### 4.1 Stage 6R current-vLLM recurrence bridge

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_recurrence_reconciliation_stage6r/
    ├── stage6r_external_bridge.cpp
    ├── build_bridge.sh
    ├── stage6r_capture_current_recurrence.py
    ├── stage6r_recurrence_body_benchmark.py
    ├── test_stage6r_current_bridge.py
    ├── commands.sh
    ├── frozen_recurrence_contract.md
    ├── frozen_recurrence_contract.json
    └── current_kernels/vllm/
        ├── abi.json
        ├── launch.json
        └── sha256.txt
```

新服务器上用固定的 vLLM commit/ROCm/Triton 环境重新 capture current recurrence
HSACO，再用 `sha256.txt` 验证；不把旧机器的 `kernel.hsaco` 当作源码提交。

### 4.2 Stage 6S hierarchical-solve bridge

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_bf16_recurrence_full_contract_stage6s/
    ├── stage6s_external_solve_bridge.cpp
    ├── build_stage6s_external_solve_bridge.sh
    ├── stage6s_full_contract_audit.py
    ├── stage6s_correctness_matrix.py
    ├── stage6s_eager_public_leaderboard.py
    ├── stage6s_trace_analysis.md
    ├── commands.sh
    ├── frozen_full_contract.md
    ├── frozen_measurement_contract.json
    ├── bf16_boundary_contract.md
    ├── bf16_boundary_contract.json
    ├── codex_final_summary.md
    └── final_decision.json
```

### 4.3 Stage 5B solve 的 source contract

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_bt64_hierarchical_solve_stage5b/
    ├── commands.sh
    ├── commands_s0_v1.sh
    ├── implementation_contract.md
    ├── implementation_contract.json
    ├── s0_v1_final_decision.json
    └── source/solve_s0/README.md
```

实际 kernel source 是 `vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`；
新服务器上按 contract 重新编译并生成固定路径的 solve HSACO。

### 4.4 asm-v0 adapter（Stage 6S Graph A / X2 兼容依赖）

`qwen_gdn_bt64_bf16_recurrence_full_stage6s.py` 会导入 asm-v0 adapter，且 Graph A
会使用它。为避免新分支导入即缺文件，保存其可重建 source closure：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
└── codex_qwen_asm_v0_integration/
    ├── assembly/qwen_gdn_bt64_gfx942_asm_v0.s
    ├── assembly/build.sh
    ├── assembly/source_origin.md
    ├── avelang_integration/qwen_gdn_bt64_gfx942_asm_v0.py
    ├── avelang_integration/qwen_gdn_bt64_gfx942_asm_v0_bridge.cpp
    ├── avelang_integration/build.sh
    ├── avelang_integration/kernarg_abi.json
    └── tests/test_qwen_gdn_bt64_gfx942_asm_v0.py
```

## 5. 两个 Q@H probe 的小型结果证据

不把整个 compiler dump 放进备份分支，只保存能直接说明这次指标对齐结论的文件：

```text
test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128/
├── correctness.json
├── static_isa_counts.json
├── artifact_hashes.sha256
└── code_object_notes.txt
```

不保存该目录的 `compiler_ir/`、`exact_lto/`、`.hsaco` 和大型 ISA/MIR 工件；需要
时由 probe 在新机器重新生成。`qwen_t8192_native_shaped_qh_parity.md` 是这次
“logical Q@H 对齐但 LDS consumer/MFMA operand 物理链不 exact”的结论主文档。

## 6. 新分支应包含的最终报告

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
├── qwen_gdn_benchmark_policy.md
├── qwen_gfx942_bt64_full_pipeline_stage2_report.md
├── qwen_gfx942_bt64_bf16_recurrence_full_contract_stage6s_report.md
├── qwen_gfx942_bt64_kkt_solve_handoff_stage6x_report.md
├── qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md
└── qwen_t8192_native_shaped_qh_parity.md
```

这些报告提供 full graph、Stage 6S、Stage 6X、X2+Z5B 和 Q@H 对齐的背景、contract
及最终结论，不需要再把 Stage 1--7 所有历史 report 复制进去。

## 7. 明确排除

新分支不纳入：

- `rocprof_outputs/`、`.rocprofv3/`、raw sessions、`golden_capture/`；
- 所有 `.hsaco`、`.o`、`.so`、`.bin`、`.llir`、`.ttir`、`.ttgir`、`.mir`、`.mlir`；
- `compiler_ir/`、`exact_lto/`、`machine/`、`isa/` 大型 dump；
- `vllm_compare/` 中未列入第 2 节的历史 `v*.py`、`profile_*`、`dump_*`、一次性
  地址/packet/register probe；
- `compile_bug/` 中 Stage 1--7 的完整实验目录；
- Kimi/Qwen 模型权重。

这些内容如需审计，另做服务器上的压缩归档；不进入用于快速 clone 的源码分支。

## 8. 重建顺序（未来在新服务器执行）

```text
1. clone avelang 到新服务器
2. 按环境清单 clone 固定 SHA 的 vLLM/SGLang/AITER
3. 构建 Stage 5B solve source，生成 solve HSACO
4. 运行 Stage 6R capture，生成并校验 current-vLLM recurrence HSACO
5. hipcc 构建 Stage 6R / Stage 6S / asm-v0 bridge
6. 运行 Stage 6S correctness，再运行 Stage 6S eager benchmark
7. 运行 Stage 6X / X2 / X2+Z5B full graph correctness 和 benchmark
8. 独立运行两个 T=8192 Q@H probe，重新生成小型证据 JSON
```

## 9. 当前状态

- 新备份分支：已创建为 `backup/qwen-kda-repro-2026-09`。
- 文件 allowlist：已完成并已包含 Stage 6S/X2 full graph 和两个 Q@H probe。
- 暂存状态：64 个 allowlist 新增文件，加上 4 个迁移元数据文件；allowlist 中另外
  23 个文件原本已存在于当前 HEAD，因此不产生新的 staged diff。
- 旧分支是否已有这些文件：不影响本分支计划。
- 本轮：**未 commit、未 push、未删除、未移动、未修改任何 Qwen 实现。**

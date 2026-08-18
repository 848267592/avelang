# Avelang Qwen GDN gfx942 X2+Z5B 提交清单

## 版本身份

**冻结日期：2026-08-18**

性能提交候选名称：`X2+Z5B`。

它是当前最快的、已经完成完整 Eager public-API correctness 和性能测试的 Qwen
GDN experimental 图，适用 contract：

```text
gfx942
B=1, Hk=4, Hv=8, K=V=128
q/k/v: BF16; g/beta/initial_state/final_state: FP32
BT=64; T>=64 and T % 64 == 0
```

它**不是** production default，也不等于 standalone Z5B chunk-o。Z5B 只有接在 X2
前四阶段之后才是本清单中的完整候选。

## 一句话选择

| 场景 | 版本 |
|:--|:--|
| 单一性能提交，主测 T>=1024 | `X2+Z5B` |
| 只测 T=512 | `X2` |
| 可做实测范围内的长度 dispatch | T=512: `X2`; T>=1024: `X2+Z5B` |
| 保守 production/default | `v24 BT16` |

没有经过验证的全长度 selector，也没有将任何 selector 写进 production。

## 最小上传范围

上一版清单同时承担“性能证据归档”和“fork 上传说明”两项职责，范围偏大。若目标
只是让个人 fork 能编译并运行 X2+Z5B，应只提交：

1. 下文列出的 15 个本地 Python module import closure；
2. 下文列出的两个 compiler runtime source 文件；
3. external recurrence 的 launcher source、build script 和 immutable HSACO。

`libstage6r_external_bridge.so` 是由 bridge source 在目标 ROCm 环境构建得到的产物，
不应作为必须提交的二进制。性能 JSON/CSV、报告、主学习复盘、pytest、benchmark、
MLIR generator test 和 language-reference 文档也不是运行时必需项；它们应作为单独的
验证或文档 commit，而不是塞进最小性能代码 commit。

当前 module-level import 仍较宽。因此即使其中一些 legacy symbol 没进入所选的五个
dispatch，也必须上传，除非另做一次正确性验证过的 packaging/import-refactor。不要为了
减少文件数临时删 import 后把未经验证的版本称作 2026-08-18 的 X2+Z5B。

## 完整图与入口

发布入口：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py`

公开函数：

```python
qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(
    q, k, v, g, beta,
    initial_state=..., scale=..., output_final_state=...
)
```

唯一和 X2 不同的调用是尾部：

```text
X2:      Stage6W BF16 chunk-o
X2+Z5B:  Stage6Z Z5B direct-Q-cache BF16 chunk-o
```

完整 dispatch 图：

```text
1. qwen_gdn_chunk_cumsum_avelang_v6_standalone
2. qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2
3. qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u
4. qwen_gdn_bt64_stage6s_recurrence_bridge
5. qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache
```

第 4 项是当前 vLLM BF16 recurrence 的 hash-guarded external HSACO bridge；不是
Avelang source recurrence。不要将该事实从 README、benchmark 或提交说明中删掉。

## 2026-08-18 完整 Eager 性能

同一批 fresh-process Eager public API、current HIP stream、无 Graph。每 T：5 session，
每 session 10 Williams block，每 block 运行六种三方顺序；每实现共 300 次完整调用。
以下为 HIP-event median；比值是 Avelang / vLLM，低于 1 表示 Avelang 更快。

| T | X2 ms | X2/vLLM | X2+Z5B ms | X2+Z5B/vLLM | vLLM ms |
|--:|--:|--:|--:|--:|--:|
| 512 | `0.193427` | `0.554x` | `0.204803` | `0.587x` | `0.348857` |
| 1024 | `0.233246` | `0.663x` | `0.227578` | `0.647x` | `0.351882` |
| 2048 | `0.317913` | `0.801x` | `0.291232` | `0.733x` | `0.397110` |
| 4096 | `0.473723` | `0.915x` | `0.433223` | `0.837x` | `0.517688` |
| 8192 | `0.847997` | `1.116x` | `0.753357` | `0.991x` | `0.759887` |
| 16384 | `1.583108` | `1.288x` | `1.392826` | `1.133x` | `1.229464` |

相对原 X2，X2+Z5B 的 paired gain 95% event CI：

| T | X2 - X2+Z5B gain us |
|--:|:--|
| 512 | `-8.90`, `[-11.68, -5.81]` |
| 1024 | `+5.53`, `[3.02, 8.66]` |
| 2048 | `+26.39`, `[23.16, 30.20]` |
| 4096 | `+40.46`, `[38.00, 43.13]` |
| 8192 | `+94.75`, `[92.38, 97.48]` |
| 16384 | `+192.25`, `[189.04, 195.66]` |

拟合 slope：X2 `5.620 us/chunk`，X2+Z5B `4.839 us/chunk`，vLLM
`3.630 us/chunk`。

v24 不在上表的同一轮重跑中。其最新 Eager primary leaderboard 的 v24/vLLM 比值为
T=`512/1024/2048/4096/8192/16384`：
`0.921x/1.174x/1.495x/2.012x/2.811x/3.439x`。该历史数据足以证明 v24 不应作为
性能提交候选，但不要把它与上表当作同一 session 的严格统计比较。

## 正确性与证据

- 新 full test：
  `test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py`
- 结果：`47 passed in 29.12s`，MI300X / ljd ROCm 7.2.2 容器。
- 覆盖：T=`64/512/2048/8192/16384`，random/high-dynamic/cancellation/neutral-gate，
  有/无 initial state，finite，zero-`V_new`，NaN output prefill 组件 gate。
- X2 final state：FP32 bit-exact。
- X2+Z5B 对 X2 output 最大差异：`0.0009765625`，小于 output gate `1/128`。
- 对 native vLLM 4 个完整 contract case：output `<=1/128`，final state `<=0.02`。

性能 harness：

`test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_eager.py`

完整报告与原始 CSV/JSON（原始目录保留在本地实验工作区，不作为必要发布文件）：

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_x2_z5b_chunko_full_eager/`

## 必须提交的运行时源码

以下是发布入口的本地 `qwen_*.py` import closure。前六项实际参与五阶段图；其余
项是目前模块级 import closure，保留它们能保证不需要临时删 import 才能运行。

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py  # release entry
  qwen_gdn_bt64_kkt_solve_handoff_stage6x.py                  # X2 fused KKT+solve
  qwen_gdn_bt64_bf16_solved_boundary_stage6u.py               # fused BF16 W/U
  qwen_gdn_bt64_bf16_recurrence_full_stage6s.py               # external recurrence ABI bridge
  qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py # Z5B
  qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py               # shared ABI validation/constants
  qwen_gdn_chunked_avelang_v6_vllm_layout_fixed.py             # cumsum and legacy symbols
  qwen_gdn_bt64_nonrecurrence_mfma_v2.py                       # fixed shape/constants exports
  qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py             # Stage 6U module dependency
  qwen_gdn_bt64_gfx942_asm_v0_experimental.py                  # Stage 6S module dependency
  qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py             # target validation dependency
  qwen_gdn_bt64_native_wu_chunko_mfma_v1.py                    # nonrecurrence module dependency
  qwen_gdn_chunked_avelang_v13_mfma_layout_fixed.py            # nonrecurrence module dependency
  qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py            # nonrecurrence module dependency
  qwen_gdn_solve_bt64_hierarchical_fp32_v1.py                  # nonrecurrence module dependency
```

以下不是 runtime 必需项，但建议作为第二个 verification/docs commit：

```text
test/examples/linear_attention/vllm_compare/
  test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py
  bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_eager.py
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md
```

原始 CSV/JSON 目录仅作为本地或 release asset 保存，不进入 fork 代码提交。

## 必须提交的 external recurrence artifact

X2+Z5B 的 recurrence 不可只提交 Python。以下目录中的 HSACO、ABI 元数据与
launcher source/build script 也是运行时依赖：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_recurrence_reconciliation_stage6r/
    stage6r_external_bridge.cpp
    build_bridge.sh
    current_kernels/vllm/kernel.hsaco
    current_kernels/vllm/abi.json
    current_kernels/vllm/launch.json
    current_kernels/vllm/sha256.txt
```

`abi.json`、`launch.json`、`sha256.txt` 实际位于
`current_kernels/vllm/`，不是 bridge 目录根部。`libstage6r_external_bridge.so`
不提交：它由 `build_bridge.sh` 在目标机器的 ROCm/HIP 环境现场生成。

HSACO SHA256 必须为：

```text
632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e
```

Python bridge 会在运行时检查该 hash、symbol
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`、grid `(4,8,1)`、WG `128` 和
dynamic LDS `40960 B`。任何 mismatch 必须 fail closed，不能 fallback。

## 必须提交的 Avelang compiler/intrinsic patch

X2 的 fused hierarchical solve 依赖 `al.amdgpu.mfma_16x16x4_f32_f32`。这是
X2+Z5B 唯一新增且必须存在的 compiler 功能；Z5B 本身只使用先前已有的
BF16 MFMA32 和 generic shared/view lowering。

| file | 需要保留的修改 |
|:--|:--|
| `lib/IR/Intrinsics/amdgpu_mfma_signatures.h` | 注册 `M=16,N=16,K=4,A/B=f32,C=f32`；每 lane A/B 一个 FP32、accumulator 四个 FP32。 |
| `lib/IR/Intrinsics/amdgpu_intrinsics.mlir` | 增加 `rocdl.mfma.f32.16x16x4f32` wrapper；从 Avelang `vector<1xf32>` A/B 提取 element 0，再调用 LLVM 所需标量 `f32` operand。 |
| `lib/IR/mlir_generator_test.cc` | 回归测试；不影响 runtime，但应随 compiler patch 提交。 |
| `docs/content/language-reference/hardware-intrinsics.md` | API 文档；不影响 runtime。 |

这四个文件当前 diff 是 40 insertions、2 deletions。它们不改 AMDGPU RA、不改
allocator、不改手写汇编，也不包含 Z5B 的 Q-cache schedule。验证应看到：

```text
v_mfma_f32_16x16x4_f32
```

在 HSACO 中出现，而不是 scalar FMA fallback。

## 明确不要混入此提交的工作树修改

以下不属于 X2+Z5B runtime/性能证据，不能因为同一 worktree 中存在就一起提交为
“实现该版本所需”：

- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.*`；
- `lower_qwen_gdn_recurrence_step_pass.*`、`lower_qwen_k64_pipeline_stage_pass.*`、
  `lower_qwen_kfrag_lds_pass.*`；
- `qwen_persistent_recurrence_pass.*`、`qwen_recurrence_schedule_plan.h`、
  `qwen_modulo_software_pipeline_pass.*`；
- `static_physical_layout.*`、`bounded_packet_schedule_pass.*`；
- R1--R5、C0--C26、block-dot V2、Z6/Z7 等后续 research sources、reports 与
  generated HSACO；
- `amdgpu_backend.cc` 的 LTO debug capture 改动：它对审计有用，但不是运行 X2+Z5B
  的语义依赖。

将这些另开 branch 或 commit；否则个人 fork 的提交将不能准确表达“X2+Z5B 的最小
可复现实验”。

## 建议的提交分组

1. `compiler: expose gfx942 fp32 mfma16x16x4 intrinsic`
   - runtime 最小集：`amdgpu_mfma_signatures.h`、`amdgpu_intrinsics.mlir`；
   - 同一 commit 推荐带上 generator test 和 API 文档。
2. `qwen: add X2+Z5B runtime`
   - 15 个 Python module closure、bridge C++/build script、immutable recurrence HSACO。
3. `qwen: verify X2+Z5B full Eager performance`
   - full test、benchmark、单份集成报告；raw CSV/JSON 保留在本地或另存 release asset。

不要把 generated Python `__pycache__`、temporary LTO dump、Docker 私有目录或其他
用户的 worktree 改动提交到 fork。

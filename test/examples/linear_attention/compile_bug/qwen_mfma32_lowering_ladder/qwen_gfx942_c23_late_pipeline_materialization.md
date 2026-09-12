# Qwen gfx942 C23 Late Pipeline Materialization

## 结论

**`STOP_C23_LATE_PIPELINE_NOT_CONTROLLABLE`**。C23 没有生成 LLVM、MIR、
ISA 或 HSACO，因此没有进行正确性、PMC 和性能测试。这个停止不是资源阈值，
而是任务预注册的硬门槛：必须先在最终 ISA 中看到
`next load -> current MFMA -> delayed wait`。

## C21 首次退化点

C21 高层源在 [qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline.py:116)
已把 Q owner、Q@H 和两个 Q@K 放进同一 K32 loop；但它没有 first-class
pending packet。第一次失去 pipeline 的位置是
[lower_qwen_block_dot_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc:2890) 的
`emitC19FullPhysicalProducer`：每个 H/K producer 立刻 lower 成 global
packet load、LDS store、barrier，之后才形成 MFMA consumer。于是 packet SSA
的第一个 consumer 就是 LDS store，AMDGPU 必须在 store 前等待 VMEM。

最终 C21 ISA 给出直接证据：line 363 global load，371 `vmcnt(0)`，372 LDS
store，374 barrier，378 第一个 MFMA；随后 line 394 K load，395 wait，396
LDS store，399 barrier，422 K MFMA。故根因属于 **AveLang lowering 的
producer SSA / dependency graph（Case I + III）**，不是 LLVM 或 AMDGPU
scheduler 把一个已经存在的合法 window 排坏。

## C23 最小实现与停止

本轮只在 C21 的 H/K lowering 边界实现 `C23LatePipelineMaterializer`：H
load 后发起一个 predicated BF16x8 K0 packet，保持 packet SSA，计划在 H
MFMA 后再提交 K0 到已有 LDS。没有修改 C21 source superloop、layout、
ownership、barrier 计划、MFMA 几何、数学、ABI、RA 或 production。

该实现暴露了现有 per-block-dot greedy rewrite 的控制边界。带有跨 op 的
predicated vector packet 时，C23 T64 compile-only 在 `get_llvm_ir` 退出
`139`；原始日志在
[c23_t64_compile.stdout.log](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_gfx942_c23_late_pipeline_materialization/machine/c23_t64_compile.stdout.log)。因此这个 lowering 阶段不能安全地
拥有一个跨 H/K logical op 的 pending packet，且无法输出可验证的 C23 ISA。
把 C21 或 native ISA 当作 C23 结果会掩盖这个事实，所以所有后续测试均明确
标记为 `not_run`。

## Native 对照

fresh T2048 selected native 是 WG256、4 waves、2 stages；T8192 是 WG128、2 waves、
2 stages，T16384 与 T8192 选择相同哈希。T2048 final ISA 的 line 219
先 issue `buffer_load_dwordx4`，line 222 和 230 仍执行当前 MFMA，直到 line 236
才首次等待并在 237--242 publish 到 LDS。T8192 对应关系是 line 255--256
issue、259/268/269/271 current MFMA、278 首次 VMEM wait、279--290 LDS publication。
这正是 C23 需要 materialize 而未能 materialize 的依赖窗口。

## 已生成证据

- `stage6z_c23_pipeline_first_divergence.json`
- `stage6z_c23_native_identity_t2048.json`
- `stage6z_c23_native_identity_t8192.json`
- `stage6z_c23_native_identity_t16384.json`
- `stage6z_c23_native_overlap_timeline.json`
- `stage6z_c23_liveness_delta.json`
- `stage6z_c23_overlap_evidence.json`
- `stage6z_c23_correctness.json`
- `stage6z_c23_formal_body.json`
- `stage6z_c23_longtext_slope.json`
- `stage6z_c23_machine_resources.json`
- `stage6z_c23_pmc.json`
- `stage6z_c23_causal_delta.json`
- `stage6z_c23_regression_results.json`

本轮到此停止。下一步若重开，不应再增加 C24 chunk-o source 变体；需要先在
compiler 中设计一个 region-level 的 pending packet / pipeline token 表示，令
packet ownership 脱离 per-op greedy rewrite，再谈 machine scheduling。

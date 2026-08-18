# Qwen persistent recurrence：R4-tail / current-vLLM 每 chunk 机器工作差额

结论：R4-tail 的主差额不是 MFMA、也不是 tail issue。本轮严格的动态计数显示两者每个 physical V32 tile 都执行 `64` 条 BF16 MFMA；R4-tail 的最大绝对超额是 VALU，其次是 VMEM、SALU，LDS 只高约 25%。我按该结论只实现了一次 full-op、specialized-block-dot 内的 typed BF16x4 fragment lowering 尝试；其 MLIR/MIR 变化被后端完全折叠，最终 HSACO 和规范化 ISA 均与 R4-tail 一致，因此按硬门槛 No-Go，未推广也未以其作性能结论。

本轮冻结的契约为 `gfx942_bt64_bv32_joint_v4_tail_issue`、BT64、WG128、BV32 physical tile、单一 first-class recurrence loop、BF16 `v_new/v_decay` 边界、FP32 carried state、typed BF16x8 producer、单一 53,248 B LDS bank 和 tail issue/commit。没有扫描 BV、issue 位置、distance、software pipeline 或第二 LDS。

## 口径与原始证据

- GPU：MI300X / gfx942；B=1、Hk=4、Hv=8、K=V=128、BF16、head-first=false、nonzero-W。
- R4 与 vLLM 都是 grid `(4, 8, 1)`、WG128；每个 recurrence chunk 有 `Hv * (V / 32) = 32` 个 physical V32 tile。故 T=2048 为 32 chunks / 1,024 tiles，T=8192 为 128 chunks / 4,096 tiles。
- PMC 通过无 graph 的 eager Python recurrence 调用收集，且只用于动态指令/资源，不使用 profiler 的时间字段。四份原始 CSV 在 `test/examples/linear_attention/vllm_compare/r4_tail_vs_current_vllm_gap_artifacts/pmc/`。
- current-vLLM 的 exact TTIR/TTGIR/LLVM/ISA/HSACO 是已有 capture：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/`。该 exact capture 没有可替代的 final-isel MIR；报告不以重新编译的不同 Triton 版本伪造 MIR。
- R4 的 exact MLIR/LLVM/MIR/ISA/HSACO 位于 `test/examples/linear_attention/vllm_compare/tail_issue_artifacts_t2048/`。

## 严格归一化 gap budget

下表的每 dispatch 是一个完整 persistent recurrence dispatch；所有动态计数来自该 dispatch 的一个稳定 R4 或 WG128 current-vLLM kernel dispatch。`/MFMA` 是该行 `/chunk` 再除以 2,048 MFMA/chunk；`/V32` 是 `/chunk` 再除以 32。它避免把 grid、loop trip count 或 T=8192 的四倍 chunks 误读为单 tile 工作差。

| T | metric | R4-tail / dispatch | current-vLLM / dispatch | R4 / chunk | vLLM / chunk | R4 / V32 | vLLM / V32 | R4:vLLM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | MFMA | 65,536 | 65,536 | 2,048 | 2,048 | 64 | 64 | 1.000 |
| 2048 | VALU | 2,239,872 | 1,395,584 | 69,996 | 43,612 | 2,187.375 | 1,362.875 | 1.605 |
| 2048 | SALU | 163,072 | 64,832 | 5,096 | 2,026 | 159.250 | 63.313 | 2.515 |
| 2048 | VMEM | 202,240 | 58,368 | 6,320 | 1,824 | 197.500 | 57.000 | 3.465 |
| 2048 | LDS | 381,952 | 305,472 | 11,936 | 9,546 | 373.000 | 298.313 | 1.250 |
| 8192 | MFMA | 262,144 | 262,144 | 2,048 | 2,048 | 64 | 64 | 1.000 |
| 8192 | VALU | 8,906,112 | 5,462,912 | 69,579 | 42,679 | 2,174.344 | 1,333.719 | 1.630 |
| 8192 | SALU | 630,016 | 230,720 | 4,922 | 1,802.5 | 153.813 | 56.328 | 2.731 |
| 8192 | VMEM | 798,208 | 230,400 | 6,236 | 1,800 | 194.875 | 56.250 | 3.464 |
| 8192 | LDS | 1,524,736 | 1,214,784 | 11,912 | 9,490.5 | 372.250 | 296.578 | 1.255 |

T=2048 的每 MFMA 非-MFMA 工作是 R4/vLLM：VALU `34.178/21.295`、SALU `2.488/0.989`、VMEM `3.086/0.891`、LDS `5.828/4.661`。T=8192 的数值稳定为 `33.974/20.840`、`2.403/0.880`、`3.045/0.879`、`5.816/4.634`。这证明长序列差额是 steady state 而非一次性 prologue。

资源也没有指向 occupancy cliff：R4 是 VGPR/AccVGPR/SGPR `128/160/112`、scratch=0、LDS=53,248 B；vLLM 是 `104/160/96`、scratch=0，capture metadata 的 LDS=40,960 B。rocprof 的 vLLM `LDS_Block_Size=0` 与其 HSACO metadata 冲突，故 LDS 容量以 metadata 为准。两者的 sampled occupancy 约为 R4 `0.63–0.65`、vLLM `0.60–0.64`，不能支持“R4 因 occupancy cliff 变慢”的说法。

静态 ISA 只作结构佐证，不能替代上表的动态归一化：R4/vLLM 的 `ds_read=182/150`、`ds_write=120/183`、`s_waitcnt=136/147`、`s_barrier=12/32`。R4 有 64 条 `v_accvgpr_read_b32` 和 64 条 `v_accvgpr_write_b32`；其 exact final-isel MIR 中有 2,460 个 `COPY` 和 356 个 `REG_SEQUENCE`。vLLM exact capture 不包含同源 final-isel MIR，故该两项不作不可靠的数值比。R4 静态 MFMA 为 48、vLLM 为 64，但循环/控制流展开不同；动态 `64/V32` 才是有效比较，且 pred/update 都没有超过 5%。

## 差额回溯到完整 recurrence 阶段

| phase | R4-tail 机器/IR 位置 | current-vLLM 位置 | 差额含义与 candidate fix |
| --- | --- | --- | --- |
| W/state pred operand preparation | R4 source `repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py:212-228`；post-block-dot MLIR 的 BF16x8 load、`vector.extract`/`from_elements`；LLVM `postopt_llvm.ll:1708-1749` MFMA cluster | TTGIR `kernel.ttgir:375-393` 的 W buffer load、local load、two pred dots | 两者总 pred MFMA 相同；R4 的通用 fragment/index lowering 进入 VALU/SALU 预算。 |
| pred K0/K1 reduction | 同上，`mfma_32x32x8` 的 two token tiles、four K packs | TTGIR `381,393`（循环携带 dot） | 不是重复 MFMA 或 wave ownership 问题。 |
| corrected / BF16 boundary | R4 source `:230-250`；`corrected = U - pred`、BF16 round-trip、`vdecay_stage` | TTGIR `406-447` | 这是必要数学边界；尚无它单独占据动态超额的证据。 |
| K/update operand preparation | R4 C++ `lower_qwen_block_dot_pass.cc:334-395`：token-major LDS 的八个 `memref.load` gather、`from_elements`，再进 retiled MFMA；MLIR 带 `token_major_bf16x8` / `lds_mediated_retile` 属性 | TTGIR `458-472`：K global/local load 和 update dot；tail `482-487` 的 in-thread transpose/local store | 这是 VMEM/VALU/SALU 最大来源的最强可定位候选，但不能只机械替换 `ds_read`；需要 producer ownership、LDS physical layout、dot operand encoding 和 address fragment 一起表示。 |
| update MFMA K0/K1 / FP32 feedback | R4 source `:252-271` 与 `lower_qwen_block_dot_pass.cc:359-395`；LLVM update MFMA clusters `:2348-2773` | TTGIR `458-474` | update MFMA 数同样没有超额；R4 的 AGPR↔VGPR 读写和 generic fragment/address IR 是后续应看的非-MFMA路径。 |
| tail issue / same-bank commit | R4 source `:273-291`；planner `qwen_persistent_recurrence_pass.cc:570-667` | TTGIR `478-488` | R4 的 barrier 静态数反而少于 vLLM；没有证据把 2x 余差归因给 tail wait/barrier。 |

这给出完整的地址/层级链：R4 ISA 中 MFMA clusters 和其间的 address/AGPR 指令 → `tail_issue_t2048.isa.s`、`exact_lto_final_isel.mir` → `mlir/post_block_dot_lowering.mlir`、`mlir/postopt_llvm.ll` → 上表的 kernel source 和 `lower_qwen_block_dot_pass.cc`。Triton 链是 exact ISA `disassembly.txt` → `kernel.llir` → `kernel.ttgir` 上述行号 → current-vLLM kernel source。Triton 无 exact MIR 是明确的采集边界，而非用不匹配工具链补写的结论。

具体 ISA 范围也吻合该映射：R4 的 pred MFMA clusters 位于 `0x3030–0x3150` 和 `0x3e54–0x3eec`，update clusters 位于 `0x4d14–0x5078` 和 `0x5300–0x56a0`；它们之间正是地址、LDS gather、AGPR↔VGPR 和 BF16 pack 指令。Triton 对应 pred ranges 为 `0x30a0–0x3374`、`0x3ca0–0x406c`，update 为 `0x4a70–0x4d78`、`0x5834–0x59b4`。这些 range 只用于 phase attribution；上表的 PMC 才是动态数量依据。

## 唯一实现候选与 No-Go

候选是 `typed BF16x4 dot fragment`：在 specialized block-dot 内，把每个 MFMA operand 的四个 `vector.extract + vector.from_elements` 替换成 `vector.extract_strided_slice`。它同时覆盖 pred 与 update，保持 R4-tail 的所有数学、producer、LDS、barrier、MFMA、state feedback 和 tail-commit 不变；日志确认仍为 `mode=specialized operand=persistent_typed_block`。

它确实改变了 post-block-dot MLIR：baseline/candidate 的 `vector.extract` 是 `104/40`，`vector.from_elements` 是 `33/9`，candidate 新增 32 个 `vector.extract_strided_slice`。candidate 的 T=64 P2 microscope 输出（h、pred FP32/BF16、v_new、v_decay、per-chunk/final state）字节相等。证据在 `test/examples/linear_attention/vllm_compare/r4_tail_typed_fragment_artifacts_t2048/`。

但最终 gate 失败：baseline 与 candidate HSACO SHA-256 同为 `7d7eb721de8109aee0264393854ef814f732ce12a7ead1b8f6c8a5c6453a47ea`。以相同 `llvm-objdump --no-show-raw-insn` 输出、仅去掉文件名 banner 后，二者 ISA SHA-256 同为 `c19050518e51fbcd6e993e955825896ba488f5704baef76af1508d45709ed255`。所以这个高层 vector 表示没有删除任何最终 ISA 指令；候选已从源码撤回，没有跑其余 P2/PMC/性能，也没有宣称性能改善。

## 无 graph 的 fresh-process Eager 复核

新脚本 `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_r4_tail_vs_current_vllm_eager.py` 使用每个 implementation/length/session 新解释器、两 session、5 warmup、20 HIP-event samples，且没有 graph capture 或 private HSACO/ctypes launch。结果 JSON 在 `r4_tail_vs_current_vllm_gap_artifacts/benchmark/`：

| T | R4-tail ms | current-vLLM Eager ms | R4:vLLM |
| --- | ---: | ---: | ---: |
| 1024 | 0.2204 | 0.1051 | 2.10 |
| 2048 | 0.3785 | 0.1538 | 2.46 |
| 8192 | 1.4085 | 0.4548 | 3.10 |

1024→8192 slope 为 R4 `10.608 us/chunk`、vLLM `3.122 us/chunk`，比值 3.40。该 API 口径保留 current-vLLM 返回张量的公开 allocator 路径，因此它是 public-Eager 复核，不应替代上述更严格的 kernel dynamic-work 归一化；它没有被用作 candidate 的性能结论。

## 最终回答与下一主矛盾

1. 最大的每 chunk 机器工作差额是 VALU（T=8192 多 26,900 条/chunk），位置集中在 typed producer 到 specialized dot operand 的 generic address/fragment/gather 表达，而非 pred/update MFMA。
2. 差额跨 pred operand preparation 和更明显的 K/update operand preparation；BF16 boundary、MFMA 数和 tail issue 目前没有因果证据是主矛盾。
3. 本轮没有可保留的 ISA 删除：唯一候选只删除了 MLIR 的 generic extract/pack 表达，最终删除 ISA 指令数为 0，因此撤回。
4. 性能改善为 0%，未达到 10%，原因是最终机器码严格等价，不是测试噪声。
5. 这不是剩余 2–3x 差距的主要矛盾的结论来自两项独立证据：MFMA 已严格相同，单纯 fragment 表示又被后端折叠。下一轮应在 first-class persistent recurrence + specialized block-dot 内建立真正的 typed dot-fragment/address lowering：它必须共同描述 BF16x8 producer lane ownership、token-major K 的 LDS physical offset/bank/swizzle、consumer dot operand encoding 与 gather/address formation，并以最终 ISA 实际减少 VALU/VMEM/SALU 为先决条件。不得先改 `ds_read`、消费端单点替换、issue 位置、distance、BV 或 LDS 容量。

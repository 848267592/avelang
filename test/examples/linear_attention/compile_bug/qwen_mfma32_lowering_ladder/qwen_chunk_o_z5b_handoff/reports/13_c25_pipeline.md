# Qwen gfx942 C25: Current-Ready / Next-Pending Pipeline Proof

## 结论

**`STOP_C25_CURRENT_READY_OVERLAP_NO_PERF`。** C25 成功把 C24 缺失的依赖顺序保留到最终 ISA：当前 H 在 next K issue 前已经 READY，next K 在四条当前 H MFMA 期间保持 PENDING，随后才 wait/commit。所有正确性 gate 通过，但 T=2048 到 T=16384 均慢于 Z5B，且长文本 slope 从 Z5B 的 0.915028 上升到 C25 的 0.973350 us/chunk。因此这条假设已完整验证且止损；不启动 C26。

C25 是有效的 compiler dependency/schedule proof，不是性能 baseline。Z5B 继续是 isolated chunk-o performance baseline。

## Native 机制

fresh native selected artifact 在 T=2048 为 BK32/BV64/WG256/2 stages，在 T=8192、4096、16384 为 BK32/BV64/WG128/2 stages。TTGIR 的 local_alloc 与 loop iter_args 证明 current operand 由旋转 stage 准备；ISA 又在 current MFMA 前后交错 next packet 的 load/wait/commit。因此分类为 **HYBRID**：stage rotation 为主，iteration 内也存在 overlap；不是仅根据 `num_stages=2` 推断。

T=2048 机械窗口：next load 约 219/224/234，current MFMA 约 222/230/244，必要 partial wait 约 236。T=8192：next load 约 255/256，current MFMA 约 259/268/269/271，`vmcnt(4)` 约 278。详见 `stage6z_c25_native_current_ready_timeline.json`。

## C24 到 C25

C24 已有 IssuePacket/CommitPacket SSA，却先 issue K_next、后 issue H_current。H publication 的 `vmcnt(0)` 同时等待 H 与更早发出的 K，故 K 无法跨 H MFMA 保持 outstanding。

C25 保留 C21 mapping、C24 packet representation、BF16 ABI、MFMA/K32 数学和所有 V/scoreV/g 路径。它只把 H 作为 current packet 在 K issue 前 publish：

```text
H_LOAD(346) -> H_READY_WAIT vmcnt(0)(352) -> H_LDS_COMMIT(353)
-> K_NEXT_LOAD(358) -> H_MFMA x4 (384..399)
-> K_FINAL_WAIT vmcnt(0)(402) -> K_COMMIT(403)
```

这满足 `NEXT_LOAD < CURRENT_MFMA_BEGIN < CURRENT_MFMA_END < NEXT_WAIT < NEXT_COMMIT`。real H_READY_WAIT 是 `vmcnt(0)`，但它发生在 K issue 之前，所以不会 drain K；这不是 real partial-wait，而是 Ready/Pending stage rotation。

## Partial-Wait 可行性

synthetic H-old/K-young proof 由 AMDGPU wait analysis 自然生成 `s_waitcnt vmcnt(1)`：`H load -> K load -> vmcnt(1) -> H LDS/MFMA -> vmcnt(0) -> K LDS`。未硬编码 waitcnt。故 gfx942/LLVM 可以表达 older-current 等待、younger-next 保持 outstanding。

但 real C25 选择更稳的 current-ready 方式：先令 H ready，再 issue K。它实现 region-level H-to-K0 two-stage schedule，而非完整 native TTGIR cross-iteration memdesc rotation；报告不把两者混为一谈。

## Liveness 与资源

C25 exact LTO code object：VGPR=188、AGPR=80、SGPR=36、LDS=24576 B、private=0、VGPR/SGPR spill=0。H packet 的代表性 `v[38:41]` 从 346 到 353；K packet 的 `v[34:37]` 从 358 跨过 4 条 H MFMA，到 403 commit。没有证据表明 C25 新增了完整 QH/QK accumulator group；这里只报告 machine def/use 级别范围，并不伪造 cycle-accurate LiveIntervals。

T=2048 dynamic PMC（每 CTA）：Z5B=MFMA/VMEM/LDS/VALU/SALU `160/672/672/7072/768`；C21=`160/304/688/6422/626`；C25=`160/304/688/6470/658`；native=`320/224/904/5912/1296`。native 的每 CTA MFMA 是两倍，说明 native CTA ownership 不同，不能据此宣称 same-CTA 工作相同。T=8192 native 选择 WG128，此时 per-CTA 同样只作描述。

## 正确性

T=64/2048/8192 均以 fresh process 对 Z5B 做 random、zero-V-new、caller-owned NaN prefill、structured Q/H/K、token/value pattern。所有 H/output 都 BF16 byte-exact，finite，无 caller-owned output 泄漏；prologue、steady、epilogue 均覆盖。

## 正式 Body Timing

口径：caller-owned preallocated output、current HIP stream、no Graph、warmup=10、repeat=50、7 个 fresh Python process sessions、balanced rotating order、HIP event。它是 isolated body diagnostic，不是 Eager public API 排名。

| T | chunks | Z5B ms | C21 source control ms | C25 ms | native selected ms | C25-Z5B us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 32 | 0.067260 | 0.071026 | 0.072107 | 0.042303 | +4.788 |
| 4096 | 64 | 0.103874 | 0.110444 | 0.108120 | 0.055643 | +4.947 |
| 8192 | 128 | 0.157775 | 0.167188 | 0.166006 | 0.090554 | +9.394 |
| 16384 | 256 | 0.275089 | 0.296300 | 0.292114 | 0.146617 | +16.745 |

T=8192 C25 比 Z5B 慢 5.22%（paired 95% CI `[7.191, 10.395] us`）；T=16384 慢 6.19%（CI `[16.426, 17.285] us`）。四个长度均未出现正收益。

## 长文本 Slope

| arm | intercept ms | slope us/chunk |
| --- | ---: | ---: |
| Z5B | 0.041196 | 0.915028 |
| C21 source control | 0.042221 | 0.991817 |
| C25 | 0.042785 | 0.973350 |
| native selected | 0.027424 | 0.469623 |

`slope_gap_closed = -0.130942`。负值表示 C25 使 Z5B-to-native slope gap 扩大约 13.1%，而非关闭。当前 C25 的 local overlap 没有消除 Z5B 更大的 global/materialization/address work，因此不能把机器窗口 overlap 当作端到端 latency hiding 的充分条件。

## 直接回答

1. Native current operand 由 previous rotating stage 准备，并在 next issue 前 ready。
2. Native 为 HYBRID：stage rotation 主导，包含 iteration 内 load/MFMA overlap。
3. C24 先 K 后 H，H 的 `vmcnt(0)` 同时 drain K。
4. 可以：synthetic 自然生成 `vmcnt(1)`。
5. synthetic PASS。
6. real H-to-K partial-wait 未采用；real 成功的是 current-ready stage rotation。
7. PASS：H READY、K PENDING 真实进入 ISA。
8. 显式为 H prologue ready、steady K pending/H consume、epilogue K commit。
9. PASS：358 < 384..399 < 402 < 403。
10. PASS：H 353 commit，早于 K 358 issue。
11. K pending 跨 4 条 H MFMA。
12. packet 是短 H/K packet；C25 code object 保持 VGPR188/AGPR80，spill=0。
13. T64/T2048/T8192 全 PASS，BF16 exact。
14. 正式数值见上表。
15. T8192/T16384 分别慢 5.22%%/6.19%%。
16. Z5B/C25/native slope 为 0.915028/0.973350/0.469623 us/chunk。
17. slope gap closed = -0.130942，实际扩大。
18. overlap 已由 ISA 证明，但性能无收益；没有把收益归因给 overlap。
19. 最终为 `STOP_C25_CURRENT_READY_OVERLAP_NO_PERF`。

## 止损

C25 到此关闭。按任务约束，不自动启动 C26、packet/barrier/VALU sweep、RA tuning、新 layout、新 superloop 或新 FullPhysicalRegion。

## 证据文件

- `stage6z_c25_native_current_ready_timeline.json`
- `stage6z_c25_partial_wait_synthetic.json`
- `stage6z_c25_real_partial_wait_overlap.json`
- `stage6z_c25_ready_pending_plan.json`
- `stage6z_c25_real_overlap_evidence.json`
- `stage6z_c25_pipeline_survival.json`
- `stage6z_c25_liveness.json`
- `stage6z_c25_correctness.json`
- `stage6z_c25_formal_body.json`
- `stage6z_c25_longtext_slope.json`
- `stage6z_c25_machine_resources.json`
- `stage6z_c25_pmc.json`
- `stage6z_c25_causal_delta.json`
- `stage6z_c25_regression_results.json`

Raw compiler and native artifacts are under `codex_qwen_gfx942_c25_current_ready_next_pending/`.

# Qwen gfx942 C20：C19 冻结性能验证与 Triton 资源差距审计

## 1. 最终决策

本轮是只读的正式验证，没有修改 C19 kernel、Avelang compiler、LLVM/ISA、packet、LDS、barrier、scheduler、RA、producer-consumer mapping、selector 或 production dispatch。

最终状态：

```text
STOP_C20_BODY_PERFORMANCE_NO_GO
PUBLIC_EAGER_NOT_RUN
```

原因不是“C19 没有降低机器工作”。C19 确实把一部分机器工作推向 native：在新鲜 T=2048 PMC 中，C19 为 `MFMA=160、VMEM=304、LDS=688、VALU=6174、SALU=618 / CTA`。但是正式 7-session、fresh-process、caller-owned、current-stream、no-Graph 的 HIP-event body 中，C19 的中位 session median 为 `0.095181003 ms`，Z5B 为 `0.066898998 ms`，C19 是 Z5B 的 `1.4228x`，反而慢约 `42.28%`。预注册的 C20 Body GO 条件因此不成立，不能把 PMC 改写成性能成功，也没有资格进入 Public Eager。

这轮仍然完成了 C20 要求的可用审计：C19 frozen identity、正式 body 对照、T=2048 fresh PMC、code-object/profiler 资源、ISA family、VMEM/LDS role ledger、register gap 以及明确的下一步候选均已落盘。

## 2. C20 冻结边界与身份

| 项目 | C19 冻结事实 |
|:--|:--|
| target | gfx942 / wave64 |
| shape | BT64 / BV64 / BK32 |
| launch | WG256、每 chunk-head 2 CTA |
| math | `v_mfma_f32_32x32x8_bf16`、K32 accumulation order、BF16 ABI、FP32 accumulator |
| C19 source SHA256 | `e3d5c81800fa8f8e1e4c8ca2f9148c0befff46c5bfcfc8e49c4531fd5ee92975` |
| lowered LLVM SHA256 | `a8579c7ad73de7daf3f0e6ae3b18bf5914c5c5807e60b9f7089e24bc90395407` |
| final ISA SHA256 | `c9279f37f79665db2173677e1c7ce3d5ff6294463cec1df41145618d4cfe5ddb` |
| C19 HSACO SHA256 | `14ffa78200ff995448c8abbe1c5484375aae1a2f75c8412a937b33d4aa628907` |
| C20 source/compiler mutation | none; artifact hash guard remained valid |

C20 的 worktree 本来就包含前面实验的 dirty files，因此不能用“git clean”冒充冻结证据。冻结证据是 C19 source/LLVM/ISA/HSACO 的 hash 和 replay driver 的 HSACO hash guard；当前 worktree digest 见 `stage6z_c20_frozen_c19_identity.json`。

## 3. 正式测试口径

正式 source body sweep 使用 Z5B、P2 和实际 public selector 选出的 native arm：T=512/1024/2048/4096/8192/16384，每个 T 7 个 fresh Python process session，warmup=10、repeat=50、预分配 caller-owned output、current HIP stream、无 CUDA Graph。每个 session 的 raw per-arm order 都被保存，order 旋转覆盖在 JSON 中；原始 harness 的 aggregate `rotating_orders` 字段曾为空，这是 metadata serialization bug，不能用它推断未旋转，raw `order` 字段才是权威证据。

C19 由于当前 active logical block-dot binding 不接受原 C19 source 的第三 operand 类型，不能在 C20 期间修改 compiler 来绕过这个问题。C19 使用 exact hash-guarded T=2048 HSACO replay，7 sessions，同样的 warmup/repeat/stream/output 口径。C19 这个 HSACO 是 T=2048-specific，所以不能把一个 T=2048 code object 假装成 C19 的 T=512～16384 sweep。

## 4. Body 性能结果

### 4.1 Source arm sweep：Z5B、P2、实际 selected native

以下是各 T 的 median of session medians，单位 ms：

| T | chunks | Z5B | P2 | native selected |
|--:|--:|--:|--:|--:|
| 512 | 8 | 0.052358001 | 0.058146499 | 0.037696000 |
| 1024 | 16 | 0.052758001 | 0.058587000 | 0.038337000 |
| 2048 | 32 | 0.066498499 | 0.072888501 | 0.042963499 |
| 4096 | 64 | 0.101651002 | 0.110564500 | 0.055983000 |
| 8192 | 128 | 0.157654501 | 0.176222004 | 0.090314001 |
| 16384 | 256 | 0.275528997 | 0.304932997 | 0.140649505 |

这里的 native selected 是每个 T 先走真实 public selector，再 pin 住实际配置进行 body 计时；不是把 T=2048 的 selector 插值到所有长度。

### 4.2 C19 frozen HSACO T=2048 formal control

| arm | median of 7 session medians (ms) | 相对 Z5B | 与 Z5B paired difference (us) |
|:--|--:|--:|--:|
| Z5B | 0.066898998 | 1.0000x | 0 |
| P2 frozen HSACO | 0.099067003 | 1.4808x | 32.168 |
| C18 frozen HSACO | 0.097945001 | 1.4641x | 31.046 |
| **C19 frozen HSACO** | **0.095181003** | **1.4228x** | **28.282** |
| native selected | 0.042303002 | 0.6323x | -24.596 |

C19 相对 Z5B 的 7 个 paired differences 全部为正，bootstrap CI 和 raw differences 见 `stage6z_c20_body_performance.json`。因此这是稳定的 No-Go，不是单个异常 session。

### 4.3 slope

source sweep 的最小二乘拟合为 latency(ms) = intercept + slope × chunks：

| arm | slope (us/chunk) | intercept (ms) |
|:--|--:|--:|
| Z5B | 0.916624 | 0.040745 |
| P2 | 1.017184 | 0.044780 |
| native selected | 0.429548 | 0.031575 |
| C19 | N/A | C19 没有跨 T 的可比 body sweep |

C19 因 T2048 单点 body 已失败，不运行任何“补长文本 C19”的私有替代测试，也不运行 Public Eager。

## 5. Correctness 与回归

C19 已有 full correctness matrix：T=64/128/512/1024/2048/4096/8192/16384 均 BF16 byte-exact、finite；T=64/8192/16384 caller-owned output、zero-V-new、NaN-prefilled output 均通过，NaN count 为 0。C20 T=2048 frozen replay 中 C18/P2/C19 对 Z5B exact，所有 arms finite。C20 source sweep 的 Z5B/P2/native selected 输出也均 finite。

这里要准确区分：C19 correctness 是已有的完整 source/runtime regression 证据；C20 的 C19 性能是 frozen T2048 HSACO replay；C20 没有声称当前 active binding 能再次编译 C19 source。

## 6. T=2048 fresh PMC：每 CTA

所有 raw aggregate counter 除以 `Grid_Size / Workgroup_Size = 512`，没有使用静态 ISA 推导动态数字：

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR | profiler Accum_VGPR | profiler SGPR | LDS metadata B | occupancy % | scratch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| z5b | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 112 | 32768 | 14.750070 | 0 |
| p2 | 160 | 448 | 576 | 8092 | 776 | 84 | 92 | 112 | 32768 | 14.856463 | 0 |
| c18 | 160 | 416 | 800 | 8010 | 776 | 84 | 92 | 112 | 32768 | 14.982923 | 0 |
| c19 | 160 | 304 | 688 | 6174 | 618 | 88 | 88 | 112 | 24576 | 15.024617 | 0 |
| native_same_shape_WG256 | 160 | 140 | 480 | 3376 | 660 | 100 | 36 | 96 | 0 | 9.511416 | 0 |


新鲜 C19 PMC 与 C19 历史 diagnostic 的小差异来自不同 profiler session；C20 当前结论只使用上表新鲜采集值。外部 HSACO trace 的 `trace_count=8`、native WG256 为 7，表示 collector 结果中匹配到的 trace rows 数，不改变 aggregate PMC 的 CTA normalization。

## 7. Code-object resource 与 profiler resource 必须分开

| arm | code VGPR | code AGPR | code SGPR | code LDS/private | code spill | profiler VGPR | profiler Accum_VGPR | profiler SGPR |
|:--|--:|--:|--:|:--|:--|--:|--:|--:|
| Z5B | 104 | 32 | 28 | 32768 / 0 | 0 | 76 | 100 | 112 |
| P2 | 132 | 48 | 30 | 32768 / 0 | 0 | 84 | 92 | 112 |
| C18 | 116 | 32 | 30 | 32768 / 0 | 0 | 84 | 92 | 112 |
| C19 | 136 | 48 | 28 | 24576 / 0 | 0 | 88 | 88 | 112 |
| native WG256 | 132 | 32 | 89 | readobj 0; metadata 12288 | 0 | 100 | 36 | 96 |

native WG256 的 `chunk_fwd_kernel_o.json` 报告 `shared=12288`，而它的 HSACO readobj 文本报告 `.group_segment_fixed_size: 0`；这不是把 0 当成真实 LDS 的理由，而是两个 artifact layer 的 metadata discrepancy，故同时保留。native actual selected T2048 是另一个 WG128/2-wave/3-stage code object，code VGPR/AGPR/SGPR 为 220/64/76，不能与 WG256 same-shape 资源混写；它只用于实际 selector 的性能对照。

## 8. 从 Z5B 到 native 的机器 gap 关闭情况

按 fresh T2048 aggregate PMC 的同 shape WG256 对照：

| 指标 | Z5B | C19 | native WG256 | Z5B→native gap | C19 已移动量 | gap closure |
|:--|--:|--:|--:|--:|--:|--:|
| MFMA | 160 | 160 | 160 | 0 | 0 | N/A (gap=0) |
| VMEM | 672 | 304 | 140 | 532 | 368 | 69.17% |
| LDS | 672 | 688 | 480 | 192 | -16 | -8.33% |
| VALU | 7072 | 6174 | 3376 | 3696 | 898 | 24.30% |
| SALU | 768 | 618 | 660 | 108 | 150 | 138.89% |


解释：VMEM gap 关闭约 69.17%，说明 C19 full physical ownership 确实删除了相当一部分重复/窄化 producer 工作；VALU gap 只关闭约 24.30%，仍有明显 address/layout/fragment feeding 成本；LDS 不是改善项，C19 反而比 native 多 208/CTA；MFMA 数学工作完全一致。这里是机器工作 gap closure，不是 latency gap closure，也不能据此宣称某一类指令就是唯一的时间因果。

## 9. ISA static audit：只作为机器图证据

`stage6z_c20_isa_resource_gap.json` 保存了逐 mnemonic 与 family count。C19 final ISA 有 `mfma32=56、ds_read=112、ds_write=112、global/buffer load=104、barrier=46、waitcnt=147` 的量级；native WG256 有 `mfma32=40、ds_read=80、ds_write=40、buffer_load_dwordx4=14、buffer_store_dwordx2=4、barrier=11、waitcnt=48` 的量级。C19 的 56 个 lexical MFMA 不能直接写成动态 MFMA；新鲜 PMC 才给出两者均为 160/CTA。native 的 40 lexical MFMA × 4 waves 与 160/CTA 一致，而 C19 的控制流/continuation 使 lexical count 不能单独作为动态执行模型。

关键 ISA 事实：

- C19 没有 `ds_bpermute`，所以当前 gap 不是 R3 那类跨 lane shuffle 爆炸；
- C19 仍有更多显式 `ds_read`/`ds_write`、barrier/waitcnt 和窄/阶段化 operand feeding；
- native WG256 使用 `buffer_load_dwordx4`、`ds_read2_b64`、`ds_write2st64_b64` 等 typed/wide packet family；
- C19 的 global/LDS lexical family 与 dynamic PMC 要分开看，不能用 static count 直接计算访问字节。

## 10. VMEM role ledger

`stage6z_c20_vmem_role_breakdown.json` 对 Q、H、K、V-new、g、output 建立了 logical tile、source phase、producer owner、native artifact 和证据等级。每个 logical tile 的 bytes 是契约级 tile bytes：Q/K/H=16 KiB，V-new/output=8 KiB，g=256 B；它们不是 hardware transaction bytes。

当前可以确认的事实：

1. C19 Q 由 FullPhysicalRegionPlan 单次 producer、Q@H/Q@K 双 consumer 管理，C19 source/report 明确删除了 Z5B 的 Q duplicate producer；
2. C19 H/K/V 也由统一 physical plan 接管，但 aggregate VMEM 没有 per-operand counter，不能把 C19 的 304/CTA 精确拆成 K 或 H；
3. g 仍有 score target/source/final scaling 多个 logical consumer role；这只是 source-level provenance，不是 304 中某个精确硬件份额；
4. native TTGIR 中同一个 Q operand `%b_q_275` 明确 feeding 两个 dot consumer，且 native ISA 有 wide packet load；native 的 140/CTA 是总量，不提供按 role 的 hardware split；
5. 因此本轮不能严谨地选“C19 VMEM 最大 offender 一定是 K/g/H”。最大的已证实事实是 **remaining aggregate VMEM/VALU feeding gap**，而非已证明的单 operand causal share。

## 11. LDS role ledger

C19 的实际 arena 是 `384 x 32 BF16 = 24576 B`：rows 0..255 是 Q physical cache；rows 256..319 是 H/K phase band；Q last-use 后部分 rows 再复用给 score 阶段。这个 reuse 改变了 footprint，但没有让所有 producer-consumer LDS instruction 消失。

新鲜 dynamic LDS 为 C19 `688/CTA`、native WG256 `480/CTA`，约 `1.433x`。这不是“LDS footprint 相同后仍多”的结论，因为 C19 exact allocation 是 24 KiB，而 native WG256 artifact 的 Triton metadata 是 12 KiB、readobj 是 0 的 metadata conflict。更稳妥的结论是：C19 的 physical arena 已比 C18/P2 小，但 native 仍有更紧的 typed shared/dot path 和更少的 phase publication/consumer traffic；单凭总 LDS PMC 不能再分配到 Q/H/K/V 某一条子路径。

## 12. Register/liveness gap

C19 有 exact-LTO MIR，可审计其 machine def/use、phase-separated score accumulator、address temporaries 和最终物理寄存器；native 工件只有 TTIR/TTGIR/LLVM/ISA，没有可比的 exact-LTO MIR，因此不能编造 native LiveIntervals。

C19 code object 是 `VGPR/AGPR/SGPR=136/48/28`，native WG256 readobj 是 `132/32/89`；C19 因此多 4 个 code VGPR、16 个 code AGPR，但少 61 个 code SGPR。profiler 字段则是 C19 `88/88/112` 对 native `100/36/96`。两套数不是同一层面的 register accounting，不能混写成“C19 只有 88 个 AGPR”或“native 只有 36 个物理 AGPR”。C19 没有 private segment/spill；剩余 gap 主要表现为 operand feeding 和 address/layout 工作，不是 C19 已经发生了 spill cliff。

## 13. Public Eager 决策

C19 没有达到 Body GO，所以按预注册规则没有运行 Public Eager。`stage6z_c20_public_eager_performance.json` 明确记录 `executed=false`。因此本报告不提供虚假的“C19 Eager vs vLLM”数字，也不把 isolated body 比例冒充完整公开 API 速度。

## 14. 最多三个有证据支持的下一候选

本轮不实现任何候选，只登记：

1. **C19→native typed/wide operand feeding gap**：同 shape 下 native 有 wide packet + dot encoding，C19 仍有更多显式 LDS publication/consumer instruction；这是由 static ISA + dynamic LDS/VALU/VMEM 共同支持的结构差距，但尚未能按一个 logical operand 精确归因。
2. **C19 remaining layout/address/fragment VALU**：C19 `6174/CTA` 对 native `3376/CTA`，而 MFMA 相同、无 ds_bpermute；说明剩余差距不能只归咎于数学 MFMA 数量，可能来自通用 physical address/fragment feeding。仍需下一轮专门 provenance，不应在本轮自动修改。
3. **C19 shared phase traffic / barrier-window overlap**：C19 static barrier/waitcnt 与 dynamic LDS 均高于 native，且 C19 24 KiB arena 与 native 12 KiB metadata 不同；这是第三候选，因没有 per-role counter，因果等级低于前两项。

这些是 measured gap 与 causal inference 分开的候选，不是“下一轮必须改 K”或“必须改 g”的结论。

## 15. 机器可读产物

本报告同目录下的产物：

- `stage6z_c20_frozen_c19_identity.json`
- `stage6z_c20_body_performance.json`
- `stage6z_c20_latency_slope.json`
- `stage6z_c20_code_object_resources.json`
- `stage6z_c20_profiler_resources.json`
- `stage6z_c20_pmc_t2048.json`
- `stage6z_c20_machine_gap_closure.json`
- `stage6z_c20_isa_resource_gap.json`
- `stage6z_c20_vmem_role_breakdown.json`
- `stage6z_c20_lds_role_breakdown.json`
- `stage6z_c20_register_gap.json`
- `stage6z_c20_public_eager_performance.json`
- `stage6z_c20_regression_results.json`

新增的只读汇总脚本是 `finalize_qwen_gfx942_c20_audit.py`。它不会编译、不会加载 kernel、不会运行 GPU；它只读取上述 frozen artifacts 和已保存的 C20 measurement JSON。

## 16. 复现与停止点

正式 body 原始结果：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gfx942_c20_formal_performance.py --help
```

T=2048 frozen control 和 fresh PMC 的原始 JSON/CSV 位于：

```text
codex_qwen_gfx942_c20_formal_performance/
  stage6z_c20_t2048_frozen_controls.json
  pmc_t2048_fresh/{z5b,p2,c18,c19,native_wg256}/
```

C20 在明确的 `STOP_C20_BODY_PERFORMANCE_NO_GO` 处停止。没有创建 C20 优化 variant，没有改生产路径，也没有运行 Public Eager。

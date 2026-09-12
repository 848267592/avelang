# Current Triton Pred -> V-new -> Update Machine Pipeline Audit

## 审计范围与身份

本轮是只读审计：没有改动 Avelang、Triton、HSACO、launch、算法或 benchmark。
审计对象是 Stage 6R 保存的 current-vLLM recurrence，而不是历史 asm-v0，严格
选择 SHA256 为：

```text
632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e
chunk_gated_delta_rule_fwd_kernel_h_blockdim64
```

它的固定 contract 是 gfx942、`BT=64`、`BV=32`、`WG=128`、两个 wave、
`num_stages=2`、动态 LDS `40960 B`，W/U/V-new 为 BF16，g/initial-state/
final-state 为 FP32。完整身份和路径见
`codex_qwen_current_triton_pred_vnew_update_pipeline_audit/artifact_identity.json`。

本次使用的是该 code object 同一编译实例的 TTIR、TTGIR、LLVM IR、AMDGCN 与
ISA。捕获物没有 AMDGPU pre/post-RA MIR；本报告明确将 MIR 项标为 N/A，没有为了
得到 MIR 重新编译一个可能不再等价的 HSACO。

## 直接答案

### Pred 和 update 是否真的分阶段？

**对同一个 BT64 chunk：是，数学上严格分阶段。** 源码和 TTIR 的顺序是：

```text
H snapshot
  -> pred = W0 @ H1 + W1 @ H2
  -> corrected = V - pred
  -> BF16 V-new global store
  -> decay(corrected)
  -> H1 += K0 @ V-decay
  -> H2 += K1 @ V-decay
```

证据是 `kernel.ttir:166-187` 的两次 pred `tt.dot` 与 V-new store，随后
`kernel.ttir:198-235` 的 decay 和两次 K update `tt.dot`。所以 Triton 并没有把
同一 chunk 的 pred/update 改写成可违反 recurrence 依赖的并行阶段。

**但对相邻 chunk：不是全局先完成所有 pred、再完成所有 update。** TTGIR 已经把
chunk `i+1` 的 W0/W1、V、g、K0/K1 全局加载安排到 chunk `i` 的 pred/update
计算窗口中。这是一条 persistent loop 的软件流水，而不是两个全序列 kernel。

### V-new 有没有 global handoff/reload？

没有。TTGIR 先得到 `%b_v_395 = V - pred`，随后将其 BF16 版本写入公开的
`v_new` buffer（`kernel.ttgir:407-414`）；但 update 端接着从同一 `%b_v_395`
计算 decay，并得到 `%b_v_428` BF16 V-decay（`419-447`）。update dot 使用这个
仍在本 CTA 内的值（`460-472`），没有读取刚写出的 `v_new` global buffer。

因此该 global store 是 ABI-visible output，不是 update 的 producer-consumer
handoff。这是 current Triton 相对“写 V-new，再另一 kernel 读 V-new”的明确结构差异。

## 每个 steady-state chunk 的实际 TTGIR 时间线

下面的 `i` 是正在完成 recurrence 的 chunk；`i+1` 是预取目标。

| 顺序 | chunk | 工作 | TTGIR 证据 |
|---:|:---:|:---|:---|
| 0 | 0 | prologue 把 W0/W1、V、g、K0/K1 读入并将 W/K 放进 LDS。 | `240-326` |
| 1 | i | 将 persistent FP32 H1/H2 转 BF16，写 H snapshot。 | `329-343` |
| 2 | i+1 | 提前发起 W0/W1 global load。 | `375, 387` |
| 3 | i | 从 loop-carried W0/W1 LDS memdesc local-load，构造 H BF16 dot operand，做 pred。 | `376-393` |
| 4 | i+1 | V、g、K0、K1 的 global load 已在当前 pred/update 周期内发起。 | `406, 426, 435, 458, 470` |
| 5 | i | `V - pred`，写 BF16 V-new；不 reload。 | `407-414` |
| 6 | i | 从仍 live 的 corrected 值形成 V-decay，并缩放 persistent state。 | `419-447` |
| 7 | i | 从 loop-carried K0/K1 LDS memdesc 和 transient V-decay dot operand 读取，执行两个 update dot。 | `459-474` |
| 8 | i+1 | 把早已发起的 W0/W1、K0/K1 结果写入同一套 LDS bank，成为下一轮的 current operand。 | `478-488` |

也就是说，Triton 的窗口是**一 chunk ahead 的 global-load lookahead**。在当前
chunk 的 update 之前，下一 chunk 的两个 K64 global load 都已出现：K0 在
`kernel.ttgir:458`，K1 在 `:470`；而当前的 K0/K1 update dot 分别是 `:462` 与
`:472`。这回答了“两个 K64 是否提前预取”：**是，成对地为下一 chunk 提前 load；
不是在当前 K0 dot 内另建一套 K1 ping-pong LDS stage。**

## LDS buffer 同时存在什么？

TTGIR 的四个 loop-long shared memdesc（`222-225`）和 LLVM `global_smem` GEP
base 共同给出以下 40 KiB map：

| 区间 | 大小 | 逻辑对象 | 用途 |
|:--|--:|:--|:--|
| `0..8191` | 8 KiB | W0 | 当前 chunk pred 的第一半 W；末尾被 next W0 覆盖。 |
| `8192..16383` | 8 KiB | W1 | 当前 chunk pred 的第二半 W；末尾被 next W1 覆盖。 |
| `16384..24575` | 8 KiB | K0 | 当前 chunk update 的 `K[0:64]`；末尾被 next K0 覆盖。 |
| `24576..32767` | 8 KiB | K1 | 当前 chunk update 的 `K[64:128]`；末尾被 next K1 覆盖。 |
| `32768..40959` | 8 KiB | transient operand band | pred 的 BF16 H operand 与 update 的 BF16 V-decay operand 按 phase 复用。 |

因此 W0/W1/K0/K1 四个 64x64 BF16 block 在 loop 内同时占用 32 KiB；高 8 KiB
是 `shared2` 的 phase-local operand band。详细 machine-readable map 见
`lds_lifetime.json`。

一个容易误读的点：metadata 的 `num_stages=2` 并不等于两套完整 W/K LDS buffer。
W/K 在 TTGIR 中是 `memdesc<1x64x64xbf16>`，并在 tail 使用同一 loop-carried
memdesc 写入 `i+1`。它实现的是**寄存器保存的 global-load lookahead + 单一
物理 LDS bank**，而不是 64 KiB 的双 LDS ping-pong。

## Load / LDS / MFMA 是否交错？

是，有三层证据。

1. **TTGIR 的 loop 排列**：next W/V/g/K global load 与 current W/K `local_load`
   和 `tt.dot` 相互穿插，见上表的 `375-487`。
2. **LDS phase**：current K 位于 16/24 KiB band；current V-decay 使用 32 KiB
   band。LLVM IR 的 `addrspace(3)` GEP 明确保留 `16384`、`24576`、`32768` 基址。
3. **ISA operand consumption**：例如 `disassembly.txt:2609-2635` 先读 LDS
   operand，然后以 `lgkmcnt(7)` 到 `lgkmcnt(0)` 渐进地发射八条
   `v_mfma_f32_32x32x8_bf16`；接下来的 `2655-2678` 对另一个 K band 重复
   `ds_read_b64 -> MFMA`。这不是“所有 LDS read 完成后才开始所有 MFMA”的平铺
   序列。

静态 ISA 还含 32 个 `s_barrier`、150 个 `ds_read`、183 个 `ds_write`、36 个
buffer load 和 64 条 BF16 MFMA。这些是静态指令计数，不应误读成每个 chunk 的
动态 barrier 数。T=2048 的已保存 rocprof 动态计数是 MFMA `65536`、VMEM
`58368`、LDS `305472`；其来源是 Stage 6R 的 exact current-vLLM profile。

## Pred、V-new、update operand 的重叠

| 值 | 逻辑生命周期 | 是否跨越 global handoff |
|:--|:--|:--|
| H1/H2 FP32 | 初始 state load 后通过 `scf.for iter_args` 跨全部 chunk；update 结果成为下一 chunk 输入。 | 不经过 H snapshot。 |
| pred accumulator | 两个 pred dot 后形成 `%b_v_379`；到 `%b_v_395 = V-pred` 后不再需要原 acc。 | 否。 |
| corrected / V-new | `%b_v_395` 同时供公开 BF16 V-new store 和 decay；BF16 V-decay 随后短暂存在。 | store 不被 reload。 |
| update accumulator | `%b_h1_443/%b_h2_453` 在 pred corrected 已产生后创建，最后成为 loop yield state。 | 不经过 global H。 |
| next W/K vectors | global load 后暂留在寄存器，直到 loop tail 写入单 LDS bank。 | 是 lookahead，不是另一 LDS stage。 |

TTGIR 因此显示的是“persistent state + next operand load + current pred/update”的
宽窗口，而不是“pred acc 和 update acc 两个完整 AGPR tile 长时间并存”。本次没有
精确 MIR，不能把该 SSA lifetime proxy 夸大成物理 vreg live interval 结论。

## Triton 是否靠更宽调度窗口隐藏 latency？

**定性答案：是，有直接编译工件证据。** `num_stages=2`、prologue、loop-carried
W/K LDS memdesc，以及 `i+1` global load 出现在 `i` 的 pred/update 前，证明
它至少提供一 chunk 的 global-memory lookahead。ISA 中多个 outstanding-memory
`vmcnt` wait 和 MFMA 中递减 `lgkmcnt` 也与此一致。

**但不能量化成“节省了 X us”。** 没有 PC sampling、wave timeline 或精确
cycle-level simulator，静态 TTGIR/ISA 只能证明调度机会和依赖顺序，不能给出
latency hiding 的百分比归因。

## 对 Avelang 后续研究的约束

这个审计给出的可复用目标不是再做一个局部 LDS swizzle，而是保持以下严格等价的
流水形状：

1. persistent FP32 H1/H2 只作为 recurrence feedback；H BF16 snapshot 不参与反馈；
2. V-new global store 保留 ABI，但 update 直接消费同 kernel 仍 live 的 corrected/
   V-decay，而非 reload；
3. 当前 chunk 使用一组 W0/W1/K0/K1 LDS operand；
4. 下一 chunk 的 W0/W1/K0/K1 在当前计算窗口预发起 global load，末尾才覆盖同一
   LDS bank；
5. K0/K1 是成对的 next-chunk lookahead，不应误实现为两个完整 K LDS ping-pong
   bank；
6. 是否能兑现收益必须通过 same-work phase-lifetime A/B 验证，不能由这份静态审计
   直接推断性能增益。

## 复现

```bash
cd /home/jiandongliu/project/avelang
sha256sum \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/kernel.hsaco

nl -ba test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/kernel.ttgir \
  | sed -n '222,488p'

nl -ba test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/disassembly.txt \
  | sed -n '2579,2678p'
```

完整结构化证据在
`codex_qwen_current_triton_pred_vnew_update_pipeline_audit/`。

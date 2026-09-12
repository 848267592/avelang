# Direct-K64 P3：Feedback Source Decomposition

## 问题

P2 在 `T=2048`、chunk 23 首次观察到 `pred_f32` 误差
`5.206061e-05`，略高于冻结的 `5e-05` 门槛。P0 已通过 nonzero-W MFMA32
lane/fragment mapping，P1 单 chunk composition 也通过，因此不能仅凭 P2 结果
把错误归因于 pred lowering。

P3 的目标是严格区分两种假设：

1. pred kernel/lowering 在 composed feedback 下本身错误；
2. earlier update 已产生 state 差异，后续正确的 pred 只是消费了不同的 BF16 operand。

这轮没有性能计时、没有改 kernel、没有改 layout、没有改 allocator/RA 或 production。

## 固定条件

P3 从与 P2 相同的确定性 `T=2048` 输入重放 chunk `0..22`，在进入目标
chunk 23 前冻结：W、U、K、g、CTA layout、`BT64/BV32/WG128`、MFMA32 pred、
C0 typed-block update 和同一个 P1 JIT kernel。

三臂均运行：

`_qwen_gdn_direct_k64_bv32_full_p1_kernel`

P1 中 pred 在 update 前执行；P3 只判定 exported pred/V-new 字段，因此同一调用中
较晚的 update 不可能反过来影响所比较的 pred 值。

| arm | pred 输入 state |
|:--|:--|
| A | P2 实际 feedback FP32 state |
| B | 独立 reference FP32 state |
| C | `BF16(actual_state)` 再扩展为 FP32 |

每一个 arm 都同时有自己的 local reference。报告必须同时看：

- `local_error`：kernel 输出对相同 state 的参考，判断 pred kernel 是否正确；
- `trajectory_error`：arm 输出对 B/reference trajectory，复现 P2 的差异来源。

## 状态输入比较

在目标 chunk 前，actual 与 reference state 已经不同：

| 项 | 值 |
|:--|--:|
| FP32 state max abs | `2.197903e-04` |
| FP32 state mean abs | `3.571382e-05` |
| BF16 bitwise 不同元素 | `68,018 / 131,072` |
| BF16 不同占比 | `51.8936%` |
| BF16 value max abs | `4.8828125e-04` |

首个 BF16 bit 不同坐标为 `[0,0,0,0]`：actual FP32 `5.720514e-04`，reference
FP32 `5.988115e-04`；量化后分别为 `5.722046e-04` 和 `5.989075e-04`。
这证明小于 `1e-4` 的 FP32 feedback 差异确实跨越了 BF16 rounding boundary；不能再
假设两个 FP32 state 的微差会产生同一个 MFMA operand。

## 三臂结果

| 比较 | pred FP32 max abs | 解释 |
|:--|--:|:--|
| A local：`pred(actual)` 对 actual-local reference | `1.862645e-09` | 通过 |
| B local：`pred(reference)` 对 reference-local reference | `1.862645e-09` | 通过 |
| C local：`pred(BF16(actual))` 对 C-local reference | `1.862645e-09` | 通过 |
| A trajectory：`pred(actual)` 对 reference trajectory | `5.206061e-05` | 精确复现 P2 |
| B trajectory：`pred(reference)` 对 reference trajectory | `1.862645e-09` | 恢复到 P0 级 |
| C trajectory：snapshot 对 reference trajectory | `5.206061e-05` | 与 A 相同 |
| A vs C pred FP32 | `0` | bit-exact |

三个 arm 均 finite，且 local checks 全通过。A/C local BF16 pred 的最大误差
`7.629395e-06`、V-new 最大误差 `1.907349e-06` 均远低于 P0 BF16 门槛。

## 结论

这是预注册的 **Case A**：

```text
pred(reference_state) 恢复 P0 精度
pred(actual_state) 对其自身 local reference 也恢复 P0 精度
pred(actual_state) == pred(BF16(actual_state))
actual/reference 的 BF16 state 已大量不同
```

所以结论是：

1. 当前 P0/P1 的 pred lane mapping 与 lowering 没有证据表明出错；
2. pred 确实按预期消费 BF16 quantized state，不存在“实际仍隐式读取 FP32 state”的
   证据；
3. P2 的 chunk 23 误差是前序 update/state trajectory 已分叉后的数学结果；
4. P3 不能单独判定 update 的哪一部分产生了第一个 seed，只能把问题从 pred
   收缩到 update/state 路径。

这排除了继续修改 pred mapping、pred LDS layout、pred address lowering 或 RA 的理由。

## 下一步：仅 update seed decomposition

下一轮仍然不做性能实验。应从 chunk 0 开始对同一固定输入导出并比较：

1. BF16 `V-new`；
2. BF16 `V-decay = round_bf16(V-new * exp(g_last-g))`；
3. update MFMA delta，参考应消费完全相同的 BF16 V-decay 与 BF16 K；
4. FP32 state scale `state * exp(g_last)`；
5. final `state_after = scale + delta`。

第一处差异才是后续 correction 的对象。届时再区分 BF16 V-decay rounding、MFMA
delta mapping/accumulation，或 state scale/add；不能直接修改性能 schedule。

## 后续结果

该 update seed decomposition 已完成：U0 证明修复前 kernel 绕过了 BF16 V-new
boundary；修复后 V-decay contract 误差为 `0`，P2 `T=2048` 也恢复为 `32/32`
chunks 通过。完整修复记录见
[BF16 V-new boundary fix report](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_update_bf16_vnew_boundary_fix_report.md)。

## 证据与复现

- [P3 稳定 JSON](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_feedback_source_decomposition_p3/p3_feedback_source_decomposition.json)
- [P3 source](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_feedback_source_decomposition_p3.py)
- [P0-P2 前序报告](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_full_recurrence_p0_p2_correctness_report.md)

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_feedback_source_decomposition_p3.py
```

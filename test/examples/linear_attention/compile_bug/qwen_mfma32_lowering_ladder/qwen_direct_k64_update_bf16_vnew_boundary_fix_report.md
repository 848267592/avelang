# Direct-K64 Update：BF16 V-new 边界正确性修复

## 结论

P2 `T=2048` chunk 23 的严格 pred 门槛失败已经定位并修复。根因不是 pred
mapping、pred lowering、RA、LDS layout 或 MFMA geometry，而是 P1 experimental
融合路径违反了 current-vLLM BF16 recurrence ABI 的一个数据边界：它写出了 BF16
`V-new`，却让 update 继续消费写出前的 FP32 `corrected`。

修复后，P1 的 random nonzero-W 单 chunk `V-decay` 误差从 `4.8828125e-04` 降到
`0`，delta 从 `4.376692e-05` 降到 `3.492460e-09`；P2 feedback ladder 在
`T=128/512/2048` 全部通过，T=2048 的 32 个 chunk 均执行完成。

本修复仅作用于 experimental Direct-K64 P1/P2 full-recurrence correctness
ladder。它**不**修复或宣称修复 old full-v29 nonzero-W、production selector、
external HSACO 或完整 public forward。

## 问题如何产生

修复前 P1 代码等价于：

```python
corrected = f32(u_bf16) - pred_f32
v_new = bf16(corrected)
v_decay = bf16(corrected * exp(g_last - g))  # 错误的 update 输入
```

而冻结的 current-vLLM BF16 ABI 是：

```python
v_new = bf16(corrected)
v_decay = bf16(f32(v_new) * exp(g_last - g))
```

这两个表达式并不等价。`corrected` 与 `bf16(corrected)` 可在 BF16 rounding
boundary 的两边；后续 decay、dot update 与 state feedback 会将这个差异带入下一
chunk 的 pred operand。

## U0：修复前的第一处错误证据

U0 复用未修复 P1 的同一 random nonzero-W 输入，比较 kernel 输出与两种公式：

| 项 | 与 unrounded 公式 | 与 BF16 V-new contract |
|:--|--:|--:|
| V-new | `0` | `0` |
| V-decay | `0` | `4.8828125e-04` |
| delta | `2.79e-09` | `4.376692e-05` |
| state_after | 与 `scale + kernel_delta` 精确一致 | 对 contract 差 `4.376844e-05` |

因此第一处非零差异就是 V-decay，而不是 delta MFMA 或 state scale/add：

```text
FP32 corrected bypassed BF16 V-new boundary
  -> wrong BF16 V-decay
  -> matching wrong delta
  -> state feedback diverges
  -> subsequent correct pred reads different BF16 state
```

这也解释了 P3 的结果：actual/reference state 在 chunk 23 前已经有 68,018 个
BF16 元素不同，`pred(actual_state)` 与 `pred(BF16(actual_state))` bit-exact，
而 `pred(reference_state)` 回到 P0 精度。

修复前原始记录：[U0 pre-fix JSON](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_update_seed_u0/u0_pre_fix_update_seed_audit.json)。

## 最小修复

改动位于 [P1 source](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1.py)：

```python
corrected = f32(u_bf16) - pred
corrected_bf16 = bf16(corrected)
corrected_from_bf16 = f32(corrected_bf16)

v_new[...] = corrected_bf16
vdecay_stage[...] = bf16(corrected_from_bf16 * decay)
v_decay[...] = bf16(corrected_from_bf16 * decay)
```

没有改动：pred schedule、Direct-K64 MFMA32 update、`block_dot` lowering、
CTA ownership、state layout、K32 accumulation order、barrier、allocator/RA 或
任何 production 路径。

## U0：修复后确认

同一 U0 审计在修复后得到：

| 项 | 修复后误差 |
|:--|--:|
| kernel V-decay vs unrounded 公式 | `4.8828125e-04` |
| kernel V-decay vs BF16 V-new contract | `0` |
| kernel delta vs BF16-boundary delta | `3.492460e-09` |
| state_after vs scale + contract delta | `3.725290e-09` |

也就是说，kernel 明确不再使用 unrounded corrected；BF16 ABI 边界已经实际进入
update consumer，而不只是写到一个未被使用的 `V-new` 输出。

证据：[U0 post-fix JSON](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_update_seed_u0_bf16_vnew_fix/u0_update_seed_audit.json)。

## P1 单 chunk 回归

六个 P0 machine case 都通过。前五个 one-hot/scan/sparse/permutation 用例全零。
random nonzero-W：

| 中间量 | 修复前 | 修复后 |
|:--|--:|--:|
| pred FP32 | `1.863e-09` | `1.863e-09` |
| V-new | `0` | `0` |
| V-decay | `4.883e-04` | `0` |
| delta | `4.377e-05` | `3.492e-09` |
| state_after / final_state | `4.377e-05` | `3.725e-09` |

证据：[P1 post-fix summary](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1_bf16_vnew_fix/p1_single_chunk_composition_summary.json)。

## P2 feedback 回归

P2 仍是 correctness microscope，不是 runtime split 或性能测试。修复后：

| T | chunk 完成 | pred FP32 max，最后 chunk | state_after max，最后 chunk | 结果 |
|--:|:--|--:|--:|:--|
| 128 | 2 / 2 | `2.906e-06` | `6.594e-06` | 通过 |
| 512 | 8 / 8 | `3.398e-06` | `8.567e-06` | 通过 |
| 2048 | 32 / 32 | `1.629e-05` | `4.347e-05` | 通过 |

T=2048 不再在 chunk 23 停止，所有值 finite。最后仍可见小量 BF16 feedback
rounding 差异，例如 V-new/V-decay `2.441406e-04`；它们没有突破冻结的
BF16/FP32 门槛，也不再形成此前的系统性 update seed。

证据：

- [P2 T=128 post-fix](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2_bf16_vnew_fix_t128/p2_recurrence_feedback_summary.json)
- [P2 T=512/2048 post-fix](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2_bf16_vnew_fix/p2_recurrence_feedback_summary.json)

## 决策

P0-P2 的 current-BF16-ABI Direct-K64 correctness ladder 已恢复通过。可以解除
“先修 pred mapping”的阻塞，但仍不得将它当作 production-ready full operator：

- old full-v29 nonzero-W correctness 没有被本修复覆盖；
- P2 的 host feedback microscope 不等于单-kernel full sequence；
- 本轮没有性能测量，不能推导任何 latency 变化；
- current-vLLM external recurrence HSACO 与 Direct-K64 Avelang research kernel
  仍是不同路径。

下一条研究动作可以回到原本被正确性阻塞的 **Triton-matched pred/update
phase-lifetime A/B**，但需先固定此 BF16 V-new boundary，并以 P0/P1/P2 回归作为
前置门槛；不得顺手改变 MFMA、CTA ownership、RA 或 LDS layout。

## 复现

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

# 当前 source 的 post-fix U0 audit
python3 test/examples/linear_attention/vllm_compare/audit_qwen_gdn_direct_k64_bv32_update_seed_u0.py \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_update_seed_u0_bf16_vnew_fix

python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1.py \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1_bf16_vnew_fix
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2.py \
  --T 128 512 2048 \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2_bf16_vnew_fix
```

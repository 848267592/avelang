# A_20260925_67_BW16_MQK_FRAGMENT_FIX

## 结果

A66 的第一个 Mqk 分歧已定位并修复：INV 结束后，LDS overlay 中仍是
`u_pre`，而不是 post-INV `U` 的 FP32→FP16 表示。A67 只在这个边界重新
发布 `convert(acc_u, f16)`；没有修改 Mqk 的 BW16/W4/S2 lane mapping、
矩阵 packet、MFMA operand ownership 或 MFMA accumulator continuation。

## 修复前审计

T32/NT1 顺序审计中：

1. post-INV U FP32：通过，max abs `2.384185791015625e-7`；
2. U FP32→FP16：bit exact，6144 elements；
3. U FP16 LDS overlay：首次失败，5609 个 bit mismatch，first index `[16]`。

因此只修第 3 个边界；最终 JSON 保留在 [A67 gate](A_20260925_67_mqk_fragment_fix/A_20260925_67_T32_MQK_GATE.json)。

## 修复后 T32/NT1 gates

| 顺序 | 边界 | 结果 |
|---:|---|---:|
| 1 | post-INV U FP32 | max abs `2.38e-7` |
| 2 | U FP32→FP16 | bit exact |
| 3 | U FP16 LDS overlay | bit exact |
| 4 | Mqk `[32,32]` FP16 LDS | bit exact |
| 5 | 首个 Mqk MFMA A/B FP16 + raw VGPR | 全部 bit exact |
| 6 | 首个 Mqk MFMA partial | max abs `9.31e-10` |
| 7 | 第二个 Mqk MFMA accumulated | max abs `9.31e-10` |
| 8 | Qd+Mqk FP32 output accumulator | max abs `9.31e-10` |

随后 existing integrated trace 的 Kd/pred/u_pre/INV/Qd、Kr1、Kr2、output
和 final_state 全部通过严格 `1e-5` 检查；没有下一个语义分歧，因此停止在
T32/NT1，不运行 full T matrix、资源/性能、K1 或 segmentation。

机器参考为 AITER BW16/W4/S2 commit
`7bb44998274fa679ece0a37bf8426ab780b1837`，HSACO
`85470da1877fcd49db1b9adee0129fff9787e7953d221817198ce2434cd0ac16`；
数学参考为 A56 BW64 K2 semantics。

文件：

- [A67 source](../avelang_impl/prefill/A_20260925_67_BW16_MQK_FRAGMENT_FIX.py)
- [A67 test](../tests/A_20260925_67_BW16_MQK_FRAGMENT_FIX_TEST.py)
- [final JSON gate](A_20260925_67_mqk_fragment_fix/A_20260925_67_T32_MQK_GATE.json)
- [exact snapshot](A_20260925_67_mqk_fragment_fix/A_20260925_67_T32_MQK_SNAPSHOT.pt)

A56、A66 及其已通过的 Kd/pred/u_pre/INV/Qd 路径均未修改。

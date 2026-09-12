# R4-tail State-KV / Dual-dot 候选结论

## 结论

`gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_dual_dot` 完成了一个完整、
单 recurrence-loop 的正确性候选，但没有通过本轮 ISA 硬门槛，结论为 **No-Go**。
因此没有采集 PMC，也没有运行 public-Eager 性能；这不是性能结论，而是避免对
不满足目标机器路径的候选继续花费 profiling 时间。

production T=2048、`emit_audit=False` HSACO：

```text
_state_kv_kernel.hsaco
sha256 fc6029bab9fd6d0f0cd81f06e895e9ef6b7158ab87e7bbbdb577664b98242761
grid=(32,1,1), workgroup=(128,1,1), gfx942
```

它仍有 16 条 `global_load_ushort` 和 16 条
`global_store_short_d16_hi` 位于两个 T32 pred-finalize/U/v_new 区域；这些
分别是 2 个 token tile x 每 tile 8 次静态展开。该区域没有对应的
`global_load_dwordx4` 或 `global_store_dwordx4`。所以 `U/v_new` 并未形成
无 exec-loop 的 b64/b128 packet access，首要验收条件失败。

## 历史复用检查

没有发现可复用的完整 R4-tail `KxV` persistent-state/dual-dot 候选。

唯一相近记录是
`compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_three_way_recurrence_update_report.md`。
它的 MFMA32 update 产生 `K[64,64] @ V[64,32] -> [64,32]`，随后经显式
transpose/add 回 `H1/H2[V32,K128]`。也就是说它仍把 persistent feedback
保留在 VxK，正是本候选不能复用的转换，且不具备 R4-tail 的完整 nonzero-W
pred/update recurrence。

## 实现

新增的 intrinsic
`block_dot_bf16_f32_staged_vdecay_preloaded_k_state_kv` 只给既有
staged-vdecay/preloaded-K block-dot 加上 `avelang.block_dot.state_kv` 语义。
在 `lower_qwen_block_dot_pass.cc` 的 R4 preloaded-K retile 分支中，它只交换
MFMA A/B operand roles；没有复制 MFMA、增加 barrier、增加 LDS bank 或使用
private ring。

候选内的 loop-carried FP32 fragments 是 KxV：lane 标识 K row，16 个
accumulator slots 标识 V32 columns。update 是

```text
K^T[K,T] @ v_decay[T,V] -> KxV
```

并将结果直接反馈到 KxV FP32 fragments。H/final-state 仍以原 ABI `[V,K]`
写出。为了保留 R4 已验证的 pred MFMA dot-operand encoding，pred 前在原有
同尺寸 8 KiB LDS allocation 中实体化一次 BF16 VxK operand view；它不是
第二个 feedback bank，但也正是 U ownership 没有改变的原因。

相关文件：

- `repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv.py`
- `run_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv_body.py`
- `lib/IR/Intrinsics/amdgpu_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`

## 正确性

在 ljd ROCm 7.2.2 / MI300X(gfx942) 容器、full nonzero-W 输入上运行：

| T | P2 microscope | h/pred/v_new/v_decay/state/final_state |
|---:|---|---|
| 64 | pass | 全部 byte-equal |
| 128 | pass | 全部 byte-equal |

相对 device-contract 的非 byte-equal 项仅为 MFMA FP32 归约舍入：T=128 的
`pred_f32 <= 9.31e-10`、`state_after/final_state <= 5.59e-09`；BF16
`v_new` 和 `v_decay` 仍 byte-equal。

## 最终 ISA 审计

生产 code object metadata：

| 项目 | 值 |
|---|---:|
| MFMA32 静态数 | 48 |
| `global_load_dwordx4` | 32（W/K packet 路径；不是 U） |
| `global_load_ushort` | 16 |
| `global_store_short_d16_hi` | 16 |
| `global_store_short` | 80 |
| `ds_bpermute` | 0 |
| VGPR | 332 |
| AccVGPR | 76 |
| SGPR | 37 |
| LDS | 53,248 B |
| private/scratch | 0 B |
| VGPR/SGPR spill | 0/0 |

U/v_new 的首个静态对为 ISA `0x4254` 的 `global_load_ushort` 与 `0x42ac`
的 `global_store_short_d16_hi`；相同模式在 `0x4378..0x4a00` 和
`0x4bc4..0x54e4` 展开。它们不是宽 packet。

候选的 dual-dot 不是 IR-only：final ISA 的 update MFMA cluster（例如
`0x560c` 起）A/B VGPR operands 已与原 R4 `K,V-decay` order 交换，且编译
日志继续报告 `mode=specialized operand=persistent_typed_block`。不过这项真实
变化不能弥补 U/v_new 未变宽，且 VGPR 从 R4-tail production 的 128 增至 332，
资源方向也不成立。

ISA 中仍存在 exec-mask 控制（包括 `0x5bd4 -> 0x1148` 的 backward
`s_cbranch_execz`，以及尾部的 exec save/restore）；U/v_new 窄访问本身已足够
否决，不把这些控制流归咎为宽访问失败的唯一原因。

## 决策

本候选证明“只改变 persistent FP32 state/update orientation”不足以改变
pred 输出到 U/v_new 的 lane ownership；它会留下窄 scalar I/O，且引入严重
VGPR 压力。不要继续沿此 state-KV 变体做 PMC、Eager benchmark 或更多 layout
微调。

若开始下一轮，应关闭这一 I/O-orientation 路线，转向已量化的
update operand preparation VALU 超额路径；前提是先提出能同时保持 R4
specialized dot-operand encoding 和 resource-cliff gate 的同源实现。

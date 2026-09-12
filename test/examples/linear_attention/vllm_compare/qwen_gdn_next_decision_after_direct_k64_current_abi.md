# Qwen GDN Next Decision After Direct-K64 Current-ABI Repro

## Decision

`direct-K64` 在 current-vLLM BF16 storage ABI 下通过 update-suffix correctness，且
`Scratch=0`、VGPR/SGPR spill 为零。它证明 broad K 不是 direct-K 的唯一可行形式，
也证明 v29 的 `AccVGPR=384` / `736 B` 不是 BF16 direct-K 本身必然产生的资源 cliff。

但它不晋级为 recurrence 或 full-GDN candidate：当前可验证 Avelang source 是
`V64 x K128` CTA、显式 32x32 LDS stage/MFMA loop；原生 Triton 是 `V32 x K128` CTA
和 `tt.dot(64x64, 64x32)`。两者没有共享同一 high-level IR 或 scheduler。

## Do Not Infer

- 不要将该 isolated body latency 与 native recurrence 或 eager public API 相减。
- 不要声称单一 LDS/vector load lowering 已被证明是 v29 overflow 根因。
- 不要恢复 broad-K 或把 direct-K repro 接入 full v29。

已有 same-source late-B-fragment A/B 的结论仍有效：两条最终路径收敛到相同 LLVM/ISA，
没有展示最后一个 B load 的 lowering 能解决 full live-region pressure。

## Next Single Action

实现一个 block-operand / block-dot compiler feature gate，而不是再做 source tile sweep。
它必须让下面的同一 high-level op 保持到 AMDGPU lowering：

```python
block_dot_bf16(K[64,64], V_decay[64,32], state[32,64])
```

然后固定同一 source、same current ABI、CTA ownership、barrier、shared allocation 与
MFMA schedule，只比较 generic 和 specialized lowering。只有这个 A/B 才能判定性能/资源
差异是否由 Avelang lowering 导致。

## Evidence

- [direct-K64 report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_current_abi_alignment_report.md)
- [direct-K64 source](repro_qwen_gdn_direct_k64_update_current_abi.py)
- [same-source late-B-load A/B](../compile_bug/qwen_mfma32_lowering_ladder/qwen_kfrag_same_source_lowering_ab_report.md)
- [v29 LTO spill audit](../compile_bug/qwen_mfma32_lowering_ladder/qwen_v29_full_mir_and_pred_streaming_report.md)

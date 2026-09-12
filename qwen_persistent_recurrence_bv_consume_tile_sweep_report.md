# Qwen persistent recurrence：BV-consume 融合输出 tile 扫描

基线为 `gfx942_bt64_bv32_joint_v4_tail_issue`。本轮完成了 `BV_consume={16,32,64}` 的完整 nonzero-W recurrence 候选、正确性、代码生成制品、PMC 与 fresh-process body benchmark。结论是：**三个候选中 BV32 最快，但 T=2048/8192 与 R4-tail 的差异仅 -0.15%/-0.59%，不足以提升基线；保留 R4-tail/BV32。** BV16 和 BV64 分别慢约 14.7% 与 170.8%（T=8192）。

## 实现范围

没有改写 R4 数学骨架、next-W/K issue 点或 LDS bank：仍为一个 first-class persistent recurrence loop、BT64、单 bank 53,248B LDS、typed BF16x8 W/K producer、LDS-mediated K retile、BF16 `v_new/v_decay` 边界、FP32 loop-carried state feedback 与 tail issue/commit。

`qwen_persistent_recurrence_pass.cc` 为三个 lowering mode 建立显式计划属性，而非只改变 launch metadata：

| logical BV | `physical_mfma_bv` | physical V32 consumer | ownership |
| ---: | ---: | ---: | --- |
| 16 | 32 | 1，padded/masked | R4 两 wave；inactive V16 lanes 掩码 |
| 32 | 32 | 1 | R4 两 wave |
| 64 | 32 | 2，serial 于同一 recurrence iteration | 两个 BV32 pair、4 wave、256 threads |

因此 BV16 不是可缩小到 MFMA16 的 accumulator experiment（硬件 MFMA geometry 被题目固定为 32x32x8）；它是一个真实编译/执行不同、但物理 V32 padded 的 logical-BV16 对照。BV64 则在同一完整 recurrence core 中顺序执行两个 V32 subtiles；每个 subtile 都遵循 `pred K=128 -> corrected -> BF16 v_new/v_decay -> K update -> FP32 feedback`，之后才进入下一个 V32 subtile。没有 private packet ring、第二 LDS、software pipeline 或 distance=2。

MLIR 证据位于各 `post_recurrence_joint_planner.mlir`：三个计划分别含 `bv_consume`、`physical_mfma_bv=32`、mapping；BV64 额外含 `physical_v32_subtiles_per_recurrence_iteration=2`、`waves=4` 与 `workgroup_size=256`。它们均有一个 `ave.gpu.amdgpu_qwen_persistent_recurrence` region，且 `next_w_before_pred`/`next_k_after_pred` 仍是 `tail_issue_control`；故没有把候选误变成分离 kernel 或 software-pipeline 核。

## 正确性和制品

构建成功：`ninja -C build-software-pipeline python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`。

三个候选均在 T=64/128/512/2048 的 full nonzero-W P2 通过：`h`、pred FP32、pred BF16、`v_new`、`v_decay`、每 chunk state 和 final state 相对于 host P2 microscope 都 byte-equal；device contract 也在既有容差内通过。

T=2048 制品已导出到 `test/examples/linear_attention/vllm_compare/bv_consume_artifacts_t2048/`：每个 BV 目录有 planner/MLIR、pre/post LLVM、link argv/prelink bitcode、HSACO、ISA、`exact_lto_final_isel.mir` 及 HSA readobj；PMC CSV 在 `pmc/`。三份 final-isel MIR 都是 `stack: []`，没有 `SI_SPILL_AV32_SAVE`/`SI_SPILL_AV64_SAVE`。HSA metadata 均为 private segment 0、VGPR/SGPR spill 0、LDS 53,248B。

## T=2048 PMC（每 kernel 8 次采样中位数）

| 版本 | WG | VGPR | AccVGPR | SGPR | scratch/LDS | MFMA | VALU | VMEM | LDS | raw OccupancyPercent |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| R4-tail BV32 | 128 | 128 | 160 | 112 | 0 / 53248 | 65536 | 2239872 | 202240 | 381952 | 0.6376 |
| BV16 | 128 | 96 | 168 | 112 | 0 / 53248 | 131072 | 4771584 | 407552 | 935936 | 1.2746 |
| BV32 | 128 | 128 | 160 | 112 | 0 / 53248 | 65536 | 2239936 | 202240 | 381952 | 0.6354 |
| BV64 | 256 | 52 | 212 | 112 | 0 / 53248 | 98304 | 3227968 | 187392 | 475648 | 0.6433 |

`OccupancyPercent` 是 rocprof 原始字段；BV16 的 grid 是两倍，因而不应把它与单 dispatch 的静态 residency 混为一谈。资源 cliff 没有由 spill/scratch 触发，但 BV64 的 AccVGPR 从 160 增至 212，且动态 MFMA/VALU/LDS 上升；这已经解释其不能作为融合输出 tile 的性能候选。

ISA 也确实不同而不只是 plan 属性：静态 `MFMA/barrier/waitcnt/VMEM-load/LDS/acc-read+write` 分别为 R4-tail `48/12/136/75/302/128`、BV16 `40/10/102/83/308/128`、BV32 `40/10/106/59/267/128`、BV64 `48/12/90/51/264/128`。所有候选 scratch 指令为零；accumulator 到 VGPR 的显式 read/write 数均为 128，未发现新增 register-class copy。动态 PMC 而非静态条数说明了 BV16/BV64 的实际工作量回退。

## Fresh-process body benchmark

协议：每个 `(implementation,T,session)` 新解释器与 source-mode cache 隔离；HIP events；无 graph capture；分配和编译均不计时；5 warmup、20 repeat、2 session，取 session median 的 median。

| 版本 | T=1024 ms | T=2048 ms | T=8192 ms | T=8192 / R4-tail | 1024->8192 ms/chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4-tail BV32 | 0.214428 | 0.378001 | 1.410125 | 1.000000 | 0.01067586 |
| BV16 | 0.237693 | 0.433594 | 1.617833 | 1.147298 | 0.01232268 |
| BV32 | 0.207989 | 0.377431 | 1.401803 | 0.994098 | 0.01065905 |
| BV64 | 0.509586 | 0.982560 | 3.819003 | 2.708273 | 0.02954836 |

BV32 是 sweep 内最快项，且 PMC 与 R4-tail 基本相同；长序列斜率差只有约 0.16%。两 session 不足以将这个差异作为可推广收益，故不替换 R4-tail。BV16 的 physical-V32 padding 使 CTA 数、MFMA、VMEM 与 LDS 增加；BV64 虽减少部分 VMEM，却以更多 AccVGPR、MFMA、VALU 和 LDS 换取，造成稳定的大幅回退。

## 结论

本轮已证明 BV 参数穿透到 first-class plan、四 wave ownership、MLIR、final-isel MIR、ISA 与 PMC，而不是 metadata-only；同时也给出物理限制：固定 MFMA32 下，BV16 只能 padded/masked，BV64 的双 V32 fusion 没有降低总 core 成本。性能上应继续以 R4-tail/BV32 为基线；本轮不扩大搜索、不做 distance=2，也不引入 software pipeline。

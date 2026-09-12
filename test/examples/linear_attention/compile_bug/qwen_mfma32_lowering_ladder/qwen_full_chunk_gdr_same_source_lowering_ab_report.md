# Full Chunk_GDR Same-Source Generic-vs-Late-B-Fragment Lowering A/B

## 结论

这次已经完成了所要求的 **exact full `chunk_gdr`** A/B，而不是此前的
reduced/full-like repro。

被测 kernel 是同一份已有 full-v29 源码中的：

```text
_qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32
```

它包含真实的 MFMA32 pred、`pred_partial`、`v_decay_t_bf16`、state recurrence、
`h` 写回、final-state 写回、32 个 chunk 和完整的 update MFMA16 路径。

两次编译共享完全相同的高级源码与 `pre_kfrag_branch.mlir`。唯一分叉发生在
该快照之后：

```text
A: persistent B-fragment consumer -> generic view/vector lowering
B: persistent B-fragment consumer -> late dedicated LDS B-fragment lowering
```

结果是一个明确的 **negative control**：B 的 dedicated op 在中后端被
canonicalize 为 A 的等价实现。两侧 post-opt LLVM SHA256 完全相同，规范化
ISA 完全相同，物理寄存器、scratch、动态 PMC、high-AGPR copy 计数完全相同，
输出也 bit-exact。

所以可以严格排除的狭义假设是：

```text
full chunk_gdr 的 AccVGPR=384 / scratch=736 B，主要由最后一个
generic vector B-load 而不是 direct LDS B-load 的选择直接造成。
```

不能据此得出“所有 v29 问题是高级代码问题”或“所有问题都不是 Avelang
lowering”的结论。它只能说明：**当前这一个 late dedicated B-load lowering
没有在 final machine code 中保留不同实现，因而不可能单独解释或修复 full
resource cliff。** 根因仍可能位于更早的 producer/shared-buffer/consumer
materialization 或 full pred+update live-set composition；这些都属于更宽的
compiler/lowering graph 问题或与 schedule 共同作用的问题，不能由本 A/B 分离。

## 为什么这不是重复实验

开始前审计了已有证据：

- `qwen_kfrag_same_source_lowering_ab_report.md` 是 R3 reduced full-like
  repro，不是完整 `chunk_gdr`。
- `qwen_late_bfrag_lowering_report.md` 是更小的 reduced repro。
- `qwen_full_kfrag_rewrite_regression_root_cause_report.md` 比较的是不同
  full-v29 source/rewrite 图，未满足 same-source A/B。

因此此前没有一个实验同时满足：同一 **full** kernel、同一 full schedule、
同一 launch、同一 frozen inputs，以及仅在 late B-fragment consumer lowering
处分叉。本报告补齐该缺口。

## 冻结契约

| 项目 | A generic | B late dedicated | 约束 |
|:--|:--|:--|:--|
| Python/AveLang kernel source | 同一文件、同一函数 | 同一文件、同一函数 | 不改源码 |
| T / chunks | 2048 / 32 | 2048 / 32 | 相同 |
| grid / workgroup | `(32,1,1)` / `(128,1,1)` | 相同 | 相同 |
| `num_warps` | 2 | 2 | 相同 |
| 输入 | 同一落盘 bundle | 同一落盘 bundle | SHA256 相同 |
| 逻辑阶段 | pred, correction, K staging, update, recurrence, h/final-state | 相同 | 相同 |
| MFMA schedule | 32x32x8 pred + 16x16x16 update | 相同 | 相同 |
| shared allocation / lifetime | 相同源码表达 | 相同源码表达 | 不修改 |
| 唯一环境变量 | `AVELANG_QWEN_KFRAG_LATE_BLOAD=0` | `=1` | pre-branch 后分叉 |

冻结输入文件为 `frozen_inputs.pt`：

```text
SHA256 19d24ed32ab20611972c11196e2d30e4850259f83fa6d97aab19e71df5460e00
```

## IR 分叉审计

`pre_kfrag_branch.mlir` 是专门在分叉前落盘的 MLIR。两侧完全相同：

```text
7fd1f760121f1d391ce9a380a3807fc871ce32e87f8cac4d811b1c7a0e17726d
```

其中有 4 个 persistent `amdgpu_qwen_update_kfrag_load` consumer。分叉后
两条 MLIR 当然不同：A 出现 generic vector load，B 保留 4 个 dedicated
`qwen_kfrag.direct_lds_b64`。但两侧 alloca 数都为 59。

| 审计点 | A generic | B dedicated | 含义 |
|:--|--:|--:|:--|
| pre-branch persistent op | 4 | 4 | 相同高层 producer/consumer |
| generic `vector.load` | 16 | - | A consumer 表达 |
| `qwen_kfrag.direct_lds_b64` | - | 4 | B consumer 表达 |
| alloca | 59 | 59 | 无额外临时 alloca |
| preopt LLVM | 不同 | 不同 | 预期：consumer lowering 仍不同 |
| postopt LLVM SHA256 | `21512384...ab89c6` | `21512384...ab89c6` | 完全相同 |

这里的关键不是“B op 没生成”，而是它在 LLVM 优化后不再保留与 A 可区分的
machine-level 形态。因此这是对该 lowering 实现的有效审计，而不是没有执行 B。

## 语义与静态 ISA Gate

两侧都报告 persistent rewrite fired。以冻结输入比较 full 输出：

| 输出 | max abs | mismatch | 结果 |
|:--|--:|--:|:--|
| `h` FP32 | 0 | 0 | bit-exact |
| `final_state` FP32 | 0 | 0 | bit-exact |

完整 full-v29 对外部 reference 的 nonzero-W recurrence 问题仍是既有问题；
本实验只验证 A 与 B 是否严格保持 **同一 full-v29 语义**，不能把该既有问题
归因于 B-fragment lowering。

最终 ISA 去除 objdump 输入路径行后 SHA256 完全相同：

```text
58dccce7226de341a8318d80d34eeb39834c207e6033b466baaefaa972eb56a3
```

| ISA 静态指标 | A | B |
|:--|--:|--:|
| MFMA16 | 128 | 128 |
| MFMA32 | 16 | 16 |
| `ds_read_b64` | 256 | 256 |
| `ds_write*` | 243 | 243 |
| global/buffer load | 195 | 195 |
| `s_barrier` | 17 | 17 |
| `v_accvgpr_write_b32` | 274 | 274 |
| high AGPR writes, index >=100 | 156 | 156 |
| max explicit AGPR write | a255 | a255 |

这同时冻结了用户指定的 MFMA schedule、LDS/global 工作量、barrier、launch
以及 high-AGPR copy 现象。

## T=2048 性能和资源

正常 HIP-event smoke 的 30-repeat median 是 1.325991 ms（A）与 1.323668 ms
（B），差 2.323 us，约 0.18%。两侧独立进程且当前 final ISA 完全相同，不能把
这种量级解释为 lowering 收益。

rocprof 每侧包含 8 个 matching dispatch；trace median 仅相差 3.1645 us
（B 快约 0.24%），同样不能在完全相同 ISA 与资源下解释为因果差异。

| metric | A generic | B dedicated | delta B-A |
|:--|--:|--:|--:|
| normal median | 1.325991 ms | 1.323668 ms | -2.323 us |
| trace median | 1304.8395 us | 1301.6750 us | -3.1645 us |
| WG / grid work-items | 128 / 4096 | 128 / 4096 | 0 |
| VGPR | 128 | 128 | 0 |
| AccVGPR | 384 | 384 | 0 |
| SGPR | 112 | 112 | 0 |
| scratch | 736 B | 736 B | 0 |
| LDS block | 61440 B | 61440 B | 0 |
| occupancy | 0.642270% | 0.643355% | collector variation |
| SQ_INSTS_MFMA | 294912 | 294912 | 0 |
| SQ_INSTS_VALU | 3180992 | 3180992 | 0 |
| SQ_INSTS_SALU | 567808 | 567808 | 0 |
| SQ_INSTS_VMEM | 601984 | 601984 | 0 |
| SQ_INSTS_LDS | 1242304 | 1242304 | 0 |

## 对编译器团队的可辩护表述

可以说：

1. 同一 full `chunk_gdr` 高层源码和 pre-branch AveLang MLIR 下，current
   generic 与 late dedicated consumer lowering 都能得到 bit-exact 输出。
2. B 的 dedicated op 确实存活到 late lowering，但 current lowerer 最终把它
   canonicalize 成与 A 相同的 postopt LLVM 和 ISA。
3. 因此最终 load instruction selection **不是** full-v29 `AccVGPR=384`、
   `Scratch=736 B` 的充分解释；替换它不会改变寄存器分配、spill 或 trace。
4. full cliff 的直接 machine evidence 仍在：a100..a255 高 AGPR copy、384
   AccVGPR 和 190 VGPR spill words/736 B private segment（见
   `qwen_v29_full_mir_and_pred_streaming_report.md`）。本实验说明仅动最后
   B-load 无法消除这套组合压力。

不能说：

```text
已经证明 Avelang lowering 是 full-v29 问题的唯一根因。
```

因为本实验让两个 lowerer 收敛为同一最终 codegen；它排除的是一个窄假设，
不能把 full pred/update schedule 和更早的 lowering/materialization 逐一归因。

## 下一步

不要继续在这个 final B-load 上做 source 或 RA 调参。若要为“更早 lowering
而非高级 schedule”取得更强正向证据，下一次实验必须保持本报告的 same-source
gate，但使 B 的 dedicated representation **有意存活至 post-LTO MIR/ISA**，且
final ISA 只在 B-fragment load 序列不同。只有那时，若 AGPR/spill/trace 随之
显著变化，才可以提出强的 lowering 因果结论。

在当前结果下，最严谨的结论是：B-fragment final-load lowering 不是可单独
修复 full `chunk_gdr` cliff 的位置。

## 文件与复现

驱动：

- `audit_qwen_full_chunk_gdr_kfrag_lowering_ab.py`
- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp.py`

产物目录：

```text
test/examples/linear_attention/rocprof_outputs/
  qwen_full_chunk_gdr_kfrag_lowering_ab/
```

核心证据：

- `A_generic/ir/pre_kfrag_branch.mlir`
- `B_specialized/ir/pre_kfrag_branch.mlir`
- `A_generic/ir/postopt_llvm.ll`
- `B_specialized/ir/postopt_llvm.ll`
- `A_generic/link/amdgpu-link-0.linked.isa`
- `B_specialized/link/amdgpu-link-0.linked.isa`
- `A_generic/rocprof/A_generic_counter_collection.csv`
- `B_specialized/rocprof/B_specialized_counter_collection.csv`
- `summary.json`

复现命令：

```bash
cd /workspace/project/avelang
PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
PYTHONDONTWRITEBYTECODE=1 \
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
audit_qwen_full_chunk_gdr_kfrag_lowering_ab.py \
  --T 2048 --warmup 10 --repeat 30 --rocprof-warmup 2 --rocprof-repeat 5
```

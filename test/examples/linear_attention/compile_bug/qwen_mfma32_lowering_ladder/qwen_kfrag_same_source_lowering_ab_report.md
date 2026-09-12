# Qwen K-Fragment Same-Source Lowering A/B Audit

## 结论

这是一条严格的反向证据，而不是“lowering 已被证明为唯一根因”的证据。

在同一份 AveLang 高级源码、同一份 pre-branch MLIR、同一份冻结输入、同一
tile ownership、同一 shared allocation、同一控制流和同一数学结果下，generic
与 specialized 两个 B-fragment lowering 分支在 LLVM 优化后收敛成完全相同的
LLVM IR 和相同的反汇编指令流。rocprof 的寄存器、scratch 和动态指令计数也
完全相同。

因此，本实验明确排除下面这个狭义假设：

    full-v29/reduced-repro 的压力主要由最后一条 vector.load versus direct
    LDS load 的选指差异造成。

它没有排除更宽泛的命题：producer-consumer rewrite 所形成的完整
producer/shared-buffer/consumer/live-range 图，或其在更早的
AveLang/MLIR lowering 中的 materialization，可能造成 full-v29 的
Accum_VGPR=384 和 736 B scratch。那是另一个更大的 A/B 问题，不能用本结果
偷换结论。

## 实验契约

被测源码只有一个：repro_qwen_kfrag_full_loop_regression.py。

被测 variant：R3_rewrite_plus_state_update_writeback。它保留真实 MFMA32
pred、pred_partial、v_decay、共享 state 和 MFMA16 update 的组合 live region。
两个运行都使用同一份已落盘的 CPU 输入 bundle：

    SHA256 0a28dba70a1a15b0fd9bb508b5c423cc45fa083e61bb6daaf5ae1342ea99be14

两个编译运行只在下列环境变量不同：

    AVELANG_QWEN_KFRAG_LATE_BLOAD=0  A: generic
    AVELANG_QWEN_KFRAG_LATE_BLOAD=1  B: specialized

分叉在 pre_kfrag_branch.mlir 落盘之后才发生。该文件两侧 SHA256 完全相同：

    845c7e95c8ba6262d2a85dcde494972391fd80827bf64c0d44a32df385380ec5

在这个点之前，源码、AveLang IR、producer、shared allocation、tile ownership、
global K load、LDS store、MFMA、barrier 和控制流完全相同。

## 两条分支

高层 persistent op 相同：

    frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(...)
    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a, frag, acc)

A 的 rewrite 将 consumer 变为 generic vector.load。

B 保留 dedicated amdgpu_qwen_update_kfrag_lds_load 到 late pass；late pass 用
addrspace(3) BF16 element-GEP 加 vector-of-four BF16 LLVM load 表达 consumer。
它不再使用早期版本的裸 byte-address arithmetic、inttoptr 加字节偏移或强制
align 8。这个调整是为了让 B 与 generic 的 BF16 元素地址语义完全等价，而不是
引入第二个高层变化。

## Repro 有效性修复

首次执行发现旧 diagnostic repro 本身不适合作为 exact gate：

1. 两个 wave 曾写入同一个 sink 坐标。
2. 同一 wave 的 16 个 lane_col 也曾竞争同一个 sink 坐标。
3. 每个 wave 只初始化了自己的 [32,64] pred staging tile 的一半。
4. state 初始化和跨-wave pred_partial 消费缺少必要同步。

这些会使 generic 对 generic 的同输入重复运行也产生不同输出，不能归因于
compiler。修复后：

- sink 使用 wave_id/local_tile/lane_group/lane_col/r 的唯一坐标；
- 每个 wave 完整填充其 [32,64] pred tile；
- 增加 state producer-to-pred 和 pred-partial-to-v-decay 的必要 barrier；
- 两个 A/B 分支使用同一修复后的源码。

这些修复没有改变 A/B 的唯一分叉点。

## Gate 结果

| gate | A generic | B specialized | 结果 |
|---|---:|---:|:---|
| pre-branch MLIR SHA256 | 845c...0ec5 | 845c...0ec5 | 通过 |
| persistent op count before branch | 4 | 4 | 通过 |
| shared alloca count after lowering | 32 | 32 | 通过 |
| frozen input checksums | 相同 | 相同 | 通过 |
| sink exact comparison | - | max abs 0, mismatch 0 | 通过 |
| static MFMA16 | 64 | 64 | 通过 |
| static MFMA32 | 8 | 8 | 通过 |
| static ds_read_b64 | 128 | 128 | 通过 |
| static ds_write | 187 | 187 | 通过 |
| static global load | 146 | 146 | 通过 |
| static barrier | 5 | 5 | 通过 |

post_kfrag_rewrite.mlir 和 preopt_llvm.ll 不同，这是预期的分叉。但
postopt_llvm.ll 的 SHA256 完全相同：

    ff9d9951c92792873e88b4bdd93f544e80008d7c51c3653a3961887249448421

两个 ISA 文本文件的原始 SHA256 因 objdump 第一行输入路径不同而不同。删去
该路径行后反汇编完全一致；diff -u 只显示该文件路径行。

## T=2048 rocprof

每侧都是相同 R3 kernel、相同 frozen input、WG=128、grid=4096，并收集
MFMA、VALU、SALU、VMEM、LDS 和 occupancy。

| metric | A generic | B specialized | 差异 |
|---|---:|---:|---:|
| trace median | 738.978 us | 743.024 us | +4.046 us |
| VGPR | 116 | 116 | 0 |
| AccVGPR | 148 | 148 | 0 |
| SGPR | 112 | 112 | 0 |
| scratch | 0 B | 0 B | 0 |
| LDS block | 61440 B | 61440 B | 0 |
| occupancy | 0.642285% | 0.642390% | collector noise |
| MFMA | 147456 | 147456 | 0 |
| VALU | 1471488 | 1471488 | 0 |
| SALU | 282944 | 282944 | 0 |
| VMEM | 239616 | 239616 | 0 |
| LDS inst | 705728 | 705728 | 0 |

两侧是独立 rocprof process；在最终 LLVM、ISA 和所有动态 PMC 相同的前提下，
4.046 us 不能解释为 lowering 性能差异。

## 对编译器团队能说什么

可以高置信度说：

1. current late dedicated B-fragment op 在这个测试中最终被 canonicalize 为与
   generic load 等价的 LLVM/ISA。
2. 因而“把最后的 generic vector load 换成 direct LDS load”本身不会解决该真实
   pred/update live-region 的 AccVGPR 压力。
3. 过去 generic 128/144/0.5292 ms versus direct 124/148/0.5634 ms 的
   reduced-repro 结果不应解读成 direct LDS lowering 已证明更差；它没有满足
   同输入、无 race、same-work 和 post-LTO equivalence gate。
4. full-v29 的 AccVGPR=384 / 736 B scratch 仍更符合 producer-consumer graph 与
   组合 live set 的问题，而非一条 consumer load 指令的最终选择。

不能说：

    本实验已经证明所有 full-v29 性能问题都是 compiler lowering bug。

因为本实验的 A/B 在后端优化后没有保留两个不同的最终实现。它是一个有效
negative control，不是产生 resource delta 的 positive causal test。

## 如果需要接近铁证的下一步

下一次 A/B 必须仍满足本报告的全部冻结条件，但还要增加一个额外 gate：

    post-LTO final ISA/MIR 除 B-fragment consumer load sequence 外必须不同，
    其余 body 必须相同。

即专用 op 不能在 LLVM 优化前被 canonicalize 掉；它必须以一个受约束、但语义
完全等价的 LDS-load representation 留到 AMDGPU instruction selection。随后才
比较 physical AGPR allocation、live intervals、spill 和 trace。

若这种真正不同的 final lowering 在 full-like repro 中令 AccVGPR/scratch 明显
下降，同时所有高层 gate 都相同，才可以向编译器团队提交强正向因果证据。若它
仍无收益，则应停止追逐最后一条 B-load，转而审计 producer materialization、
跨 phase live range 和 full recurrence composition。

## 产物和复现

驱动：

- audit_qwen_kfrag_same_source_lowering_ab.py
- repro_qwen_kfrag_full_loop_regression.py

compiler audit hook：

- lib/Target/GPU/lower_to_llvm.cc
- lib/Dialect/AveLang/Transforms/lower_qwen_kfrag_lds_pass.cc

artifact root：

    test/examples/linear_attention/rocprof_outputs/qwen_kfrag_same_source_lowering_ab/

关键 artifact：

    A_generic/ir/pre_kfrag_branch.mlir
    B_specialized/ir/pre_kfrag_branch.mlir
    A_generic/ir/preopt_llvm.ll
    B_specialized/ir/preopt_llvm.ll
    A_generic/ir/postopt_llvm.ll
    B_specialized/ir/postopt_llvm.ll
    A_generic/link/amdgpu-link-0.linked.isa
    B_specialized/link/amdgpu-link-0.linked.isa
    A_generic/rocprof/A_generic_kernel_trace.csv
    B_specialized/rocprof/B_specialized_kernel_trace.csv
    frozen_inputs.pt

基础 smoke：

    cd /workspace/project/avelang
    PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python
    PYTHONDONTWRITEBYTECODE=1
    python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_kfrag_same_source_lowering_ab.py
      --warmup 5 --repeat 20 --skip-rocprof

# Stage 6Z：通用 `block_dot_bf16_f32` Full-Scope Generalization

## 结论

本轮完成了一个可编译、可运行、可审计的通用 full-scope block-dot 候选，但
没有形成新的性能 baseline。

| arm | T=2048 body | T=8192 body | 相对 Z5B | 结论 |
|:--|--:|--:|--:|:--|
| Z5B | `0.067641 ms` | `0.157294 ms` | baseline | 当前 isolated baseline |
| BDV2-G generic | `0.069043 ms` | `0.166847 ms` | `+2.07% / +6.07%` | No-Go |
| BDV2-S specialized | `0.068962 ms` | `0.167048 ms` | `+1.95% / +6.20%` | No-Go |
| native selected chunk-o | `0.042744 ms` | `0.090875 ms` | diagnostic | 未接入 |

BDV2 的价值在编译器表示和证据链，而不是本轮 latency：

1. K/H 使用同一个 target-independent logical block-dot API；没有新增 Qwen
   专用 intrinsic，也没有把 K/H 拆成两个 planner。
2. generic 和 specialized 共享同一个 BDV2 source，source SHA256 相同。
3. specialized 的差异穿过 lowered LLVM、pre-LTO AMDGCN、exact-LTO MIR、
   final ISA 和 HSACO；不再是 Z7B 那种两臂最后得到同一个 HSACO 的实验。
4. correctness 在 `T=64/512/1024/2048/4096/8192/16384` 以及
   caller-owned zero-V/NaN-prefill cases 全部通过。
5. T=2048 PMC 显示 BDV2 两臂每 CTA 的 MFMA 仍为 `160`，VMEM 从 Z5B 的
   `672` 降为 `448`，LDS 从 `672` 降为 `464`；但 VALU 从 `7072` 增至
   `8410`，latency 没有改善，长文本反而退化。

最终判断：**full-scope 表示和 late lowering 可以落地，但当前
producer-layout-consumer 实现仍不是 native-style path。Z5B 继续是 Stage 6Z
isolated baseline；BDV2 只保留为编译器实验工件，不接入 X2 或 production。**

---

## 1. 实验问题与冻结边界

### 1.1 从 Z7B 继续的原因

Z7B 已经把 Z5B 中的 `Q@H.T` 和 `Q@K.T` 换成同一个高层
`block_dot_bf16_f32` logical block-dot contract。Z7B correctness 通过，lowered
LLVM 有 generic/specialized 差异，但 final HSACO 相同，说明表示在 machine
lowering 之前收敛。

本轮需要一个更完整、可复用的表示，让 op 在同一个 lowering 边界同时拥有：

```text
logical global B block
  -> producer ownership
  -> typed BF16 packet
  -> physical shared placement
  -> dot operand
  -> MFMA32 consumer
```

### 1.2 冻结内容

- gfx942 / wave64；BT64、BV64、BK32；WG256、2 CTA/chunk-head；
- BF16 Q/K/H/V-new，FP32 g/output accumulator；
- dedicated full-Q LDS cache，约 16 KiB；总 shared allocation 约 32 KiB；
- MFMA32 geometry、K32 accumulation order、causal mask；
- phase-separated `inter_acc -> score_half0 -> score_half1 -> intra_acc`；
- global layout、caller-owned BF16 output、V-new/g/output producer；
- allocator/RA、X2、production selector、external HSACO。

BDV2 只改变 K/H 的 producer 表达和 late lowering 入口。

---

## 2. 通用 full-scope 表示

### 2.1 API

BDV2 复用既有 `AMDGPUBlockDotBF16F32Op`。新增两个 source-facing helper：

```python
al.amdgpu.block_dot_bf16_f32_logical(...)
al.amdgpu.block_dot_bf16_f32_logical_transposed(...)
```

两者都创建同一个 `ave.gpu.amdgpu_block_dot_bf16_f32` op，只添加 generic
metadata；不是新的硬件指令，也不是 Qwen/chunk-o 专用 intrinsic。

### 2.2 metadata contract

| 字段 | 语义 |
|:--|:--|
| `operand_mode=full_scope` | producer/consumer 由一个 lowering scope 管理 |
| `scope=logical_block` | op 表达 logical block，不是已展开 fragment |
| `operand_role=B` | block 作为 MFMA B operand |
| `source_role=K/H` | K 或 H 是 logical source role |
| `transpose=none/rhs_transposed` | logical transpose 关系 |
| `logical_shape=32x32x32` | logical M/N/K identity |
| `logical_source_operand=source_block` | 第三个 operand 是 logical global source |
| `lhs_residency=existing_shared` | A operand 是已有 resident shared block |
| `rhs_producer=global_packet` | producer 由 target lowering 选择 |
| `reuse_key=lhs_ssa_block` | A block 的 reuse identity |
| `layout_intent=typed_shared_dot` | 允许目标选择 physical layout |

字段只描述 logical shape、role、transpose、residency 和 reuse；没有在 Qwen
pass 中硬编码 4096 个元素地址。

### 2.3 K/H 统一 planner

BDV2 source 中的调用是：

```python
inter_pair = al.amdgpu.block_dot_bf16_f32_logical_transposed(
    q_cache, phase, h, ...
)
score_pair = al.amdgpu.block_dot_bf16_f32_logical(
    q_cache, phase, k, ...
)
```

K 和 H 都进入 `emitFullScopeProducer`，最后都复用
`emitGenericOperandBPair`。K/H 的差别只有 rank/layout 和 transpose metadata，
没有两个专门的 producer pass。

Q cache 仍然是 source 中唯一的 Q global producer：

```text
global Q -> dedicated shared Q cache -> resident A
```

K logical op 由整个 CTA 调用，以便 producer barrier CTA-uniform；真正发出 K
producer/consumer 的 ownership 是 `value_half == 0`，即 wave 0 和 wave 2，保持
Z5B 原来的 value-half ownership。

---

## 3. 代码和 pipeline

| 文件 | 作用 |
|:--|:--|
| `vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py` | BDV2 source 和两臂 launch |
| `vllm_compare/check_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py` | correctness |
| `vllm_compare/bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py` | body benchmark |
| `vllm_compare/profile_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py` | rocprof PMC |
| `vllm_compare/dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py` | LLVM/LTO/MIR/ISA/HSACO |
| `lib/IR/Intrinsics/amdgpu_module.cc` | 旧 API 和 logical helper |
| `lib/Dialect/AveLang/IR/AveLangOps.td` | full-scope op contract |
| `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc` | generic/specialized lowering |

pipeline 为：

```text
logical source -> block-dot op + metadata -> LowerQwenBlockDotPass
  -> memref/vector/SCF/GPU -> LLVM -> ROCm LTO -> MIR/ISA
```

specialized mode 使用 BF16x8 typed global load 和 packed shared store；generic
mode 使用同一 ownership 的 scalar BF16 load/store；两臂使用同一 MFMA-B consumer。

---

## 4. same-source 和 machine convergence 证据

### 4.1 source identity

T=2048 两臂 source SHA256 相同：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

唯一编译选择是：

```text
AVELANG_BLOCK_DOT_LOWERING=generic|specialized
```

### 4.2 initial MLIR 工具限制

当前 Docker binding 的 `get_mlir()` 在 initial MLIR printer 阶段 SIGSEGV。artifact
script 使用 `--skip-initial-mlir`，machine summary 明确写：

```text
initial_mlir = skipped by command line
```

因此不能把 source hash 冒充 pre-branch MLIR hash，也不能声称拿到了 initial
MLIR 的字节级证据。这是本轮的工具限制，报告保留为证据缺口。

### 4.3 分层 hash

| 层次 | generic | specialized |
|:--|:--|:--|
| source | `aee02e0309152...52047e` | 同左 |
| lowered LLVM | `a8d17cb4febf60857a8407fe806da27b6c071ba8a24c9ed595d060ee8b44de65` | `df3dbcceedb48c8fafc56a7889a05356ea6e0e56da1820a208f654e798d5e8f2` |
| pre-LTO AMDGCN | `f78058976f9fce06252b1d0cd812479d6674955aade58fadf915d4f01174051c` | `ff7e54624eb0dd88671bf868a4f4147e943591a634b4389ab38caf60aa3407d6` |
| final ISA | `785ec66447b988dc5b759c37b56bbb54d17cc970f0b1e9fd3d850a5833d8f523` | `a8deb1d3b6a2c57a6cf63ab22978b608c745f9c0e04cf03b876ae5e46c311938` |
| HSACO | `6b2b07fbf707d4e8013db21bf112d38d354a96c74aeb3287bb9e319a74cc2c35` | `d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a` |

本轮 fresh artifact 已经得到不同 HSACO，满足“如果第一轮相同则继续修正”的
要求。

### 4.4 lowered LLVM 差异

对 `lowered_llvm.ll` 做文本审计得到：

| 文本模式 | generic | specialized |
|:--|--:|--:|
| `load <8 x bfloat>` | 4 | 10 |
| `load bfloat` | 50 | 2 |
| `store <8 x bfloat>` | 0 | 2 |
| `store bfloat` | 20 | 4 |

这是 textual IR count，不是 dynamic VMEM transaction count。它证明 typed producer
在 LLVM 文本中没有完全收敛；后面的 static ISA family count 恰好相同，说明
machine pipeline 仍然重排了大量工作。

### 4.5 exact-LTO MIR

两臂都保存 `kernel_section_00.mir` 到 `kernel_section_19.mir`：section 00 是
pre-greedy，01 是 post-greedy，02 是 post-virtregrewriter，08/09 是 later
no-vregs/prologepilog；10-19 是重复 LTO pipeline 的另一 section。

generic pre-greedy section 00 为 4078 行，specialized 为 4072 行。两臂的
`SI_SPILL_AV32_SAVE`、`SI_SPILL_AV64_SAVE` 和 spill virtual register 列表均为
零，exact LTO replay return code 为 0。

### 4.6 static ISA

| static lexical count | generic | specialized |
|:--|--:|--:|
| MFMA32 | 56 | 56 |
| `s_barrier` | 44 | 44 |
| global load family | 140 | 140 |
| global store family | 16 | 16 |
| `ds_read` family | 56 | 56 |
| `ds_write` family | 92 | 92 |

静态数量相同不等于两臂相同：final ISA 字节和 HSACO hash 已不同；也不能用
static count 冒充 dynamic PMC。

---

## 5. correctness gate

### 5.1 ownership 调试过程

初始实现曾将 K producer/consumer 写成 `tid < 128`，只覆盖 wave 0；Z5B 的
value-half ownership 实际是 wave 0 和 wave 2。初始 T=64 generic 出现
`max_abs=0.004642...`，不是可忽略噪声。

修复为：

```text
wave = tid // 64
lane = tid % 64
value_half = wave % 2
K active producer/consumer iff value_half == 0
```

修复后两条 arm 都恢复完整 ownership 和 CTA-uniform barrier，所有 correctness
case 通过。

### 5.2 全长度

| T | generic vs Z5B | specialized vs Z5B | finite |
|--:|:--:|:--:|:--:|
| 64 | BF16 byte-exact | BF16 byte-exact | pass |
| 512 | BF16 byte-exact | BF16 byte-exact | pass |
| 1024 | BF16 byte-exact | BF16 byte-exact | pass |
| 2048 | BF16 byte-exact | BF16 byte-exact | pass |
| 4096 | BF16 byte-exact | BF16 byte-exact | pass |
| 8192 | BF16 byte-exact | BF16 byte-exact | pass |
| 16384 | BF16 byte-exact | BF16 byte-exact | pass |

两臂在 T=64/8192/16384 的 zero-V、NaN-prefilled、caller-owned BF16 output
检查也全部通过，没有放宽容差或跳过错误。

---

## 6. T=2048 dynamic PMC

rocprofv3 include matching kernel，并采集 `SQ_INSTS_MFMA/VALU/SALU/VMEM/LDS`
和 `OccupancyPercent`。profile 的 `Grid_Size=131072`、WG=256，对应 512 个
CTA；下表是原始 counter 除以 512 得到的 per-CTA 结果，绝不是 static ISA 推导。

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | profiler VGPR | profiler AccVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 14.498718% |
| BDV2-G | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.634712% |
| BDV2-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.448873% |

原始 JSON 位于：

```text
codex_qwen_bt64_stage6z_bdv2/bdv2_z5b_T2048_rocprof.json
codex_qwen_bt64_stage6z_bdv2/bdv2_bdv2_generic_T2048_rocprof.json
codex_qwen_bt64_stage6z_bdv2/bdv2_bdv2_specialized_T2048_rocprof.json
```

### 6.1 code object 与 profiler 指标分开

不能把 profiler `Accum_VGPR_Count` 当成 readobj 的 AGPR count。exact HSACO
metadata 为：

| arm | code VGPR | code AGPR | SGPR | LDS | private | spill |
|:--|--:|--:|--:|--:|--:|--:|
| BDV2-G | 136 | 48 | 30 | 32768 B | 0 B | 0 |
| BDV2-S | 132 | 48 | 30 | 32768 B | 0 B | 0 |

profiler 的 `76/100` 和 `88/88` 另属 collector 指标体系，单独保留。

### 6.2 解释

BDV2 的 dynamic MFMA 不变、VMEM/LDS 下降，说明 full-scope producer 没有改变
数学 MFMA 工作，并减少了部分 phase materialization；但 VALU 增加约 19%，SALU
略增，register profile 也改变。它把一部分内存/同步工作换成了更复杂的 index、
packet、layout 和 ownership 计算，在本硬件和当前后端上没有转化为 latency 收益。

BDV2-G 与 BDV2-S 的 dynamic PMC 相同。specialized typed packet 差异虽存在于
LLVM、MIR 和 HSACO，但当前后端仍编排出同样的 dynamic instruction 工作量，
没有达到 native-style 的低地址/低 VALU feeding。

---

## 7. body benchmark

口径：current HIP stream、caller-owned preallocated output、no Graph、fresh
Python process、warmup=10、repeat=50、7 sessions、rotating order。native 是
same-shape selected chunk-o body diagnostic，不是 public full-graph 排名。

| T | Z5B | BDV2-G | BDV2-S | native | G/Z5B | S/Z5B | Z5B/native |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2048 | `0.067641 ms` | `0.069043 ms` | `0.068962 ms` | `0.042744 ms` | `1.0207x` | `1.0195x` | `1.5825x` |
| 8192 | `0.157294 ms` | `0.166847 ms` | `0.167048 ms` | `0.090875 ms` | `1.0607x` | `1.0620x` | `1.7309x` |

paired session difference：

| T | BDV2-G - Z5B | BDV2-S - Z5B |
|--:|--:|--:|
| 2048 | `+1.402 us` | `+1.322 us` |
| 8192 | `+9.553 us` | `+9.754 us` |

端点 `(T8192-T2048)/96 chunks`：

| arm | slope |
|:--|--:|
| Z5B | `0.933891 us/chunk` |
| BDV2-G | `1.018802 us/chunk` |
| BDV2-S | `1.021724 us/chunk` |
| native | `0.501364 us/chunk` |

因此 BDV2 长文本 slope 比 Z5B 高约 9%，不晋级。

T=2048 rocprof trace median 是 Z5B `41.822 us`、BDV2-G `42.583 us`、BDV2-S
`42.704 us`。trace 含 profiler 扰动，只作机器诊断，正式 latency 以 HIP-event
body 表为准。

---

## 8. 对编译器团队的意义

本轮支持以下较窄的结论：

- 同一 high-level source、同一 MFMA 数学工作和同一 ABI 下，AveLang late
  lowering 可以改变 LLVM/MIR/ISA/HSACO；
- full-scope producer 变化可以减少一部分 dynamic VMEM/LDS；
- 但当前 specialized producer 也引入了更高 VALU，不能把剩余 native gap 全部
  归因于“一个 load lower 错了”。

因此它是“lowering 是可控制性能因素”的证据，不是“剩余全部差距已经证明为
lowering 单一原因”的铁证。还存在 ownership、address/layout feeding、wave
schedule 和 native fused pipeline 的组合差距。

Z7B 的 final-HSACO 收敛问题已被 BDV2 修正为 distinct HSACO，但 specialized
仍没有达到 native machine work。下一次若继续，应先做 LLVM/AMDGPU/LTO
convergence 和 address/layout feeding 归因，不应再添加 Qwen-specific intrinsic，
也不应同时修改 ownership、MFMA geometry、RA 或 full-v29。

---

## 9. 编译器回归测试

新增无设备静态 contract tests：

```text
test/examples/linear_attention/vllm_compare/
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py
```

覆盖：

- 旧 `block_dot_bf16_f32` 注册仍存在；
- logical/transposed helper 创建同一个 block-dot op；
- full-scope metadata、generic fallback 和 specialized path 存在；
- K/H 经过同一个 `emitFullScopeProducer`/`emitGenericOperandBPair`；
- resident Q 只在 dedicated cache fill，logical calls 接收 `q_cache`；
- 禁止出现 `qwen_chunk_o_dot`、`chunk_o_operand_op`、`z8_special_dot`。

旧 direct-K64 block-dot test 仍保留并作为回归测试；BDV2 的 device correctness
由 `check_...full_scope.py` 单独完成。

---

## 10. 复现命令和 artifact

```bash
cmake --build /tmp/avelang-z7b-build3 -j 16

export PYTHONPATH=/tmp/avelang-z7b-build3/python:\
/workspace/project/avelang/test/examples/linear_attention:\
/workspace/project/avelang:/opt/avelang/python:/opt/avelang

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/\
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py -s
```

correctness、benchmark 和 PMC 工件位于：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_bdv2/
  codex_qwen_bt64_stage6z_bdv2_machine_generic/
  codex_qwen_bt64_stage6z_bdv2_machine_specialized/
```

每个 machine 目录保存 `source.py.txt`、`lowered_llvm.ll`、`pre_lto_amdgcn.s`、
`final_isa.s`、HSACO、LTO bitcode、20 个 MIR section、replay argv 和 machine
summary。initial MLIR skipped 的事实保存在 summary 中。

---

## 11. 最终 gate

| gate | BDV2-G | BDV2-S |
|:--|:--:|:--:|
| full correctness | PASS | PASS |
| zero-V/NaN caller-owned output | PASS | PASS |
| same source | PASS，source hash 相同；initial MLIR 工具缺口 | 同左 |
| LLVM/MIR/ISA/HSACO 分叉 | PASS | PASS |
| MFMA/CTA 数学不变 | PASS，160/CTA | PASS，160/CTA |
| scratch/spill | PASS，0 | PASS，0 |
| T=2048 优于 Z5B | FAIL | FAIL |
| T=8192 优于 Z5B | FAIL | FAIL |

BDV2 不晋级，不建 selector，不接入 X2，不改 production。Z5B 仍是当前
Stage 6Z isolated research baseline。

# Qwen persistent recurrence：software-pipeline 基础设施与原型审计

日期：2026-07-29。基线是 `gfx942_bt64_bv32_joint_v4`（R4）。本次没有新增 `joint_v6`，没有手写 Qwen ISA 序列，也没有改变 R4 的数学、BV32 ownership、typed producer 或 LDS-mediated K retile。

## 结论

已接入一个真正的 distance-one modulo software-pipeline 原型：
`gfx942_bt64_bv32_software_pipeline`。它为完整 recurrence 生成了
prologue、带四个 loop-carried packet token 的 steady-state `scf.for`，以及
drain epilogue；而非 R5 式把一条 SSA load 往前移动。

该候选通过 T=64/128/512/2048 的 full nonzero-W correctness，并导出了同一
T=2048 候选的 MLIR、LLVM、exact-LTO MIR、HSACO、ISA 与 PMC。不过它尚未通过
性能 gate：fresh-process body benchmark 比 R4 慢 9.7–18.1%。因此结论是
“pipeline 基础设施/正确性通过，性能候选 No-Go”，而不是宣称取代 R4。

## 已有基础设施审计

| 位置 | 已有能力 | 为什么旧 recurrence 没有使用 |
| --- | --- | --- |
| AveLang | `FormQwenPersistentRecurrencePass`、planner、late K64 stage lowering、block-dot/recurrence-step lowering | persistent region 只保存语义和 R4 stage token；没有一个 pass 把整个 `scf.for` 展开为 pipelined prologue/kernel/epilogue。R5 的 `scheduleJointV5Superblock` 只移动 next-K recipe。 |
| 当前 MLIR 安装 | SCF、`IRMapping`、循环 iter-arg 基元可用 | 安装中没有可直接调用的通用 SCF loop-pipelining expansion pass/API；更重要的是这里的 stage token、block-dot 与 barrier/memref 效果不是纯 SCF arithmetic loop。 |
| 当前 Triton AMD | 通用 `PipeliningUtility`、`PipelineExpander`，以及 AMD `Pipeline.cpp`、wait/barrier lowering | Triton 的 utility 假定它自己的 load/async-copy 语义、dependence/alias 模型和循环 representation；AveLang 的 opaque token 与 R4 shared-bank consumer 不匹配，不能直接注册一个 pass 即可复用。 |

最适合复用的是 Triton 的通用算法骨架：依赖距离、`iter_args` 重命名、
prologue/steady/epilogue expansion。没有复制其 Qwen 或机器码；AveLang 实现利用
MLIR `scf::ForOp`、`IRMapping` 和现有 typed packet producer。AMD 的 wait insertion
仍由后端在实际 global/LDS 依赖上生成。

阻塞旧版本的因素是组合而非单点：

- recurrence 是自定义 region，W/K 是 opaque stage token，`block_dot` 也是 opaque op；
- frontend 将部分标量地址保存为 private `memref.store/load`，没有 MemorySSA/alias
  证明时不能安全把该 recipe 穿过计算区；
- R4 的 LDS bank 与 `gpu.barrier` 是真实同步/alias 边界；
- 原有 pass 顺序中没有一个通用 expansion pass 在 stage token 仍可见时运行。

这也解释了 R5 的 No-Go：它只做同一迭代内的 issue motion，没有循环值重命名、
packet ring、prologue、steady-state 或 epilogue。

## 实现

新增的通用层是
`lib/Dialect/AveLang/Transforms/qwen_modulo_software_pipeline_pass.cc`。
它只识别“有四个 prologue packet 和四个 next packet 的 recurrence”，复制 opaque
完整 core，而不是枚举 Qwen 指令。它运行在 planner 之后、recurrence/block-dot late
lowering 之前。`lower_qwen_k64_pipeline_stage_pass.cc` 随后将 loop-carried token
实现为每 lane 的 private `memref<4xvector<8xbf16>>` packet ring，并复用 R4 的
typed LDS commit/retile consumer。

调度的一个 steady iteration `i` 是：

| 阶段 | 自动形成的工作 | 跨迭代距离 |
| --- | --- | --- |
| 0 | 4 个 `next W0/W1/K0/K1` global packet stage load，写入 ring slot | `i -> i+1`，1 |
| 1 | 用 `iter_args` 中的四个 token 从 ring 取 packet，R4 LDS commit，`gpu.barrier` | 1 |
| 1 core | 当前 W local load + pred MFMA；BF16 V-new/V-decay；当前 K local load + update MFMA；FP32 state feedback | 消费 packet `i` |
| epilogue | 消费最后一个 ring entry、barrier、最后一次完整 core；不再 issue next packet | drain |

对 T=2048（32 chunks），调度器快照中的可核验 SSA 是：

- prologue issue token：`%51/%55/%59/%63`；
- steady loop：`scf.for ... iter_args(%arg13=%51, %arg14=%55, %arg15=%59, %arg16=%63)`；
- steady commit 消费 `%arg13..%arg16`，core 产生下一组 `%109/%113/%117/%121` 并 yield；
- epilogue commit 消费 `%66#0..%66#3`；
- recurrence attr 明确为
  `prologue(issue+consume)->steady(iter_args,distance=1)->epilogue`。

因此 packet 的 global-load-to-LDS-consumer distance 是 1，而不是“提前的同一 SSA
load”。`issue_decision=retained_due_to_memory_dependence` 出现在不能由当前 pass
证明可穿越 private scalar store/load 的三个 recipe 上；它保留合法 issue 位置，但不
取消跨迭代 packet ring 或 drain。这是有意的保守 dependence 决策，不是回退到 R5。

编译后的 ISA 也不是只剩一个提前 load：可在 `0x7bbc..0x7c70` 看到一组
`global_load_dwordx4`、`s_waitcnt vmcnt(7)`、barrier，随后在 `0x7ce8` 起看到
`ds_read_b128`/`s_waitcnt lgkmcnt`/`v_mfma_f32_32x32x8_bf16` 的消费序列。完整 ISA
共含 122 `global_load`、288 `ds_write`、504 `ds_read`、128 MFMA、328 `s_waitcnt` 和
32 `s_barrier`。waitcnt 是实际 global/LDS def-use 在 AMD lowering 后生成，barrier
来自 scheduler/保留的 R4 ownership 边界。

## 验证

### Correctness

独立的软件候选与 device contract reference 及 P2 host microscope 对照：

| T | chunks | finite | result |
| ---: | ---: | --- | --- |
| 64 | 1 | true | pass |
| 128 | 2 | true | pass |
| 512 | 8 | true | pass |
| 2048 | 32 | true | pass |

T=2048 对 device reference 的最大误差为 BF16 outputs `2.44140625e-4`、
FP32 final state `4.3474138e-5`（在既有 gate 内）；对 P2 microscope 为 byte-equal。

### Compiler / machine evidence

同一只候选 kernel（不运行 reference kernel 覆盖 snapshot）产生的完整证据在：

`test/examples/linear_attention/vllm_compare/swp_final_artifacts_t2048/`

- `mlir/post_software_pipeline_scheduler.mlir`：stage attrs、prologue/steady/epilogue 与四个 `iter_args`；
- `mlir/preopt_llvm.ll`、`mlir/postopt_llvm.ll`：pipeline 前后 LLVM；
- `link/amdgpu-link-0.argv.txt` 和 `link/amdgpu-link-0.prelink.bc`：精确可重放 LTO 输入；
- `exact_lto_mir/`：replay 的 pre/post greedy、virtregrewriter、prologepilog MIR；所有 20 个抽取 section 的 `SI_SPILL_AV32_SAVE` 和 `SI_SPILL_AV64_SAVE` 都为 0；
- `hsaco/_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel.hsaco`、`software_pipeline_t2048.isa.s` 与筛选过的 `software_pipeline_t2048.interleave.s`：精确 code object 与 ISA。

### T=2048 PMC

采集命令使用 kernel regex，只取候选 kernel；最后五个 replay dispatch 的动态计数完全一致：

| PMC | 每 dispatch |
| --- | ---: |
| SQ_INSTS_MFMA | 65,536 |
| SQ_INSTS_VALU | 2,353,664 |
| SQ_INSTS_SALU | 160,416 |
| SQ_INSTS_VMEM | 264,640 |
| SQ_INSTS_LDS | 385,024 |
| OccupancyPercent（median） | 0.638220 |

运行时记录还给出 grid=4096、workgroup=128、LDS=53,248 B、scratch=1,024 B、VGPR=128、AccumVGPR=248、SGPR=112。原始 CSV 位于
`swp_final_artifacts_t2048/pmc/t2048_counter_collection.csv`。

### Fresh-process body benchmark

每一个实现都在独立 Python process 中 JIT/预热，随后仅用 HIP events 测 body；
分配、JIT 和 graph capture 不在计时区。这样避免 AveLang lowering mode 不进入 in-process
JIT cache key 而造成的假 A/B。

| T | software pipeline (ms) | R4 (ms) | Triton direct (ms) | pipeline / R4 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.146878 | 0.133949 | 0.065998 | 1.097x |
| 1024 | 0.300817 | 0.269761 | 0.094440 | 1.115x |
| 2048 | 0.575895 | 0.499322 | 0.148691 | 1.153x |
| 8192 | 2.371473 | 2.007732 | 0.441907 | 1.181x |

每项为两个 fresh-process session median 的 median，5 warmups、20 repeats。原始 rows、
summary 和 ratio 在
`test/examples/linear_attention/vllm_compare/swp_benchmark/fresh_process_body.json`。

## 复现入口

```bash
PYTHONPATH=build-software-pipeline/python:python \
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_software_pipeline.py \
  --T 64 128 512 2048 --seed 20260729 --out-dir swp_correctness_final_build

PYTHONPATH=build-software-pipeline/python:python \
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_software_pipeline.py \
  --T 512 1024 2048 8192 --warmup 5 --repeat 20 --sessions 2
```

后续性能工作应针对 packet-ring/private-memory 开销和可证明的 scalar-address
dependence，不能再以“移动单条 load”冒充 software pipeline。

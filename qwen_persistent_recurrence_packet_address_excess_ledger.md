# Qwen persistent recurrence：packet/address excess ledger（R4-tail vs exact current-vLLM）

日期：2026-08-04  
结论状态：**已在 `ljd_qwen_vllm_avelang_rocm722` Docker 内完成 public-Eager `emit_audit=False` T=2048 production capture；没有新增 layout、没有修改 lowering/pass、没有创建新 kernel 变体。PC/range 动态归因仍因 ATT `Wave incomplete` 未闭合，90%/70% 门槛保持 No-Go。**

> 说明：第 1–9 节保留了进入 Docker 前的历史阻断记录。第 10–13 节是本轮同一 production HSACO 的新证据，并覆盖第 1.4、5、6、8 节中“尚未生成 production HSACO”的历史状态。

用户要求的门槛是“至少解释 90% 的动态 VMEM 差额和 70% 的动态 VALU/SALU 差额”。当前证据能够完整枚举两套 ISA 的静态 memory sites，并能复核 PMC 的 dispatch aggregate；但不能把 PMC 的动态指令计数按 ISA 地址分桶。更重要的是，R4 的 exact ISA 是 `emit_audit=True` correctness 编译，而 PMC body 使用 `emit_audit=False`；二者不能直接相乘。故本 ledger 不把静态条数或 audit-only store 猜成生产 body 的动态差额，达不到门槛就停止。本轮只纠正 launch geometry、记录 capture/tool 的硬阻断，并没有修改任何 layout、lowering、planner 或 kernel 数学。

## 1. 证据和口径

### 1.1 制品

| 对象 | exact 制品 | 可用层级 |
| --- | --- | --- |
| R4-tail | `test/examples/linear_attention/vllm_compare/tail_issue_artifacts_t2048/tail_issue_t2048.isa.s` | ISA、exact-LTO MIR、MLIR、LLVM、HSACO metadata |
| R4-tail LLVM | `.../tail_issue_artifacts_t2048/mlir/postopt_llvm.ll` | pointer argument、load/store 的 source/LLVM 归因 |
| R4-tail MIR | `.../tail_issue_artifacts_t2048/exact_lto_final_isel.mir` | final-isel COPY/AGPR 证据 |
| current-vLLM | `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/disassembly.txt` | exact ISA、TTIR、TTGIR、LLVM IR、HSACO |
| current-vLLM TTIR/TTGIR | `.../current_kernels/vllm/kernel.ttir`, `kernel.ttgir` | source loc、Triton loop/consumer 对应关系 |
| PMC | `test/examples/linear_attention/vllm_compare/r4_tail_vs_current_vllm_gap_artifacts/pmc/{r4_t2048,r4_t8192,vllm_t2048,vllm_t8192}` | `SQ_INSTS_{VMEM,VALU,SALU,...}` aggregate |

current-vLLM exact capture 没有同一工具链产生的 final-isel MIR；因此没有用其他版本或其他 kernel 的 MIR 代替。

### 1.2 两个不能混用的 capture

R4 correctness capture 的 source 是 `repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` 的 `_run_kernel(..., emit_audit=True)`，因此 exact ISA 中包含 `pred_f32`、`pred_bf16`、`v_decay`、`state_after` 这些 audit sink。生产 body `run_body()` 明确使用 `emit_audit=False`，source 注释也明确说明这些 accesses compile-time elided。current-vLLM capture 没有这些 audit tensors。

另外，PMC counter/kernel trace 中 R4 的 `Grid_Size=4096`；当前 source 的 `p0.GRID = H_V * (KDIM // BV) = 32`。这不是不一致：HSA/ROCm 的 `grid_size` 是 work-items，不是 workgroup 数。current-vLLM 的 `(512,8)` 也是 4096 个 work-items。两者都对应 32 个 128-thread workgroups，故“PMC aggregate ÷ 32 个 physical V32 CTA × chunk 数”是正确的归一化口径；旧报告把这个事实列为 blocker 的句子已废止。

### 1.3 Launch geometry：`Grid_Size` 的确切语义

本机 `/opt/rocm-7.2.3/include/hsa/hsa.h:2978–3004` 明确写的是：`workgroup_size_{x,y,z}` 和 `grid_size_{x,y,z}` 的单位都是 **work-items**，而不是 workgroups。`rocprofiler_kernel_dispatch_info_t` 也把这两个字段作为 runtime workgroup/grid size。原始 kernel-trace schema 同时给出 `Workgroup_Size_X/Y/Z` 与 `Grid_Size_X/Y/Z`；counter schema 的标量 `Grid_Size` 是三维 grid 的乘积，`Workgroup_Size` 是三维 local size 的乘积。

| dispatch | source logical grid（CTA/workgroup） | HIP/HSA global size（work-items） | HIP/HSA local size | trace `Grid_Size` | trace `Workgroup_Size` | global/local = workgroups |
| --- | --- | --- | --- | ---: | ---: | ---: |
| R4-tail `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` | `(p0.GRID,1,1)=(32,1,1)`，其中 `p0.GRID=H_V*(KDIM/BV)=8*(128/32)=32` | `(4096,1,1)` | `(128,1,1)` | 4096 | 128 | 32 |
| current-vLLM `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `(ceil(V/BV),B*H,1)=(4,8,1)` | `(4*128,8*1,1)=(512,8,1)` | `(128,1,1)` | `512*8*1=4096` | 128 | 32 |

R4 的调用链在 `repro_qwen_gdn_persistent_recurrence_r4_tail_issue.py:48–55` 明确传入 `((r4.GRID,1,1),(r4.WORKGROUP,1,1))`；常量定义在 `repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0.py:25–31`。current-vLLM 的 launch manifest 在 `stage6r_capture_current_recurrence.py:234–245`，其 `[4,8,1]` 是 logical CTA grid。故 R4 `Grid_Size=4096, WG=128` **确实等于 32 个 workgroups**；current-vLLM `(512,8)` 也确实是 4096 work-items、32 个 workgroups。`source GRID=32` 与 trace `Grid_Size=4096` 不再是阻断项。

### 1.4 Capture manifest 与同源性状态

当前 git `HEAD=7432ee26a8869f090cdd8dbac454675fd0c5e22d`（worktree 本身已有用户改动）。本轮只读取这些文件；没有把工作树中的其他改动纳入新 capture。

| capture | plan / constexpr | `emit_audit` | symbol | logical grid / HSA global / local | HSACO SHA256 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **R4 production body（本轮要求）** | `gfx942_bt64_bv32_joint_v4_tail_issue`; B=1,Hk=4,Hv=8,K=128,V=128,BT=64,BV=32 | false | `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` | `(32,1,1)` / `(4096,1,1)` / `(128,1,1)` | **未生成** | BLOCKED：当前进程没有 `torch`/`avelang`/`vllm`，`/opt/venv` 不存在，rocprof 报 KFD descriptor 无效；不能声称有 production HSACO |
| R4 archived correctness | 同上 | **true** | `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` | trace `(4096,1,1)` / `(128,1,1)` | `7d7eb721de8109aee0264393854ef814f732ce12a7ead1b8f6c8a5c6453a47ea` | 仅 audit artifact，不能与 body PMC 同源 |
| current-vLLM archived exact capture | Triton BV32, BT64, `num_warps=2,num_stages=2`; source constants `(12..23)=(8,4,128,128,64,32,true,false,true,true,true,false)` | 无 AveLang audit sink | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `(4,8,1)` / `(512,8,1)` / `(128,1,1)` | `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e` | exact TTIR/TTGIR/LLVM/ISA/HSACO；benchmark/PMC 仍没有写入该 SHA 的运行时 manifest |

R4 source SHA256：`repro_qwen_gdn_persistent_recurrence_r4_tail_issue.py` = `4366270d77a97574f7d3419b97bae1e271aa5831e09ae73b6da5dd830f4429cd`；其直接复用的 `repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` = `41af7be6c89d10b4a4159ccbbe5d01cf2ad9abe3eed2c7fbfa254a5ccad972c8`；P0 常量文件 = `ff0d4bf2461de4c6769a8dc6344f67c88288e6a6e8ddab876e5aaa28304f08de`。R4 archive 的 HSA notes 是 gfx942、LDS 53248 B、scratch 0、VGPR 228、AGPR 32、SGPR 43，但它的 kernarg/ISA 包含 audit outputs。现有 public-Eager benchmark JSON 与 PMC CSV 没有保存 symbol、HSACO SHA 和全部 constexpr，故不能追溯证明它们使用上述某一个 HSACO；本轮不补写伪造的 manifest。

current-vLLM archived capture 的生成脚本 `stage6r_capture_current_recurrence.py` SHA256 是 `d19dd4f7f7d47d450e72591eac35d75d727dabfb7625833b4f9e69aaf6811c12`；其 `kernel.ttir`/`kernel.ttgir`/`kernel.llir`/`kernel.amdgcn` SHA256 分别为 `90b813a4f39fbe2cf19840af7c0f11a476b863e22ef06e884c16a7801301f662`、`6f6219947a7f5ffe999ebc451706bf39e95a7569a124ef3461e075c55b76d177`、`4ecb237d3bfd374ccfa88caf478329ca578d2d00f4b41966201342a1e3023c40`、`c2d666a966a80971ed26cf1dd906e797fb37dab98456d36bdcc318779765b1e6`。这些 hash 只标识 archived current-vLLM capture，不把它提升为本轮新生成的 R4 production capture。

### 1.5 Profiler/运行环境探测

- `rocprofv3 --version`：1.1.0，git revision `c2d94761153e1033a91744842dfc66eddd631fde`，ROCm 7.2.3；help 明确支持 `--kernel-trace`、`--runtime-trace`、`--hsa-trace`、`--pmc`、PC sampling（instructions/cycles/time，stochastic/host_trap）以及 gfx9 Advanced Thread Trace（`--advanced-thread-trace/--att`）。
- legacy `rocprof --version`：ROCm 7.2.3，ROCProfiler 2.0。
- `rocprofv3 --list-avail true`/`-L true`：重复输出 `Invalid KFD descriptor: -1`，GPU:0–7 的 Name 均为空；因此当前环境没有可用的 KFD/GPU，无法运行 ATT、PC sampling 或重新采集 production PMC。
- 实际 ATT smoke probe（`rocprofv3 --advanced-thread-trace true --kernel-trace true -- /bin/true`）在启动阶段失败：`rocprof-trace-decoder library path not found in ['', '/opt/rocm-7.2.3/lib']`；即使忽略 GPU/KFD 问题，当前安装也没有可用 decoder。
- `rocprof-compute --help`：因 `pandas`、`astunparse`、`dash` 等依赖缺失而退出，不可用。
- 当前 `python3` import：`torch`、`avelang`、`vllm` 均 `ModuleNotFoundError`；`/opt/venv` 不存在。故公开 Eager 入口无法启动，不能从 `emit_audit=False` 生成单一 HSACO。

原始 `r4_t2048_kernel_trace.csv`/`vllm_t2048_kernel_trace.csv` 的 schema 是 `Workgroup_Size_X/Y/Z,Grid_Size_X/Y/Z`；对应 counter CSV 的 schema 是 `Grid_Size,Workgroup_Size,...,Counter_Name,Counter_Value`。已有 CSV 是 kernel-wide dispatch trace/PMC，不含执行 PC。上述环境阻断是本轮 production capture 和 PC/range 动态归因均未完成的直接原因。

## 2. PMC aggregate（只作动态总量，不作地址分桶）

以下数字直接来自 counter CSV；没有把 profiler 的 latency 当作结论。

| T | kernel | trace grid（global work-items） | VMEM / dispatch | VALU / dispatch | SALU / dispatch | MFMA / dispatch | LDS / dispatch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | R4-tail | 4096 | 202,240 | 2,239,872 | 163,072 | 65,536 | 381,952 |
| 2048 | current-vLLM | 4096 | 58,368 | 1,395,584 | 64,832 | 65,536 | 305,472 |
| 8192 | R4-tail | 4096 | 798,208 | 8,906,112 | 630,016 | 262,144 | 1,524,736 |
| 8192 | current-vLLM | 4096 | 230,400 | 5,462,912 | 230,720 | 262,144 | 1,214,784 |

trace 的 4096 是 work-items，不是 workgroups。对本问题，一个 workgroup 是一个 physical V32 CTA；每个 dispatch 有 32 个 CTA，`T/BT` 是 recurrence chunks。因此正确的 per-physical-V32-per-chunk 分母是 `32*(T/64)`，而不是 4096 本身。归一化结果如下（仍然只是 kernel-wide aggregate）：

| T | denominator = CTA × chunks | metric | R4 / physical V32 / chunk | current-vLLM / physical V32 / chunk | R4 excess |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2048 | `32*32=1024` | VMEM | 197.500 | 57.000 | **+140.500** |
| 2048 | `32*32=1024` | VALU | 2187.375 | 1362.875 | **+824.500** |
| 2048 | `32*32=1024` | SALU | 159.250 | 63.3125 | **+95.9375** |
| 2048 | `32*32=1024` | MFMA | 64.000 | 64.000 | 0 |
| 8192 | `32*128=4096` | VMEM | 194.875 | 56.250 | **+138.625** |
| 8192 | `32*128=4096` | VALU | 2174.344 | 1333.719 | **+840.625** |
| 8192 | `32*128=4096` | SALU | 153.8125 | 56.3281 | **+97.4844** |
| 8192 | `32*128=4096` | MFMA | 64.000 | 64.000 | 0 |

因此历史报告中的约 `139 VMEM / 840 VALU / 98 SALU` 是“每 recurrence chunk、每 physical V32 tile”的正确数值，但它来自 aggregate 的归一化，**不是**某个 PC bucket 的归因。

## 3. R4-tail / current-vLLM 的静态 VMEM ledger

`N_dyn/V32` 一列必须由 per-PC 动态计数产生；当前 PMC 只有 kernel-wide `SQ_INSTS_VMEM`，所以按要求填为“不可证明”，不以静态 site count 冒充。`bytes` 是由公开 tensor shape 得到的每个 logical packet payload，不是把一个 `dwordx4` 误算成整个 tensor。

| path | R4 ISA range / type | current-vLLM ISA range / type | R4 static sites | Triton static sites | width | logical bytes / physical V32 | reload / reuse | R4 vs Triton 分类 | LLVM/MLIR/source |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| W | `0x1c74–0x1cac`: 4 个 `global_load_dwordx4`（current/prologue）；`0x216c–0x21c0`: 8 个 `global_load_dwordx4`（tail next-W） | `0x1fa0–0x1fa8`: 2 个 `buffer_load_dwordx4`（prologue）；`0x2fdc–0x2ffc`: 4 个 `buffer_load_dwordx4`（loop W） | 12（其中 tail 8） | 6（2+4） | b128 per lane (`dwordx4`) | W `[BT64,K128] bf16` = 16,384 B/chunk | R4 steady 每 chunk issue 一次；最后 `next_start=min(...)` 会再次 issue 最后 packet；Triton 每 chunk 直接消费一次 | 同字节但 R4 有额外 tail issue；不是 b128→b16 的宽度问题 | R4 source `repro...r4_joint_v4.py:164–173,205–207`; MLIR `post_block_dot_lowering.mlir` stage load/commit；Triton TTGIR 375–393、`loc #loc113/#loc91` |
| K | `0x1c74–0x1cac` 的另一 pointer base 4 个 `global_load_dwordx4`；`0x21cc–0x2214`: 8 个 `global_load_dwordx4`（tail next-K） | `0x1eac–0x1f54`: 8 个 `buffer_load_dwordx4`（prologue）；`0x3b2c–0x3e3c`: 8 个 `buffer_load_dwordx4`（loop K） | 12（其中 tail 8） | 16（8+8） | b128 per lane | K `[BT64,K128] bf16` = 16,384 B/chunk | R4 1 tail packet/chunk，epilogue 重发；Triton prologue/body 各自一次、无独立 tail global issue | R4 的 extra 是 packet issue/重复尾部提交路径；不是 MFMA 重复（动态 MFMA 相等） | R4 source `:164–173,252–253`; C++ `lower_qwen_block_dot_pass.cc:334–395`; Triton TTGIR 311–326、458–487 |
| U | `0x3184,0x330c,0x34f0,...,0x4ab0`: 16 个 `global_load_ushort` | `0x31e0–0x3200`: 4 个 `buffer_load_dwordx4` | 16 | 4 | R4 b16；Triton b128 | U/V `[BT64,V32] bf16` = 4,096 B/chunk | 两者逻辑上各读一次；R4 把同一 packet 切成 16 个窄 load | **同字节但更窄的 load**；不是 logical reload | R4 source `:230–249`; LLVM `postopt_llvm.ll` corrected load；Triton TTGIR 406–414，TTIR loc `#loc136/#loc43` |
| state initial/load | `0x1af0–0x1bf0`: 16 个 `global_load_dwordx4` | `0x1984,0x19d8,0x1a34,0x1a84,0x1c24,0x1c68,0x1cb4,0x1d08`: 8 个 `global_load_dwordx4` | 16 | 8 | b128 | initial state `[V32,K128] f32` = 16,384 B/dispatch（一次） | 1 次；steady state 无 reload | R4 静态 site 数更多，但一次性，长序列 slope 影响小 | R4 LLVM args `%16`/`initial_state`，source `:144–152`; Triton TTIR 94、106 (`h0` loads) |
| g / decay input | `0x22e8`: 1 个 `global_load_dword` (`g_last`)；`0x318c–0x4abc`: 16 个 `global_load_dword` (`g[token]`) | `0x2018,0x2070` 为 prologue g；`0x3788,0x37b8` 为 loop g | R4 17 b32 | Triton 4 b32 | b32 | `g_last` 4 B + 64-token g 256 B = 260 B/chunk | logical reuse=1；R4 用 16 个 token/group load 表达同一 64-token packet；Triton 是 vector+scalar | **同字节但更窄/更碎的 load + 地址/控制差异**；不能称为 logical reload | R4 source `:178,232–241`; LLVM `%17=g`; Triton TTIR 198–204、TTGIR 422–435 |
| output: H snapshot | `0x235c–0x2fe8`: 64 个 `global_store_short` | `0x2be0,0x2c24,0x2eb0,0x2ef0`: 4 个 `buffer_store_dwordx4` | 64 | 4 | R4 b16；Triton b128 | H `[V32,K128] bf16` = 8,192 B/chunk | 每 chunk 写一次；无 packet reuse | **同字节但更窄的 store**；R4 address/control 也更碎 | R4 source `:187–195`; Triton TTIR 166–180、TTGIR 338–343 |
| output: v_new | `0x3304,0x34e8,...,0x4c4c`: 16 个 `global_store_short` | `0x35b0,0x3644,0x36e0,0x377c,0x5124,0x51e4,0x5298,0x5358`: 8 个 `buffer_store_dwordx2` | 16 | 8 | R4 b16；Triton b64 | `[BT64,V32] bf16` = 4,096 B/chunk | 每 token tile 一次；无 logical reload | **同字节但更窄的 store** | R4 source `:247`; Triton TTIR 184–187、TTGIR 410–414 |
| decay output（audit-only） | `0x328c,0x346c,...,0x4bd8` 中 16 个 `global_store_short_d16_hi` | exact current-vLLM capture 没有 `v_decay` sink | 16 | 0 | b16 high-half | `[BT64,V32] bf16` = 4,096 B/chunk，仅 `emit_audit=True` | 每 chunk 一次；生产 body compile-time elided | **额外 global round-trip（audit contract）**，不能归因 packet reload | R4 source `:248–249`; LLVM `%19=v_decay`; Triton TTIR 无对应 op |
| pred audit output（audit-only） | 同一 `global_store_short_d16_hi` 家族另 16 个；`0x31d0–0x4b4c` 16 个 `global_store_dword` | 无对应 sink | 32（b16 16 + b32 16） | 0 | b16/b32 | pred FP32 4,096 B + pred BF16 2,048 B/chunk，仅 audit | 每 chunk 一次；生产 body compile-time elided | **额外 global round-trip（audit contract）** | R4 source `:243–246`; LLVM `%15=pred_f32,%20=pred_bf16`; Triton TTIR 无对应 op |
| state store after chunk（audit-only） | `0x5848–0x60a8`: 47 个 `global_store_dword` | 无对应 sink | 47 | 0 | b32 | state-after `[V32,K128] f32` = 16,384 B/chunk，仅 audit | 每 chunk 一次；生产 body compile-time elided | **额外 global round-trip（audit contract）** | R4 source `:281–291`; LLVM `%13=state_after`; Triton TTIR 无对应 op |
| state final/output | `0x60e0–0x658c`: 48 个 `global_store_dword` | `0x5ce0–0x5e6c`: 8 个 `buffer_store_dwordx4` | 48 | 8 | R4 b32；Triton b128 | final state `[V32,K128] f32` = 16,384 B/dispatch（一次） | 1 次；无 steady reload | **同字节但更窄的 store + 地址/控制差异**；一次性 | R4 source `:293–301`; LLVM `%24=final_state`; Triton TTIR/TTGIR 240–244 |
| tail next-W/K | `0x216c–0x2214`: 16 个 `global_load_dwordx4` | 无 separate tail global issue；Triton 仅在 loop body 读 current W/K，loop 尾部 478–487 是 LDS store/transpose | 16 | 0（独立 tail path） | b128 | W+K = 32,768 B/chunk | R4 每 chunk 一次，epilogue 对最后 packet 重发 | **额外 global round-trip / issue placement 差异**；需 PC 动态计数才能判断它占总 VMEM 多少 | R4 source `:205–207,252–253`; planner `qwen_persistent_recurrence_pass.cc:570–667`; Triton TTGIR 478–487 |
| other | `0x2160`: 1 个 `global_store_dword`；其 pointer/semantic 在现有 source/LLVM 对照中未形成唯一闭环 | 未找到对应 Triton site | 1 | 0 | b32 | 未证明 | 未证明 | **未归因，禁止猜测** | exact ISA `0x2160`；需要同一 build 的 debug pointer map |

R4 exact ISA 静态总量为：`global_load_dwordx4=40`、`global_load_dword=17`、`global_load_ushort=16`、`global_store_dword=112`、`global_store_short=80`、`global_store_short_d16_hi=32`；另有 4 个 helper format op。current-vLLM 为：`global_load_dwordx4=8`、`global_load_dword=4`、`buffer_load_dwordx4=36`、`buffer_store_dwordx2=8`、`buffer_store_dwordx4=16`。这些是静态 site 枚举，不是 per-V32 动态数。

### 3.0 用户指定格式的动态 ledger

下表把“不可证明”显式保留在用户指定的列中；不能用静态 site count 填入 `R4/V32` 或 `Triton/V32`。`excess VMEM` 和 `excess VALU/SALU` 只有 aggregate（见第 2、4 节），没有 path-level PC bucket。

| path | R4/V32 | Triton/V32 | bytes | width | reload factor | excess VMEM | excess VALU/SALU | root cause |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| W | 不可证明 | 不可证明 | 16,384/chunk | b128 | R4 tail 1/chunk，epilogue 重发 | 不可分桶 | 不可分桶 | tail next-W issue；需动态 PC count |
| K | 不可证明 | 不可证明 | 16,384/chunk | b128 | R4 tail 1/chunk，epilogue 重发 | 不可分桶 | 不可分桶 | tail next-K issue；需动态 PC count |
| U | 不可证明 | 不可证明 | 4,096/chunk | R4 b16 / Triton b128 | logical 1；R4 窄切片 | 不可分桶 | 不可分桶 | 同字节但更窄的 load |
| state initial/load | 不可证明 | 不可证明 | 16,384/dispatch | b128 | 1 次 | 不可分桶 | 不可分桶 | 一次性初始 state |
| g / decay input | 不可证明 | 不可证明 | 260/chunk | b32 | logical 1；R4 16-way 碎片化 | 不可分桶 | 不可分桶 | 窄/碎 load 与地址控制 |
| H output | 不可证明 | 不可证明 | 8,192/chunk | R4 b16 / Triton b128 | 1/chunk | 不可分桶 | 不可分桶 | 同字节但更窄的 store |
| v_new | 不可证明 | 不可证明 | 4,096/chunk | R4 b16 / Triton b64 | 1/chunk | 不可分桶 | 不可分桶 | 同字节但更窄的 store |
| decay output（audit） | 不可证明 | 0（无 sink） | 4,096/chunk | b16 | 1/chunk，仅 audit | 不可分桶 | 不可分桶 | audit-only global round-trip |
| pred audit（audit） | 不可证明 | 0（无 sink） | 6,144/chunk | b16+b32 | 1/chunk，仅 audit | 不可分桶 | 不可分桶 | audit-only global round-trip |
| state-after（audit） | 不可证明 | 0（无 sink） | 16,384/chunk | b32 | 1/chunk，仅 audit | 不可分桶 | 不可分桶 | audit-only global round-trip |
| final state | 不可证明 | 不可证明 | 16,384/dispatch | R4 b32 / Triton b128 | 1 次 | 不可分桶 | 不可分桶 | 一次性窄 store/address control |
| tail next-W/K | 不可证明 | 0（无独立 tail） | 32,768/chunk | b128 | 1/chunk，尾部重发 | 不可分桶 | 不可分桶 | issue-placement / extra round-trip 候选 |
| other (`0x2160`) | 不可证明 | 0 | 未证明 | b32 | 未证明 | 不可分桶 | 不可分桶 | pointer/semantic 未闭合，禁止猜测 |

### 3.1 VMEM 差额可证明到什么程度

- 静态 **site 覆盖 100%**：两份 disassembly 中所有 `global/buffer load/store` 都已列出或落入 `other`。
- 逻辑 bytes 的 shape 覆盖包含 W、K、U、g、H、v_new、initial/final state；audit-only outputs 单独列出，没有混进 production body。
- 但是动态 `SQ_INSTS_VMEM` 没有 PC/address 维度，且 R4 exact ISA 与 PMC body 的 `emit_audit`/grid 口径不同。因此“R4 相对 Triton 动态 VMEM 差额中有多少来自 W、K、U、g、tail”目前**不能证明到 90%**。不能以静态 `16 vs 4` 或 `64 vs 4` 直接乘 loop trip count。

## 4. VALU/SALU ledger（静态可定位，动态占比未闭合）

下表统计 exact disassembly 中的 opcode site，给出明确的机器工作候选；右侧的 `动态 excess` 不能从当前 PMC 分配到该类。

| path | R4 static evidence | current-vLLM static evidence | source/IR mapping | R4 extra aggregate（T=8192/physical V32/chunk） | dynamic coverage |
| --- | --- | --- | --- | ---: | --- |
| global address formation | `v_lshl_add_u64=179`、`v_lshl_or_b32=13`；大量 `v_add3_u32` 分布在 `0x20d0–0x2f00`、`0x3180–0x4c50` | `v_lshl_add_u64=28`、`v_lshl_or_b32=18`、`v_add_lshl_u32=26` | R4 LLVM `getelementptr` blocks；Triton TTGIR W/U/K/g `arith.addi/muli/tt.addptr` | 未分桶 | 不可证明 |
| LDS address formation | R4 `v_lshlrev_b64=48`、`v_lshlrev_b32=61`，对应 token-major K retile 和 same-bank offsets | Triton 多个 `v_add_u32/v_add3` 与 LDS offset `32768/34816` | R4 `lower_qwen_block_dot_pass.cc:334–395`; Triton TTGIR 317–326, 482–487 | 未分桶 | 不可证明 |
| lane/packet index | R4 `v_bfe_u32=81`、`v_lshlrev_b32=61`、`v_or3_b32=10` | `v_bfe_u32=64`、`v_add_u32=150`、`v_or3_b32=14` | R4 source lane/mfma ownership `:83–142`; Triton TTIR index construction `:80–140` | 未分桶 | 不可证明 |
| vector extract/insert/pack | R4 `v_perm_b32=64`、`v_pk_mul_f32=32`、`v_pk_add_f32=32` | Triton `v_perm_b32=144`、`v_pk_mul_f32=8`、`v_pk_add_f32=32` | R4 final-isel MIR `COPY/REG_SEQUENCE`；R4 block-dot fragment lowering；Triton TTGIR `convert_layout/trans` | 未分桶 | 不可证明 |
| AGPR/VGPR bridge | R4 `v_accvgpr_read_b32=64`、`v_accvgpr_write_b32=64` | Triton `read=112`、`write=64` | R4 exact MIR/ISA MFMA accumulator handoff；Triton update clusters `0x4a70–0x4d78`, `0x5834–0x59b4` | 未分桶 | 不可证明 |
| loop/control | R4 12 `s_barrier`，exact MIR 356 `REG_SEQUENCE`/2,460 `COPY`；Triton 32 `s_barrier`，无 exact MIR | planner loop `qwen_persistent_recurrence_pass.cc:570–667`; Triton SCF loop TTGIR 328–489 | aggregate SALU delta `+97.484`，但无法分配到 branch、wait、index 或 address | 不可证明 |
| other | R4 0x2160 store 与若干 source/ISA 未闭合的 control path | current exact disassembly 无对应 | 需同一编译 build 的 debug metadata | 未分桶 | 不可证明 |

T=8192 的 aggregate（按 `32 CTA × 128 chunks` 归一化到 physical V32/chunk）为：

| metric | R4/V32/chunk | Triton/V32/chunk | excess |
| ---: | ---: | ---: | ---: |
| VALU | 2,174.344 | 1,333.719 | **+840.625** |
| SALU | 153.813 | 56.328 | **+97.484** |
| VMEM | 194.875 | 56.250 | **+138.625** |

静态 opcode site 已覆盖上述机器类别，但没有动态 PC count，所以无法满足“至少解释 70% 的 VALU/SALU 差额”。特别是 `v_perm`/AGPR read/write 的静态数量不能直接乘以 loop trip count；R4 与 Triton 的 loop unroll、exec mask 和 grid/source 口径不同。

## 5. Production PC/range 动态归因状态

用户要求的是 production `emit_audit=False` ISA 的实际 PC 执行次数，再按 ISA range 统计 VMEM/VALU/SALU，并用 kernel-wide PMC 闭合。当前只有静态 ISA site 和 kernel-wide aggregate；没有生产 R4 HSACO，也没有可运行的 GPU/KFD。因此下表中的 `dynamic executions` 不能填 0（0 会被误读为未执行），统一标为 **未采集**。

| bucket | production PC range / opcode 证据 | dynamic executions / dispatch | dynamic VMEM/VALU/SALU | logical bytes / width | source → LLVM/MLIR |
| --- | --- | --- | --- | --- | --- |
| W current/prologue | R4 `0x1c74–0x1cac`, `global_load_dwordx4` | 未采集 | 未采集 | W packet, b128 | `repro...r4_joint_v4.py:164–173` → post-block-dot load/commit |
| W tail | R4 `0x216c–0x21c0`, `global_load_dwordx4` | 未采集 | 未采集 | W packet, b128 | `repro...r4_joint_v4.py:205–207` → `qwen_persistent_recurrence_pass.cc:570–667` |
| K current/prologue | R4 current/prologue `global_load_dwordx4` sites around `0x1c74–0x1cac` | 未采集 | 未采集 | K packet, b128 | `repro...r4_joint_v4.py:164–173,252–253` → `lower_qwen_block_dot_pass.cc:334–395` |
| K tail | R4 `0x21cc–0x2214`, `global_load_dwordx4` | 未采集 | 未采集 | K packet, b128 | same source → recurrence planner tail |
| U | R4 `0x3184...0x4ab0`, `global_load_ushort` | 未采集 | 未采集 | 4096 B/chunk, b16 | `repro...r4_joint_v4.py:230–249` → corrected load |
| g/decay | R4 `0x22e8` and `0x318c...0x4abc`, `global_load_dword` | 未采集 | 未采集 | 260 B/chunk, b32 | `repro...r4_joint_v4.py:178,232–241` → `%17=g` |
| H output | R4 `0x235c–0x2fe8`, `global_store_short` | 未采集 | 未采集 | 8192 B/chunk, b16 | `repro...r4_joint_v4.py:187–195` |
| v_new output | R4 `0x3304...0x4c4c`, `global_store_short` | 未采集 | 未采集 | 4096 B/chunk, b16 | `repro...r4_joint_v4.py:247` |
| initial state | R4 `0x1af0–0x1bf0`, `global_load_dwordx4` | 未采集 | 未采集 | 16384 B/dispatch, b128 | `initial_state` args `:144–152` |
| final state | R4 `0x60e0–0x658c`, `global_store_dword` | 未采集 | 未采集 | 16384 B/dispatch, b32 | `repro...r4_joint_v4.py:293–301` |
| pred operand preparation | static VALU/AGPR ranges in exact audit ISA | 未采集 | 未采集 | n/a | block-dot fragment lowering |
| K/update operand preparation | static LDS/VALU ranges in exact audit ISA | 未采集 | 未采集 | n/a | `lower_qwen_block_dot_pass.cc:334–395` |
| BF16 bridge | static `v_cvt_*`, `v_accvgpr_*` sites | 未采集 | 未采集 | n/a | pred finalize / `v_new` boundary |
| state feedback | static MFMA/AGPR/read-write ranges | 未采集 | 未采集 | n/a | FP32 loop-carried state |
| loop/control | `s_barrier`, `s_waitcnt`, branches | 未采集 | 未采集 | n/a | `qwen_persistent_recurrence_pass.cc:570–667` |

### 5.1 Aggregate closure and the requested 139/840/98 attribution

Raw PMC totals close at the kernel level for the four archived runs (for example T=8192: R4 VMEM 798208, VALU 8906112, SALU 630016, MFMA 262144; current-vLLM VMEM 230400, VALU 5462912, SALU 230720, MFMA 262144). They do **not** close against a sum of PC buckets because no PC counts exist. Consequently:

- `+138.625 VMEM`, `+840.625 VALU`, `+97.4844 SALU` are normalized aggregate differences per physical V32 tile per recurrence chunk;
- the fraction attributable to W, K, U, state, decay, output, tail, or “other” is **unknown**, not zero;
- static site coverage is 100% for the disassemblies, but dynamic causal coverage is 0% for VMEM and 0% for VALU/SALU under the requested PC-count definition;
- the 90% VMEM and 70% VALU/SALU gates are therefore not met.

The audit correctness ISA has 16 static `v_decay` stores + 32 static pred (`pred_f32`/`pred_bf16`) stores + 47 static `state_after` stores = **95 audit-only global-store sites**. This is the only defensible “audit-only instruction” count available. It is a static count from the disqualified `emit_audit=True` artifact; because the required production R4 HSACO was not generated, verified production deletion is **not available**. These 95 sites must not be multiplied by trip count or included in the 139 VMEM difference.

## 6. 阻断原因和停止决定

1. **production R4 capture 未生成**：当前进程缺少 `torch`/`avelang`/`vllm`，没有 `/opt/venv`，且 rocprof 无有效 KFD/GPU；因此无法从公开 Eager `emit_audit=False` 路径产生唯一 HSACO、MLIR/LLVM/MIR/ISA/HSA manifest。
2. **R4 exact ISA/PMC body 不同 constexpr**：现存 ISA 是 `emit_audit=True`，production body 是 `emit_audit=False`。audit-only global stores 不能拿来解释生产 VMEM；95 个静态 audit sites 已单独剔除。
3. **PMC 没有 per-PC/per-address 动态计数**：只有 `SQ_INSTS_VMEM/VALU/SALU` kernel aggregate，不能把 `0x216c–0x2214`、`0x3184` 等 range 分别计数。
4. **benchmark/PMC 与 HSACO identity 未闭合**：现有 JSON/CSV 没有 symbol、HSACO SHA、完整 constexpr，不能证明它们没有在 session 间重新 JIT。
5. **Triton 没有 exact final-isel MIR**：不能用其他版本 MIR 推导 COPY/REG_SEQUENCE 的动态开销。

因此本轮没有实现新 layout、没有修改 `lower_qwen_block_dot_pass.cc`、`qwen_persistent_recurrence_pass.cc` 或任何 planner。任何“U 窄 load 已占 VMEM 差额 X%”“tail next-W/K 已占 Y%”“address formation 已占 VALU 差额 Z%”的数字，在补齐上述 capture 前都属于猜测，按要求不输出。

## 7. 继续前必须补齐的最小证据

下一步不是 layout 实验，而是重新生成一个**与 PMC 完全相同**的 R4 production-body capture：

1. `emit_audit=False`；
2. 记录实际 launch grid，并使 source、HSACO metadata、kernel trace 三者一致；
3. 导出该 capture 的 MLIR、LLVM、exact-LTO MIR、ISA；
4. 使用支持 PC/range 维度的动态 trace，至少对上表各 ISA range 产生 `SQ_INSTS_VMEM`、`VALU`、`SALU` 分桶；
5. 重新计算 per-dispatch、per-recurrence-chunk、per-physical-V32 三种归一化；
6. 只有 VMEM 动态差额覆盖 ≥90%、VALU/SALU 覆盖 ≥70% 后，才允许选择一个 layout/lowering 候选。

在这些证据完成前，任何新 layout、packet 重排、ds_read 替换或 address lowering 都保持 No-Go。

## 8. 本轮决策

不选择 A/B/C/D 中任何一个实现方向，也不创建性能候选。geometry 已确认是 32 个 workgroups，不能再以 grid 语义为由修改归一化；但 production HSACO、同源 benchmark/PMC manifest 和 PC/range 动态 trace 尚未存在，故无法回答“139/840/98 分别来自哪些 PC bucket”。下一轮唯一允许的动作是恢复与 GPU/KFD/公开 Eager 依赖相同的运行环境，先生成 `emit_audit=False` 的单一 R4 HSACO 并把其 SHA、symbol、grid/WG、constexpr 写入 benchmark、PMC、trace 三份日志；在动态 ledger 达到 90%/70% 覆盖前，继续保持 No-Go。

## 9. 本轮 production capture 前置复核（2026-08-04）

本轮重新执行了前置检查；第一项环境门槛即失败，因此按用户要求停止，没有重新读取旧 audit ISA、没有运行公开 Eager、没有编译或创建任何 HSACO。

| 检查项 | 实际结果 | 缺失/阻断 |
| --- | --- | --- |
| Python `torch` / `avelang` / `vllm` | `/usr/bin/python3`、仓库 `.venv/bin/python`、`/home/qiuchuyu/atrex-bench/.venv/bin/python` 均 `ModuleNotFoundError` | ROCm PyTorch、AveLang Python 包、vLLM runtime 未安装 |
| HIP device | `torch` 无法 import；无法执行 `torch.cuda.is_available()` | 无 HIP runtime 可用的 Python stack |
| KFD/DRM | `rocminfo`: `Unable to open /dev/kfd read-write: No such file or directory`；`/dev/kfd`、`/dev/dri/render*` 不存在；当前用户不在 `video` 组 | 主机/容器没有 MI300X device passthrough 或权限 |
| rocprofv3 GPU/counter | 1.1.0 / ROCm 7.2.3 可执行，但 `--list-avail true` 重复输出 `Invalid KFD descriptor: -1`，GPU 名称为空 | 无有效 KFD，不能列 GPU/counter 或采 PMC |
| PC sampling / ATT | help 中存在 PC sampling 与 gfx9 ATT 开关，但没有 GPU；ATT smoke probe 失败：`rocprof-trace-decoder library path not found in ['', '/opt/rocm-7.2.3/lib']` | decoder 包未安装，且 device 仍缺失 |
| rocprof-compute | 启动即报 `pandas`、`astunparse`、`dash`、`sqlalchemy` 等依赖缺失 | profiler-compute Python 依赖未安装 |

### 9.1 修复命令（必须在有 MI300X 的主机/ROCm 容器执行）

以下命令只是恢复环境，不是本轮编码；当前沙箱没有权限也没有设备，未执行这些安装/主机操作。

1. 在 MI300X 主机确认驱动和节点，并把用户加入 DRM 访问组；重新登录后检查设备：

   ```bash
   sudo modprobe amdgpu
   sudo usermod -aG video "$USER"
   # 重新登录后：
   test -e /dev/kfd && compgen -G '/dev/dri/renderD*' >/dev/null
   rocminfo | rg -n 'gfx942|Name:'
   ```

   若使用容器，必须将设备和组传入，而不是在容器内手工创建节点：

   ```bash
   docker run --rm -it --ipc=host --shm-size=16g \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     <validated-rocm7.2.2-avelang-vllm-image> bash
   ```

2. 使用仓库已验证的 ROCm/vLLM 镜像链恢复 Python stack。对应 Dockerfile 是 `docker/Dockerfile.vllm_rocm722_base` → `docker/Dockerfile.avelang_vllm_rocm722`；MI300X 构建时把默认的 `PYTORCH_ROCM_ARCH=gfx90a` 覆盖为 `gfx942`：

   ```bash
   docker build -f docker/Dockerfile.vllm_rocm722_base \
     --build-arg PYTORCH_ROCM_ARCH=gfx942 \
     -t qwen-vllm-rocm722-base:gfx942 .
   docker build -f docker/Dockerfile.avelang_vllm_rocm722 \
     --build-arg VLLM_ROCM722_BASE_IMAGE=qwen-vllm-rocm722-base:gfx942 \
     -t avelang-vllm-rocm722:gfx942 .
   ```

   镜像内必须先通过 `import torch, avelang, vllm`、`torch.cuda.is_available()` 和 `torch.cuda.get_device_name(0)`，再开始 capture。

3. 补齐 ATT decoder 和 rocprofiler-compute 依赖（版本必须与 ROCm 7.2.3 匹配）：

   ```bash
   sudo apt-get update
   sudo apt-get install -y rocprof-trace-decoder-ubuntu-24.04
   python3 -m pip install -r /opt/rocm-7.2.3/libexec/rocprofiler-compute/requirements.txt
   rocprofv3 --list-avail true
   rocprofv3 --advanced-thread-trace true --kernel-trace true -- /bin/true
   rocprof-compute --help
   ```

只有上述检查全部通过，才能回到第 7 节的 production capture 清单；在此之前不允许把旧 audit ISA、旧 PMC 或静态 site 数重新解释成 bucket 差额。

## 10. Docker 内 production capture（本轮新证据）

### 10.1 环境和入口

本轮实际执行环境是运行中的容器 `ljd_qwen_vllm_avelang_rocm722`，共享 checkout 为 `/workspace/project/avelang`。不是宿主机的 `/usr/bin/python3`，也不是容器中旧的 `/opt/avelang` binding；Python 搜索路径优先使用当前 checkout 的 `build-software-pipeline/python` binding 和 `python/avelang` source package。

| 检查 | production capture 实际值 |
| --- | --- |
| Python | `/opt/venv/bin/python` |
| torch / HIP | `2.10.0+rocm7.2.2.git40d237bf` / `7.2.53211` |
| GPU | 8 × AMD Instinct MI300X，`gfx942`；`torch.cuda.is_available()=True` |
| KFD/DRM | `/dev/kfd` 与 render nodes 可用，`rocminfo` 可见 gfx942 |
| rocprofv3 | 1.1.0，ROCm 7.2.2；可采 kernel trace/PMC/ATT |
| AveLang path | `/workspace/project/avelang/build-software-pipeline/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so` |
| git | `7432ee26a8869f090cdd8dbac454675fd0c5e22d` |

公开入口是 `bench_qwen_gdn_r4_tail_vs_current_vllm_eager.py` 的 `_launch("r4_tail")`，没有 private HSACO launch；R4 仍是 `gfx942_bt64_bv32_joint_v4_tail_issue`，`emit_audit=False`。

### 10.2 唯一 T=2048 production module 和 manifest

完整 manifest：
`test/examples/linear_attention/rocprof_outputs/qwen_r4_tail_production_t2048_capture_20260804_v2/production_manifest.json`。

| 字段 | 值 |
| --- | --- |
| shape | B=1, Hk=4, Hv=8, K=128, V=128, BF16, BT=64, BV=32 |
| constexpr | `num_tokens=2048`, `num_chunks=32`, `emit_audit=false`, `persistent_semantic=true` |
| logical grid | `(32,1,1)` |
| HSA global/local | `(4096,1,1)` / `(128,1,1)` |
| workgroups | 32（4096 work-items ÷ 128） |
| symbol | `_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel` |
| HSACO | `hsaco/_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel.hsaco` |
| HSACO SHA256 | **`5ebff98fd16fe1fa47501fbeb7dd4bd738f716d6445c0d0fd52d71e0621740c7`** |
| cache key | manifest `jit_calls[0].cache_key`；包含上述 12 pointer ABI + 4 constexpr |

`hsaco/*.hsaco` 的首次编译和 retry 编译 SHA 相同。public-Eager T=2048 probe、PMC、ATT 的 runtime link 输出 SHA 均为同一值：

```text
link_runtime_t2048/amdgpu-link-0.linked.out  5ebff98f...
pmc/link/amdgpu-link-0.linked.out             5ebff98f...
att_gpu0/link/amdgpu-link-0.linked.out        5ebff98f...
```

因此 T=2048 的 benchmark probe、PMC dispatch 和 ATT code object 同源；T=1024/8192 是不同 `num_tokens/num_chunks` constexpr specialization，不能声称复用 T=2048 的二进制，但仍走同一 public-Eager R4-tail lowering。

### 10.3 同一 HSACO 的 IR/MIR/ISA 审计链

以下文件从上述 HSACO/同一次 link 产生，没有使用 `emit_audit=True` correctness kernel 替代：

| 层 | 文件 |
| --- | --- |
| post-planner MLIR | `v2/mlir/post_recurrence_joint_planner.mlir`（SHA `0533935e...`） |
| post-block-dot MLIR | `v2/mlir/post_block_dot_lowering.mlir`（SHA `495ad3a2...`） |
| post-LLVM | `v2/mlir/postopt_llvm.ll`（SHA `88bcfe42...`） |
| exact-LTO final-isel/MIR | `v2/exact_lto/kernel_section_08.mir`、`kernel_section_09.mir`；`summary.json` 显示 spill saves=0 |
| production ISA | `v2/production.isa.s`（SHA `76e51ddd...`） |
| HSA metadata | `v2/hsa_metadata.txt`（SHA `a17a101c...`） |

production HSA metadata：`.private_segment_fixed_size=0`、`.group_segment_fixed_size=53248`、`.vgpr_count=288`、`.agpr_count=32`、`.sgpr_count=35`、spills=0。rocprof dispatch resource row 同时记录 `VGPR_Count=128`、`Accum_VGPR_Count=160`、`SGPR_Count=112`；这是 profiler 的 allocation/residency 口径，不与 HSA code-object max 字段混写。

在 production ISA 中搜索 `pred_f32|pred_bf16|v_decay|state_after` 无匹配。12 个 pointer 参数仍保留是共享 ABI；对应 audit sinks 和 stores 已在 `emit_audit=False` 编译中消失，不能把旧 audit ISA 的 95 个静态 store sites 算入 production VMEM。

## 11. Production PMC、benchmark 和 trace

### 11.1 T=2048 production PMC

原始 CSV：
`v2/pmc/trace/0364d3a007f9/753325_counter_collection.csv`。

5 个 R4 production dispatch 的 kernel row 完全相同：`Grid_Size=4096`、`Workgroup_Size=128`、LDS=53248、Scratch=0，symbol 为 production symbol。每 dispatch：

| counter | dynamic instructions / dispatch | 每 physical-V32/chunk（分母 `32*32=1024`） |
| --- | ---: | ---: |
| `SQ_INSTS_VMEM` | 202,240 | 197.500 |
| `SQ_INSTS_VALU` | 2,239,872 | 2,187.375 |
| `SQ_INSTS_SALU` | 163,072 | 159.250 |
| `SQ_INSTS_LDS` | 381,952 | 373.000 |
| `SQ_INSTS_MFMA` | 65,536 | 64.000 |

与已有 current-vLLM T=2048 aggregate（同 grid/WG、同 shape）归一化后为 VMEM=57.000、VALU=1362.875、SALU=63.3125、LDS=298.3125、MFMA=64.000；R4 production excess 是 **+140.500 VMEM、+824.500 VALU、+95.9375 SALU、+74.6875 LDS，MFMA=0**。历史 T=8192 aggregate 的约 `+138.625/+840.625/+97.484` 仍只代表 kernel-wide normalization，不是 PC bucket。

### 11.2 fresh-process public-Eager body benchmark

原始结果：`v2/benchmark/summary.json`。每个实现、长度、session 都是 fresh process；无 graph capture；编译不计入 HIP event；没有 private launch。

T=2048 的 public-Eager identity sidecar：`qwen_r4_tail_production_t2048_identity.json`；它记录了同一个 symbol、grid/WG、`emit_audit=false` 和 canonical HSACO/link SHA。`summary.json` 的 T=1024/2048/8192 specialization 本身不写二进制 hash，不能把不同 `num_tokens` 的 module 误说成同一 HSACO。

| T | R4-tail median (ms) | current-vLLM Eager median (ms) |
| ---: | ---: | ---: |
| 1024 | 0.2230915 | 0.1078402 |
| 2048 | 0.4095785 | 0.1581248 |
| 8192 | 1.4136603 | 0.4575195 |

1024→8192 slope：R4-tail **10.6301 µs/chunk**，current-vLLM **3.12214 µs/chunk**。这只是冻结基线的结果，不创建新候选。

### 11.3 ATT/PC sampling 的真实状态

PC sampling（stochastic/host-trap）在本机均返回 `Given PC sampling configuration is not supported on any of the agents`，所以没有伪造 PC sample。

ATT 使用 `/tmp/rocprof-trace-decoder-016/.../librocprof-trace-decoder.so`，原始文件：
`v2/att_gpu0/trace/0364d3a007f9/755536_62421_shader_engine_0_14.att`（193,416 B）。其 code object `...code_object_id_5.out` 的 SHA 与 production HSACO 同为 `5ebff98f...`。用 production ISA 作为 decoder callback 后，得到两个 wave（CU=1，SIMD=0/3），指令数分别 7,844 和 7,843，真实 PC 非零地址各 2,003 个；不是 stub ISA 的静态乘法。

decoder 的 INFO 记录是：

```text
INFO=3 = "Wave incomplete: The trace was cutoff before all waves ended."
```

两个部分 wave 的 category aggregate 为：

| category | 两 wave 合计 |
| --- | ---: |
| SMEM | 6 |
| SALU | 804 |
| VALU | 10,136 |
| FLAT（含 global/flat VMEM） | 938 |
| LDS | 1,678 |
| IMMED | 1,775 |
| JUMP/NEXT | 174/176 |

这不是完整 dispatch：例如简单按 64 waves 放大，FLAT 只有 30,016，不能闭合 production PMC 的 VMEM=202,240；INFO=3 也明确说明 wave 在结束前被截断。故该 trace 只能证明 production code object 的实际 PC/控制流确实被采到，不能给出完整 dispatch 的每-range dynamic count。

## 12. PC/range bucket 闭合结论

production ISA 的静态 range anchor 已保留在第 5 节，并以 production 文件重新核对：

| bucket | production ISA anchor | 动态状态 |
| --- | --- | --- |
| initial state / W-K prologue | `0x1B00–0x1C00`、`0x1C7C–0x1CB4` 的 `global_load_dwordx4` | ATT 部分 wave 可见；dispatch 计数未闭合 |
| LDS initial commit | `0x1D64–0x1E14` 的 `ds_write_b128` | 部分 wave；未闭合 |
| tail W/K issue | `0x24DC–0x2588` 的 16 个 `global_load_dwordx4` | 部分 wave；未闭合 |
| H/output stores | `0x2A54` 起的 `global_store_short` 家族 | 部分 wave；未闭合 |
| pred/update LDS/MFMA | `0x5200–0x55D8` 的 `ds_read_*`/`v_mfma_*` | 部分 wave；未闭合 |
| final state / loop-control | `0x55E0` 之后的 branch/store/control | 部分 wave；未闭合 |

由于 INFO=3，不能把这些 PC 的部分 wave 次数外推成每 dispatch 次数。因此本轮严格结果是：

- production VMEM 动态差额覆盖：**0% 可闭合**（静态 site 覆盖仍为 100%，但不满足动态门槛）；
- production VALU/SALU 动态差额覆盖：**0% 可闭合**；
- `+140.500/+824.500/+95.9375`（以及历史 `+138.625/+840.625/+97.484`）只能报告为 kernel-wide PMC aggregate，不能归因到 W/K/U/state/decay/output/tail 任一 bucket；
- 因此不能回答“139 VMEM、840 VALU、98 SALU 分别来自哪些 PC bucket”，也不能根据静态 ISA 猜测下一 layout。

## 13. 最终决策（只决策，不编码）

本轮已经解决此前的环境、production HSACO 和同源性阻断：T=2048 benchmark probe、PMC 和 ATT 使用同一个 symbol、grid/WG、constexpr specialization 和 HSACO SHA。production ISA 也确认 audit-only sinks 已消失。

但 ATT 的官方 `Wave incomplete` 使每-PC/range dynamic executions 无法与 kernel-wide PMC 闭合；PC sampling 硬件不支持。故用户要求的“VMEM ≥90%、VALU/SALU ≥70%”未满足，**A/B/C/D 均不选，不修改任何真实机器路径，不创建性能候选**。下一步若继续，只能先换用能返回完整 wave-end 的 ATT/PC trace（或修复 profiler/decoder 版本匹配）并重新取得同一 production HSACO 的完整分桶；在闭合前仍禁止 layout、lowering、planner 和 kernel 数学改动。

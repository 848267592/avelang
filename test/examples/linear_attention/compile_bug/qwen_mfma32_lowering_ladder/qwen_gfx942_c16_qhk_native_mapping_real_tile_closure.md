# Qwen gfx942 C16：Q/H/K Native Mapping Recovery 与 Real-Tile Numerical Closure

## 结论

本轮 C16 已完成，最终状态为：

    C16_QHK_REAL_TILE_GO_FOR_FULL_CHUNKO

这个 GO 只表示 selected-native physical mapping 已恢复，并且 Q、H、K 每个
role 都经过了真实 GPU 的 global -> LDS -> MFMA 数值闭环。它授权下一轮另行
设计 full chunk-o integration；它不等于 full chunk-o 已接入，也不等于性能或
Public Eager 晋级。

本轮没有做：

- full chunk-o integration；
- T2048/T8192 latency、PMC 或 Public Eager；
- V mapping 重做；
- packet width、LDS layout、barrier、scheduler、recurrence 或 RA sweep；
- production selector 或 production dispatch 修改。

完成的硬证据：

| gate | 结果 |
|:--|:--|
| selected target | T2048、WG256、gfx942、wave64、MFMA32x32x8 BF16 |
| Q physical mapping | 5/5 patterns，debug 和 raw MFMA 均 bit-exact |
| H physical mapping | 5/5 patterns，debug 和 raw MFMA 均 bit-exact |
| K physical mapping | 5/5 patterns，debug 和 raw MFMA 均 bit-exact |
| Q dual consumer | 5/5；两个 consumer 都 exact；单一 physical producer |
| C15 V regression | 10/10 exact，V 未修改 |
| C13/C14 static/codegen tests | 7/7 |
| direct block-dot regressions | 14/14 |
| private/spill | 所有 C16 role private=0，spill=0 |
| generic ds_bpermute | 0 |
| final ISA runtime div/rem | 0 |

机器可读结果：

- stage6z_c16_qhk_native_mapping.json
- stage6z_c16_q_real_tile.json
- stage6z_c16_h_real_tile.json
- stage6z_c16_k_real_tile.json
- stage6z_c16_q_dual_consumer.json
- stage6z_c16_machine_evidence.json
- stage6z_c16_regression_results.json

## 1. 本轮要解决的问题

C15 已经证明 V 可以走真实的：

    ordinary global V
     -> distributed ownership
     -> physical shared placement
     -> packed LDS store
     -> CTA barrier
     -> encoded LDS read
     -> MFMA32

但 C15 不能把 Q/H/K 直接等同处理。此前 Q/H/K 只有 C13 typed
representation 和 C14 synthetic codegen，没有同时满足：

1. selected native 的 per-role lane/register producer mapping；
2. exact shared byte offset；
3. dot operand consumer-visible fragment mapping；
4. ordinary global source 的真实 GPU 数值闭环；
5. Q 的一个 physical source 同时服务 Q@H 与 Q@K。

C16 因此只做 mapping recovery 和 numerical closure，不把任何结果扩展成
full kernel 性能结论。

## 2. 冻结的 selected native target

本轮只使用 T2048 的 WG256 selected native artifact，没有混入长文本 T8192
的 WG128 specialization。

| 属性 | selected native |
|:--|:--|
| architecture | gfx942 |
| wavefront | 64 |
| BT/BV/BK | 64/64/32 |
| workgroup | 256 |
| TTGIR num_warps | 4 |
| TTGIR num_stages | 3 |
| native shared | 24576 B |
| native scratch | 0 B |
| MFMA | v_mfma_f32_32x32x8_bf16 |
| TTGIR SHA256 | b53b6a9fbe10654a88654abbf6cc49be1b1af677547bf0600695bd5ff4d8495c |
| native HSACO SHA256 | 9cc107ecf4f9b8529694fd07f4532174ba98593aab206fbddf628dbf53bfb95c |

原始工件目录：

    codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/

关键文件为 chunk_fwd_kernel_o.ttgir、chunk_fwd_kernel_o.llir、
chunk_fwd_kernel_o.amdgcn、chunk_fwd_kernel_o.hsaco。

## 3. Mapping 恢复方法

本轮没有从变量名猜 mapping。证据顺序是：

1. selected TTGIR 的 literal encoding；
2. selected LLVM 中的 tid/wave/lane shift、mask、OR/XOR 和 LDS base；
3. final ISA 对这些公式的指令对应；
4. C16 的独立 logical reference；
5. 真实 GPU 的 BF16 LDS readback 与 raw MFMA 输出。

selected TTGIR 的核心 encoding：

    blocked  = sizePerThread [2,8], threadsPerWarp [8,8],
               warpsPerCTA [4,1], order [1,0]
    blocked1 = sizePerThread [8,1], threadsPerWarp [4,16],
               warpsPerCTA [1,4], order [0,1]
    blocked2 = sizePerThread [1,8], threadsPerWarp [16,4],
               warpsPerCTA [4,1], order [1,0]
    mma      = version 3, warpsPerCTA [2,2],
               instrShape [32,32,8], isTransposed true
    shared   = vec 4, perPhase 2, maxPhase 8, order [1,0]
    shared1  = vec 1, perPhase 1, maxPhase 1, order [1,0]
    shared2  = vec 4, perPhase 2, maxPhase 8, order [0,1]

三条 role mapping 的完整规则和 sample entries 在
stage6z_c16_qhk_native_mapping.json 中保存。

## 4. Q exact mapping

Q logical tile 是 [64,32]，使用 blocked2：

    wave              = row // 16
    lane_high         = row % 16
    lane_low          = (col // 8) % 4
    lane              = lane_high * 4 + lane_low
    packet_slot       = (col % 8) // 4
    element_in_packet = col % 4
    register_slot     = packet_slot * 4 + element_in_packet

一个 lane 在 Q 的 innermost 维持有 8 个 BF16，分成两个 BF16x4 packet。
shared physical element offset 为：

    phase        = (row >> 1) & 7
    block_no     = (row >> 4) & 7
    swizzle      = phase xor block_no
    inner_group  = col >> 2
    physical_col = ((inner_group xor swizzle) << 2) | (col & 3)
    element_off  = row * 32 + physical_col
    byte_off     = element_off * 2

代表性 entry：

| Q logical (row,col) | wave | lane | register | packet | shared byte offset |
|:--|--:|--:|--:|--:|--:|
| (0,0) | 0 | 0 | 0 | 0 | 0 |
| (0,4) | 0 | 0 | 4 | 1 | 8 |
| (0,8) | 0 | 1 | 0 | 0 | 16 |
| (16,0) | 1 | 0 | 0 | 0 | 1032 |
| (32,0) | 2 | 0 | 0 | 0 | 2064 |
| (48,24) | 3 | 3 | 0 | 0 | 3112 |
| (63,31) | 3 | 63 | 7 | 1 | 4062 |

Q consumer 是 dot opIdx=0、kWidth=4。selected LLVM 的 Q operand base
使用 %120，四个 MFMA K-width fragment 使用：

    %120, %120 xor 16, %120 xor 32, %120 xor 48

C16 Q real-tile kernel 使用真实 global Q -> vector<4xbf16> -> shared ->
barrier -> encoded read -> MFMA32，而不是把 source 直接当成已经排好的
fragment。

## 5. H exact mapping 和 transpose

H 的 distributed producer ownership 与 Q 相同，仍然是 blocked2：

    wave              = row // 16
    lane              = (row % 16) * 4 + ((col // 8) % 4)
    packet_slot        = (col % 8) // 4
    register_slot      = packet_slot * 4 + (col % 4)

H 使用 shared1：

    vec=1, perPhase=1, maxPhase=1, order=[1,0]

H producer 的 shared element offset：

    element_off = row * 32 + col
    byte_off    = element_off * 2

代表性 entry：

| H logical (row,col) | wave | lane | register | packet | shared byte offset |
|:--|--:|--:|--:|--:|--:|
| (0,0) | 0 | 0 | 0 | 0 | 0 |
| (0,4) | 0 | 0 | 4 | 1 | 8 |
| (0,8) | 0 | 1 | 0 | 0 | 16 |
| (16,0) | 1 | 0 | 0 | 0 | 1024 |
| (48,24) | 3 | 3 | 0 | 0 | 3120 |
| (63,31) | 3 | 63 | 7 | 1 | 4094 |

### H transpose 的准确解释

这里的 transpose 不是 generic runtime transpose，也不是第二份完整 LDS
transpose tile：

- producer 仍按 blocked2 的 lane/register ownership 读入 H；
- shared1 是 vec=1、perPhase=1、maxPhase=1，没有 Q/K 的 rotating swizzle；
- consumer 是 dot opIdx=1；
- selected LLVM 的 H B fragment base 是 %127，后续为 %127、%127 xor
  16、%127 xor 32、%127 xor 48；
- selected H final ISA 没有 ds_bpermute，也没有 H body 的 generic
  v_perm_b32。

本轮把它归类为固定的 consumer-side LDS address/register recipe。它由
native literal layout 和固定 operand lowering 表达，不是任意 runtime matrix
transpose，也没有证据表明它需要跨 lane 的通用交换。

H raw 输出采用独立的 I @ H^T reference 检查，5 个 pattern 全部 exact。

## 6. K exact mapping

K logical tile 是 [32,64]，使用 blocked1：

    wave              = col // 16
    lane_high         = col % 16
    lane_low          = row // 8
    lane              = lane_high * 4 + lane_low
    packet_slot       = (row % 8) // 4
    element_in_packet = row % 4
    register_slot     = packet_slot * 4 + element_in_packet

这与 Q/H 相反：wave 沿 logical column 展开，lane 的低两位选择 K row 的
8-element group。

shared2 使用 order=[0,1]：

    shared_logical_coordinate = [col, row]
    element_off               = col * 32 + row
    byte_off                  = 2 * (col * 32 + row)

代表性 entry：

| K logical (row,col) | wave | lane | register | packet | shared coordinate | byte offset |
|:--|--:|--:|--:|--:|:--|--:|
| (0,0) | 0 | 0 | 0 | 0 | (0,0) | 0 |
| (4,0) | 0 | 0 | 4 | 1 | (0,4) | 8 |
| (8,0) | 0 | 1 | 0 | 0 | (0,8) | 16 |
| (0,16) | 1 | 0 | 0 | 0 | (16,0) | 1024 |
| (31,63) | 3 | 63 | 7 | 1 | (63,31) | 4094 |

K consumer 使用 dot opIdx=1、kWidth=4。selected LLVM 的 K physical base
可以写成：

    t113 = (tid << 6) & 1984
    t114 = (tid << 2) & 56
    t115 = (tid >> 2) & 8
    t116 = (tid << 5) & 2048
    t117 = t113 | t116
    t118 = t117 | t115
    Kbase = shared_base + t118

之后仍然是 base 加 16/32/48 byte 的四个 K-width reads。C16 pass 中的
emitC16NativeKPacketElementOffset 使用相同的有限 plan algebra。

### K packet 说明

selected native 的 blocked1 语义是连续的八-BF16 packet。C16 real-tile
source tensor 是普通 row-major [32,64] diagnostic tensor，因此一个 logical
K packet 的四个 source elements 在该测试输入中不是相邻地址；C16 source
先执行四个 scalar global BF16 reads，再用 vector<4xbf16> 一次写入 shared。

这必须与两件事区分：

1. 它没有把四个值逐元素 scatter 到 LDS；post-block MLIR 中是
   vector.store vector<4xbf16>，带 c16.packed_lds_store；
2. 它没有声称 C16 是 full native performance path，也没有声称 diagnostic
   source 的 global packet width 已经等价于 Triton native packet。native packet
   形状由 TTGIR/LLVM 恢复，full integration 需要在真实 Q/K global ABI 下
   另行接入。

因此 C16 关闭了 scalar LDS scatter fallback 问题，但没有把 diagnostic
source load 误写成 native performance claim。

## 7. Q single physical producer / dual consumer

Q 是额外 gate。普通 Q real-tile 只验证一个 consumer，不能证明 Q@H 和
Q@K 不会各自复制一次 Q。

新增 audit-only kernel：

    one source Q tile
      -> Q shared physical tile
      -> consumer A: block_dot_bf16_f32_operand
      -> consumer B: block_dot_bf16_f32_operand_transposed

第二个 transposed spelling 只用于给两个 consumer 不同的 IR identity。在
AVELANG_C16_Q_DUAL=1 下，两者都选择 Q physical role，第二个 op 不再执行
producer。

post-block MLIR 机械证据：

| 证据 | 数量 |
|:--|--:|
| c16.q_dual_producer | 1 |
| c16.real_tile_producer | 2 |
| c16.packed_lds_store | 2 |
| block-dot MFMA operand ops | 2 |
| gpu.barrier | 2 |

两个 producer/store marker 是一个 Q tile 的两个 BF16x4 packet，不是两次
Q tile producer pass。第二个 block-dot op 没有 c16.q_dual_producer。

Q dual 的 5 个 patterns 均满足：

    debug exact = true
    consumer A raw exact = true
    consumer B raw exact = true
    high fragment exact = true
    finite = true
    single physical producer = true

dual HSACO SHA256：

    e39a7eb571cc4cd8bb02bf20ce76c2a25837d1b4023afa91213dffdf4c66828f

## 8. C13/C14 representation 是否足够

结论：足够表达 C16 的 Q/H/K exact mapping，不需要新增
QwenQMappingAttr、QwenHMappingAttr 或 QwenKMappingAttr。

实际被 C16 block-dot op 消费的 attrs：

    c13.distributed = ave.distributed_encoding
    c13.shared      = ave.shared_encoding
    c13.mfma        = ave.mfma_encoding
    c13.dot         = ave.dot_operand_encoding
    c13.transform   = ave.static_transform
    c14.static_physical
    c14.mfma_callee = _avelang_amdgpu_rocdl_mfma_f32_32x32x8bf16_1k
    c16.real_tile_role = Q/H/K

c16.real_tile_role 是实验 gate，不是 public role-specific layout
abstraction。physical meaning 仍由 C13 typed attrs 和 compiler-owned
C14StaticPhysicalPlan 承载。

实现位置：

- emitC16NativeSharedElementOffset：
  lower_qwen_block_dot_pass.cc:265
- emitC16NativeKPacketElementOffset：
  lower_qwen_block_dot_pass.cc:341
- Q/H/K 统一 producer helper：
  lower_qwen_block_dot_pass.cc:635
- C16 role-parametric consumer lowering：
  lower_qwen_block_dot_pass.cc:2041
- Q dual producer suppression：
  lower_qwen_block_dot_pass.cc:371-386、2049-2105

重要性质：

- shape、vec、phase、order、dot opIdx 和 transform 都由 compiler plan
  静态决定；
- C++ 中的除法只读取静态 plan 常量，不生成 runtime DivUI/RemUI；
- final ISA 没有 v_div、s_div、v_rem、s_rem；
- final ISA 没有 generic ds_bpermute；
- 没有 private array 保存完整 tile；
- 没有第二份完整 transpose LDS tile；
- 没有用 identity readback 代替真实 MFMA closure。

## 9. C16 source 和真实 GPU closure

实验源文件：

    test/examples/linear_attention/vllm_compare/repro_qwen_gfx942_c16_qhk_native_mapping.py

关键结构：

    ordinary BF16 source
     -> role-parametric C16 producer
     -> real 64x64 workgroup LDS
     -> real CTA barrier
     -> C16 encoded offset readback
     -> existing MFMA32 operand lowering
     -> raw FP32 fragment output

identity stage 只作为已知的 MFMA control operand。它在 source 中用 256-thread
CTA 的 16 轮覆盖完整 4096-element 64x64 tile，避免旧版本的越界初始化。

Q/H/K 每个 role 同时检查：

1. BF16 debug readback 与 source byte-exact；
2. raw MFMA low fragment 与独立 Python logical reference exact；
3. high fragment exact；
4. finite；
5. first mismatch（若有）直接报告。

独立 reference 没有调用 compiler mapper：

- Q：source[row, col % 32]；
- H：source[col, row % 32]，对应 I @ H^T；
- K：source[row % 32, col]，对应 I @ K。

## 10. Correctness 结果

### Q/H/K regular real tile

| role | patterns | debug exact | raw low/high exact | finite | max abs |
|:--|--:|--:|--:|--:|--:|
| Q | 5 | 5/5 | 5/5 | 5/5 | 0 |
| H | 5 | 5/5 | 5/5 | 5/5 | 0 |
| K | 5 | 5/5 | 5/5 | 5/5 | 0 |

patterns 是 zero、one_hot、row_code、col_code、checker。

### Q dual consumer

| check | result |
|:--|:--:|
| patterns | 5 |
| debug byte exact | 5/5 |
| consumer A raw exact | 5/5 |
| consumer B raw exact | 5/5 |
| high fragment exact | 5/5 |
| finite | 5/5 |
| single physical producer | 5/5 |

### C15 V 回归

C15 V real physical tile 使用原有 source 和 mapping，没有改 V。C16 后重新在
GPU 上运行 10 个 structured patterns：

    debug byte exact: 10/10
    raw fragment exact: 10/10
    finite: 10/10

## 11. Machine evidence

### 11.1 HSACO metadata

| kernel | VGPR | AGPR | SGPR | LDS | private | VGPR spill | SGPR spill |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Q | 48 | 16 | 18 | 16384 B | 0 | 0 | 0 |
| H | 36 | 16 | 14 | 16384 B | 0 | 0 | 0 |
| K | 44 | 16 | 14 | 16384 B | 0 | 0 | 0 |
| Q dual | 48 | 16 | 20 | 16384 B | 0 | 0 | 0 |

这些是 code-object metadata，不是 rocprof 的 Accum_VGPR_Count。本轮没有
运行 rocprof，因此没有把 code-object AGPR 字段冒充动态 profiler 指标。

### 11.2 exact-LTO MIR

每个 regular role 和 Q dual 都保存完整 kernel_section_00.mir 到
kernel_section_19.mir。代表段：

| section | 含义 |
|:--|:--|
| 06 | pre-greedy MIR |
| 07 | post-greedy MIR |
| 08 | post-virtregrewriter |
| 09 | post-prologepilog |

所有 role 的 post-greedy 证据：

    SI_SPILL_AV32_SAVE = 0
    SI_SPILL_AV64_SAVE = 0
    SI_SPILL_AV32_RELOAD = 0
    SI_SPILL_AV64_RELOAD = 0

post-virtregrewriter/prologepilog 段没有 virtual register 残留，private
segment 仍为 0。

### 11.3 final ISA lexical evidence

下面是各 real-tile HSACO selected kernel body 的 static lexical count。
它们只证明 lowering 形成了真实指令，不是动态工作量或性能数据。

| role | MFMA32 | ds_read | ds_write | barrier | waitcnt | global-load family | global-store family | ds_bpermute | v_perm |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Q | 4 | 25 | 34 | 2 | 19 | 4 | 14 | 0 | 9 |
| H | 4 | 7 | 49 | 2 | 18 | 3 | 13 | 0 | 0 |
| K | 4 | 5 | 49 | 2 | 19 | 10 | 18 | 0 | 4 |

Q dual 的两个 source operations 各自对应 MFMA column-half 序列；post-block
MLIR 的 two-operand evidence 比 static lexical count 更适合证明 dual
consumer，因为 final ISA 同时包含 diagnostic output code。

整份 C16 diagnostic ISA 中仍可能有 ds_write_b16，这是 identity-stage
初始化或 BF16 debug readback 的 lexical code，不能直接解释成 real tile
producer scalar scatter。real producer 的 source-of-truth 是 post-block MLIR
的 vector.store vector<4xbf16> 以及 c16.packed_lds_store attr。

## 12. C13/C14/C15 和 block-dot 回归

### C13/C14 compiler tests

在 ljd_qwen_vllm_avelang_rocm722 容器中用重新 build 的 build-vllm-rocm722
运行：

    ave-lang-parser-test                 PASS
    mlir_generator_test                   PASS
    static_physical_layout_test          PASS
    lower_to_llvm_test                   PASS
    gpu_outlining_test                   PASS
    amdgpu_codegen_test                  PASS
    c14_static_physical_codegen_test     PASS

总计 7/7。

### direct block-dot regression

已有 direct-K64 block-dot 测试集合收集到 14 个 test，结果：

    14 passed, 0 failed

覆盖 direct K64 block-dot A/B、BV32 cooperative ownership、typed operand A/B
和 precomputed V-decay control。

### C15 V

fresh GPU rerun 为 10/10，C15 V pass 和 source 都没有修改。

## 13. 修改内容

### compiler

修改：

    lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc

主要内容：

1. 新增 emitC16NativeSharedElementOffset，按 selected Q/H shared order、
   vec、phase、maxPhase 生成有限地址代数；
2. 新增 emitC16NativeKPacketElementOffset，表达 selected K packet 的
   col * 32 + row physical placement；
3. Q/H/K 共用 emitC16RealQHKTile producer helper；
4. Q/H 使用 vector<4xbf16> global load + LDS vector store；
5. K 将 source gather 的四个 BF16 值组合成 vector<4xbf16> 后写 LDS；
6. C16 consumer 把 C13 distributed/shared/MFMA/dot/transform attrs 继续
   带到 existing first-class MFMA operand boundary；
7. Q dual gate 只在 audit env 下抑制第二个 producer，保持两个 consumer。

没有新增 public Qwen layout attr，也没有修改 allocator、RA、production
selector、V mapping 或 full chunk-o。

### experimental source

修改：

    test/examples/linear_attention/vllm_compare/repro_qwen_gfx942_c16_qhk_native_mapping.py

它包含 Q/H/K 三个 regular real-tile kernel 和一个 Q dual-consumer audit
kernel。每个 launch 都显式设置 AVELANG_C16_REAL_TILE_ROLE，避免编译器
环境变量残留造成 role 错配。

## 14. 为什么这是 GO，但还不是性能 GO

本轮证明：

1. Q/H/K 的 selected native lane/register/shared/dot mapping 可以由 C13/C14
   generic typed representation 承载；
2. Avelang 可以把这些 attrs 传到真实 GPU machine graph，而不是在 MLIR
   阶段丢失；
3. Q/H/K 真实 global/LDS/barrier/MFMA numerical closure 成立；
4. Q 的一个 physical LDS source 可以被两个不同 consumer 使用。

但本轮没有证明：

- full chunk-o 的 phase ownership 已经正确组合；
- full recurrence/chunk-o live range 不会产生 spill；
- full pipeline 的 VMEM/LDS/VALU 与 native 接近；
- C16 source packet 的 global ABI 已经是 full native packet layout；
- full kernel latency 会改善。

因此当前正确表述是：

    native physical mapping and numerical representation: closed
    full chunk-o performance and integration: not run

## 15. 最终决策和边界

最终决策：

    C16_QHK_REAL_TILE_GO_FOR_FULL_CHUNKO

下一轮可以基于已闭环的 Q/H/K/V physical contracts 单独设计 full chunk-o
integration。下一轮必须重新建立 full correctness gate，不能把 C16 real-tile
exact 直接当成 full recurrence exact。

本轮明确没有：

- 创建 Q2/H2/K2 mapping variant；
- 跑 C16 performance；
- 跑 PMC；
- 接 Z5B；
- 接 X2/full chunk-o；
- 建立 Public Eager 结果；
- 修改 production。

## 16. 复现命令

### build

    docker exec ljd_qwen_vllm_avelang_rocm722 sh -lc \
      'cd /workspace/project/avelang && cmake --build build-vllm-rocm722 -j2'

### C16 Q/H/K numerical closure

    docker exec ljd_qwen_vllm_avelang_rocm722 sh -lc \
      'cd /workspace/project/avelang && \
       export PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare && \
       export AVELANG_BLOCK_DOT_LOWERING=specialized && \
       export AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT=1 && \
       python3 test/examples/linear_attention/vllm_compare/repro_qwen_gfx942_c16_qhk_native_mapping.py \
         --dump-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_gfx942_c16_qhk_native_mapping_final_v2'

### C13/C14 static regression

    ctest --test-dir build-vllm-rocm722 --output-on-failure -R \
      'ave-lang-parser-test|mlir_generator_test|static_physical_layout_test|lower_to_llvm_test|gpu_outlining_test|amdgpu_codegen_test|c14_static_physical_codegen_test'

### C15 V regression

    export AVELANG_C15_REAL_TILE=1
    python3 test/examples/linear_attention/vllm_compare/repro_qwen_gfx942_c15_real_physical_tile_v.py \
      --dump-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_gfx942_c15_real_tile_c16_regression

## 17. 工件索引

本轮完整工件目录：

    codex_qwen_gfx942_c16_qhk_native_mapping_final_v2/

目录包含：

    Q/{MLIR,LLVM,exact_lto,MIR,ISA,HSACO metadata}
    H/{MLIR,LLVM,exact_lto,MIR,ISA,HSACO metadata}
    K/{MLIR,LLVM,exact_lto,MIR,ISA,HSACO metadata}
    Q_DUAL/{MLIR,LLVM,exact_lto,MIR,ISA,HSACO metadata}
    correctness.json
    source_identity.json

C16 到此停止。任何 full chunk-o 实验必须以本报告的四个 physical contract
和新的 full-sequence correctness 为输入，不能跳过 mapping closure 直接复用
静态 synthetic 结论。


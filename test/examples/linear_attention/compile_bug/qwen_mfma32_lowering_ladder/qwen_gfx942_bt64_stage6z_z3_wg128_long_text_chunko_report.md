# Qwen gfx942 BT64 Stage 6Z Z3：WG128 长文本 chunk-o 候选

## 结论

**Z3 No-Go，不能晋级，也没有接入 X2 full graph。**

这轮先完成了 native vLLM public API 的逐长度 fresh selector capture，确认
`T=4096/8192/16384` 的确选中 `BT64/BV64/BK32、WG128、2 waves、2 stages`，所以
WG128 候选有真实 selector 依据。随后实现唯一的 Z3 source candidate，继承 Z2
的 direct-fragment 路径，只把四 wave 的 value-half ownership 改成两 wave 顺序
处理两个 32-value half，并保持 BF16 ABI、MFMA32 geometry、`[64,64]` tile ownership
和现有 phase barrier 语义。

Z3 的短长度 one-arm correctness 已通过：`T=64/512/2048/4096` 与 Z2 byte-exact。
但长文本 gate 没有通过：`T=8192` 没有形成可接受的 candidate 结果，`T=16384`
candidate 进程发生 GPU memory access fault，返回 `RC=134`。按照预注册规则，
correctness 未通过就不能运行性能，因此没有生成 Z3 latency、PMC 或 selector
promotion 结果。

所以当前 Stage 6Z 的准确状态是：

| 候选 | 状态 |
|:--|:--|
| Z1 WG256 | correctness 通过、性能正收益，但 41 barriers，No-Go |
| Z2 WG256 phase-aware direct-fragment | 当前 isolated 最快，27 barriers，No-Go |
| Z3 WG128 long-text | 短文本通过，长文本 correctness No-Go |
| length-based WG selector | 未建立 |
| X2 full-graph 接入 | 未执行 |

## 1. 冻结边界

Z3 没有修改以下内容：

- X2 immutable current-vLLM recurrence HSACO；
- R4 recurrence；
- Qwen chunk-o 数学和 causal mask；
- BF16 `q/k/v_new/h/out`、FP32 `g` 和 FP32 accumulator；
- `BT64/BV64/BK32`；
- 每个 `[64,64]` 输出 tile 一个 CTA、每 chunk-head 两个 CTA；
- MFMA32 `v_mfma_f32_32x32x8_bf16`；
- K32 accumulation order；
- public ABI 和输出 layout；
- allocator/RA、compiler barrier-elision 和 production selector。

Z3 唯一 intended variable 是 workgroup ownership：

```text
Z2: 4 waves / WG256，wave 0..3 分别持有 row/value half
Z3: 2 waves / WG128，每个 wave 负责一个 row half，两个 value half 顺序处理
```

Z3 不采用 `w_next/k_next` 普通 local array、不增加第二套完整 LDS、不做
double-buffer、不批量删除 barrier，也没有把 native WG128 的 Triton ISA 直接复制
进 AveLang source。

## 2. 为什么先做 selector capture

Stage 6Z 的 gate 要求不能从 T2048/T8192 插值猜 selector。因此每个长度都在
fresh process 中实际调用 native vLLM `chunk_fwd_o`，并保存 selected kernel 的：

```text
source / TTIR / TTGIR / LLVM IR / AMDGCN ISA / HSACO / code-object readobj
```

工件根目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_native_chunko/native_refresh/
```

每个 `T{T}/native_capture.json` 中的 `runtime_selected_config` 是实际命中的配置，
而不是从 metadata 反推：

| T | BK | BV | num_warps | WG | num_stages | shared metadata | static MFMA32 | static `s_barrier` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 32 | 4 | 256 | 2 | 10240 B | 40 | 11 |
| 1024 | 32 | 64 | 4 | 256 | 2 | 12288 B | 40 | 11 |
| 2048 | 32 | 64 | 4 | 256 | 2 | 12288 B | 40 | 11 |
| 4096 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 |
| 8192 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 |
| 16384 | 32 | 64 | 2 | 128 | 2 | 12288 B | 80 | 11 |

native selector 的 source/IR/ISA 还显示：T512/T1024/T2048 与 T4096/T8192/T16384
使用不同 code object；长文本 WG128 不是一个人为指定的静态假设。

native selected code object 的静态资源如下。这里特意使用 `.agpr_count`、
`.vgpr_count` 等 code-object 字段，不把它们称为动态 rocprof `Accum_VGPR`：

| T | `.vgpr_count` | `.agpr_count` | `.sgpr_count` | private segment | spill |
|---:|---:|---:|---:|---:|:--|
| 512 | 128 | 32 | 87 | 0 B | 0 |
| 1024 | 132 | 32 | 89 | 0 B | 0 |
| 2048 | 132 | 32 | 89 | 0 B | 0 |
| 4096 | 220 | 64 | 76 | 0 B | 0 |
| 8192 | 220 | 64 | 76 | 0 B | 0 |
| 16384 | 220 | 64 | 76 | 0 B | 0 |

本轮 native refresh 没有对窄 chunk-o body 采集同口径动态 `Accum_VGPR`，所以动态
该字段为 N/A。六个 selected code object 都没有 private segment 或 spill。

## 3. native 速度对照

selector capture 之后额外运行了 native body harness：

```text
vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z_eager_body.py
```

契约是 fresh process、current HIP stream、no Graph、warmup=5、repeat=20。首次
public call 负责编译/autotune，明确排除在时间外。每个长度同时测：

1. direct preallocated body：预分配 `o`，直接调用已选择的 `chunk_fwd_kernel_o`；
2. public `chunk_fwd_o`：包含正常 output allocation，作为 eager public 对照。

这不是完整 Qwen public full graph 排名，只是 Z3/Z2 isolated chunk-o 的 native
速度参照。

| T | chunks | native direct body ms | native public `chunk_fwd_o` ms |
|---:|---:|---:|---:|
| 512 | 8 | 0.037696 | 0.045167 |
| 1024 | 16 | 0.039178 | 0.045467 |
| 2048 | 32 | 0.043826 | 0.050555 |
| 4096 | 64 | 0.056483 | 0.063634 |
| 8192 | 128 | 0.089814 | 0.098987 |
| 16384 | 256 | 0.151886 | 0.154569 |

六点线性拟合：

```text
native direct body = 0.030376 ms + 0.469496 us/chunk
native public body = 0.038122 ms + 0.455653 us/chunk
```

这组 native 数据的作用是确认长文本 selector 和比较基线，不能拿来替代 Z3 的
correctness gate，也不能因为 native 本身正常就推断 Z3 ownership 正确。

## 4. Z3 source 结构

新 source：

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_z3_wg128.py
```

Z3 仍使用 Z2 的 16 KiB `phase` shared buffer 和 direct `phase_vec` fragment view：

### Phase A：Q/H 与 inter accumulator

- 两个 wave 分别负责 32-row half；
- Q/H staged 到 phase 后 CTA-wide synchronize；
- 每个 wave 依次计算两个 value half 的 inter accumulator；
- 不重新引入 Z1 的 `frag_words` LDS round trip。

### Phase B：Q/K score

- 两个 source half 依次计算；
- 每个 wave保持自己的 score accumulator；
- score half 写回 phase，保留 Z2 原有 score serialization boundary；
- 没有批量删 `al.syncthreads()`。

### Phase C：score/V-new

- V-new phase publication 后，两个 wave 顺序计算两个 value half；
- 最终直接 BF16 store；
- 没有 FP32 output staging 或额外 global reload。

这个方案的目的不是改变数学，而是让 WG128 下每个 wave 有足够的工作覆盖同一个
`[64,64]` tile，同时减少 wave 数。它并没有复制 Triton 的完整 TTGIR local-buffer
deallocation schedule。

## 5. correctness 过程与第一次错误

### 5.1 第一次 T64 smoke 发现的 source coverage bug

Z3 初版沿用了 WG256 的重复覆盖次数：A/B `rep=8`、C `rep=16`。WG128 下这些循环
只覆盖一半 shared staging，属于明确的 source indexing 错误，不是性能结果。该问题
已修正：

```text
A/B: rep 8 -> 16
C:   rep 16 -> 32
```

修正后的 T64 finite smoke 通过。之后没有把多 kernel 同进程 pytest 的异步 GPU fault
当成可靠的 correctness 结论，而是使用一臂一进程 validator：

```text
vllm_compare/verify_qwen_gdn_bt64_native_chunko_stage6z_z3_single_process.py
```

### 5.2 accepted short-text results

validator 的比较对象是同一组 seeded 输入上的 Z2 reference 和 Z3 candidate，比较
BF16 output 的 byte equality：

| T | Z3 vs Z2 | max abs | 状态 |
|---:|:--|---:|:--|
| 64 | byte-exact | 0 | pass |
| 512 | byte-exact | 0 | pass |
| 2048 | byte-exact | 0 | pass |
| 4096 | byte-exact | 0 | pass |

这四个结果只能证明短文本上当前输入和当前编译结果没有暴露错误，不能外推到
T8192/T16384，也不能证明 WG128 general correctness。

### 5.3 long-text hard stop

| T | 结果 | 解释 |
|---:|:--|:--|
| 8192 | 没有可接受的 candidate 结果 | candidate gate 未形成完整 accepted record |
| 16384 | GPU memory access fault，RC=134 | 不是数值误差，进程被 GPU fault 终止 |

T16384 candidate 实际触发 GPU memory access fault。它不是 `max_abs` 超阈值，不能
通过修改 tolerance 处理。T8192 也没有一条完整、可作为正式通过证据的 candidate
记录。因此长文本 correctness gate 失败，按规则不运行 Z3 body benchmark、不采集
Z3 PMC、不做 WG selector。

## 6. 为什么不能把 Z3 当作“只差一点修复”

Z3 的唯一目标是判断 native 长文本 WG128 ownership 是否可以直接由当前 Avelang
source 复现。现在的证据同时包含：

- 短长度 byte-exact；
- 长文本 memory fault；
- 没有接受的 T8192 candidate；
- 没有可用于性能归因的 Z3 ISA/PMC；
- 没有验证其 shared address/phase reuse 在长文本边界下仍然有效。

因此不能把它写成“WG128 已正确但性能尚未测”，也不能继续用性能门槛替代
correctness。继续工作前必须先定位：

1. Z3 长文本实际 launch grid/`program_id` 到 chunk/head 的映射；
2. A/B/C phase 的所有写入范围是否覆盖完整 `[256,32]`；
3. score 与 V stage 的 phase 地址是否存在越界/别名；
4. 生成的 LLVM/MIR 是否有超出 shared allocation 的地址；
5. WG128 下 loop-unroll 和 `lane_group/wave_id` 组合是否引入未定义值。

这一步应该是 correctness/address audit，不应直接添加更多 WG、BV 或 barrier 变体。

## 7. Gate 判定

| gate | 结果 | 判定 |
|:--|:--|:--|
| native selector 逐长度 fresh capture | 完成 | pass |
| T64/512/2048/4096 Z3 byte-exact | 完成 | pass |
| T8192 accepted correctness | 未完成 | **fail** |
| T16384 correctness | GPU memory fault | **fail** |
| scratch/spill | 未进入正式 candidate 评测 | N/A |
| static barrier <19 | 未进入可接受 candidate | N/A |
| T8192/T16384 比 Z2 快 | 按 correctness stop rule 未测 | N/A |
| X2 full graph / public Eager integration | 未执行 | 保持冻结 |

**最终结论：Z3 不晋级。**

## 8. 与 Z2 的关系

Z2 仍是当前 isolated Avelang chunk-o 的最快版本：

- T2048 相对 Z1 加速约 8.09%；
- T8192 相对 Z1 加速约 15.93%；
- dynamic MFMA 不变；
- LDS、VALU、SALU、AccVGPR 和 trace 均下降；
- 但 static barrier 仍为 27，大于预注册的 `<19`，所以仍是 No-Go。

Z3 没有推翻 Z2，也没有证明 WG128 ownership 在 AveLang 中不可实现；它只证明了
当前这份最小 source 映射还不能通过长文本 correctness gate。下一步若继续，只能
围绕 fault 做精确 address/machine audit；不能直接建立 WG256/WG128 selector，也不能
接入 X2。

## 9. 复现入口

```bash
# native selector/IR/ISA capture 已存在于 native_refresh/T{T}/

# native chunk-o speed
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z_eager_body.py \
  --T 2048 --warmup 5 --repeat 20

# Z3 short-text one-arm correctness（示例）
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/verify_qwen_gdn_bt64_native_chunko_stage6z_z3_single_process.py \
  --mode reference --T 2048 --reference /tmp/z2_T2048.pt

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/verify_qwen_gdn_bt64_native_chunko_stage6z_z3_single_process.py \
  --mode candidate --T 2048 --reference /tmp/z2_T2048.pt
```

Z3 的完整 multi-length pytest 入口保留在：

```text
test_qwen_gdn_bt64_native_chunko_stage6z_z3.py
```

但由于该 candidate 已触发长文本 GPU fault，不应把它当成通过的性能测试入口。

## 最终调试更新：长文本故障已定位并修复

本节覆盖前文的 `5.3 long-text hard stop`、`7. Gate 判定` 和 `8. 与 Z2 的关系`
中的 Z3 长文本未通过结论。前文记录的是修复前状态；下面是修复后的独立复核结果。

### 1. 第一处错误地址

本轮先没有重新跑性能。使用 host-only 地址枚举器，按 Z3 实际 source 公式展开每个
阶段边界，并把修复前公式作为 `--include-buggy-phase-b` 对照；它不启动 GPU kernel，
也不依赖容差或异步输出。

旧 Phase B 的 K staging 为：`idx=tid+rep*128`，`row=floor(idx/32)`，
`source_token=source_half*32+row`，`global_token=chunk_start+source_token`。
旧代码对 K 使用 `rep=0..15`，所以 `row=0..63`；但 consumer 只读取
`phase[score_stage_base+64+lane_col,...]`，其中 `lane_col=0..31`。因此
`row=32..63` 是无 consumer 的死 staging，却仍执行 global K load。

最后一个 chunk 中 `chunk_start=T-64`，第一个非法点是
`source_half=1,row=32,col=0,source_token=64,global_token=T,rep=8,tid=0`。

| T | 第一个非法访问 | 对象 | LDS 是否越界 |
|---:|:---|:---|:---|
| 8192 | `k[token=8192,key_head,feature]` | Phase-B 死 K staging | 否 |
| 16384 | `k[token=16384,key_head,feature]` | Phase-B 死 K staging | 否 |

这解释了 T=16384 的 GPU memory fault，以及 T=8192 偶尔不暴露的原因：是否立即报告
取决于相邻分配和运行时状态。它不是数值误差，也不是 phase shared allocation 越界。

### 2. 源码修复

修复文件为：

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z3_wg128.py
```

Phase B 保留 Q 的 `rep=0..15`，把 K 改成 `rep=0..7`。这样 K 的 row 恰为 `0..31`，
覆盖 consumer 可见的 32 行；MFMA、consumer index、barrier、phase layout 和输出数学
不变。对应 source 位置是该文件 Phase B 的 `:105-139`，修复后的 K loop 在 `:122-128`。

调试过程中还发现 Z2 reference source 有相同的死 K overread，只是此前未在所有进程和
分配布局下暴露。实验 reference 同步修复为 WG256 下 Q `rep=8`、K `rep=4`，文件为：

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py
```

这不是 production path，也没有把修复后的 Z2 重新计入旧性能排名；报告中的 Z2
历史 timing/PMC 仍明确标为修复前历史测量。

### 3. 地址审计结果

工件目录为 `codex_qwen_bt64_stage6z_z3_wg128_debug/`，关键文件是
`address_audit_buggy.json/csv`、`address_audit_fixed.json/csv` 以及
`vllm_compare/audit_qwen_gdn_bt64_native_chunko_stage6z_z3_addresses.py`。

| 路径 | T | checks | valid | invalid | first OOB | phase rows |
|:---|---:|---:|---:|---:|:---|:---|
| buggy Phase-B K | 8192 | 256 | 240 | 16 | `source_half=1,row=32,col=0,token=8192` | `0..255` |
| buggy Phase-B K | 16384 | 256 | 240 | 16 | `source_half=1,row=32,col=0,token=16384` | `0..255` |
| fixed | 64/512/2048/4096/8192/16384 | 224 each | 224 | 0 | none | `0..255` |

launch 映射仍为 `programs=num_chunks*H_V*2`、`v_block_idx=pid%2`、
`value_head_idx=(pid//2)%H_V`、`chunk_idx=pid//(2*H_V)`。没有证据表明 LDS row、
`wave_id`、`lane_group` 或 `program_id` 映射越界。

### 4. LLVM、MIR、ISA 和 HSACO 证据

本轮新增的是 compile-only capture，`launch_executed=false`、`rocprof_executed=false`。
机器工件位于 `codex_qwen_bt64_stage6z_z3_wg128_debug/machine/`，包括
`lowered_llvm.ll`、`pre_lto_amdgcn.s`、`final_isa.s`、`z3_fixed.hsaco`、
`machine/exact_lto/` 和 `machine/machine_evidence.json`。LLVM 保留 `[16384 x i8]`
addrspace(3) phase allocation，也保留 source_half/row/lane_col 对应的 GEP/index
链；exact-LTO replay return code 为 0。

| code-object/ISA 字段 | T=8192 fixed Z3 |
|:---|---:|
| HSACO SHA256 | `a3b05564781bbb191038b506345183a5219e7f6959fd0a1dc5fbfafbaa92206c` |
| `.agpr_count` | 32 |
| `.vgpr_count` | 228 |
| `.sgpr_count` | 38 |
| `.group_segment_fixed_size` | 16384 B |
| `.private_segment_fixed_size` | 0 B |
| `.vgpr_spill_count` / `.sgpr_spill_count` | 0 / 0 |
| static `v_mfma_f32_32x32x8_bf16` | 56 |
| static `s_barrier` | 21 |
| static global load/store | 336 / 32 |
| static LDS read/write | 50 / 288 |
| static `ds_bpermute` | 0 |

`exact_lto/summary.json` 中的 pre-greedy、post-greedy 和 virtregrewriter sections
均没有 `SI_SPILL_AV32_SAVE` 或 `SI_SPILL_AV64_SAVE`。这些是静态 ISA/code-object
数据，不是 rocprof 动态计数。初始 AveLang MLIR printer 在当前 Docker 绑定的单独
T=64 probe 中直接 core dump，因此没有伪造 `initial_mlir.mlir`；source-level audit、
LLVM、MIR、ISA 和 HSACO 工件仍然完整保留。

### 5. 修复后 correctness gate

每个长度使用独立进程；Z3 与修复后的 Z2 reference 做 BF16 byte-exact 比较。

| T | Z3 vs fixed Z2 | max abs | finite | zero-V caller-owned output |
---:|:---:|---:|:---:|:---:|
| 64 | byte-exact | 0.0 | pass | pass |
| 512 | byte-exact | 0.0 | pass | - |
| 2048 | byte-exact | 0.0 | pass | - |
| 4096 | byte-exact | 0.0 | pass | - |
| 8192 | byte-exact | 0.0 | pass | pass |
| 16384 | byte-exact | 0.0 | pass | pass |

zero-V 检查使用预填充 NaN 的 caller-owned output，确认输出被覆盖且最终 finite；它是
补充的边界/输出复用检查，不替代 nonzero 的主 correctness gate。

### 6. 最终判定

本次故障是 Z3/Z2 实验 source 中 Phase-B K staging repetition 与 consumer-visible
row 数不一致造成的越界 global load，不是 Avelang LLVM/AMDGPU lowering、RA 或硬件
MFMA 的问题。修复后 Z3 已通过六个长度的 correctness gate，且 compile-only machine
audit 显示 scratch/spill 为 0。

按本轮 stop rule，本轮没有执行 Z3 性能、PMC、selector 或 X2 full graph。因此最终
状态是：

```text
Z3 correctness: PASS
Z3 compile/resource audit: PASS, scratch/spill=0
Z3 performance promotion: PENDING, no measurement in this debug pass
X2 integration: NOT RUN
```

Z3 不再是“长文本 correctness No-Go”，但还不是新的 isolated performance winner。
下一次若继续，只能用修复后的 source 重新做预注册 body/PMC gate，不能使用修复前 fault
或 Z2 修复前历史性能数据替代新测量。

## 修复后正式重测：Z3 性能与晋级结论

上一节的 `performance: PENDING` 已由本节覆盖。此前 Z3 fault 和 Z2 历史 timing
均不作为本节依据；本节使用 Phase-B dead K overread 修复后的 source，并以固定
Z2、native selected chunk-o 为同口径对照。

### Correctness and machine status

Z3 在 T64/512/1024/2048/4096/8192/16384 的独立进程检查全部 finite，和
Stage6W reference 的最大误差分别为 `1.16e-10`、`1.53e-05`、`1.53e-05`、
`1.53e-05`、`1.53e-05`、`1.53e-05`、`3.05e-05`，均小于 `1/128`；Z3 与
fixed Z2 每个长度 BF16 byte-exact。T64/8192/16384 的 zero-V + NaN
caller-owned output reuse 也全部通过。

重新编译的 fixed Z3 工件在
`codex_qwen_bt64_stage6z_fixed_rerun/machine/z3_T8192/`，HSACO SHA256 为
`a3b05564781bbb191038b506345183a5219e7f6959fd0a1dc5fbfafbaa92206c`。
静态 code-object 为 `VGPR=228, AGPR=32, SGPR=38, LDS=16384 B`，
private segment=0，VGPR/SGPR spill=0；ISA 为 56 MFMA32、21 `s_barrier`、
336 global load、32 global store、50 LDS read、288 LDS write、0
`ds_bpermute`。exact-LTO MIR 的 greedy/virtregrewriter sections 没有 spill。

### Paired body timing

口径是 caller-owned isolated body、current stream、no Graph、warmup=10、
repeat=50、5 fresh-process session、旋转顺序。

| T | fixed Z2 ms | fixed Z3 ms | native selected ms | Z3/Z2 |
|---:|---:|---:|---:|---:|
| 512 | 0.061431 | 0.074491 | 0.036194 | 1.213x |
| 1024 | 0.063634 | 0.078456 | 0.037756 | 1.233x |
| 2048 | 0.077475 | 0.094661 | 0.042763 | 1.222x |
| 4096 | 0.110724 | 0.126388 | 0.056224 | 1.142x |
| 8192 | 0.177203 | 0.220769 | 0.091917 | 1.246x |
| 16384 | 0.317692 | 0.393445 | 0.141591 | 1.238x |

拟合 slope：Z2 `1.048349 us/chunk`，Z3 `1.309077 us/chunk`，native
`0.438248 us/chunk`。Z3 在预注册的三个长文本点全部慢于 Z2：T4096
`+14.15%`，T8192 `+24.60%`，T16384 `+23.84%`。

### Dynamic PMC explanation

| T | arm | MFMA | VMEM | LDS | VALU | SALU | trace us | occupancy | profiler AccVGPR |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | Z2 | 81920 | 475136 | 475136 | 5836800 | 548864 | 55.322 | 15.738% | 32 |
| 2048 | Z3 | 81920 | 458752 | 462848 | 5555200 | 421376 | 73.229 | 8.510% | 164 |
| 8192 | Z2 | 327680 | 1900544 | 1900544 | 23347200 | 2195456 | 158.797 | 36.798% | 32 |
| 8192 | Z3 | 327680 | 1835008 | 1851392 | 22220800 | 1685504 | 208.591 | 18.134% | 164 |
| 16384 | Z2 | 655360 | 3801088 | 3801088 | 46694400 | 4390912 | 299.926 | 40.650% | 32 |
| 16384 | Z3 | 655360 | 3670016 | 3702784 | 44441600 | 3371008 | 391.542 | 20.403% | 164 |

Z3 的 MFMA 数量保持与 Z2 一致，VMEM/LDS/VALU/SALU 小幅减少，但 AccVGPR
从 32 增至 164，occupancy 明显下降，trace 反而增加约 30%。这说明 WG128
ownership 的当前 AveLang lowering 形成了更差的 accumulator/resource shape；
“每 CTA 工作项更少”没有转化成更低的 body latency。

### Final gate

Z3 的 correctness 和 scratch/spill gate 已通过，但 static barrier=21 不满足
`<19`，且 T4096/8192/16384 均未快于 Z2。因此 Z3 **正式 No-Go**：不建立
WG256/WG128 selector，不接入 X2 full graph，不运行 X2 public Eager，也不继续
增加新的 WG/barrier 变体。fixed Z2 只作为当前有效 Avelang isolated baseline，
其旧的修复前性能数字不得继续引用。

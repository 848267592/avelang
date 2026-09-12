# Qwen gfx942 Stage 6Z：BDV2-P1 通用 VALU/Layout Planner 实验

## 结论先行

本轮完成了一个**正确、通用、可回归**的 `block_dot_bf16_f32` planner
优化，但没有形成新的性能 baseline。

P1 的控制范围很窄：在同一份 BDV2 full-scope source 和同一个
`ave.gpu.amdgpu_block_dot_bf16_f32` op 上，增加一个可切换的
`LogicalBlockLayoutPlan`，让 K/H 共享一次 affine ownership/index plan；同时在
P1 分支尝试直接生成每次 MFMA 所需的 `<4 x bf16>` fragment，避免旧路径先生成
`<8 x bf16>` 再逐元素 `extractelement/insertelement` 重建。

结果分成两层：

1. 在 lowered LLVM 文本层，P1 specialized 的 fragment reconstruction 明显减少：
   `extractelement 130 -> 66`，`insertelement 160 -> 96`。
2. 在最终 AMDGPU/LTO 层，P1 specialized 与 BDV2 specialized 生成完全相同的
   HSACO 二进制、相同的 normalized ISA body、相同的 code-object 资源和相同的
   T=2048 动态 PMC。因此这次没有证明“P1 已降低最终机器工作”，而是精确证明了
   当前 planner 表示在 LLVM 之后仍被 AMDGPU/LTO 重新物化/收敛。

P1 全部 correctness 通过，但性能结果为 No-Go：

| T | Z5B | BDV2-S | P1-S | native diagnostic |
|--:|--:|--:|--:|--:|
| 2048 | `0.066899501 ms` | `0.069463000 ms` | `0.069863502 ms` | `0.042483000 ms` |
| 8192 | `0.157854497 ms` | `0.167768501 ms` | `0.167308502 ms` | `0.090734500 ms` |

相对 Z5B，P1 在 T=2048 为 `1.044305x`，在 T=8192 为 `1.059891x`；相对
native diagnostic 分别为 `1.644505x` 和 `1.843935x`。P1 相对 BDV2-S 在 T=2048
慢 `0.4005 us`，T=8192 的 session-median 低 `0.4600 us`，但没有 machine graph
变化，也没有跨长度稳定的正收益证据，不能晋级。

最终状态：

```text
Z5B       = Stage 6Z isolated performance baseline
BDV2-S    = full-scope 通用 block-dot infrastructure，performance No-Go
BDV2-P1   = correctness PASS，LLVM 表示改进，final machine No-Go
native    = same-shape diagnostic，不是 AveLang production path
```

P1 保留为 compiler infrastructure / regression evidence，不接入 X2、selector
或 production。

---

## 1. 实验问题与冻结边界

### 1.1 要回答的问题

BDV2 把 K/H 的 logical global block、producer ownership、shared placement 和
MFMA-B consumer 放进同一个通用 `block_dot_bf16_f32` full-scope lowering 后，
T=2048 的动态 VALU 从 Z5B 的 `7072/CTA` 增加到 `8410/CTA`，增加 `1338/CTA`，
同时 VMEM 从 `672/CTA` 降到 `448/CTA`，LDS 从 `672/CTA` 降到 `464/CTA`。

P1 不是重新设计 block-dot，也不是重新做 raw `buffer_load_x4`。它只审计并尝试
控制新增 VALU 的两个最可能来源：

- full-scope producer/consumer 是否在不同 helper 中重复计算同一 affine map；
- MFMA B operand 是否经历了不必要的 BF16 vector view 和 fragment reconstruction。

### 1.2 冻结内容

以下内容在 Z5B、BDV2-G、BDV2-S、P1-G、P1-S 之间保持不变：

- gfx942、wave64；
- BT64、BV64、BK32；
- WG256、每 chunk-head 两个 CTA；
- BF16 Q/K/H/V-new/output ABI，FP32 g 和 accumulator；
- dedicated full-Q LDS cache；
- phase-separated accumulator 顺序：`inter_acc -> score_acc0 -> score_acc1 -> intra_acc`；
- causal mask、K32 accumulation order、MFMA32 geometry；
- global layout、caller-owned output、正确性比较口径；
- allocator/RA、production selector、X2 recurrence HSACO；
- 不增加新的 Qwen/chunk-o 专用 op。

P1 只有一个 lowering 环境开关：

```text
AVELANG_BLOCK_DOT_LAYOUT_PLANNER=legacy
AVELANG_BLOCK_DOT_LAYOUT_PLANNER=bdv2_p1_affine
```

`legacy` 是 BDV2 当前 lowering，`bdv2_p1_affine` 是 P1。generic/specialized
仍由原有的：

```text
AVELANG_BLOCK_DOT_LOWERING=generic|specialized
```

选择。P1-G 与 P1-S 的 high-level source 相同；本轮性能主 arm 是 P1-S，因为
它保留 BDV2-S 的 gfx942 typed producer。

---

## 2. 现有数据流与 provenance 审计

### 2.1 source 级别的两个 full-scope helper

BDV2 full-scope lowering 的主要入口是：

```text
emitFullScopeProducer
    -> producer ownership / global packet / shared placement

emitGenericOperandBPair
    -> shared B row / MFMA B fragment / MFMA32 call
```

在旧 BDV2 路径里，两个 helper 各自重新构造了部分相同的值：

```text
tid
  -> wave = tid // 64
  -> lane = tid % 64
  -> laneCol = lane % 32
  -> laneGroup = lane // 32
  -> rowHalf = wave // 2
  -> valueHalf = wave % 2
  -> kStage / sourceHalf / stage offset
```

之后又分别构造：

```text
row = producer index
packet = producer index % packet width
packetCol = packet * 8
feature = kStage * 32 + packetCol
qRow / bRow / LDS element offset
```

这类整数运算不是 MFMA 数学，也不是 causal mask；它属于 ownership、global
address、LDS physical address 和 fragment feeding 的控制工作。由于这些变量的
shape、wave64、packet width 和 ownership contract 在本实验中是静态的，P1 的
合理控制杆是将它们保存为一个通用 planner object，而不是在每个 consumer 中再
创建一组新的 SSA index。

### 2.2 P1 的通用 planner

P1 新增内部对象：

```cpp
struct LogicalBlockLayoutPlan {
    bool isH;
    Value tid, kStage, wave, lane, laneCol, laneGroup;
    Value rowHalf, valueHalf;
    Value producerLinear, packetRow, packet, packetCol, feature;
};
```

它位于 `lower_qwen_block_dot_pass.cc`，不是 Qwen-specific address table。K/H
共用这个 object；只有 `isH`、source role 和 transpose metadata 改变逻辑 block
解释。P1 在 `lowerFullScopeOperandMode` 中创建一次，然后传给：

```text
emitFullScopeProducer(..., plannedLayout)
emitGenericOperandBPair(..., plannedLayout)
```

K 的 `valueHalf` 也直接复用 planner 的 SSA value，不再在 K consumer 分支里重新
从 `thread_id` 计算一遍。

对应代码位置：

- `lower_qwen_block_dot_pass.cc:101-114`：前置 index helper 声明；
- `lower_qwen_block_dot_pass.cc:116-185`：`LogicalBlockLayoutPlan` 和 affine map；
- `lower_qwen_block_dot_pass.cc:991-1035`：consumer 复用 planner value；
- `lower_qwen_block_dot_pass.cc:1185` 附近：producer 复用 planner value；
- `lower_qwen_block_dot_pass.cc:1365-1422`：full-scope 创建、传递和 K ownership 复用。

### 2.3 P1 的 typed fragment 尝试

BDV2 legacy consumer 逻辑是：

```text
LDS load <8 x bf16>
  -> extract BF16[0:4]
  -> build <4 x bf16>
  -> MFMA
LDS load <8 x bf16>
  -> extract BF16[4:8]
  -> build <4 x bf16>
  -> MFMA
```

P1 在 `plannedLayout != nullptr` 时改为：

```text
LDS load <4 x bf16>  // offset 0
  -> MFMA
LDS load <4 x bf16>  // offset 4
  -> MFMA
```

specialized arm 直接使用 typed vector local load；generic arm 保留合法的 scalar
fallback，但也沿用同一 `<4 x bf16>` fragment contract。P1 没有改变 MFMA operand
的逻辑值、调用次数、K32 顺序或 accumulator phase。

对应代码位置：

- `lower_qwen_block_dot_pass.cc:1065-1113`：P1 direct fragment path；
- `lower_qwen_block_dot_pass.cc:1115-1149`：BDV2 legacy `<8 x bf16>` + slice path。

---

## 3. same-source / same-contract 证据

### 3.1 source identity

BDV2 legacy 和 P1 使用同一个 source 文件：

```text
test/examples/linear_attention/vllm_compare/
  qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope.py
```

source SHA256：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

P1 选择发生在 late compiler lowering environment，不是 Python source fork。
source、ABI、launch、tile ownership、数学和 output contract 因而保持一致。

### 3.2 initial MLIR 的已知工具缺口

当前 Docker runtime binding 的 `get_mlir()` 在 initial MLIR printer 阶段会
SIGSEGV。本轮 dump 使用：

```text
--skip-initial-mlir
```

因此本报告**不声称**保存了 initial/pre-branch MLIR 的字节级 hash。可证明的
same-source 证据是 source SHA256 相同；可证明的 lowering 分叉证据是同一 source
在不同 planner 环境下得到不同 lowered LLVM/pre-LTO 文件。这个工具缺口不能被
source hash 冒充为 initial MLIR hash。

### 3.3 artifact 位置

BDV2-S：

```text
codex_qwen_bt64_stage6z_bdv2_machine_specialized/
```

P1-S：

```text
codex_qwen_bt64_stage6z_bdv2_p1_machine_specialized/
```

两者都保存：

- `source.py.txt`；
- `lowered_llvm.ll`；
- `pre_lto_amdgcn.s`；
- exact-LTO `kernel_section_00.mir` 至 `kernel_section_19.mir`；
- `exact_lto/linked.hsaco.0.*` bitcode；
- `exact_lto/replay_argv.json`；
- `final_isa.s`；
- `machine_summary.json`；
- code-object notes 和 replay stdout/stderr。

---

## 4. VALU provenance 分层结果

### 4.1 lowered LLVM 文本统计

下面是对 `lowered_llvm.ll` 的 lexical pattern count。它们不是动态 PMC，也不等同
于执行次数；用途是定位表示在哪一层膨胀或收缩。

| LLVM textual pattern | Z5B | BDV2-S | P1-S | P1 相对 BDV2-S |
|:--|--:|--:|--:|--:|
| `udiv` | 0 | 9 | 9 | 0 |
| `urem` | 0 | 8 | 8 | 0 |
| `mul` | 65 | 65 | 65 | 0 |
| `add` | 105 | 104 | 116 | +12 |
| `select` | 7 | 5 | 5 | 0 |
| `extractelement` | 96 | 130 | 66 | -64 |
| `insertelement` | 96 | 160 | 96 | -64 |
| `bitcast` | 8 | 4 | 4 | 0 |

这张表有三个重要含义：

1. BDV2 相对 Z5B 新增了 `udiv/urem` ownership/index map；这与 full-scope
   producer/consumer 需要动态 tid/wave/lane 解释相符。
2. BDV2-S 相对 Z5B 的 fragment reconstruction 变重：`extractelement` 增加
   34，`insertelement` 增加 64。它解释了为什么“VMEM/LDS 少了”不自动意味着
   VALU 少了。
3. P1 确实删除了 BDV2-S 中 64 个 `extractelement` 和 64 个 `insertelement`，
   但 planner 的 affine values 仍然带来 12 个额外 LLVM `add` 文本模式；这个
   变化尚未被优化成最终机器收益。

Z5B 的 LLVM 文件来自：

```text
codex_qwen_bt64_stage6z_z5b_machine_stage1/lowered_llvm.ll
```

BDV2-S/P1-S 文件分别来自前述两个 machine artifact 目录。

### 4.2 pre-LTO AMDGCN 词法证据

对 specialized arm 的 `pre_lto_amdgcn.s` 做同一统计，得到：

| 指令族 | BDV2-S | P1-S | 变化 |
|:--|--:|--:|--:|
| `v_add*` | 68 | 72 | +4 |
| `v_lshl_add*` | 34 | 37 | +3 |
| `v_mul*` | 5 | 5 | 0 |
| `v_and*` | 30 | 31 | +1 |
| `v_lshr*` | 22 | 23 | +1 |
| `v_cndmask*` | 18 | 18 | 0 |
| `v_mov*` | 757 | 776 | +19 |
| `v_perm*` | 0 | 0 | 0 |

这是 pre-LTO 文件的静态 lexical 统计，不能直接换算成动态 VALU。它反而说明
P1 当前的 planner SSA 没有在 pre-LTO 阶段实现真正的 CSE/hoisting；部分 affine
map values 以额外的机器候选形式出现。

### 4.3 按功能归类

| VALU/整数工作族 | 证据 | 判断 |
|:--|:--|:--|
| row/col/token/k/value index arithmetic | LLVM `udiv/urem/add/mul`；pre-LTO `v_add/v_lshl_add/v_lshr/v_and` | BDV2 full-scope 的主要新增非数学工作；P1 共享 SSA 但没有在最终层消除 |
| tid/wave/lane ownership/predicate | planner 的 `wave/lane/laneCol/laneGroup/rowHalf/valueHalf`，以及 K 的 value-half predicate | full-scope ownership 必需，但重复 materialization 可优化；不能归为 MFMA 数学 |
| global byte/address generation | producer 的 `feature`、source stride、token/head/column 乘加 | producer contract 必需；当前 lowering 仍把部分 affine recipe 动态化 |
| shared/LDS physical address | `qRow/bRow/packetCol/elementOffset` 与 `getelementptr addrspace(3)` | producer/consumer layout 必需；native 使用更紧凑的 typed mapping，AveLang 目前有额外地址 feeding |
| BF16 packet extract/insert/view | BDV2-S LLVM `extract 130/insert 160`；P1 `66/96` | P1 明确减少了 LLVM 层的重建链，但未传递到 final ISA/PMC |
| transpose/permutation | 当前 pre-LTO 没有 `v_perm` 爆炸；无证据表明它是本轮主因 | 不选择它作为 P1 控制杆 |
| accumulator / MFMA math | MFMA/accumulator 数学固定为 160 MFMA/CTA | 不是 P1 归因对象，不能把它误算作 layout overhead |
| exp/causal/gating math | source math 冻结 | 不在本轮差异中 |

因此，对 `8410 - 7072 = 1338 VALU/CTA`，最强证据指向的是：

```text
full-scope affine ownership/address recipe
+ shared physical placement / consumer row calculation
+ BF16 fragment reconstruction
```

而不是 MFMA、exp 或 causal 数学。仅凭这些静态模式无法把 1338 条动态 VALU
逐条分配到每一类；动态归因必须以 PMC 和 MIR/ISA def-use 继续做更细的
instruction provenance。

### 4.4 native 对照的边界

native same-shape WG256 artifact 位于既有 Stage6Z native capture 工件中。T=2048
真实 dynamic PMC 为：

```text
MFMA=160, VMEM=140, LDS=480, VALU=3376, SALU=660 / CTA
```

它说明相同 logical block 数量和 MFMA 数量下，native 的 producer ownership、typed
shared/dot operand 和地址复用更紧凑。但本轮没有足够的 tensor provenance 证明
native 每个 lane 的完整来源，所以不从 native 缺失的变量名猜测精确 lane map。
native 只作为 machine-work 参照，不被当作 AveLang source 结构的逐项证明。

---

## 5. P1 分层 machine convergence

### 5.1 source/LLVM/pre-LTO

P1-S 与 BDV2-S：

- source SHA 相同；
- lowered LLVM 不同；
- pre-LTO AMDGCN 不同；
- P1 的 LLVM fragment load 已经由 `<8 x bf16>` slice 变为 `<4 x bf16>` typed load；
- P1 的 planner values 出现在 producer 和 consumer 的共同 control flow 中。

代表性 LLVM 变化：

```text
BDV2-S:
  load <8 x bf16>
  extractelement x 8
  insertelement x 4
  MFMA

P1-S:
  load <4 x bf16>
  MFMA
```

### 5.2 final ISA/HSACO

P1-S `machine_summary.json`：

| 字段 | BDV2-S | P1-S |
|:--|--:|--:|
| code VGPR | 132 | 132 |
| code AGPR | 48 | 48 |
| SGPR | 30 | 30 |
| LDS | 32768 B | 32768 B |
| private segment | 0 B | 0 B |
| VGPR spill | 0 | 0 |
| SGPR spill | 0 | 0 |
| static MFMA32 | 56 | 56 |
| static `s_barrier` | 44 | 44 |
| static global load | 140 | 140 |
| static global store | 16 | 16 |
| static `ds_read` | 56 | 56 |
| static `ds_write` | 92 | 92 |
| HSACO SHA256 | `d17483b9...c9091af5a` | `d17483b9...c9091af5a` |

两个 `final_isa.s` 文件的第一行只是不同 artifact 的绝对 HSACO 路径。去掉
这个 disassembler header 后，ISA body SHA256 完全相同：

```text
7a02fc2af0542b4a2177681e1cd51663f98d9f0357ef664c3cc067988dafaf70
```

因此应当严格写成：

```text
P1 改变了 LLVM/pre-LTO 表示，但 AMDGPU/LTO 最终 machine graph 收敛回 BDV2-S。
```

不能写成“P1 final ISA 与 BDV2-S 不同”，也不能以两个 disassembler 文件头的路径
差异伪造机器分叉。P1 的 `exact_lto` replay return code 为 0；没有 spill save/reload
和 private scratch。

### 5.3 dynamic PMC

T=2048 使用 rocprof fresh capture，`Grid_Size=131072`、WG256，即 512 CTA；动态
counter 用 512 做 CTA 归一化。以下都是硬件 PMC，不由 static ISA 推导：

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | profiler VGPR | profiler AccumVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 14.498718% |
| BDV2-G | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.634712% |
| BDV2-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.448873% |
| P1-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.549514% |
| native WG256 | 160 | 140 | 480 | 3376 | 660 | 见 native capture | 见 native capture | 见 native capture |

P1-S 与 BDV2-S 逐项相同。P1 没有保住 LLVM 层的 fragment shrink 到硬件计数，
也没有降低 BDV2 的 8410 VALU。

---

## 6. Correctness 与回归

### 6.1 BDV2/P1 full matrix

P1-G 和 P1-S 均完成：

| 检查 | 结果 |
|:--|:--:|
| T=64/512/2048 vs Z5B BF16 byte-exact | PASS |
| T=1024/4096/8192/16384 vs Z5B BF16 byte-exact | PASS |
| finite | PASS |
| T=64/8192/16384 caller-owned output | PASS |
| zero-V-new | PASS |
| NaN-prefilled caller-owned output | PASS |
| MFMA 数学工作保持 | PASS，dynamic 160/CTA |

correctness driver 使用相同 output contract，没有靠放宽 tolerance 或跳过输出
检查通过。

### 6.2 编译器/旧 block-dot 回归

Docker authoritative run：

```text
python3 -m pytest -q \
  test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  test_qwen_gdn_direct_k64_block_dot_ab.py -s
```

结果：

```text
7 passed in 26.46s
```

静态 contract test 覆盖：

- `block_dot_bf16_f32` 旧注册仍在；
- logical/transposed helper 仍创建同一个 block-dot op；
- `LogicalBlockLayoutPlan` 和 `bdv2_p1_affine` 存在；
- K/H 仍通过同一 `emitFullScopeProducer`/`emitGenericOperandBPair`；
- P1 typed fragment marker 存在；
- 没有新增 Qwen-specific block-dot intrinsic。

旧 direct-K64 block-dot regression 同时通过，说明 P1 的 planner 选择没有破坏
已有 direct-K64 API。

---

## 7. fresh-process body benchmark

### 7.1 口径

- caller-owned preallocated output；
- current HIP stream；
- no CUDA Graph；
- warmup=10、repeat=50；
- 7 个 fresh-process sessions；
- rotating arm order；
- private body diagnostic，不是 Eager public API 最终排名。

原始 JSON：

```text
codex_qwen_bt64_stage6z_bdv2_p1/bench_T2048_sessions7_primary.json
codex_qwen_bt64_stage6z_bdv2_p1/bench_T8192_sessions7_primary.json
```

### 7.2 latency 与倍数

| T | Z5B ms | BDV2-S ms | P1-S ms | native ms | BDV2-S/Z5B | P1-S/Z5B | P1-S/native |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2048 | `0.066899501` | `0.069463000` | `0.069863502` | `0.042483000` | `1.038319x` | `1.044305x` | `1.644505x` |
| 8192 | `0.157854497` | `0.167768501` | `0.167308502` | `0.090734500` | `1.062805x` | `1.059891x` | `1.843935x` |

P1-S 与 Z5B 的 session-level paired differences：

| T | P1-S - Z5B（us，7 sessions） |
|--:|:--|
| 2048 | `+2.484, +3.546, +2.043, +3.705, +2.525, +2.443, +2.364` |
| 8192 | `+11.097, +8.593, +10.777, +10.195, +8.693, +16.044, +9.193` |

这些差值在两个长度均为正，P1 没有显示相对 Z5B 的稳定改善。

### 7.3 per-chunk endpoint slope

用 `(latency(T8192)-latency(T2048))/96` 计算：

| arm | slope（us/chunk） | 相对 native |
|:--|--:|--:|
| Z5B | `0.947448` | `1.885019x` |
| BDV2-S | `1.024016` | `2.037356x` |
| P1-S | `1.015052` | `2.019523x` |
| native | `0.502620` | `1.000000x` |

P1 的 slope 比 BDV2-S 略低，但差异来自 session 噪声和同一个最终机器图，不能
视为 planner 取得了硬件收益；P1 slope 仍高于 Z5B 和 native。

T=16384 没有运行 body benchmark。原因是本轮预注册条件是 T=2048 或 T=8192
先出现稳定正收益后才补长文本；P1 在两者都没有相对 Z5B 正收益，因此没有把
条件性长文本测试伪装成晋级证据。T=16384 correctness 已完成。

---

## 8. 对“1338 VALU 增量”的最终归因

### 8.1 已证实的部分

BDV2 的 `8410 - 7072 = 1338 VALU/CTA` 不是 MFMA 数量增加：
两者都是 `160 MFMA/CTA`。也不是 static barrier 或 global I/O 简单增加：BDV2
反而把 dynamic VMEM/LDS 降低到 `448/464`。

最强证据是：

1. BDV2 LLVM 新增了 `udiv/urem` ownership map；
2. BDV2 LLVM 增加了 fragment `extract/insert` reconstruction；
3. BDV2 pre-LTO 包含大量 affine address、LDS offset、packet/fragment feeding
   的整数指令族；
4. P1 只删除 LLVM 的 fragment reconstruction 后，pre-LTO 仍有更多 affine
   arithmetic，最终 HSACO/PMC 完全不变。

因此当前最可信的分类排序是：

```text
1. full-scope affine ownership/address + shared physical placement
2. BF16 vector view / fragment reconstruction
3. 其他 layout feeding / register packet preparation
4. 冻结的 MFMA/exp/causal 数学（不是新增来源）
```

### 8.2 哪些是必需工作，哪些是当前多余工作

必需或至少有语义依据的工作：

- 每个 wave/lane 的 ownership 判断；
- K/H producer 到 shared 的合法地址；
- producer/consumer layout 不同的时候做必要的 physical mapping；
- MFMA32 要求的 `<4 x bf16>` operand 形成。

当前看起来多余或可被更好表示的工作：

- producer 和 consumer 分别重建同一个 affine index recipe；
- 在已知 typed fragment boundary 后先做 `<8xbf16>` load 再拆成两个 `<4xbf16>`；
- 每个 packet/consumer 重复生成相同的 `kStage/sourceHalf/packetCol` 组合；
- planner 已经知道的 static layout 信息仍以普通动态 integer SSA 传递到后端。

不能从现有证据直接断言所有 1338 条都属于 compiler bug。更准确的结论是：
BDV2 full-scope 表示把一部分内存工作转换成了当前 AveLang lowering 的地址和
fragment feeding 工作，而 P1 证明当前的 `<4xbf16>` typed intent 尚未可靠地保留
到最终 AMDGPU machine graph。

### 8.3 对未来 Q MFMA-A/V-new 的复用

P1 的可复用部分是 planner contract，不是 Qwen-specific fast path：

```text
LogicalBlockLayoutPlan
  -> role/transpose/shape/target
  -> producer ownership
  -> physical shared placement
  -> typed dot fragment
```

未来 Q MFMA-A、V-new 或其他 BF16 block-dot 只需要提供 logical block identity、
transpose、residency 和 target encoding，就可以复用同一个 planner。不能通过
增加 `kSpecialCase`、`hSpecialCase` 或 kernel 名称判断来获取收益。

---

## 9. 决策

### 9.1 P1 是否是性能优化成功？

不是。P1 的 LLVM 文本更紧凑，但：

- final normalized ISA body 与 BDV2-S 相同；
- HSACO SHA256 相同；
- code-object VGPR/AGPR/SGPR/LDS 相同；
- dynamic MFMA/VMEM/LDS/VALU/SALU 相同；
- body latency 没有稳定改善。

所以 P1 不能声称删除了最终机器 VALU，也不能成为新的 Stage6Z performance
baseline。

### 9.2 是否保留 P1 infrastructure？

保留。它提供了三个有价值的结果：

1. K/H 确实共享一个通用 planner object；
2. source/LLVM 层能表达 typed `<4xbf16>` consumer intent；
3. exact-LTO 证据显示该 intent 在 AMDGPU/LTO 层被收敛，下一步控制点应在
   representation preservation / lowering boundary，而不是继续改 Qwen source。

### 9.3 下一步唯一建议

不要再做另一个 source-level K/H variant，也不要改 RA。下一轮若继续，应只做
一个通用 compiler convergence experiment：让 `LogicalBlockLayoutPlan` 生成的
typed dot-fragment 在 AMDGPU lowering 到 MIR 之前保持 first-class operand/layout
身份，并用同一 source 的 generic/specialized A/B 证明：

```text
planner intent survives LLVM -> AMDGPU/MIR -> ISA
```

如果仍在 LLVM 到 AMDGPU lowering 之间被拆回普通 vector/load/extract，才能进一步
把差距明确归因到 representation loss。这个实验仍应保持 K/H 同一 planner、旧
block-dot V1 兼容，不增加 Qwen-specific op。

---

## 10. 复现命令

```bash
# 构建当前 compiler
cmake --build /tmp/avelang-z7b-build3 -j 16

# correctness + old direct-K64 regression
export PYTHONPATH=/tmp/avelang-z7b-build3/python:\
/workspace/project/avelang/test/examples/linear_attention:\
/workspace/project/avelang:/opt/avelang/python:/opt/avelang

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_ab.py -s

# P1 machine artifact，initial MLIR 因 runtime SIGSEGV 跳过并显式记录
python3 test/examples/linear_attention/vllm_compare/\
dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py \
  --variant specialized \
  --planner bdv2_p1_affine \
  --T 2048 \
  --out-dir test/examples/linear_attention/compile_bug/\
qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_bdv2_p1_machine_specialized \
  --skip-initial-mlir

# 7 fresh-process sessions，正式 P1 body JSON 已保存
python3 test/examples/linear_attention/vllm_compare/\
bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_specialized bdv2_p1_specialized native

python3 test/examples/linear_attention/vllm_compare/\
bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 8192 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_specialized bdv2_p1_specialized native
```

所有 artifact 和原始 session JSON 保存在：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_bt64_stage6z_bdv2_p1/
  codex_qwen_bt64_stage6z_bdv2_p1_machine_generic/
  codex_qwen_bt64_stage6z_bdv2_p1_machine_specialized/
```


# Qwen gfx942 C15-TILE：Real Physical Tile Numerical Closure

## 1. 本轮结论

本轮严格按“V first”执行，没有跑 body latency、rocprof、PMC、full chunk-o
接入或 Eager public API。结论分成两层：

1. **V 的真实 physical-tile 路径通过。** 普通 global BF16 `V[64,64]` 经过
   C13 distributed ownership、C14 fixed transform、真实 addrspace(3) shared
   allocation、packed LDS store、CTA barrier、encoded LDS read 和现有
   `v_mfma_f32_32x32x8_bf16`，在 GPU 上完成了 10 组结构化 pattern 的闭环。
2. **Q/H/K 不能晋级为真实数值路径。** 现有 C13/C14 工件可以让 Q/H/K 的
   typed recipe 通过 synthetic codegen，但没有恢复出每个 role 的完整
   native lane/register producer-consumer 表。继续实现会把“静态属性能过编译”
   误写成“真实 global-to-LDS-to-MFMA 数值正确”。因此本轮正式收口为：

```text
STOP_C15_REAL_TILE_CODEGEN_INSUFFICIENT
```

这不是 V mapping 错误，也不是 MFMA32 intrinsic 不支持；它是 Q/H/K 的真实
producer/consumer 物理映射证据仍不完整。按照预注册 stop rule，不运行性能，不接
full chunk-o，不修改 production。

机器可读结果：

- [stage6z_c15_v_real_tile_mapping.json](./stage6z_c15_v_real_tile_mapping.json)
- [stage6z_c15_real_tile_correctness.json](./stage6z_c15_real_tile_correctness.json)
- [stage6z_c15_real_tile_machine_evidence.json](./stage6z_c15_real_tile_machine_evidence.json)
- [stage6z_c15_regression_results.json](./stage6z_c15_regression_results.json)

## 2. 冻结边界与没有做的事情

本轮没有修改：

- full chunk-o 或 X2/full recurrence 接入；
- Z5B、Stage 6Z isolated baseline 或任何生产 selector；
- Q/H/K 的性能 variant；
- packet width、shared swizzle、transpose、scheduler、barrier、waitcnt；
- recurrence、allocator、RA、MFMA geometry；
- generic runtime fallback。

本轮也没有用性能结果替代数值证据。所有 `MFMA=8`、`ds_read=24` 等数字都是
这个 V-only repro 的 static ISA lexical count，不是 full chunk-o dynamic PMC。

## 3. 实验源与真实数据流

实验源为：

[`repro_qwen_gfx942_c15_real_physical_tile_v.py`](../../vllm_compare/repro_qwen_gfx942_c15_real_physical_tile_v.py)

source 仍调用已经存在的：

```python
al.amdgpu.block_dot_bf16_f32_operand(
    a_stage, b_stage, v, debug, g, tid,
    zero_i32, zero_i32, zero_i32, zero_i32, zero_i32,
    zero_f32, acc, acc,
)
```

只有 `AVELANG_C15_REAL_TILE=1` 时，compiler 才把第三个普通 rank-2 BF16
global operand 识别为 C15 V tile。它不是新的 Qwen public op。源代码保留了
普通 `V[64,64]` global tensor、真实 `make_shared` 的 A/B tile、CTA barrier、
MFMA output 和 debug/raw output。

数据流是：

```text
ordinary global V[64,64] BF16
    -> C13 distributed owner
    -> C14 fixed in-thread transform
    -> amd_rotating_shared [vec=4, perPhase=1, maxPhase=16]
    -> packed vector<4xbf16> LDS store
    -> barrier
    -> encoded vector<4xbf16> LDS read
    -> MFMA32 B operand
    -> raw FP32 accumulator / debug readback
```

这里的 `debug` 不是性能路径输出，它是为了把“producer/shared transport”和
“MFMA fragment mapping”分开验证。`raw` 是保留的 16-lane accumulator fragment，
用来检查 MFMA 输出 lane/fragment 到逻辑 tile 的关系。

## 4. V 的 lane 到 logical tile 映射

本轮采用 selected native chunk-o 的 literal V recipe，而不是从变量名猜测：

```text
wave      = tid >> 6
lane      = tid & 63
lane_row  = lane >> 3
lane_col  = lane & 7
row_base  = (wave << 5) + (lane_row << 2)
column    = (lane_col << 3) + packet, packet=0..7
```

每个 packet 由四个连续 source row 组成：

```text
V[row_base + element, column], element=0..3
```

因此覆盖量为：

```text
128 threads * 8 packets * 4 BF16 = 4096 elements = 64 * 64
```

这不是“每 lane 猜一个地址”。每个线程的 8 个 packet、每 packet 的 4 个
element 在 GPU 上都由 debug readback 检查；10 组模式会同时暴露 row、column、
wave、lane 和 packet 维度的错位。

### 4.1 LDS physical offset

C14 `SharedEncoding` 为：

```text
kind      = amd_rotating_shared
vec       = 4
perPhase  = 1
maxPhase  = 16
order     = [0,1]
rotating  = true
```

producer 将 source logical coordinates 以 `(column, row_base)` 交给 C14 shared
offset recipe，产生 element offset，再拆成：

```text
physicalRow = offset >> 6
physicalCol = offset & 63
```

并以 `vector<4xbf16>` 写入 B shared tile。consumer 使用同一个 typed recipe 读取
该 packed fragment。这里没有创建第二个完整 transpose buffer，没有
`ds_bpermute`，也没有重新回到 generic `DivUI/RemUI` full-region planner。

### 4.2 MFMA raw output oracle

identity-like A 使 raw MFMA 输出可以直接检查 V 元素。对 tid 和 result slot `r`：

```text
row = wave * 32 + (lane & 31)
col = wave * 32 + ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
raw[tid,r] = float(V[row,col])
```

这个公式只存在于 Python reference 和报告中，没有藏进 compiler lowering。

## 5. Correctness 结果

执行命令：

```bash
cd /workspace/project/avelang
export PYTHONPATH=/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare
export AVELANG_C15_REAL_TILE=1
export AVELANG_BLOCK_DOT_LOWERING=specialized
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gfx942_c15_real_physical_tile_v.py \
  --dump-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_gfx942_c15_real_tile
```

V pattern matrix：

| pattern | debug V readback | raw max abs | raw exact | finite |
|:--|:--:|--:|:--:|:--:|
| all_zero | pass | 0 | pass | pass |
| one_hot_element | pass | 0 | pass | pass |
| one_hot_token_row | pass | 0 | pass | pass |
| one_hot_value_column | pass | 0 | pass | pass |
| token_distinct | pass | 0 | pass | pass |
| value_distinct | pass | 0 | pass | pass |
| packet_distinct | pass | 0 | pass | pass |
| wave_distinct | pass | 0 | pass | pass |
| lane_distinct | pass | 0 | pass | pass |
| small_integer | pass | 0 | pass | pass |

特别重要的是：本轮不是只检查 `debug == V`。raw fragment 也逐 slot 与独立
Python reference 比较，10/10 pattern 的 `raw_max_abs` 都是 `0.0`。

## 6. MLIR -> LLVM -> MIR -> ISA 证据

### 6.1 MLIR

`post_block_dot_lowering.mlir` 中可以看到：

- 真实 `vector.store vector<4xbf16>`，带 `c15.real_tile_producer` 和
  `c15.packed_lds_store`；
- 真实 `gpu.barrier`；
- 两个带 C13 typed attrs、`c14.static_physical`、`c15.real_tile=V` 的内部
  first-class MFMA operand op；
- `c14.mfma_callee` 绑定现有 gfx942 MFMA32 intrinsic。

`final_mlir.mlir` 和 `postopt_llvm.ll` 中可以看到两个 8192-byte workgroup
global：

```text
@__wg__c15_real_v_tile_kernel_0 = addrspace(3) global [8192 x i8]
@__wg__c15_real_v_tile_kernel_1 = addrspace(3) global [8192 x i8]
```

这对应 A/B 两个真实 shared tile，总 group segment 为 16384 B。LLVM 中保留：

```text
llvm.amdgcn.s.barrier()
llvm.amdgcn.mfma.f32.32x32x8bf16.1k(...)
```

并且 shared store/load 已经是 addrspace(3) `vector<4xbf16>` 形式，而不是
global V reload。

### 6.2 Exact full-LTO MIR

backend 用 `AVELANG_AMDGPU_LINK_DEBUG_DIR` 记录了 replayable argv，随后执行了
exact LTO replay：

- `kernel_section_06.mir`: representative pre-greedy；
- `kernel_section_07.mir`: representative post-greedy；
- `kernel_section_08.mir`: representative post-virtregrewriter；
- `kernel_section_09.mir`: representative prologue/epilogue。

这次 replay 生成了 20 个同名 kernel pipeline snapshot；所有 sections 的
`SI_SPILL_AV32/AV64_SAVE` 计数为 0。`kernel_section_08/09` 已无 virtual
register，且没有 spill/reload。由于 plugin 会对 linked module 中的重复函数
pipeline 生成多组 dump，报告不把 20 组当成 20 个 runtime kernel。

### 6.3 Final ISA 和 HSACO

实际 HSACO：

```text
SHA256 ec8f2e2f3663dcdf2d82f5e8c6129229ee04977e2a47503f5eb876bc97943a51
```

HSACO metadata：

| field | value |
|:--|--:|
| target | gfx942 |
| wavefront | 64 |
| max WG | 128 |
| group segment | 16384 B |
| private segment | 0 B |
| code-object VGPR | 52 |
| code-object AGPR | 16 |
| code-object SGPR | 14 |
| VGPR spill | 0 |
| SGPR spill | 0 |

符号表另外记录 `num_vgpr=33`、`num_agpr=16`、`numbered_sgpr=8`。这两个 VGPR
表示来自不同 metadata 层，不能混写成一个数字；本报告保留两者原值。

static ISA lexical count：

| instruction family | count |
|:--|--:|
| `v_mfma_f32_32x32x8_bf16` | 8 |
| `ds_write_b64` | 8 |
| `ds_write_b16` | 32 |
| `ds_read_b64` | 24 |
| `s_barrier` | 2 |
| `s_waitcnt` | 30 |
| global load issuing instructions | 12 |
| global store issuing instructions | 16 |
| `ds_bpermute` | 0 |
| `v_perm_b32` | 19 |

`v_perm_b32` 是最终 BF16/fragment 组织所需的固定寄存器级 packing 证据；它
不是 `ds_bpermute`，也不等于 C15 已经实现了任意 generic transpose。这个事实
单独记录，避免把“无 ds_bpermute”夸大成“无任何寄存器 permute”。

以上是 static ISA count。C15 明确没有采集 dynamic PMC，所以不能把 8 个 MFMA
或 24 个 LDS read 外推成 full chunk-o 的 per-CTA 工作量。

## 7. Q/H/K 为什么没有继续冒险接入

### 7.1 已有证据能证明什么

C13/C14 的 `ChunkOPhysicalPlan` 和 typed attrs 已经包含 Q/H/K 的：

- logical shape；
- `sizePerThread`、`threadsPerWave`、`wavesPerCTA`、order；
- shared kind/vec/perPhase/maxPhase；
- fixed transform（H）；
- dot op index/kWidth；
- Q/H/K consumer relationship。

现有 `c14_static_physical_codegen_test` 也在真实 AveLang -> LLVM -> AMDGPU
codegen 中构造了 `c14_q/c14_h/c14_k/c14_v` 四个 synthetic function，CTest
通过，说明这些 typed attr 可以到 MFMA codegen。

### 7.2 仍然缺什么

`stage6z_c14_exact_native_layout_recipes.json` 明确记录：V 的 selected native
recipe 有 literal `#linear1` basis；Q/H/K 只有静态 attr 和 shared formula，
没有完整 per-role native lane/register producer-consumer table。具体缺口是：

| role | 缺口 |
|:--|:--|
| Q | op0 的真实 lane/register source ownership 与 A-fragment consumer 对应关系 |
| H | fixed transpose 后的 native lane/register 交换和实际 transaction 对应关系 |
| K | producer ownership、packed packet 到 B-fragment 的完整 lane mapping |

如果现在直接把 Q/H/K 套上 V 的 `row_base/column` 公式，会得到一个能编译的
伪闭环，但不能知道它是否是 native physical mapping，更不能保证 one-hot、
row-distinct 和 packet-distinct 输入在真实 GPU 上保持正确。因此：

- synthetic Q/H/K codegen = **pass**；
- real global -> shared -> barrier -> encoded read -> MFMA numerical closure =
  **not established**。

这正是 `STOP_C15_REAL_TILE_CODEGEN_INSUFFICIENT`，不是 `STOP_C15_V_PHYSICAL_MAPPING_INCORRECT`。

## 8. 回归结果

本轮 Docker full build 通过。已有 compiler regressions：

```text
4/4 passed
static_physical_layout_test
gpu_outlining_test
amdgpu_codegen_test
c14_static_physical_codegen_test
```

V fresh GPU closure 为 10/10 pattern pass。没有运行 chunk-o、full recurrence、
body benchmark、rocprof 或 Eager public API。

## 9. 对原问题的回答

### AveLang 能否表达真实 static full-region encoding？

V 路径给出了比 C14 synthetic 更强的答案：AveLang compiler 可以把 typed
distributed/shared/transform/dot encoding 保留到实际 shared allocation、LDS
store/read 和 gfx942 MFMA32，并且在真实 GPU 上数值闭环。

但“能表达 V 的 physical encoding”不等于“Q/H/K 的 native physical encoding
已经恢复”。Q/H/K 的剩余问题是 mapping evidence，不是普通 `make_layout` 语法
或 MFMA intrinsic 缺失。

### 这是否已经证明 full chunk-o 可以优化？

还没有。V-only repro 没有 full Q/H/K producer chain、full score/update loop、
真实 phase lifetime，也没有性能数据。它只证明了下一步所需的一个最小真实
physical tile transport/consumer control point。

### 能不能现在跑性能？

不能。本轮预注册条件要求 Q/H/K 真实闭环后才进入 full chunk-o；该条件没有
满足。任何性能数字都会把 V-only diagnostic 与 full chunk-o 混为一谈。

## 10. 下一步唯一允许的动作

不做 swizzle/packet/RA/性能 sweep。下一轮只能补齐 Q/H/K 的 machine-grounded
mapping evidence：从对应 native TTGIR/LLVM/ISA 建立 per-role lane/register
table，再为每个 role 各做一个 real global-to-shared numerical closure。若仍然
无法恢复完整映射，正式关闭 C15 physical full-chain 路线；不能用 V 的成功替代
Q/H/K 证据。

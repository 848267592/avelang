# Qwen gfx942 Stage 6Z Z7AB: Q/H/K A+B Joint Superphase

## 状态

Z7AB 已完成为一个单一、experimental-only 的 Stage 6Z 候选，并完成全部
预注册的 Z5B byte-exact 正确性、机器图、PMC 与 clean fresh-process body 测量。
结论是 **No-Go**：它正确地复用了 Q fragment，静态 LDS/barrier 也下降，但动态
VMEM、VALU 和 SALU 大幅增加，T=2048 与 T=8192 均稳定慢于 Z5B。因此不运行
T=16384 或 Eager public API，也不接入 X2、selector 或 production。

Z5B 继续是唯一 isolated Stage 6Z performance baseline；Z6G-S/I 仍是 No-Go，
不在本候选中复活。

## 目的

Z5B 已通过完整 Q LDS cache 将 Q global producer 收敛为一次，也通过 direct
Q-cache consumer 删除了 Q cache 到旧 phase-Q rows 的 republish。但是 Z5B 的
Phase A `Q@H` 与 Phase B `Q@K` 仍是相继的 source phase。native Triton 的 TTGIR
显示一个 typed Q dot operand 在同一循环中供两个 dot consumer 使用，并在 MFMA
区间穿插下一 packet 的 load。

Z7AB 只检验一个结构性假设：在不改变 BT64/BV64/BK32、WG256、2 CTA per
chunk-head、MFMA32、BF16 ABI、causal math 或 K32 reduction order 的前提下，能否
让 Phase A 与 score half 0 的 Phase B 以相同 Q fragment 进入一个 joint
superphase，并在 current consumer 释放前发起下一个 K0 packet。

它不是下一轮完整 pipeline，也不改 X2 immutable recurrence HSACO、R4
recurrence、allocator/RA、selector 或 production dispatch。

## 冻结合同

| 项目 | 值 |
|:--|:--|
| target | gfx942, wave64 |
| tile | BT64 / BV64 / BK32 |
| workgroup | 256，固定；无 WG128 fallback |
| grid ownership | 2 CTA / chunk-head |
| Q/K/H/V-new/output | BF16 |
| g | FP32 |
| matrix primitive | `v_mfma_f32_32x32x8_bf16` |
| Q producer | 一次 global fill 到完整 `[64,128]` BF16 dedicated LDS cache |
| accumulator policy | `inter_acc -> score0_acc -> score1_acc` 分相；禁止 Z4C 三组 accumulator 共同 K32 loop |
| output | caller-owned BF16 output，原 Z5B ABI 与数学不变 |

## 实现

实现位于
[`qwen_gdn_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase.py)。
它直接从最终 Z5B source fork，保留 Q cache、B1、C/D/E 与 caller-owned launch
contract。

共享内存仍是 Z5B 的 32 KiB allocation。低 16 KiB 是 persistent Q cache；上半区
只重新划分原本已有的 transient phase rows：

```text
Q cache     rows [  0, 255]  16 KiB, lifetime: full A+B
H stage     rows [320, 383]   2 KiB, current A consumer
K0 bank 0   rows [384, 415]   2 KiB, current/next B0 consumer
K0 bank 1   rows [416, 447]   2 KiB, alternate ping-pong bank
K1 stage    rows [448, 479]   2 KiB, later B1 consumer
```

没有增加完整 Q double buffer、thread-local Q tile 或 global Q reload。

### A+B0 joint superphase

每个 `k_stage` 的执行次序为：

```text
typed H BF16x8 global packet -> H LDS [320,383]
barrier
for kt in {0,1}:
    q_words = persistent-Q-LDS current K32 slice
    q_frag  = typed BF16 fragment(q_words)
    Q@H     consumes q_frag on all four waves
    Q@K0    consumes that exact q_frag on score-owner waves
issue next K0 BF16x8 packet to alternate 2 KiB LDS bank
barrier / next stage commit
```

这里的“复用”不是两个语法相似的 cache read：`q_words` 和由它构造的 `q_frag` 是
同一个 source-level SSA value，同时进入 `inter_acc` 和 `score0_acc` 两个 MFMA
consumer。score half 1 仍在该 loop 结束、`inter_acc` 与 `score0_acc` 结束后才
创建 `score1_acc`，以保持 Z5B phase-separated accumulator policy。

H 的 packet ownership 是全部 256 threads，每线程一个连续 BF16x8；K0/K1 的
packet ownership 是前 128 threads，每线程一个连续 BF16x8。K0 的下一个 stage
packet 在当前 K0 consumer 完成后、release barrier 前被 issue 到另一个 bank。
这是一份最小的 source-level one-stage lookahead；它没有声称已达到 Triton 的
machine-level issue distance。

详细 source-level consumer map 见
[`stage6z_z7ab_ab_consumer_map.json`](stage6z_z7ab_ab_consumer_map.json)。

## Native 对齐证据

同 shape native WG256 control 的 TTGIR 已有如下可直接观察事实：

1. Q、H、K 分别有 `local_alloc`，类型分别是 `1x64x32 BF16`、
   `1x64x32 BF16`、`1x32x64 BF16`。
2. 一个 `scf.for` 同时携带 inter/score accumulator 与 rotating shared
   descriptors。
3. loop 中同一 typed Q dot operand 先进入 `tt.dot(Q,H,inter)`，再进入
   `tt.dot(Q,K,score)`。
4. next packet 的 `buffer_load`/`local_store` 位于同一持续 loop 的 MFMA 区间。

final ISA 也可看见 packed `buffer_load_dwordx4`、`s_waitcnt`、`s_barrier` 和
MFMA 之间的 interleaving，例如约 `0x1b00` 的 load 位于前一条 MFMA 之后、下一
条 MFMA 之前。它支持 source-level schedule 的方向，但不单独证明每一个 tensor
element 的 provenance。

完整证据等级与未对齐项目在
[`stage6z_z7ab_native_ab_schedule.json`](stage6z_z7ab_native_ab_schedule.json) 中。

## 正确性

测试：

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z7ab.py -s
```

实际结果：`11 passed in 12.03s`。

| 检查 | 结果 |
|:--|:--|
| Z7AB vs Z5B BF16 byte-exact | T=64/512/1024/2048/4096/8192/16384 全通过 |
| finite | 全通过 |
| caller-owned output | T=64/8192/16384 通过 |
| zero-V-new + NaN-prefilled output | T=64/8192/16384 通过 |
| fixed WG256 / no fallback | 通过 |

这只证明当前 source/JIT runtime 的结果与 Z5B byte-exact；没有放宽误差门槛，
也没有跳过长文本或 caller-owned 边界。

## Machine Capture

启动的 artifact driver 在 initial `get_mlir()` 阶段输出：

```text
[z7ab-machine] reconstruct ASTSource T=2048
[z7ab-machine] dumping initial MLIR
```

然后 GPU container 被 OOM kill。宿主机重启后 `/dev/kfd` 一度缺失；恢复
`amdgpu` 后，同一 `ljd_qwen_vllm_avelang_rocm722` 容器重新可见 MI300X。
使用 `--skip-initial-mlir` 重跑 capture，避免重触发该已知调试 probe：

```text
python3 dump_qwen_gdn_bt64_stage6z_z7ab_machine_artifacts.py \
  --variant z7ab --T 2048 --skip-initial-mlir --out-dir .../machine
```

Z7AB 生成了不同于 Z5B 的 code object：

| 项目 | Z5B | Z7AB |
|:--|--:|--:|
| HSACO SHA256 | `979889…ed67` | `ff4762…6578` |
| static MFMA32 | 56 | 56 |
| static `global_load` | 192 | 128 |
| static `ds_read` / `ds_write` | 56 / 144 | 48 / 92 |
| static `s_barrier` / `s_waitcnt` | 32 / 214 | 24 / 163 |
| code-object VGPR / AGPR / SGPR | 104 / 32 / 28 | 104 / 32 / 34 |
| LDS / private / spills | 32768 B / 0 / 0 | 32768 B / 0 / 0 |

Z7AB 的 lowered LLVM、pre-LTO AMDGCN、final ISA 与 llc stop-point MIR 均保存
在 `codex_qwen_bt64_stage6z_z7ab_joint_superphase/machine/`。其中 LLVM 可见同一
Q fragment 进入两个 MFMA call，final ISA 和 HSACO 确认候选没有折回 Z5B。

`llc` 提供了 `amdgpu-isel`、`greedy`、`virtregrewriter`、`prologepilog` 和
`post-RA-sched` stop-point MIR；其中没有 `SI_SPILL_*`，metadata 也报告零 private
segment / 零 VGPR/SGPR spill。

exact linker-LTO MIR 仍不可用，但原因现在可复现且范围明确：source 中的
`AVELANG_AMDGPU_LINK_DEBUG_DIR` hook 在当前运行时 binary 没有导出 argv，capture
driver 和单次 JIT launch probe 均未生成 `*.argv.txt`。所以本报告将 llc MIR 标为
LLVM-input 的 pre/post-RA 证据，不伪称为 exact linker post-RA MIR。

静态 lexical count 仅证明 machine graph 确实改变，不能推导 runtime work。动态
结论来自下一节 PMC。

## T=2048 Dynamic PMC

Grid 为 131072 work-items，即 `131072 / 256 = 512` CTA；下表全部是同一新鲜
rocprofv3 capture 的 total 除以 512。它不是由 static ISA 估算。

| per CTA | Z5B | Z7AB | 变化 |
|:--|--:|--:|--:|
| MFMA | 160 | 160 | 0 |
| VMEM | 672 | 2464 | `+1792` |
| LDS instructions | 672 | 448 | `-224` |
| VALU | 7072 | 11114 | `+4042` |
| SALU | 768 | 5000 | `+4232` |
| profiler `Accum_VGPR_Count` | 100 | 88 | -12 |
| profiler `VGPR_Count` | 76 | 88 | +12 |
| OccupancyPercent | 14.480 | 16.708 | +2.228 pp |
| trace median us | 41.942 | 73.990 | `+32.048` |

因此问题不是 MFMA 数、spill 或 occupancy cliff。Z7AB 在静态上缩短 Q/H/K phase
代码，也让 LDS work 降低，但将 typed H/K packet producer、packet offset/address
与 dynamic control work 放在 superphase 内，导致 VMEM 变为 `3.67x`、SALU 变为
`6.51x`。Q global fill 仍为一次且没有新增 Q republish，所以这个回退不能归因于
Q cache；证据指向新增 H/K packet/load-lookahead path 尚未形成 native 的低动态
work pipeline。

## Clean Fresh-Process Body Timing

此前 GPU 恢复期间有两份外层 `docker exec` 过早返回、可能短暂重叠的 preliminary
JSON，不能作为正式证据。本表只使用随后显式 detached、确认无其他 benchmark
process 后的 `*_clean.json`：caller-owned output、current HIP stream、no Graph、
warmup=10、repeat=50、7 fresh-process rotating sessions。

| T | Z5B ms | Z7AB ms | native ms | Z7AB - Z5B | Z7AB / native |
|--:|--:|--:|--:|--:|--:|
| 2048 | `0.067841` | `0.100409` | `0.042383` | `+32.454 us`, CI `[31.773, 33.040]` | `2.369x` |
| 8192 | `0.157915` | `0.287987` | `0.090234` | `+129.744 us`, CI `[129.011, 130.228]` | `3.191x` |

两点 endpoint slope：

| arm | us/chunk |
|:--|--:|
| Z5B | `0.9383` |
| Z7AB | `1.9539` |
| native | `0.4984` |

Z7AB 的 fixed overhead 约与 Z5B 相近；回退几乎全部来自每 chunk 增加约
`1.016 us` 的工作。这与 PMC 的 extra VMEM/SALU 一致，且 seven-session CI 都远离
零。因此未运行条件性的 T=16384 或 Eager public API；它们不能改变本轮 No-Go。

机器差异 ledger 在
[`stage6z_z7ab_machine_delta.json`](stage6z_z7ab_machine_delta.json)。

## 当前决策

**No-Go。** Z7AB 是一个正确、machine-distinct 的 research candidate，但不是新的
performance baseline。保持 Z5B；不接入 X2、selector 或 production；不实现 Z7AC、
score/V-new superphase 或更多局部 schedule 变体。本轮证明的反例是：仅在 source
层把 native 的 Q dot-operand reuse 与 K0 lookahead 形状拼入 Z5B，不足以复现 native
pipeline；如果没有同时降低 typed H/K producer 的动态 VMEM/SALU，静态 LDS/barrier
下降反而会被每 chunk 的额外 machine work 压倒。

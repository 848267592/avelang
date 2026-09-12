# Qwen gfx942 Stage 6Z Z9S: Final-Machine Critical-Path Closure

## 结论

**Case B，正式关闭 Stage 6Z 的 packet scheduling 支线。** Z9S 成功证明了
AveLang 可以在不改高层 chunk-o 源码、不改数学、MFMA、LDS layout、WG 或 RA 的前提下，
把一个已有 K0 raw packet 的 `vmcnt` 推迟到 consumer point，并让这一差异保留到
MLIR、LLVM、MIR、final ISA 和不同的 HSACO hash。可是这个真实的 machine-graph 改变
没有转化为相对 Z5B 的稳定 body 性能胜利。

所以结论并不是“调度没有起作用”，而是更精确的一句：**这里的一次、严格有界的
load/MFMA overlap 不足以成为当前 Z5B 关键路径的主导控制杆。** Z5B 仍是唯一
Stage 6Z isolated performance baseline；Z9S 不接入 X2、selector、production，也不跑
条件性的 T=16384 或 public Eager。

本轮的机器可读证据是
[stage6z_z9s_dependency_schedule.json](stage6z_z9s_dependency_schedule.json) 和
[stage6z_z9s_machine_delta.json](stage6z_z9s_machine_delta.json)。

## 冻结边界

Z8W 与 Z9S 使用完全相同的高层源：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase.py`

其 SHA256 为
`66b7883f6990630644b03c35d5d0f5da6b79f3b190431d25fd467e8b149c524e`。两臂都固定
gfx942/wave64、BT64/BV64/BK32、WG256、每 chunk-head 两个 CTA、BF16
Q/K/H/V-new/output、FP32 g、MFMA32、caller-owned output、16 KiB Q cache、32 KiB
LDS、Q global producer once、A+B0 joint consumer、phase-separated score1 和同一 K32
累加顺序。

唯一分支是 compiler 环境变量：

```text
Z8W: AVELANG_STAGE6Z_PACKET_LOAD_LOWERING=waterfall_free
Z9S: 上述设置 +
     AVELANG_STAGE6Z_PACKET_SCHEDULING=bounded_consumer_point
```

Z9S 新增的不是一个 Qwen source schedule，而是通用
`bounded_packet_schedule_pass`。它只匹配一个结构受限的 `vector<4xi32>` raw-buffer
packet：地址前缀必须是封闭的 private scalar bookkeeping，packet 之后紧邻一个独立
MFMA loop 和原有 barrier。pass clone 这个私有地址 recipe，在 consumer loop 前发起
load；原 LDS commit 与 barrier 保持在 consumer point。未匹配的 raw packet helper
会无操作返回，不被错误要求“恰好一个匹配”。

这避免了三种本轮明确禁止的行为：把完整 H/K tile 留在寄存器、让多个 K stage packet
同时长期存活、或走回 R5 跨完整核心的 distance-one/full-core residency。

## 先审计，再修改

所有 PC 距离都只是 **final ISA lexical proxy**。它们不能解释 cycle、stall、VMEM
latency 或动态执行次数；动态工作使用独立 PMC，性能使用 fresh-process HIP-event
session。这个区分很重要，不然很容易把一段看上去很长的反汇编误读为“真的隐藏了
多少周期”。

### Z5B、Z8W 与 native 的结构

| arm | H/K producer 形态 | load-to-wait | next load 落在 current MFMA cluster 内 | 结论 |
|:--|:--|:--|:--|:--|
| Z5B | scalar BF16 producer fragment | producer-local 窄 load 后早 wait；没有可唯一追踪的 v4i32 packet | 未建立 | phase producer/commit 串行；精确 packet PC 不可从标量化 ISA 恢复 |
| Z8W | waterfall-free `buffer_load_dwordx4` | K0 next: `0x2D68 -> 0x2D70`，仅 `0x08` bytes | 否 | `load -> immediate wait -> LDS commit -> barrier -> MFMA` |
| Z9S | 同一个 logical K0 `buffer_load_dwordx4` | `0x2BEC -> 0x2D8C`，`0xA0` bytes | 是 | packet 跨现有 K0 两次迭代 MFMA 区间，随后才 commit |
| native selected WG256 | typed H/K packet group | 首发 `0x1954` 到 final `vmcnt(0)` `0x1A38`，`0xE4` bytes | 是 | `0x1B00/0x1B14` 的下一 packet load 穿插在 `0x1AF8` 与 `0x1B0C` 等 current MFMA 间 |

Z5B 的 H/K 已标量化，不能诚实地给出一个和 Z8W `v4i32` 完全相同的“某一个 packet
issue PC”。这里记录它的 producer-local early-wait 结构，不伪造精确 byte distance。

### Gate B/C 的直接证据

Z8W 的 K0 next packet 是：

```text
0x2D68  buffer_load_dwordx4 v[12:15]       issue
0x2D70  s_waitcnt vmcnt(0)                  immediate wait
0x2D74  ds_write_b128                       commit
0x2D84  s_barrier                           cross-wave ownership transition
0x2DA4  ds_read_b128
0x2DB8  first dependent MFMA
```

这说明高层所谓的 K0 lookahead 在 final machine 并没有获得有意义的 load/MFMA
overlap，满足 Gate B/C。barrier 并非被当作“数量太多”而删除：它仍是 LDS publication
到跨 wave consumer 的必要 ownership transition。

native 的同形状 WG256 则先在 `0x1954`、`0x198C`、`0x199C`、`0x19BC` 发起 packet
group，分阶段 wait 并写 LDS；在 current MFMA 的 `0x1AF8` 与 `0x1B0C` 之间又发起
`0x1B00 buffer_load_dwordx4`，随后 `0x1B14` 再发一个。也就是说，native 并非只靠
“更多 instruction distance”，而是确实有 load/MFMA interleaving。

## Z9S 机器图

Z9S 成功将同一逻辑 K0 packet 改为：

```text
0x2BEC  buffer_load_dwordx4 v[12:15]       early issue
         existing current K0 LDS reads/MFMA run through 0x2D70
0x2D8C  s_waitcnt vmcnt(0)                  latest legal wait
0x2D90  ds_write_b128                       original consumer-point commit
0x2DA0  s_barrier                           original ownership synchronization
0x2DB8  first dependent MFMA
```

因此 packet 从 issue 到 commit 的 lexical live interval 由 Z8W 的 `0x0C` 扩成
Z9S 的 `0xA4` bytes，且只包含一个 `v4i32` packet。它跨过的是同一个局部 K0
two-iteration MFMA region，不是完整 H/K tile、不是双 buffer、不是多个 generation，
所以没有重复 R5 的 long-live-range 失败模式。

这种差异不仅停留在源码：

| layer | Z9S evidence |
|:--|:--|
| post-pass MLIR | `post_bounded_packet_schedule.mlir` 有 `avelang.bounded_packet_schedule=consumer_point` |
| LLVM / MIR | 新 compiler pass 位于 block-dot lowering 之后；capture 保存 lowered LLVM 与 exact-LTO pre/post-RA MIR |
| final ISA | raw load 从 `0x2D68` 移至 `0x2BEC`，而 wait/commit 留到 `0x2D8C/0x2D90` |
| HSACO | Z8W `502973...514df`，Z9S `f9f249...4eea` |

两侧 code object 都是 VGPR/AGPR/SGPR=`104/32/33`、LDS=`32768 B`、private=`0 B`、
VGPR/SGPR spill=`0/0`。最终静态 MFMA、global load/store、LDS read/write、barrier
也都相同。因此这不是 RA cliff、spill 或“偷偷减数学”造成的结果。

## 正确性与动态机器工作

在 `waterfall_free + bounded_consumer_point` 下，Stage6Z + packet-load regression
为 **13 passed in 22.61s**。它覆盖 T=64/512/1024/2048/4096/8192/16384 对 Z5B 的
BF16 byte-exact、finite、caller-owned、zero-V-new 与 NaN-prefill。

T=2048 的新鲜 rocprof 动态计数除以 `131072 / 256 = 512` CTA：

| per CTA | Z5B | Z8W | Z9S | native WG256 diagnostic |
|:--|--:|--:|--:|--:|
| MFMA | 160 | 160 | 160 | 160 |
| VMEM | 672 | 448 | 448 | 140 |
| LDS | 672 | 448 | 448 | 480 |
| VALU | 7072 | 6990 | 7060 | 3376 |
| SALU | 768 | 840 | 880 | 660 |
| profiler VGPR / AccumVGPR | 76 / 100 | 88 / 88 | 88 / 88 | native capture: 100 / 32 |
| occupancy percent | 14.4945 | 14.5928 | 14.6120 | 9.5108 |
| scratch | 0 | 0 | 0 | 0 |

Z9S 保住了 Z8W 的 VMEM/LDS/MFMA reduction，但引入 `+70 VALU` 和 `+40 SALU` per CTA。
这不是静态 ISA 推导的数字。对 native，使用的是先前 exact selected WG256 capture 的
`160/140/480/3376/660`；本轮 broad regex `chunk_fwd_kernel_o` 会把多个 autotune
variant 聚合，已从 profiling harness 移除，不能作为 selected-native PMC 使用。

## 正式 body benchmark

条件：caller-owned preallocated output、current HIP stream、no Graph、warmup=10、
repeat=50、每 arm 每长度 7 个 fresh-process session、五臂 rotating order。下表为
session median 的中位数：

| T | native ms | Z5B ms | Z8W ms | Z9S ms |
|--:|--:|--:|--:|--:|
| 2048 | `0.042383` | `0.068161` | `0.068963` | `0.068261` |
| 8192 | `0.090014` | `0.157313` | `0.163643` | `0.162561` |

配对 bootstrap 的关键判断：

| comparison | T=2048 mean, 95% CI | T=8192 mean, 95% CI |
|:--|:--|:--|
| Z9S - Z5B | `+2.515 us`, `[-0.161, +7.405]` | `+5.305 us`, `[+4.704, +5.963]` |
| Z9S - Z8W | `+1.797 us`, `[-0.853, +6.507]` | `-1.207 us`, `[-2.341, -0.243]` |

Z9S 在长文本上比 Z8W 小幅快，但 T=2048 没有稳定收益，T=8192 则稳定慢于 Z5B，
所以不满足晋级条件。保留所有样本，不因 T=2048 的一个慢 session 人为删 outlier。

用这两个端点作描述性 slope：native `0.496151`、Z5B `0.928667`、Z8W `0.986250`、
Z9S `0.982292 us/chunk`。这不是多点回归；T=16384 由预注册规则不运行，因为前两个
门槛已经失败。

## 回答本轮十二个问题

1. Z8W 比 Z5B 少 VMEM/LDS、VALU 略低，却没有更快，是因为减少的工作没有被转成足以
   压过其 producer/consumer critical schedule 的收益；T=8192 仍比 Z5B 慢 `6.33 us`。
2. Z5B 是 scalar producer-local early-wait，无法有诚信地给一个 v4i32 packet 的精确
   PC；Z8W audited K0 为 `0x08` bytes；Z9S 为 `0xA0`；native packet group 的首发到
   final wait 为 `0xE4`，并有新 packet 进入 MFMA cluster。
3. Z8W 的 source K0 lookahead 没有在 final ISA 形成真实 overlap：`0x2D68` 后立刻
   `0x2D70 vmcnt(0)`。
4. 审计到的 `s_waitcnt vmcnt(0)` 是 packet data readiness，commit 后的
   `s_waitcnt lgkmcnt(0)` 与 `s_barrier` 是 LDS cross-wave ownership；不能把它们当
   cosmetic barrier 批量删除。
5. 存在安全的 consumer-point wait sinking，Z9S 已实现并 byte-exact 验证。
6. Z9S 只把一个 v4i32 packet 延长至 commit，静态 lexical span `0xA4` bytes；VGPR/AGPR
   没增加，spill/scratch 为零。
7. 它没有重复 R5：没有完整 tile、packet ring 或跨完整 pred/update core 的 residency。
8. machine work 基本保持 MFMA/VMEM/LDS；但 VALU/SALU 分别升 `70/40` per CTA。
9. 上表给出 T2048/T8192 与端点 slope；T16384 因门槛失败未跑。
10. Z9S 没有超过 Z5B。
11. 是，packet scheduling 正式关闭；禁止 distance-2/3、更多 pipeline variant 或 source
    schedule sweep。
12. 下一最大测得 gap 应先审计 **VMEM producer ownership**：Z8W `448` 对 native `140`
    per CTA，约 `3.2x`；同时记录它绑定的 VALU fragment/address feeding。这里只登记，
    本任务不实现。

## 后续登记，不自动执行

唯一登记方向是：`Z8W vs selected native remaining-work provenance`。目标按 logical
producer/feeding family 定位 `448 -> 140` 的剩余 VMEM，且同步解释 `6990 -> 3376`
VALU：Q fill、H、K0/K1、g、V-new、output、address/ownership/select、fragment
reconstruction 与 AGPR feed。禁止回到 g residency、local swizzle、B-op load width、Q
duplicate 或 waterfall 优化。


# pred_mn_swap_v4_io v3：production 性能验收

日期：2026-08-05。此轮冻结已经通过 correctness 的
`gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_pred_mn_swap_v4_io` v3；**未修改任何
layout、lowering、planner、kernel 或数学代码**。

## 同源 production 制品

T=2048、`emit_audit=False`、B=1/Hk=4/Hv=8/K=V=128、BT64/BV32、WG128、grid=32 CTA
（4096 global work-items）的 production HSACO 是：

```text
test/examples/linear_attention/rocprof_outputs/
qwen_r4_tail_state_kv_pred_mn_swap_v4_io_v3_t2048/hsaco/_state_kv_kernel.hsaco
SHA256 2286cd1029a9114c544de23b0b26f2ac29651e83dcee3d18b6a3b460160507f7
```

ATT、PMC 都由 public-Eager `run_pred_mn_swap_v4_io_body` 路径触发，而非直接
HSACO launch。完成计时后另起一个 public-Eager fresh-JIT 进程 dump 的 HSACO hash
仍为该值（`fresh_eager_identity/hsaco/`），因而 production capture 与计时的
冻结 source/plan/constexprs 同源。编译日志同时确认 complete recurrence lowering
和 `specialized operand=persistent_typed_block`。

完整 ATT 有两个完整 wave（没有 `Wave incomplete`）。其 target code object 是上述
SHA 的 code-object ID 5。按 MFMA 的 `2,048 -> 65,536` 命中比例乘 32 后，ATT 与
PMC 逐项闭合：

| 指标 | ATT 缩放 / dispatch | PMC / dispatch |
| --- | ---: | ---: |
| VMEM | 123,904 | 123,904 |
| LDS | 451,584 | 451,584 |
| MFMA | 65,536 | 65,536 |
| VALU（含 MFMA） | 1,927,712 | 1,927,712 |
| SALU（按 SQ 排除 nop/branch/wait/barrier/SMEM 的口径） | 208,096 | 208,096 |

归一化分母为 `32 CTA × 32 chunks = 1,024 physical V32/chunk`。

## 最终 ISA 和 I/O 实际收益

v3 的 U/v_new packet 有 8 个静态 `buffer_load_dwordx2` 与 8 个静态
`buffer_store_dwordx2`；ATT 分别给每类 256 raw hits，缩放后均为
**8/V32/chunk**。旧 R4-tail 的对应 `global_load_ushort` 和
`global_store_short_d16_hi` 各是 **32/V32/chunk**。所以：

| 严格 I/O 子路径 | R4-tail VMEM / V32 | v3 VMEM / V32 | 改变 |
| --- | ---: | ---: | ---: |
| U load | 32 | 8（b64） | -24 |
| v_new store | 32 | 8（b64） | -24 |
| **U/v_new 合计** | **64** | **16** | **-48（-75%）** |

这不是逻辑传输字节减少：两个方向仍各传输 64 B/V32；减少的是相同字节由四个
BF16 scalar access 合为一个 b64 access。PC 区间为 v3 `0x3e88..0x521c` 的
`buffer_load/store_dwordx2` 组；不存在 `global_load_ushort`、`ds_bpermute`、
exec 驱动 packet 回边或逐元素 gather。

## 既有 bucket 口径的对比

下表的 current-vLLM 数字来自已经闭合的完整 ATT bucket capture；R4-tail 与 v3
是同一 shape、同一 physical-V32 归一化。v3 的 total 行来自本轮同源 PMC/ATT，
不是把静态 ISA 乘循环次数。

| 指标 / V32/chunk | R4-tail | v3 | current-vLLM | v3 - R4 | v3 - vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| VMEM | 197.500 | 121.000 | 57.000 | -76.500 | +64.000 |
| VALU（含 MFMA） | 2187.375 | 1882.531 | 1362.875 | -304.844 | +519.656 |
| SALU | 159.250 | 203.219 | 63.313 | +43.969 | +139.906 |
| LDS | 373.000 | 441.000 | 298.313 | +68.000 | +142.688 |
| MFMA | 64.000 | 64.000 | 64.000 | 0 | 0 |

### pred operand preparation 的代价

为避免把 I/O 收益错误归到 producer，本轮也用 ATT 的 MFMA-bracketed operand
区间统计了 `qwen_pred_state_kv_frag_load_bf16x4`：v3 的两个 pred 区间是
`0x3c30..0x3e6c` 和 `0x4730..0x4964`；R4 的等价区间是
`0x35e8..0x38b8` 和 `0x4010..0x422c`。这里的只计 prep（含其 LDS/VALU），不把
U/v_new 的 b64 区间混入。

| pred prep（ATT PC bracket）/ V32 | R4-tail | v3 | v3 - R4 |
| --- | ---: | ---: | ---: |
| non-MFMA VALU | 183 | 66 | **-117** |
| LDS | 48 | 160 | **+112** |
| MFMA | 32 | 32 | 0 |

所以新 pred state-KV B-fragment 的真实交换成本是 **+112 LDS**，不是新增 VALU；
同时总 LDS 只升 +68，意味着其它范围净减 44 LDS。整个 kernel 的 VALU 反而少
304.844/V32。该候选的回退风险是额外 LDS/SALU，而不是 pred-prep VALU 或 MFMA。

## 资源

| 资源 | R4-tail | v3 | 判定 |
| --- | ---: | ---: | --- |
| rocprof VGPR | 128 | 96 | -32 |
| rocprof AccVGPR | 160 | 192 | +32 |
| SGPR | 112 | 112 | 不变 |
| LDS | 53,248 B | 53,248 B | 不变 |
| private / scratch | 0 / 0 | 0 / 0 | 不变 |
| spill | 0 | 0 | 不变 |

v3 不会因 VGPR 触发 occupancy cliff：VGPR 实际降低，LDS allocation 保持相同的
53,248 B，并且没有 spill/scratch。AccVGPR 增加 32 是唯一反向资源变化，但在
LDS 不变、VGPR 降低且三长度实测加速的条件下，它不能解释性能回退；本轮没有把
不相关候选的 sampled `0.650` occupancy 数字套用到 v3。

## public-Eager fresh-process benchmark

每个长度/实现/session 都是新 Python 进程；5 warmup、20 HIP-event samples，编译在
计时区间外；无 graph capture、无 private HSACO launch。R4-tail/current-vLLM 取同一
fresh-process benchmark 的两个 session 中位数；v3 也以相同协议另起两次 fresh process。

| T | R4-tail ms | v3 ms | v3 相对 R4 | current-vLLM ms |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.224043 | 0.186978 | -16.54% | 0.104976 |
| 2048 | 0.382338 | 0.327226 | -14.41% | 0.155040 |
| 8192 | 1.412989 | 1.190058 | -15.78% | 0.457039 |

`1024 -> 8192` steady-state slope：R4-tail **10.616 us/chunk**，v3
**8.956 us/chunk**（-15.64%），current-vLLM **3.143 us/chunk**。v3 在长序列仍约
2.85× vLLM 的 slope；这项候选没有关闭 update operand preparation 的第二主矛盾。

## 最终判定：GO

**GO，保留冻结的 v3 作为 R4-tail 的更快完整正确性候选。** 它在没有增加 MFMA、
scratch、spill、LDS 容量或 barrier 的情况下，真实消除了 75% 的 U/v_new VMEM，
并在 T=1024/2048/8192 全部取得 14–17% 的 public-Eager 加速。

该 GO 只证明这一完整 pred-M/N producer + T1xV4 b64 I/O 路径有效；不应把它误读为
已经追平 Triton。后续若继续优化，应优先处理剩余的 update operand preparation
非-MFMA工作，而不是重开已关闭的 distance/software-pipeline 搜索。

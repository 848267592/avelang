# C26-WDO: Chunk-o Work Decomposition And Ownership Audit

## 结论

**最终分类：`CASE C: C26_PMC_NORMALIZATION_ERROR_FOUND`。** C25 的 native `320 MFMA/CTA` 来自 substring 匹配 public-selector/autotune dispatch 群，再用硬编码 CTA 数归一化。C26 对 final direct-tail 做 exact `Kernel_Name` 匹配，并以 `Grid_Size/Workgroup_Size` 归一化：Z5B 与 native 在三个长度均为 `160 MFMA/CTA`。

修正后两边完成一个 logical `(chunk,value_head)` `[64,128]` output tile 都需要 **2 CTA**，理论和 PMC 都是 **320 MFMA/logical unit**。因此不能归因于 Z5B 多 CTA。same-work 下 Z5B 的 per-MFMA VMEM/LDS/VALU/SALU 更高，这是 feeding/materialization gap 的观察，不是本轮的 latency 因果证明。

## Same Logical Unit

一个 logical unit 是一个 chunk、一个 value head、完整 `[64 token,128 value]` BF16 output tile。两个 CTA 分别拥有 V `[0:64]` 和 `[64:128]`，都覆盖 64 个 token rows。Z5B 是 4 waves/CTA，native selected 是 2 waves/CTA；所以分别是 8 和 4 waves/logical unit。

## MFMA Oracle

每 V64 CTA：QxH=64、QxK score=64、scorexV-new=32，合计160。两个 V64 CTA 合计320。此 oracle 和三个长度的 final-tail PMC 完全吻合。

## Same-Work Dynamic PMC

| T | arm | CTA/unit | waves/unit | MFMA/unit | VMEM/unit | LDS/unit | VALU/unit | SALU/unit |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 2048 | Z5B | 2 | 8 | 320 | 1344 | 1344 | 14144 | 1536 |
| 2048 | native selected | 2 | 4 | 320 | 216 | 368 | 5548 | 832 |
| 8192 | Z5B | 2 | 8 | 320 | 1344 | 1344 | 14144 | 1536 |
| 8192 | native selected | 2 | 4 | 320 | 216 | 368 | 5548 | 832 |
| 16384 | Z5B | 2 | 8 | 320 | 1344 | 1344 | 14144 | 1536 |
| 16384 | native selected | 2 | 4 | 320 | 216 | 368 | 5548 | 832 |

T2048 per logical unit: Z5B/native 的 VMEM=6.22x、LDS=3.65x、VALU=2.55x、SALU=1.85x。MFMA 相同，故这些比值也等于 per-MFMA 比值。

## Producer And Synchronization

Q/K/g 都会跨两个 V64 CTA 重复，H/V-new/output 由 V64 分片而不重复。native 同样有两个 CTA，且无跨 CTA LDS 或 barrier，因此没有通过更大 output ownership amortize Q/K/g。两边的 shared publication 都是 CTA-local；静态 barrier/waitcnt 已单独记录，未伪装为 dynamic count。

## Fresh Body Timing

口径：7 fresh-process sessions、caller-owned output、current HIP stream、no Graph、warmup=10/repeat=50。它是 isolated direct body diagnostic，不是 public Eager 排名。

| T | Z5B ms | native selected ms | Z5B/native |
|--:|--:|--:|--:|
| 2048 | 0.067740999 | 0.043885000 | 1.5436x |
| 8192 | 0.157974504 | 0.099326998 | 1.5904x |
| 16384 | 0.276189998 | 0.143813506 | 1.9205x |

slope: Z5B=0.930197 us/chunk, native=0.440782 us/chunk。work ratio 与 slope 同向，但不能只凭相关性分配因果份额。

### Selector Identity Caveat

PMC 表对应 fresh final-tail 的 W2/BK32 selected HSACO。为避免共享 cache 固化历史选择，formal timing 的每个 fresh worker 都使用隔离的 Triton cache；因此它记录到 current selector 的真实 session-level choice。T2048 多数选择 W4（其中一例 stages=3），T8192 稳定 W2/BK32，T16384 有 W2/BK32 与 W2/BK64。所有这些 choice 仍是 BV64、两个 V64 CTA/logical unit；但 timing 不能被表述为 W2 HSACO 的单一固定-code-object latency。逐 session identity 已保存在 `stage6z_c26_latency_reference.json`。

## 18 个问题的直接回答

1. unit 是 `(chunk,value_head)` 的 `[64,128]` output。
2. Z5B 是2 CTA/unit。
3. native T2048 是2 CTA/unit。
4. native T8192/T16384也是2 CTA/unit。
5. 两边都是64 token x本地V64 output。
6. C25的160 vs320来自宽匹配和错误归一化。
7. 320/CTA不可信，exact tail是160/CTA。
8. 理论MFMA/unit=320。
9. 两边实测MFMA/unit=320。
10. 五种动态工作/unit见表和JSON。
11. work density/per-MFMA另存JSON。
12. Z5B的Q/K/g跨V64 CTA重复，H/V-new/output不重复。
13. native没有用更大CTA amortize它们。
14. shared/barrier均CTA-local，不能跨CTA复用。
15. 只能说与slope同向，不能定量归因。
16. 修正后主矛盾不是CTA partition，而是per-MFMA feeding/materialization。
17. 主分类为Case C。
18. 非Case A，不产生work-partition candidate。

## Stop

C26 是只读审计：未修改kernel、未开始C27、未做CTA/layout/packet/barrier/RA/pipeline优化，也未接入X2。下一步最多登记为相同 V64 CTA ownership 下的 operand-feeding 审计，不能重启CTA ownership假设。

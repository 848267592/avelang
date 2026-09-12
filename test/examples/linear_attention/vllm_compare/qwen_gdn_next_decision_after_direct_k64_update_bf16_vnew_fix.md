# Direct-K64 BF16 V-new Boundary Fix 后的决策

## 已解决

P3 正确地将 P2 的 pred feedback 误差归因于前序 update/state。U0 随后证明第一颗
误差种子是 P1 让 update 消费 unrounded FP32 `corrected`，而不是 ABI 规定的
BF16 `V-new`。该 source 边界已修复。

| gate | 状态 |
|:--|:--|
| U0 post-fix V-decay vs BF16 V-new contract | `0` |
| P1 six-case nonzero-W composition | 通过 |
| P2 T=128 | 2 / 2 chunks 通过 |
| P2 T=512 | 8 / 8 chunks 通过 |
| P2 T=2048 | 32 / 32 chunks 通过 |

完整前后证据见 [BF16 V-new boundary fix report](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_update_bf16_vnew_boundary_fix_report.md)。

## 下一步

正确性阻塞解除后，下一步只允许做 Triton-matched pred/update phase-lifetime A/B：

- 固定当前 BF16 V-new boundary；
- 固定 BT64/BV32/WG128、MFMA32、C0 typed block update；
- 比较 pred accumulator、V-new 与 update accumulator 的 creation/release 时机；
- 每个 arm 先过 P0/P1/P2 correctness，再做 isolated body profiling；
- 不改 production、external HSACO、allocator/RA、MFMA geometry、CTA ownership 或 LDS layout。

这不是 full-v29 promotion，也不意味着 public operator 已正确或已具备性能资格。

# Qwen gfx942 BT64 Recurrence Reconciliation: Stage 6R

## 结论

**CASE B：当前 Stage 6A vLLM full graph 选择了不同且更快的 recurrence specialization。** 这不是 asm-v0 退化、也不是 rocprof 把 Avelang 计数重复三次。asm-v0 仍然忠实对应历史 FP32/WG256 Triton lineage；当前 vLLM 则是 BF16 W/U/v_new、BV32、WG128 的新 code object。

## 计数口径

最终三个显式 replay 的每 dispatch 动态 MFMA：asm-v0 `196608`，vLLM `65536`。按 32 chunks 归一化为 `6144` / `2048`；按 chunk-head 为 `768` / `256`。两边都只取最后 3 个 dispatch 的中位数，因此 3x 是真实动态工作差异。

## 当前身份与 ABI

- asm-v0 HSACO: `eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226`
- 当前 vLLM HSACO: `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`
- 历史 Triton original: `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9`
- 二者均为 88 B kernarg、相同 pointer-slot 物理布局；但语义 ABI 不同：asm W/U/v_new 为 FP32，当前 vLLM 为 BF16。
- 当前 vLLM config：`BV=32,num_warps=2,num_stages=2`，WG128，dynamic LDS 40960 B。asm-v0：WG256，dynamic LDS 57344 B。

## ISA 与资源

当前 vLLM 静态 MFMA 为 BF16 `64`、XF32 `0`；asm-v0 为 BF16 `48`、XF32 `96`。
T=2048 fresh rocprof：asm trace `145.737 us`, VGPR/AccVGPR/SGPR `128/192/80`, VMEM/LDS/barrier `91136/588928/44`；vLLM `106.278 us`, `104/160/96`, `58368/305472/32`。两边 scratch 都为 0；完整明细见 `resource_diff.csv` 与 `instruction_diff.csv`。

## 同口径 Body 延迟

| T | asm-v0 FP32 ms | vLLM actual BF16 ms | gap us | actual bridge delta |
|--:|--:|--:|--:|--:|
| 512 | 0.049554 | 0.039199 | 10.356 | -0.207% |
| 2048 | 0.154670 | 0.114570 | 40.100 | -0.052% |
| 8192 | 0.574333 | 0.413274 | 161.059 | 0.049% |
| 16384 | 1.172622 | 0.841410 | 331.212 | 0.017% |

current-vLLM bridge 在 T=512/2048/8192/16384 都对 native vLLM 的 h/v_new/final_state bit-exact；T=2048 bridge 与 native 相差 `-0.052%`，通过 <=5% gate。历史 original、rebuilt 与 asm-v0 在同一 FP32 W/U 输入上也均 bit-exact。

## 解释

Stage 6A 的 `~40 us` T=2048 gap 可以稳定复现。以本轮原生 ABI body 数据拟合，asm/vLLM/gap 的 intercept 为 `0.008181`/`0.009630`/`-0.001449` ms，slope 为 `4.524662`/`3.230977`/`1.293686` us/chunk。Stage6A 在 asm side 将 vLLM BF16 W/U 外部转换为 FP32，asm 随后执行历史 XF32-heavy WG256 body。将 W/U 都设为同一 FP32 值时，当前 vLLM source body 与 asm-v0 数值 bit-exact；但它仍不能把这当成纯 dtype 因果控制，因为输入 dtype 也可能影响 Triton 选择的代码路径。证据只支持：当前 BF16/BV32/two-wave specialization 及其相关 lowering 是主导边界，仍存在较小的代码生成/几何差异。

## 决策

不修改 asm，不修改 compiler，不在本轮继续 chunk-o。允许的唯一下一步是单独的、opt-in 的 current-vLLM BF16 recurrence bridge 全图 contract 集成实验；它不得接入 production，且必须先证明上下游 BF16 W/U/v_new 边界的完整正确性。

## 产物

- `counter_aggregation_{raw,normalized}.csv` / `counter_aggregation_audit.md`
- `current_kernels/`, `kernel_identity_comparison.*`, `isa_diff.md`, `abi_diff.md`, `dtype_layout_diff.md`, `launch_diff.md`
- `standalone_{raw,summary,slopes,correctness}.csv`, `body_native_abi_{comparison,slopes}.csv`, `resource_diff.csv`, `instruction_diff.csv`
- `bridge_probe/`, `tests/pytest_results.txt`, `root_cause_decision.*`, `next_stage_decision.*`, `final_decision.json`

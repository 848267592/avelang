# chunk-o 报告索引

## 推荐顺序

1. `00_chunk_o_optimization_complete_report_cn.md`：完整中文复盘，先读这一份。
2. `00_stage6a_gap_audit.md`、`00b_stage6w_boundary.md`：为什么 chunk-o 被选成主瓶颈。
3. `01_z0_native_audit.md`：native Triton 的真实 ownership、tile 和 selector。
4. `02_z2_phase_aware.md`、`03_z3_wg128.md`：第一轮 Avelang source schedule 尝试。
5. `04_z4_q_residency.md`、`05_z5a_dedicated_q_cache.md`：Q residency ladder。
6. `06_z5b_direct_q_cache.md`：当前 standalone 最优 Z5B 的正式报告。
7. `07_z5b_remaining_vmem_ledger.md`：Z5B 与 native 的剩余机器工作归属。
8. `08`--`11`：g residency、fusion、compiler waterfall-free、scheduler 尝试。
9. `12_c20_formal_gap.md`、`13_c25_pipeline.md`、`14_c26_ownership_audit.md`：后续
   full-physical/pipeline 路线为什么没有替代 Z5B。
10. `15_block_dot_feasibility.md`、`16_block_dot_v2_generalization.md`：通用
    `block_dot` compiler infrastructure 的结果，不能误认为 Z5B 性能已经超过 native。

## 证据等级

报告中区分了：

- source/LLVM/ISA 静态机器图；
- rocprof 动态 PMC；
- fresh-process HIP-event body timing；
- full public Eager timing。

不同层次不能互相替代。尤其不能用静态 MFMA 或 VMEM 行数直接推导 latency。

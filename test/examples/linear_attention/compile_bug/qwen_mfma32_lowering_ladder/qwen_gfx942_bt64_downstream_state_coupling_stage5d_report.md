# Qwen GDN gfx942 BT64 Downstream State-Coupling Stage 5D Report

## 1. 结论摘要

Stage 5D 是 audit-only 实验。本轮没有修改 solve、KKT、W/U、chunk-o、
asm-v0、compiler 或 production dispatch。

现有数据可以高置信度确认一个操作层面的根因：

> v18 solve 会留下一个有利于紧随其后的 W/U -> asm recurrence -> chunk-o 的
> 瞬态执行状态；hierarchical_fp32_v1、no-solve 和短 dummy predecessor 不会。

这个状态与 solved tensor 的数值内容无关，也不是下游可见 pointer alignment 的差异。
在 T=2048、下游读取同一份 bitwise canonical data 和同一个 consumer pointer 时：

- v18 predecessor 后的连续 tail：`0.267518 ms`
- v1 predecessor 后的连续 tail：`0.332073 ms`
- v1 tail penalty：`64.555 us`

将同一个 tail 预热一次，差距缩小到 `0.120 us`；在 tail 前执行相同的 512 MiB
cache/execution-state 扰动，差距为 `-0.080 us`，也就是测量分辨率内相同。这个结果
强烈支持“前驱造成的 cache residency、频率/功耗爬升或二者组合”这一类机制，但本轮
没有拿到 cache counter 和 clock telemetry，因此不能把最终根因写成确定的 L2 cache，
也不能写成确定的 clock ramp。

Stage 5C 中 solve 单阶段约节省 `96.583 us`，无分段 event 的 public full 只节省
`18.808 us`，相差约 `77.775 us`。本轮连续 tail 单 event 实际复现了 `64.255 us`
的下游抵消，说明抵消并非主要由分段 event 伪造；剩余部分来自运行条件、测量扰动和
尚未关闭的 direct common solve-output pointer/硬件状态变量，不能强行精确分摊。

唯一 Stage 5E 建议是：

> 让 v18 和 hierarchical_fp32_v1 直接写入同一个固定预分配 output buffer，再运行
> 完整 fixed graph。

这是当前最小、低风险、可证伪的实验。它不改变数学、W/U、asm、compiler 或生产路径，
并关闭当前 copy-to-canonical 控制仍未关闭的 solve-store 物理地址/cache-set 变量。

## 2. 审计范围与工作区安全

审计目录：

`codex_qwen_bt64_downstream_state_coupling_stage5d/`

新增内容只有独立 harness、测试、profile/telemetry 驱动、CSV 和报告。未修改：

- v18 solve；
- hierarchical_fp32_v1 solve；
- Stage 4 cumsum/KKT/W/U/chunk-o；
- asm-v0 HSACO、symbol、ABI、grid/WG；
- Avelang compiler、generic lowering、LLVM/AMDGPU RA；
- production 默认 dispatch；
- v23/v24/v26/v27/v28/v29 production baseline。

开始前的工作区状态保存在 `git_before.txt` 和 `git_before.diff`。本轮没有执行
`git reset`、`git clean` 或创建 commit。

## 3. 冻结执行图

两条审计图均为八个 dispatch：

```text
chunk cumsum
  -> KKT
  -> solve
  -> W
  -> U
  -> gfx942 asm-v0 recurrence
  -> chunk-o
  -> BF16 output cast
```

GRAPH-A 使用 `_qwen_gdn_solve_kernel_v18_parallel`，workgroup 128。

GRAPH-B 使用 `_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1`，workgroup 256。

除 solve symbol 和 solve workgroup 外，两图使用相同的输入、stream、预分配输出、
dispatch 数量和顺序。W/U/chunk-o 在同一个 Python process 内调用相同 JIT function、
相同 constexpr specialization，因此使用同一 JIT cache entry。asm-v0 使用同一外部
HSACO，SHA256 为：

```text
eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226
```

T=2048 的关键 launch：

| stage | grid | WG | dynamic LDS |
|:--|--:|--:|--:|
| cumsum | 256 | 1 | 0 |
| KKT | 4096 | 64 | 0 |
| solve A | 256 | 128 | 0 |
| solve B | 256 | 256 | 0 |
| W | 2048 | 256 | 0 |
| U | 2048 | 256 | 0 |
| asm recurrence | `(4,8,1)` | 256 | 57344 B |
| chunk-o | 2048 | 256 | 0 |
| BF16 cast | identical | identical | identical |

逐 dispatch 的 tensor shape、dtype、stride、pointer、storage offset、依赖关系记录在
`graph_a_dispatch_map.json` 和 `graph_b_dispatch_map.json`。

限制：本轮没有额外导出 W/U/chunk-o JIT binary 的字节 hash；“same code object”的
证据是同 process、同 JIT callable、同 constexpr specialization 和同 cache key。
asm-v0 的 HSACO hash 是直接记录的。

## 4. Correctness 与 solve 输出审计

已实际执行的 smoke：

| T | A/B output max abs | A/B final-state max abs | canonical-vs-A output/state |
|--:|--:|--:|:--|
| 64 | `0` | `0` | `0 / 0` |
| 512 | `2.44140625e-4` | `2.7179718e-5` | `0 / 0` |
| 8192 | `2.44140625e-4` | `1.5418977e-5` | `0 / 0` |

T=2048 solved tensor A/B 统计：

| metric | value |
|:--|--:|
| max abs | `2.980232e-8` |
| mean abs | `3.80825e-10` |
| bitwise mismatch | `317998 / 1048576` |
| NaN | 0 |
| Inf | 0 |
| subnormal | 0 |

两种 solve 输出均为 contiguous FP32，shape、stride、storage offset 完全相同。微小数值
差异确实存在，所以不能仅凭误差小就宣布数据无关；数据主因是由 canonical-data
控制实验排除的。

Stage 5C 六项 integration、Stage 4 29 项回归、asm-v0/external bridge 回归没有在本轮
重新执行。原因是完成 benchmark/control 后，平台拒绝新的 Docker execution；这些项在
`pytest_results.txt` 中明确标为 N/A，不沿用旧结果冒充本轮结果。

## 5. 无分段 Event 的完整 A/B

固定预分配审计 harness 在完整 graph 边界放一个 HIP event，graph 内不插 event。
每个 T 使用 warmup=20、repeat=200、5 sessions，并覆盖 ABAB、BABA 和随机 ABBA。

| T | v18 full ms | v1 full ms | v1 gain us |
|--:|--:|--:|--:|
| 512 | `0.277933` | `0.242881` | `35.052` |
| 1024 | `0.357131` | `0.316590` | `40.541` |
| 2048 | `0.548115` | `0.470680` | `77.435` |
| 4096 | `0.724257` | `0.630296` | `93.961` |
| 8192 | `1.373361` | `1.178431` | `194.930` |
| 16384 | `2.611822` | `2.336312` | `275.509` |

T=2048 固定-buffer 数据存在明显 order/session 异常：四个 session 的 v1 gain 为
`75--96 us`，随机 ABBA session 却反转为 v18 `0.480975 ms`、v1 `0.616737 ms`。
因此不能把上表 T=2048 的 `77.435 us` 当作生产结论。

更可信的 Stage 5C public/no-segment A/B 是：

| graph | T=2048 ms |
|:--|--:|
| v18 | `0.474385` |
| v1 | `0.455577` |
| v1 gain | `18.808 us` |

完整 raw samples 在 `full_ab_raw_samples.csv`，order 分析在
`order_effect_analysis.md`。

## 6. 连续 Downstream Tail

tail 的一个 HIP event 连续包围：

```text
W -> U -> asm recurrence -> chunk-o -> BF16 cast
```

solve 在 event 之前执行，tail 内没有分段 event。

### Solve + Tail Boundary

| T | v18 ms | v1 ms | v1 gain us |
|--:|--:|--:|--:|
| 512 | `0.228479` | `0.200939` | `27.540` |
| 2048 | `0.393144` | `0.351943` | `41.201` |
| 8192 | `1.226943` | `1.032995` | `193.948` |
| 16384 | `2.354199` | `2.067292` | `286.907` |

### Tail Only

| T | after v18 ms | after v1 ms | v1 tail penalty us |
|--:|--:|--:|--:|
| 512 | `0.100089` | `0.176903` | `76.814` |
| 2048 | `0.267938` | `0.332193` | `64.255` |
| 8192 | `0.991834` | `1.004613` | `12.779` |
| 16384 | `2.220320` | `2.252848` | `32.528` |

T=2048 的 ABAB、BABA、random order 都稳定复现约 `64 us` penalty。这证明真实下游
抵消在无内部 event 的连续 tail 中存在，不能归因于 Stage 5C 分段 event 的简单相加。

## 7. Canonical Data 控制

下游始终读取同一个预分配、bitwise 固定的 `solved_canonical`：

- A：先执行 v18，丢弃输出，再消费 canonical；
- B：先执行 v1，丢弃输出，再消费 canonical；
- C：不执行 solve，直接消费 canonical；
- D：执行短 dummy predecessor，再消费 canonical。

| T | after v18 ms | after v1 ms | v1 penalty us |
|--:|--:|--:|--:|
| 2048 | `0.267518` | `0.332073` | `64.555` |
| 8192 | `0.991954` | `1.005834` | `13.880` |

T=2048 no-solve 是 `0.334737 ms`，dummy predecessor 是 `0.343390 ms`，均更接近
v1，而不是 v18。这说明有利状态是 v18 predecessor 特有的，不是 canonical 数据内容
本身，也不是“任意 predecessor 都能预热”的简单现象。

## 8. Same Pointer 与 Alignment 控制

当前 same-pointer 控制的准确语义是：

1. 两种 solve 仍分别写自己的 output buffer；
2. solve 后把真实输出 copy 到同一个 canonical consumer buffer；
3. copy 在 tail event 外；
4. downstream 只读取同一个 data pointer。

T=2048：

| predecessor | tail ms |
|:--|--:|
| v18 + copy-to-canonical | `0.267838` |
| v1 + copy-to-canonical | `0.336020` |
| v1 penalty | `68.181 us` |

copy 成本约 `9.8--10 us`，不计入 tail。这个结果排除了 downstream consumer pointer
和 visible alignment 作为主因，但没有关闭 solve 自身写入哪个物理 output pointer 这一
变量。因此不能把它误写成“两种 solve 已直接写同一 pointer”。

T=2048/8192 的 solved A、solved B、canonical 均为：

- contiguous FP32；
- storage offset 0；
- address mod 16/64/128/256 = 0；
- address mod 4 KiB/64 KiB = 0。

可见低位 alignment 不是主因；更高物理地址/cache-set 交互仍未由 direct common-out
实验关闭。timed region 内无 allocation，allocator state 没有观察到 A/B graph 差异。

## 9. Cache/Execution-State 控制

所有控制都让 downstream 消费相同 canonical pointer 和相同数据：

- none：solve 后直接 tail；
- warm：tail 前先运行一次同样的 tail；
- controlled perturbation：tail 前执行 512 MiB tensor add；
- prime：用 reduction 访问相关输入。

512 MiB 操作只称 controlled cache/execution-state perturbation，不宣称精确清空某一级
cache。

| T | control | after v18 ms | after v1 ms | v1 penalty us |
|--:|:--|--:|--:|--:|
| 2048 | none | `0.267898` | `0.336520` | `68.622` |
| 2048 | warm | `0.267638` | `0.267758` | `0.120` |
| 2048 | 512 MiB perturb | `0.274528` | `0.274448` | `-0.080` |
| 2048 | reduction prime | `0.280777` | `0.356189` | `75.412` |
| 8192 | none | `0.993256` | `1.009660` | `16.404` |
| 8192 | warm | `0.993717` | `0.993937` | `0.220` |
| 8192 | 512 MiB perturb | `1.011784` | `1.012064` | `0.280` |
| 8192 | reduction prime | `1.001168` | `1.020457` | `19.289` |

结论边界：

- warm 和大 buffer perturb 都把 A/B 差距压到 sub-us；
- reduction prime 没有压平差距；
- 这强烈支持可被 GPU 工作负载重置的瞬态执行状态；
- 没有 cache counter，不能确定是 L2/TCC/TCP 中哪一级；
- v18 本身比 v1、dummy/no-solve 更长，warm/大 buffer 也是长 workload，因此 GPU
  clock/power ramp 仍是同样合理的解释；
- 最严谨表述是 cache residency 与 clock/power ramp 尚未分离。

## 10. Profiling 与时钟项的状态

完成 benchmark/control 后，平台拒绝了新的 Docker execution，并返回 usage-limit
blocker。因此以下项目未执行，全部标为 N/A：

- 当前 gfx942 可用 cache counter 查询；
- GRAPH-A/B whole-graph rocprof timeline；
- W/U、asm、chunk-o 的 A/B trace/counter；
- dispatch gap、CU/wave 分布；
- cache/TCC/TCP counter；
- amd-smi/rocm-smi clock、memory clock、power、temperature 采样；
- rocprof instrumentation overhead 的本轮重测。

对应 CSV 已保留 N/A schema，`rocprof/README.md` 记录 blocker。未使用不存在的 counter
名称，也没有根据旧 profile 编造新 A/B counter。

由于 graph contract 已证明下游 kernel symbol/specialization 相同，静态 code object
资源应相同；但动态 instruction count、trace duration、cache behavior 和 dispatch gap
仍需要实际 rocprof 才能回答。本报告把这些字段保持为 `null`。

## 11. Measurement Perturbation

Stage 5C 同输入数据：

| timing mode | v18 ms | v1 ms | v1 gain us |
|:--|--:|--:|--:|
| full single event | `0.474385` | `0.455577` | `18.808` |
| per-stage events summed | `0.490008` | `0.485421` | `4.587` |

分段 event 相对 full single event 增加：

- v18：`15.623 us`
- v1：`29.844 us`
- 对 A/B gain 的扭曲：`14.221 us`

因此 Stage 5C 的 “W/U +16.865 us、asm +75.352 us” 只能用于定位 downstream
组合状态，不能当作真实、可加的 `92.217 us` 分解。另一方面，Stage 5D 的连续 tail
单 event 复现了 `64.255 us`，所以 instrumentation 不是抵消的主因。

rocprof 相对 HIP-event duration 的扰动本轮 N/A。

## 12. 根因矩阵摘要

| candidate | status | confidence | explanation |
|:--|:--|:--|:--|
| solved 数值内容 | rejected as dominant | high | canonical bitwise input 仍保留差距 |
| downstream pointer/alignment | rejected as dominant | high | 同 consumer pointer、同 alignment 仍保留差距 |
| allocator pool | rejected as dominant | medium-high | timed region 无 allocation，图结构相同 |
| cache residency | supported, not isolated | medium | warm/大扰动压平差距；无 cache counter |
| clock/power ramp | unresolved, supported alternative | medium | 长 predecessor/扰动可能拉高频率；无 telemetry |
| dispatch gap | unresolved | low/unknown | whole-graph timeline N/A |
| predecessor occupancy/resource state | supported at operational level | high | v18 特有；no-solve/dummy/v1 均慢 |
| stage-event instrumentation | not dominant | high | continuous tail 复现抵消 |
| hidden dispatch | rejected | high | frozen graph dispatch count/order 相同 |
| downstream code-object difference | rejected | high | 同 process 同 JIT specialization；同 asm HSACO |
| solve-store physical pointer/cache set | unresolved | medium | copy-to-canonical 未让 solve 直接写同一 out |

详细证据见 `root_cause_matrix.md` 和 `root_cause_matrix.json`。

## 13. 对 25 个核心问题的回答

1. **两条 full 图除 solve 外是否一致？** 是。八个 dispatch 中只有 solve symbol/WG
   不同。
2. **W/U 与 asm 是否使用相同 symbol/code object？** 是。W/U/chunk-o 是同 process
   同 JIT function/specialization；asm HSACO hash 完全相同。JIT binary byte hash N/A。
3. **launch 数量是否相同？** 是，均为八个。
4. **W/U/asm grid、WG、LDS 是否相同？** 是。
5. **solve output shape/dtype/stride 是否相同？** 是，contiguous FP32，storage offset 0。
6. **solve output 地址是否不同？** 原始 A/B 分别预分配，地址不同。
7. **alignment/cache-set 低位是否不同？** 检查的 mod16 到 mod64KiB 均为 0；物理
   cache-set 映射 N/A。
8. **timed region 内是否有不同 allocation？** 没有。
9. **allocator pool 是否不同？** 未观察到图相关差异；不能从公开 API 得到所有内部
   allocator/cache-set 信息。
10. **消费 bitwise 相同 solved tensor 后差距是否存在？** 是，T=2048 为
    `64.555 us`。
11. **两 solve 直接写同一预分配 pointer 后差距是否存在？** N/A，尚未执行；现有控制
    是 solve 后 copy 到同一个 consumer pointer。
12. **受控大 buffer 扰动后差距是否消失？** 是，T=2048 剩 `-0.080 us`。
13. **相同 reduction prime 后差距是否消失？** 否，T=2048 仍为 `75.412 us`。
14. **warm/cold 差距？** none `68.622 us`，warm `0.120 us`，512 MiB perturb
    `-0.080 us`。
15. **W/U 动态指令是否改变？** N/A；相同 code object 已确认，动态 counter 未采集。
16. **asm 动态指令是否改变？** N/A；相同 HSACO 已确认，动态 counter 未采集。
17. **cache hit/miss counter 是否改变？** N/A，counter 查询/rocprof 受 quota 阻塞。
18. **occupancy/wave/CU/gap 是否改变？** N/A。
19. **clock/power 是否存在稳定差异？** N/A，未得到 telemetry。
20. **rocprof/分段观察能否由无 rocprof tail 复现？** 可以，连续 tail 单 event 复现
    `64.255 us`。
21. **per-stage event 扰动？** 对 v18/v1 full 分别增加 `15.623/29.844 us`，扭曲
    A/B gain `14.221 us`。
22. **约 78 us 抵消主因？** 高置信度是 predecessor-induced transient downstream
    execution state；具体 cache 与 clock 机制尚未分离。
23. **是否可能有共同原因？** 是，cache residency、频率/功耗爬升以及未关闭的 solve
    store physical pointer 可能共同作用。
24. **下一步？** 选择 A：两个 solve 直接写同一个固定预分配 out pointer；不是 fusion。
25. **需要修改 compiler/assembly 吗？** 没有证据支持，当前不需要。

## 14. Stage 5E 唯一建议

实现一个 audit-only fixed-out harness：

```text
v18 solve -------------------> solved_common
hierarchical_fp32_v1 solve --> solved_common
                                |
                                +-> unchanged W/U -> asm -> chunk-o
```

必须让 solve kernel 本身直接接收并写同一个 `solved_common.data_ptr()`，而不是先写不同
buffer 再 copy。继续保持：

- 相同 KKT 输入；
- 相同 stream；
- 相同 graph order；
- 无内部 stage event；
- 相同 downstream pointer；
- 数学和 production 不变。

如果 direct-common-out 仍保留 tail 差距，solve-store pointer/cache-set 可排除，随后只读
采集 clock telemetry 与 whole-graph cache/dispatch counters，区分 cache 与频率状态。
如果差距消失，则 pointer/cache-set 是可行动根因。

预计可恢复收益保持 `N/A`。当前可观察上界是 T=2048 约 `64.255 us` tail penalty，
但在 direct control 前不能把它承诺为 full 收益。

## 15. 复现与证据路径

命令：

`codex_qwen_bt64_downstream_state_coupling_stage5d/commands.sh`

主数据：

- `full_ab_benchmark.csv`
- `full_ab_raw_samples.csv`
- `downstream_tail_benchmark.csv`
- `downstream_tail_raw.csv`
- `canonical_data_control.csv`
- `same_pointer_control.csv`
- `cache_state_control.csv`
- `pointer_alignment.csv`
- `solved_data_statistics.csv`
- `measurement_perturbation.csv`

结构化结论：

- `root_cause_matrix.json`
- `stage5e_decision.json`
- `final_decision.json`

未执行项和原因：

- `rocprof/README.md`
- `whole_graph_trace_analysis.md`
- `clock_power_interpretation.md`
- `pytest_results.txt`

## 16. 最终状态

- `audit_only=true`
- `ready_for_stage5e=true`
- `ready_for_production=false`
- dominant operational cause：`predecessor-induced transient downstream execution state`
- exact hardware mechanism：unresolved between cache residency, clock/power ramp, and
  direct solve-store physical-address interaction
- compiler/assembly modification：不推荐


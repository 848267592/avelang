# Qwen gfx942 Stage 6Z Z8W: Waterfall-Free Typed H/K Packet Producer Lowering

## 结论

Z8W 完成了一个严格的同源码编译器 lowering A/B，并在最终机器代码上证明：
Z7AB 的 H/K BF16x8 packet producer 把 lane-divergent byte offset 放入 AMDGPU
raw-buffer 的 scalar `soffset` 操作数，迫使后端为不同地址的 lane 生成 waterfall
convergence loop。Z8W 只在通用 raw-buffer lowering 中把该动态 offset 归并到
VGPR-compatible `vindex`，并将 `soffset` 设为立即数零。高层源码、逻辑 packet、
CTA/WG、MFMA、LDS 分配、barrier 和数学全部保持不变。

这个修复在机器工作上是明确成功：T=2048 每 CTA 的 VMEM 从 `2464` 降到 `448`，
SALU 从 `5000` 降到 `840`，VALU 从 `11114` 降到 `6990`，MFMA 和 LDS 均不变。
但它没有超过 Z5B：T=2048 慢 `0.727 us`，T=8192 慢 `6.347 us`，两个 paired 95%
CI 都不跨零。因此结论是 **compiler correctness / machine-work GO，性能晋级 NO-GO**；
Z5B 仍是唯一的 Stage 6Z isolated performance baseline。

## 固定 A/B

两臂均使用同一文件
`qwen_gdn_bt64_native_chunko_stage6z_z7ab.py`，其 SHA256 为
`66b7883f6990630644b03c35d5d0f5da6b79f3b190431d25fd467e8b149c524e`。
唯一开关是：

```text
AVELANG_STAGE6Z_PACKET_LOAD_LOWERING=current_raw|waterfall_free
```

`current_raw` 保留原 raw-buffer operand 顺序。`waterfall_free` 不创建 Qwen
专用 op，不改变 source schedule；它仅规范化通用 raw-buffer 地址：

```text
effective_byte_address = vindex + soffset

current_raw:      vindex = 0,                  soffset = divergent_byte_offset
waterfall_free:   vindex = divergent_byte_offset, soffset = 0
```

这是对任意非零 `vindex` 也成立的代数保持变换。AMDGPU MUBUF 的 `soffset` 只能是
SGPR 或立即数，而 `vindex` 可以是 VGPR；因此这恰好消除了 lane divergence 落入 scalar
operand 的非法/昂贵形式。

两臂均保持每 CTA `H=1024`、`K0=512`、`K1=512` 个 BF16x8 logical packet，合计
`2048`；MFMA 数学工作也保持 `160 MFMA/CTA`。大 kernel 的 `get_mlir()` debug 导出在
此容器内内存耗尽，因此没有伪造 MLIR A/B；同源码 hash、LLVM、exact-LTO MIR、ISA、
HSACO 和运行时证据都已保存。

## Waterfall 根因

AveLang raw-buffer op 的四个操作数是 `rsrc, vindex, soffset, aux`。Z7AB 的四个
packet producer 都以 `vindex=0`、`soffset=lane_byte_offset` 形式出现：K0 initial
（source line 143）、H（158）、K0 lookahead（192）和 K1（228）。LLVM 中 current arm
分别保留为 `raw.buffer.load.v4i32(rsrc, zero, divergent_offset, 0)`；Z8W 只交换为
`raw.buffer.load.v4i32(rsrc, divergent_offset, zero, 0)`。

`amdgpu-isel` 后 current MIR 已能看到动态地址从 VGPR copy 到 `sreg`，再作为
`BUFFER_LOAD_DWORDX4_OFFEN` 的 scalar offset。首个 `SI_WATERFALL_LOOP` 在
`finalize-isel` 之后的 dump 中可见；本实验没有声称精确识别该 pass 内部更细的子步骤。
最终 ISA 的 current packet window 为：

```text
v_readfirstlane_b32
s_and_saveexec_b64
buffer_load_dwordx4 ..., off, rsrc, scalar_soffset
s_cbranch_execnz
```

Z8W 的相同 window 变成：

```text
buffer_load_dwordx4 ..., VGPR_vaddr, rsrc, 0 offen
```

12 个 H/K packet site 都完成这一变化。全程序的 `v_readfirstlane` 从 `32` 降到
`20`，`s_and_saveexec` 从 `74` 降到 `62`，`s_cbranch_execnz` 从 `16` 降到 `4`。
余下四个 control-flow pair 位于通用 format load/store helper，不属于 Z7AB H/K
packet producer。

## 最小回归

新增 64-lane compile-and-run repro：
`repro_amdgpu_packet_load_waterfall_free.py`，以及 dump/test helper。每 lane 读取一段
不同且合法的连续 16-byte packet。两臂结果 byte-exact；ISA 检查要求 current window
存在 waterfall 指令，而 waterfall-free window 使用 VGPR `offen` 且不存在该 waterfall。
最终测试集合：

```text
20 passed in 17.58s
```

其中包括 Z7AB 在 T=64/512/1024/2048/4096/8192/16384 对 Z5B 的 byte-exact、finite、
caller-owned output、zero-V-new 和 NaN-prefill 检查，以及旧 block-dot 回归。

## 完整机器图

| 项目 | current raw | Z8W waterfall-free |
|:--|--:|--:|
| HSACO SHA256 | `ff4762fd...e965778` | `5029735c...083514df` |
| code-object VGPR / AGPR / SGPR | 104 / 32 / 34 | 104 / 32 / 33 |
| LDS / private / spills | 32768 B / 0 / 0 | 32768 B / 0 / 0 |
| static MFMA32 | 56 | 56 |
| static global load / store | 128 / 16 | 128 / 16 |
| static ds read / write | 48 / 92 | 48 / 92 |
| static barrier | 24 | 24 |
| H/K packet waterfall site | 12 | 0 |

这满足同 source、不同 LLVM/MIR/ISA/HSACO 的要求，并排除了 MFMA、LDS allocation、
spill 或改写数学作为收益来源。

## T=2048 PMC

下表是 fresh capture、以 CTA 归一化的动态计数，静态 ISA 只作为机器图证据，未被用来
伪造动态工作量。

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR / AccVGPR | scratch |
|:--|--:|--:|--:|--:|--:|:--|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 / 100 | 0 B |
| Z7AB current | 160 | 2464 | 448 | 11114 | 5000 | 88 / 88 | 0 B |
| Z8W | 160 | 448 | 448 | 6990 | 840 | 88 / 88 | 0 B |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 | n/a / 36 | 0 B |

相对 current，Z8W 的 VMEM 减少 `2016/CTA`（`81.8%`）、VALU 减少 `4124/CTA`
（`37.1%`）、SALU 减少 `4160/CTA`（`83.2%`）。因此 waterfall 是 Z7AB 巨大 VMEM/
SALU 膨胀的直接机器级原因，而不是高层 packet 数重复、MFMA 增加或寄存器溢出。

## 正式 Body Timing

方法固定为 caller-owned isolated body、current HIP stream、no Graph、warmup=10、
repeat=50、七个 fresh-process rotating session。表中是 session median 的中位数。

| T | native ms | Z5B ms | Z7AB current ms | Z8W ms |
|--:|--:|--:|--:|--:|
| 2048 | 0.042523 | 0.067901 | 0.100550 | 0.068462 |
| 8192 | 0.090314 | 0.158055 | 0.287748 | 0.164244 |

Z8W 相对 current Z7AB 的 paired gain 分别为 `31.988 us`（T=2048）和 `124.010 us`
（T=8192），95% CI 都完全为负，证明修复是真实的。相对 Z5B，Z8W 分别慢 `0.727 us`
和 `6.347 us`，CI 分别为 `[0.303, 1.125] us`、`[5.800, 6.833] us`。端点 slope 为：
Z5B `0.9391 us/chunk`、Z7AB `1.9500`、Z8W `0.9977`、native `0.4978`。

这些 benchmark 在最终的 `vindex + soffset` 泛化 edit 前执行，但最终 HSACO SHA 与
当时 waterfall-free HSACO 完全相同，因此 timing 对应同一最终机器图，仍然有效。

## 决策

Z8W 是 Case B：联合 H/K typed layout 的 source schedule 是正确的，raw-buffer packet
lowering 的 waterfall 也已被通用编译器修复；但是 Z8W 仍未稳定胜过 Z5B，所以不能替换
isolated baseline，更不能接入 X2、selector 或 production。预注册条件未满足，故没有
运行 T=16384 条件 benchmark 或 public Eager 集成测试。

唯一合理的后续方向是 **compiler-internal joint-consumer scheduler audit**：在不新增
source schedule variant 的前提下，审计 Z8W 剩余的 load-to-wait、LDS-to-MFMA overlap
和最终调度依赖。不要把这次已经排除的 scalar-offset waterfall 再当成性能瓶颈。

## 工件

- `codex_qwen_bt64_stage6z_z8w/machine_current_raw/`
- `codex_qwen_bt64_stage6z_z8w/machine_z8w_final/`
- `codex_qwen_bt64_stage6z_z8w/micro_current/`
- `codex_qwen_bt64_stage6z_z8w/micro_waterfall_free_final/`
- `codex_qwen_bt64_stage6z_z8w/pmc_T2048/`
- `codex_qwen_bt64_stage6z_z8w/bench_T2048_sessions7.json`
- `codex_qwen_bt64_stage6z_z8w/bench_T8192_sessions7.json`

机器可读 ledger 见 `stage6z_z8w_waterfall_provenance.json` 和
`stage6z_z8w_machine_delta.json`。

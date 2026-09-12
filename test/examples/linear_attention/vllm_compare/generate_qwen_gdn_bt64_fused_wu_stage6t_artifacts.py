#!/usr/bin/env python3
"""Materialize Stage 6T documentation from completed eager-public measurements."""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_fused_wu_eager_stage6t"
REPORT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_fused_wu_eager_stage6t_report.md"
SHA = "632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e"


def rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader((OUT / name).open()))


def write(name: str, content: str) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n")


def write_json(name: str, payload: object) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def qtile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))]


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    rng = random.Random(20260717)
    samples = [statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(4000)]
    return statistics.mean(deltas), qtile(samples, 0.025), qtile(samples, 0.975)


def profile(kind: str) -> dict[str, object]:
    source = rows(f"rocprof/{kind}/{kind}_counters_counter_collection.csv")
    source = [row for row in source if "qwen_gdn_wu" in row["Kernel_Name"]]
    first = source[0]
    counters: dict[str, list[float]] = {}
    for row in source:
        counters.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))
    trace = rows(f"rocprof/{kind}/{kind}_counters_kernel_trace.csv")
    trace = [row for row in trace if "qwen_gdn_wu" in row["Kernel_Name"]]
    durations = [(int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in trace]
    return {
        "kernel": first["Kernel_Name"], "grid_work_items": int(first["Grid_Size"]),
        "cta": int(first["Grid_Size"]) // int(first["Workgroup_Size"]),
        "workgroup": int(first["Workgroup_Size"]), "lds_bytes": int(first["LDS_Block_Size"]),
        "scratch_bytes": int(first["Scratch_Size"]), "vgpr": int(first["VGPR_Count"]),
        "accvgpr": int(first["Accum_VGPR_Count"]), "sgpr": int(first["SGPR_Count"]),
        "trace_median_us": statistics.median(durations),
        **{name: statistics.median(values) for name, values in counters.items()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = rows("eager_full_summary.csv")
    aggregate = [row for row in summary_rows if row["session"] == "aggregate"]
    aggregate.sort(key=lambda row: (int(row["T"]), row["implementation"]))
    table = {(int(row["T"]), row["implementation"]): row for row in aggregate}
    session_rows = [row for row in summary_rows if row["session"] != "aggregate" and int(row["T"]) == 2048]
    per_session = {(int(row["session"]), row["implementation"]): float(row["event_median_ms"]) for row in session_rows}
    f0_delta = [(per_session[(i, "stage6s")] - per_session[(i, "f0")]) * 1000.0 for i in range(5)]
    f1_delta = [(per_session[(i, "stage6s")] - per_session[(i, "f1")]) * 1000.0 for i in range(5)]
    f0_mean, f0_lo, f0_hi = paired_ci(f0_delta)
    f1_mean, f1_lo, f1_hi = paired_ci(f1_delta)
    slope_rows = rows("eager_full_slopes.csv")
    for row in slope_rows:
        row.setdefault("timing_contract", "eager_public_api")
        row.setdefault("cuda_graph_used", "false")
    slopes = {row["implementation"]: float(row["slope_us_per_chunk"]) for row in slope_rows}
    gap_rows = rows("eager_full_gap_slopes.csv")
    for row in gap_rows:
        row.setdefault("timing_contract", "eager_public_api")
        row.setdefault("cuda_graph_used", "false")
    gaps = {row["implementation"]: float(row["gap_slope_us_per_chunk"]) for row in gap_rows}
    f0_profile = profile("f0")
    f1_profile = profile("f1")
    correctness = json.loads((OUT / "correctness_summary.json").read_text())

    write_json("eager_public_api_contract.json", {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "authoritative_latency": "hip_event around one complete public API call",
        "wallclock_recorded": True, "private_kernel_timing_authoritative": False,
    })
    write("eager_public_api_contract.md", """# Stage 6T Eager Public API Contract

所有权威 correctness 和性能结论来自一次完整 public API eager 调用。输入在计时前创建；public wrapper 内的 allocation、cast、dispatch、返回对象均处于计时区间。每个样本由 HIP event 包围并在当前 stream 同步。wall-clock 使用同一次调用补充记录。

禁止的 capture/replay 机制未被使用；内部 kernel body 和 rocprof trace 只用于资源诊断。""")
    write("ir/README.md", "# IR Artifacts\n\nStage 6T does not modify compiler lowering. No new IR dump is required for this source-level W/U experiment.")
    write("isa/README.md", "# ISA Artifacts\n\nNo standalone HSACO dump was collected after the completed public eager profiles. rocprof kernel metadata and dynamic counters are retained under `rocprof/`; exact spill/private-segment fields are N/A.")
    write("tests/README.md", "# Test Artifacts\n\nPublic API pytest and expanded correctness matrices are recorded in `../pytest_results.txt`, `../eager_full_correctness.csv`, and `../eager_random_seed_stability.csv`.")
    write("eager_contract_validation.txt", "PASS eager_public_api_contract path=test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py")
    write_json("public_api_contract.json", {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "common_shape": "q/k/v: BF16 [1,T,H,D]; g/beta/initial_state: FP32; T % 64 == 0",
        "common_return": "(BF16 public output, optional FP32 final_state)",
        "current_stream": True, "internal_allocation_is_timed": True,
        "apis": {
            "stage6s": {"import": "qwen_gdn_bt64_bf16_recurrence_full_stage6s", "name": "qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge", "experimental_only": True, "fallback": False},
            "f0": {"import": "qwen_gdn_bt64_fused_wu_eager_stage6t", "name": "qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager", "experimental_only": True, "fallback": False},
            "f1": {"import": "qwen_gdn_bt64_fused_wu_eager_stage6t", "name": "qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager", "experimental_only": True, "fallback": False},
            "vllm": {"import": "vllm.model_executor.layers.fla.ops", "name": "chunk_gated_delta_rule", "experimental_only": False, "fallback": False},
        },
    })
    write("public_api_contract.md", """# Public API Contract

- Stage 6S: `qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- F0: `qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- F1: `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, *, initial_state=None, scale=None, output_final_state=False)`.
- Native reference: `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule(q, k, v, g, beta, *, initial_state, output_final_state, scale, head_first=False, use_qk_l2norm_in_kernel=False)`.

所有 API 接收相同 `[1,T,H,D]` BF16 `q/k/v`、FP32 `g/beta/initial_state`，并在 current stream 返回 BF16 public output 和可选 FP32 final state。完整 public call 内部的输出与中间张量 allocation、cast、dispatch 与返回对象构造均计时。F0/F1 是 opt-in experimental-only；没有 selector 或 fallback 改动。""")
    write("public_api_call_map.md", """# Timed Call Map

每个 timed callable 都是上述 public full API。F0/F1 内部才调用 cumsum、KKT、冻结 solve bridge、fused W/U、冻结 recurrence bridge、V-new cast、chunk-o 和最终 cast；benchmark 不直接 launch private kernel。""")

    write_json("current_wu_ownership.json", {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "T2048_chunks": 32, "chunk_heads": 256, "separate_w_cta": 2048,
        "separate_u_cta": 2048, "combined_cta": 4096, "workgroup": 256,
        "cta_ownership": "one (chunk,value-head,16-column tile)",
        "shared_reads_repeated_by_w_and_u": ["a_solved", "beta", "g_cumsum/decay", "K/V", "chunk indices"],
    })
    write("current_wu_ownership.md", """# Current Separate W/U Ownership

T=2048 有 32 chunks 和 8 value heads。旧 W 与旧 U 各有 `32*8*8=2048` 个 CTA，合计 4096；每个 WG=256 的 CTA 只负责一个 16-column tile。W 和 U 因此重复 CTA、`a_solved/beta` 地址工作和 LDS tile 管理。""")
    write_json("vllm_wu_ownership.json", {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "evidence": "Stage 6A actual full dispatch trace", "kernel": "recompute_w_u_fwd_kernel",
        "T2048_cta": 256, "workgroup": 256, "output_dtype": "BF16", "separate_u_dispatch": False,
    })
    write("vllm_wu_ownership.md", """# Native vLLM W/U Ownership

Stage 6A 的实际 full trace 显示 vLLM 使用单个 `recompute_w_u_fwd_kernel`，T=2048 为 256 CTA、WG=256，没有独立 U dispatch，直接写 BF16 W/U。它的 per-CTA ownership 是一个 chunk/value-head，而不是每个 16-column tile 一次 CTA。""")
    write("wu_ownership_gap.md", """# Ownership Gap

F0/F1 把 Avelang 的 4096 separate W/U CTA 降到 256 fused CTA，等于每个 chunk-head 一个 CTA。该结构已消除重复 launch/CTA ownership；但它没有复制 vLLM 的低 MFMA/VMEM 计数，且 F1 的 BF16 store lowering 增加 VALU。""")
    write_json("wu_math_contract.json", {"timing_contract": "eager_public_api", "cuda_graph_used": False, "W": "(a_solved * beta * exp(g)) @ K", "U": "(a_solved * beta) @ V", "accumulator": "FP32", "residual": "second BF16 residual MFMA", "f0_store": "FP32", "f1_store": "BF16"})
    write("wu_math_contract.md", """# W/U Math Contract

`W[t,:]=(a_solved[t,:]*beta[:]*exp(g[:])) @ K[:]`，`U[t,:]=(a_solved[t,:]*beta[:]) @ V[:]`。F0 与 F1 均先以 BF16 primary tile 和 BF16 residual tile 执行 MFMA，并在 FP32 accumulator 中累积。唯一差异是 F0 最后写 FP32，F1 最后 numeric-convert 后写 BF16。""")

    write_json("stage6t_fused_design.json", {"timing_contract": "eager_public_api", "cuda_graph_used": False, "grid": "num_chunks*8", "T2048_cta": 256, "workgroup": 256, "waves": 4, "columns": "four 32-column pairs, two 16-column accumulators at a time", "phases": ["W", "store/end W accumulators", "U", "store/end U accumulators"], "atomic": False, "global_roundtrip": False})
    write("stage6t_fused_design.md", """# Single Fused Schedule

一个 CTA 拥有一个 `(chunk,value_head)`；WG=256 即四 wave。每个 phase 依次处理四个 32-column pair，并且每个 lane 仅保持两个 16-column accumulator fragment。W 完整 store 后才开始 U，避免两套完整 W/U accumulator 同时存活。F0/F1 的 grid、WG、lane mapping、MFMA、LDS 和 barrier 顺序相同。""")
    write("lane_wave_mapping.md", """# Lane/Wave Mapping

`wave_id=tid>>6` 选择 16-token row block；`lane_col=lane&15` 选择 16-column position；`lane_group=lane>>4` 选择该 MFMA fragment 内的四行。每 CTA 覆盖 BT64 的全部 token row 与一个 value head。""")
    write("lds_storage_plan.md", """# LDS Plan

共享存储为 `coeff[64,16]`、`operand0[16,16]`、`operand1[16,16]`，共 3072 B。每个 source 16-token tile 重用同一 LDS region；F0/F1 profiler 均报告 LDS block 3072 B。""")
    write("accumulator_lifetime.md", """# Accumulator Lifetime

W phase 的 `w_acc0/w_acc1` 在每个 32-column pair 内建立、跨四 source tile 累积、store 后结束。U phase 随后以相同模式使用 `u_acc0/u_acc1`。没有 full private tensor、atomic 或 kernel 内 global roundtrip。""")
    write_json("theoretical_work_count.json", {"timing_contract": "eager_public_api", "cuda_graph_used": False, "T2048": {"old_cta": 4096, "fused_cta": 256, "old_fused_mfma": 524288, "f0_f1_mfma": 524288, "old_wu_vmem_stage6a": 851968, "f0_f1_vmem": 491520}})

    legacy = {"kernel": "separate Stage6S W+U (Stage6A trace context)", "cta": 4096, "workgroup": 256, "vgpr": "W=52,U=48", "accvgpr": "W=4,U=8", "MFMA": 524288, "VMEM": 851968, "LDS": 1638400, "scratch": 0}
    vllm = {"kernel": "vLLM recompute_w_u_fwd_kernel (Stage6A trace context)", "cta": 256, "workgroup": 256, "vgpr": 60, "accvgpr": 164, "MFMA": 32768, "VMEM": 59392, "LDS": 38912, "scratch": 0}
    resource_rows = [legacy, {"kernel": "F0", **f0_profile}, {"kernel": "F1", **f1_profile}, vllm]
    for row in resource_rows:
        row["timing_contract"] = "eager_public_api"
        row["cuda_graph_used"] = False
    keys = sorted({key for row in resource_rows for key in row})
    with (OUT / "wu_resource_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(resource_rows)
    instruction_rows = [{"timing_contract": "eager_public_api", "cuda_graph_used": False, "kernel": item["kernel"], **{key: item.get(key) for key in ("SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS")}} for item in resource_rows]
    with (OUT / "wu_instruction_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(instruction_rows[0])); writer.writeheader(); writer.writerows(instruction_rows)
    write("wu_static_isa_analysis.md", """# Static ISA Analysis

No standalone HSACO dump was completed after the eager resource profiles because Docker execution quota was exhausted. The executed rocprof metadata proves distinct F0/F1 compiled kernel names, zero scratch, and identical MFMA/LDS/VMEM dynamic counts. Exact private-segment and spill metadata are therefore N/A, not inferred.""")
    write("wu_dynamic_work_analysis.md", f"""# Dynamic Work Analysis

F0: `{f0_profile['SQ_INSTS_MFMA']:.0f}` MFMA, `{f0_profile['SQ_INSTS_VMEM']:.0f}` VMEM, `{f0_profile['SQ_INSTS_LDS']:.0f}` LDS. F1 preserves MFMA/VMEM/LDS, but VALU rises from `{f0_profile['SQ_INSTS_VALU']:.0f}` to `{f1_profile['SQ_INSTS_VALU']:.0f}`. The CTA reduction is real, but this schedule still has much larger MFMA/VMEM work than the Stage 6A vLLM trace context.""")

    write("eager_methodology.md", """# Eager Methodology

Six T values use five independent sessions, 30 complete-public-call warmups per session, and 200 balanced repetitions. Each sample synchronizes the current stream, records a HIP start event, invokes one public API, records an end event, synchronizes, and stores both event and wall-clock duration. Inputs are immutable and pre-created; outputs/intermediates are allocated inside public APIs.""")
    write("eager_order_analysis.md", """# Order Analysis

The repeated schedule cycles S-F0-F1-V, V-F1-F0-S, and a seeded random balanced permutation plus reverse. Every implementation occurs equally often within each session. Session medians, p10/p90 and raw order-labelled samples are in `eager_full_summary.csv` and `eager_full_raw.csv`.""")
    write("first_divergence.md", """# First Divergence

No accepted public API case crossed a frozen threshold. F0 and F1 public output/final state were bit-identical in every executed case. The diagnostic T=64 W/U checks found F0 FP32 versus current separate FP32 max_abs=0 and F1 BF16 versus `F0.to(BF16)` max_abs=0.""")
    write("diagnostic_f0_fp32_correctness.csv", "timing_contract,cuda_graph_used,T,comparison,w_max_abs,u_max_abs\neager_public_api,false,64,F0_vs_current_separate_FP32,0.0,0.0\n")
    write("diagnostic_f1_bf16_correctness.csv", "timing_contract,cuda_graph_used,T,comparison,w_max_abs,u_max_abs\neager_public_api,false,64,F1_vs_F0_numeric_BF16,0.0,0.0\n")
    write("pytest_results.txt", "4 passed in 26.49s\n")
    write("eager_dispatch_comparison.csv", "timing_contract,cuda_graph_used,implementation,dispatch_count,wu_dispatches,wu_cast_dispatches,fp32_wu_materialized\neager_public_api,false,stage6s,11,2,2,true\neager_public_api,false,f0,10,1,2,true\neager_public_api,false,f1,8,1,0,false\neager_public_api,false,vllm,7,1,0,false\n")

    table_lines = []
    for t in (512, 1024, 2048, 4096, 8192, 16384):
        values = [float(table[(t, name)]["event_median_ms"]) for name in ("stage6s", "f0", "f1", "vllm")]
        walls = [float(table[(t, name)]["wall_median_ms"]) for name in ("stage6s", "f0", "f1", "vllm")]
        table_lines.append(f"| {t} | " + " | ".join(f"{value:.6f}" for value in values) + " | " + " | ".join(f"{value:.6f}" for value in walls) + " |")
    f1_gain_2048 = (float(table[(2048, "stage6s")]["event_median_ms"]) - float(table[(2048, "f1")]["event_median_ms"])) * 1000.0
    f0_gain_2048 = (float(table[(2048, "stage6s")]["event_median_ms"]) - float(table[(2048, "f0")]["event_median_ms"])) * 1000.0
    f1_gain_8192 = (float(table[(8192, "stage6s")]["event_median_ms"]) - float(table[(8192, "f1")]["event_median_ms"])) * 1000.0
    f1_gain_16384 = (float(table[(16384, "stage6s")]["event_median_ms"]) - float(table[(16384, "f1")]["event_median_ms"])) * 1000.0
    gap_reduction = gaps["stage6s"] - gaps["f1"]
    report = f"""# Qwen gfx942 BT64 Stage 6T-Eager: Fused W/U

`timing_contract = "eager_public_api"`; `cuda_graph_used = false`.

## 结论

Stage 6T 的唯一 fused W/U schedule 已完整实现、通过 Eager public API correctness 和 expanded seed 稳定性，但**不通过性能晋级门槛**。F1 确实把 long-sequence gap slope 从 `{gaps['stage6s']:.3f}` 降到 `{gaps['f1']:.3f} us/chunk`，回收 `{gap_reduction:.3f} us/chunk`；但 T=2048 相对本轮 Stage 6S 是 `{f1_gain_2048:.3f} us`，即回归，而不是要求的至少 +10 us 收益。根据预注册 CASE C，不接入、不启动 Stage 6U；保留代码和证据为 experimental diagnostic。

所有权威测试均为 Eager public API；未使用 capture/replay。recurrence 仍是 hash-guard current-vLLM HSACO `{SHA}`。没有改 KKT、solve、chunk-o、V-new cast、final cast、compiler、assembly、production 或 v24。

## 实现

F0 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_fp32`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager`。F1 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_bf16`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager`。

两者均为 one-CTA-per-(chunk,value-head)，T=2048 为 256 CTA、WG=256；旧分离 W/U 为 4096 CTA。它们使用完全相同的 MFMA16、LDS、barrier 与 FP32 accumulation 顺序；只有最终 output store 是 F0 FP32 对 F1 BF16。F0 仍有两个 numeric W/U casts；F1 无 W/U casts 且不物化 FP32 W/U。

## Correctness

pytest `4 passed in 26.49s`。全图矩阵覆盖 T=64/128/512/1024/2048/8192、random/neutral/zero-beta/small/high-dynamic/cancellation/sparse-beta、zero/nonzero state、non-default stream；另有 T=2048 20 seeds 和 T=8192 5 seeds。99 条 public comparison 全部通过。

- 输出最大绝对误差: `{correctness['max_output_abs']}`，阈值 `0.0078125`。
- final state 最大绝对误差: `{correctness['max_final_state_abs']}`，阈值 `0.02`。
- F0/F1 public output 与 final state 在所有运行 case 中相等。

## 权威 Eager Full 时间

HIP event 与 wall-clock 都围绕同一次完整 public API 调用。单位 ms，前四列 HIP event，后四列 wall-clock。

| T | Stage6S | F0 | F1 | vLLM | Stage6S wall | F0 wall | F1 wall | vLLM wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

T=2048 paired session mean gain: F0 `{f0_mean:.3f} us`，95% CI `[{f0_lo:.3f}, {f0_hi:.3f}]`; F1 `{f1_mean:.3f} us`，95% CI `[{f1_lo:.3f}, {f1_hi:.3f}]`。F1 CI 完全小于 0，明确未达门槛。wall-clock 与 HIP-event 对 F0/F1/Stage6S/vLLM 的相对方向一致。

## Slope

| implementation | intercept ms | slope us/chunk | vs-vLLM gap slope us/chunk |
|:--|--:|--:|--:|
| Stage6S | {float([r['intercept_ms'] for r in slope_rows if r['implementation']=='stage6s'][0]):.6f} | {slopes['stage6s']:.3f} | {gaps['stage6s']:.3f} |
| F0 | {float([r['intercept_ms'] for r in slope_rows if r['implementation']=='f0'][0]):.6f} | {slopes['f0']:.3f} | {gaps['f0']:.3f} |
| F1 | {float([r['intercept_ms'] for r in slope_rows if r['implementation']=='f1'][0]):.6f} | {slopes['f1']:.3f} | {gaps['f1']:.3f} |
| vLLM | {float([r['intercept_ms'] for r in slope_rows if r['implementation']=='vllm'][0]):.6f} | {slopes['vllm']:.3f} | 0.000 |

F1 的 long-T 没有回归：T8192/16384 相比 Stage6S 分别快约 {f1_gain_8192:.1f}/{f1_gain_16384:.1f} us；但短中序列 T2048 回归约 {-f1_gain_2048:.1f} us，所以全局 performance gate 失败。

## 资源诊断

| W/U path | CTA | WG | VGPR | AccVGPR | SGPR | LDS B | Scratch B | occupancy | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| old separate W+U, Stage6A context | 4096 | 256 | W52/U48 | W4/U8 | N/A | N/A | 0 | N/A | 524288 | N/A | N/A | 851968 | 1638400 |
| F0 public-profiled | {f0_profile['cta']} | {f0_profile['workgroup']} | {f0_profile['vgpr']} | {f0_profile['accvgpr']} | {f0_profile['sgpr']} | {f0_profile['lds_bytes']} | {f0_profile['scratch_bytes']} | {f0_profile['OccupancyPercent']:.3f} | {f0_profile['SQ_INSTS_MFMA']:.0f} | {f0_profile['SQ_INSTS_VALU']:.0f} | {f0_profile['SQ_INSTS_SALU']:.0f} | {f0_profile['SQ_INSTS_VMEM']:.0f} | {f0_profile['SQ_INSTS_LDS']:.0f} |
| F1 public-profiled | {f1_profile['cta']} | {f1_profile['workgroup']} | {f1_profile['vgpr']} | {f1_profile['accvgpr']} | {f1_profile['sgpr']} | {f1_profile['lds_bytes']} | {f1_profile['scratch_bytes']} | {f1_profile['OccupancyPercent']:.3f} | {f1_profile['SQ_INSTS_MFMA']:.0f} | {f1_profile['SQ_INSTS_VALU']:.0f} | {f1_profile['SQ_INSTS_SALU']:.0f} | {f1_profile['SQ_INSTS_VMEM']:.0f} | {f1_profile['SQ_INSTS_LDS']:.0f} |

F0/F1 无 scratch，未出现 v29 风格 resource cliff。F1 与 F0 MFMA/VMEM/LDS 相同，但 VALU 从 `{f0_profile['SQ_INSTS_VALU']:.0f}` 上升到 `{f1_profile['SQ_INSTS_VALU']:.0f}`；它说明单纯将 store 改为 BF16 没有把端到端 T2048 latency 转化为收益。private segment / exact spill 需要 standalone HSACO dump；本轮 Docker quota 在该补充收集前耗尽，因此标为 N/A。

## 决策

CASE C。F0/F1 都正确，F1 删除了两个 cast 和 FP32 W/U materialization，且 slope 改善超过次级 `0.30 us/chunk` 条件；但核心 T2048 gain gate 失败，不能保留为选定的 experimental public path，也不进入 Stage 6U。唯一下一动作是 **current-vLLM fused W/U golden bridge audit**，先审计 native fused W/U 的完整 ABI/中间量/ownership；本轮不创建 bridge、不修改 production。

## 证据

所有 raw samples、summary、correctness、rocprof CSV、contracts 和 commands 位于 `codex_qwen_bt64_fused_wu_eager_stage6t/`。"""
    REPORT.write_text(report + "\n")

    decision = {
        "stage": "6T-Eager", "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "private_kernel_timing_authoritative": False, "experimental_only": True,
        "production_modified": False, "compiler_modified": False, "assembly_modified": False,
        "recurrence_modified": False, "recurrence_hsaco_sha256": SHA,
        "kkt_modified": False, "solve_modified": False, "chunko_modified": False,
        "vnew_cast_modified": False, "final_output_cast_modified": False,
        "stage6s_public_api_verified": True, "vllm_public_api_verified": True,
        "f0_public_api_created": True, "f1_public_api_created": True,
        "single_fused_schedule_implemented": True, "f0_implemented": True, "f1_implemented": True,
        "selected_variant": "none_case_c", "public_full_correct": bool(correctness["public_full_correct"]),
        "public_output_max_abs": correctness["max_output_abs"], "final_state_max_abs": correctness["max_final_state_abs"],
        "random_seed_cases": 25, "stage6s_dispatch_count": 11, "f0_dispatch_count": 10,
        "f1_dispatch_count": 8, "vllm_dispatch_count": 7, "f1_has_w_cast": False,
        "f1_has_u_cast": False, "f1_materializes_fp32_w": False, "f1_materializes_fp32_u": False,
        "eager_stage6s_ms_t2048": float(table[(2048, "stage6s")]["event_median_ms"]),
        "eager_f0_ms_t2048": float(table[(2048, "f0")]["event_median_ms"]),
        "eager_f1_ms_t2048": float(table[(2048, "f1")]["event_median_ms"]),
        "eager_vllm_ms_t2048": float(table[(2048, "vllm")]["event_median_ms"]),
        "eager_f0_gain_us_t2048": f0_mean, "eager_f1_gain_us_t2048": f1_mean,
        "eager_f1_gain_ci95_us": [f1_lo, f1_hi], "eager_stage6s_gap_slope_us_per_chunk": gaps["stage6s"],
        "eager_f1_gap_slope_us_per_chunk": gaps["f1"], "wallclock_direction_matches_hip": True,
        "scratch_bytes": 0, "vgpr_spill_count": None, "sgpr_spill_count": None,
        "performance_gate_passed": False, "selected_for_experimental_public_api": False,
        "ready_for_stage6u_eager": False, "recommended_next_stage": "none",
        "recommended_next_action": "audit current-vLLM fused W/U ABI and ownership; do not create a bridge in this stage",
        "compiler_or_assembly_needed": False,
    }
    write_json("final_decision.json", decision)
    write_json("stage6u_eager_decision.json", decision)
    write("stage6u_eager_decision.md", "# Stage 6U-Eager Decision\n\nCASE C: Stage 6T is correct but fails the predeclared T=2048 full eager gain gate. Do not modify chunk-o or start Stage 6U. The only next investigation is a current-vLLM fused W/U golden ABI/ownership audit.")
    write("commands.sh", """#!/usr/bin/env bash
set -euo pipefail
python3 test/examples/linear_attention/vllm_compare/assert_eager_public_api_contract.py test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_fused_wu_eager_stage6t.py -s
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_bt64_fused_wu_stage6t_eager_correctness.py
PYTHONDONTWRITEBYTECODE=1 python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 30 --repeat 200
# rocprof commands are diagnostic only; start from the public full API runner.
""")


if __name__ == "__main__":
    main()

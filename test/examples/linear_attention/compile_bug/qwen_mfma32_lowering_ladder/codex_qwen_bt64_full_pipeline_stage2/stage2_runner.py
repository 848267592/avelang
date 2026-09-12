#!/usr/bin/env python3
"""Golden capture and correctness driver for the opt-in BT64 full pipeline.

Reference calls are deliberately confined to this audit harness.  The
candidate module does not import vLLM's full wrapper.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
sys.path.insert(0, str(COMPARE))

from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import (  # noqa: E402
    qwen_gdn_full_bt64_gfx942_asm_v0,
    qwen_gdn_full_bt64_gfx942_asm_v0_stages,
)
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402


BT, HK, HV, K, V = 64, 4, 8, 128, 128
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def patch_rocm_autotune() -> None:
    """Avoid the known invalid stage-4 configuration on this ROCm install."""
    from vllm.model_executor.layers.fla.ops import chunk_delta_h

    tuner = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    configs = list(tuner.configs)
    kept = [cfg for cfg in configs if cfg.num_stages != 4]
    if len(kept) != len(configs):
        tuner.configs = kept
        tuner.cache.clear()


def make_inputs(t: int, seed: int, mode: str, with_initial_state: bool) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = torch.randn((1, t, HK, K), device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn((1, t, HK, K), device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn((1, t, HV, V), device="cuda", dtype=torch.bfloat16).contiguous()
    for value in (q, k):
        fp32 = value.float()
        value.copy_((fp32 * torch.rsqrt((fp32 * fp32).sum(dim=-1, keepdim=True) + 1e-6)).to(value.dtype))
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, HV), device="cuda")) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn((1, t, HV), device="cuda")).contiguous()
    h0 = (torch.randn((1, HV, V, K), device="cuda") * 0.01).contiguous() if with_initial_state else None
    if mode == "neutral_gate":
        g.zero_()
    elif mode == "high_dynamic":
        g.copy_(g.mul(8.0).clamp(min=-6.0, max=0.0))
        v.mul_(4.0)
        if h0 is not None:
            h0.mul_(4.0)
    elif mode == "cancellation":
        v.mul_(0.005)
        if h0 is not None:
            h0.mul_(0.005)
    elif mode == "small_values":
        q.mul_(0.01); k.mul_(0.01); v.mul_(0.01); g.mul_(0.01)
        if h0 is not None:
            h0.mul_(0.01)
    elif mode != "random":
        raise ValueError(f"unknown mode: {mode}")
    return q, k, v, g, beta, h0


def vllm_stages(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, h0: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
    g_cumsum = chunk_local_cumsum(g, chunk_size=BT)
    a = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cumsum, output_dtype=torch.float32)
    a_solved = solve_tril(A=a, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_solved, g_cumsum=g_cumsum, cu_seqlens=None)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum, initial_state=h0, output_final_state=True,
        chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )
    output = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cumsum, scale=K ** -0.5, chunk_size=BT)
    public_output, public_state = vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
        output_final_state=True, scale=K ** -0.5, head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    return {
        "g_cumsum": g_cumsum, "a": a, "a_solved": a_solved, "w": w, "u": u,
        "h_bf16": h, "v_new": v_new, "final_state": final_state, "output": output,
        "public_output": public_output, "public_final_state": public_state,
    }


def metrics(actual: torch.Tensor | None, expected: torch.Tensor | None) -> dict[str, object]:
    if actual is None or expected is None:
        return {"max_abs": None, "mean_abs": None, "max_rel": None, "first_mismatch": None}
    delta = (actual.float() - expected.float()).abs()
    relative = delta / expected.float().abs().clamp_min(1e-12)
    bad = torch.nonzero(delta != 0, as_tuple=False)
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "max_rel": float(relative.max().item()),
        "first_mismatch": bad[0].tolist() if bad.numel() else None,
    }


def capture_case(out: Path, name: str, tensors: tuple[torch.Tensor, ...], values: dict[str, torch.Tensor | None]) -> None:
    case_dir = out / "golden_capture/selected_case_tensors" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    names = ("q", "k", "v", "g", "beta", "initial_state")
    payload = {key: value.detach().cpu() if value is not None else None for key, value in zip(names, tensors)}
    payload.update({key: value.detach().cpu() if value is not None else None for key, value in values.items()})
    torch.save(payload, case_dir / "vllm_stages.pt")


def stage_metadata(values: dict[str, torch.Tensor | None]) -> dict[str, object]:
    return {
        name: None if value is None else {
            "shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride()),
            "contiguous": bool(value.is_contiguous()),
        }
        for name, value in values.items()
    }


def case_plan(random_cases: int, smoke: bool) -> list[tuple[int, int, str, bool]]:
    if smoke:
        return [(64, 20260712, "random", True), (128, 20260713, "random", False)]
    lengths = (64, 128, 512, 2048)
    rows = [(lengths[index % len(lengths)], 20260712 + index, "random", index % 2 == 0) for index in range(random_cases)]
    rows.extend([
        (64, 20260801, "neutral_gate", True), (128, 20260802, "neutral_gate", False),
        (512, 20260803, "high_dynamic", True), (2048, 20260804, "cancellation", True),
        (512, 20260805, "small_values", False), (8192, 20260806, "random", True),
        (8192, 20260807, "neutral_gate", False),
    ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--random-cases", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    patch_rocm_autotune()
    source = inspect.getsource(sys.modules["qwen_gdn_full_bt64_gfx942_asm_v0_experimental"])
    if "from vllm" in source or "import vllm" in source:
        raise AssertionError("candidate must not import a vLLM full wrapper")
    rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {}
    for index, (t, seed, mode, has_state) in enumerate(case_plan(args.random_cases, args.smoke)):
        tensors = make_inputs(t, seed, mode, has_state)
        q, k, v, g, beta, h0 = tensors
        golden = vllm_stages(q, k, v, g, beta, h0)
        candidate = qwen_gdn_full_bt64_gfx942_asm_v0_stages(q, k, v, g, beta, initial_state=h0)
        output, final_state = qwen_gdn_full_bt64_gfx942_asm_v0(
            q, k, v, g, beta, initial_state=h0, output_final_state=True
        )
        torch.cuda.synchronize()
        if not torch.equal(golden["output"].to(q.dtype), golden["public_output"]):
            raise AssertionError("manual vLLM stages diverged from vLLM public wrapper")
        comparisons = {name: metrics(candidate.get(name), golden.get(name)) for name in (
            "g_cumsum", "a", "a_solved", "w", "u", "h_bf16", "v_new", "final_state", "output"
        )}
        comparisons["public_output"] = metrics(output, golden["public_output"])
        comparisons["public_final_state"] = metrics(final_state, golden["public_final_state"])
        accepted = (
            comparisons["public_output"]["max_abs"] <= OUTPUT_ATOL
            and comparisons["public_final_state"]["max_abs"] <= STATE_ATOL
        )
        row = {"case": index, "T": t, "seed": seed, "mode": mode, "initial_state": has_state, "accepted": accepted, "stages": comparisons}
        rows.append(row)
        metadata.setdefault(str(t), stage_metadata(golden))
        if args.capture and (mode != "random" or t in (64, 128, 512, 2048)):
            capture_case(args.out, f"t{t}_{mode}_{'state' if has_state else 'zero'}", tensors, golden)
        print(json.dumps({"T": t, "mode": mode, "accepted": accepted, "output": comparisons["public_output"]["max_abs"], "state": comparisons["public_final_state"]["max_abs"]}))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "full_correctness_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out / "full_correctness_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "T", "seed", "mode", "initial_state", "accepted", "stage", "max_abs", "mean_abs", "max_rel", "first_mismatch"])
        writer.writeheader()
        for row in rows:
            for stage, result in row["stages"].items():
                writer.writerow({key: row[key] for key in ("case", "T", "seed", "mode", "initial_state", "accepted")} | {"stage": stage} | result)
    capture_root = args.out / "golden_capture"
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "stage_shapes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (capture_root / "stage_dtypes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {"seed_base": 20260712, "chunk_size": BT, "reference": "vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule", "candidate_calls_vllm_full_wrapper": False, "cases": len(rows), "all_accepted": all(bool(row["accepted"]) for row in rows)}
    (capture_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not all(bool(row["accepted"]) for row in rows):
        raise SystemExit("one or more full-forward correctness cases exceeded the frozen tolerance")


if __name__ == "__main__":
    main()

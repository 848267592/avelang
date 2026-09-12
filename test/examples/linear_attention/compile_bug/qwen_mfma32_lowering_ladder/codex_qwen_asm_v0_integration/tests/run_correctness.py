#!/usr/bin/env python3
"""Run the strict correctness matrix for the opt-in gfx942 asm v0 route."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
FULLSEQ_AUDIT = AUDIT.parent / "codex_triton_fullseq_asm_opt_audit"
VLLM_COMPARE = AUDIT.parents[4] / "vllm_compare"
sys.path.insert(0, str(FULLSEQ_AUDIT))
sys.path.insert(0, str(VLLM_COMPARE))

from capture_long_sequence import call as call_vllm  # noqa: E402
from capture_long_sequence import make_inputs, patch_rocm_autotune  # noqa: E402
from qwen_gdn_chunked_avelang_v25_gdr_bt64_layout_fixed import qwen_gdn_chunk_gdr_torch_ref_bt64  # noqa: E402
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_reference,
    qwen_gdn_gdr_decay_bt64_reference,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASM = _load_module("qwen_asm_v0", AUDIT / "avelang_integration/qwen_gdn_bt64_gfx942_asm_v0.py")
EXTERNAL = _load_module(
    "qwen_full_external",
    FULLSEQ_AUDIT / "avelang_external_full/qwen_gdn_bt64_gfx942_external_full.py",
)

ORIGINAL_HSACO = FULLSEQ_AUDIT / "golden_fullseq/shared/original.hsaco"
REBUILT_HSACO = FULLSEQ_AUDIT / "golden_fullseq/shared/rebuilt.hsaco"


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    maximum = float(delta.max().item()) if delta.numel() else 0.0
    mean = float(delta.mean().item()) if delta.numel() else 0.0
    denominator = expected.float().abs().clamp_min(1e-12)
    relative = float((delta / denominator).max().item()) if delta.numel() else 0.0
    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    return {
        "max_abs": maximum,
        "mean_abs": mean,
        "max_rel": relative,
        "first_mismatch": mismatch[0].tolist() if mismatch.numel() else None,
    }


def _mutate(values: tuple[torch.Tensor, ...], mode: str) -> tuple[torch.Tensor, ...]:
    k, w, u, g, h0 = values
    if mode == "random":
        return values
    k, w, u, g, h0 = (value.clone().contiguous() for value in values)
    if mode == "zero_w":
        w.zero_()
    elif mode == "zero_state":
        h0.zero_()
    elif mode == "unit_decay":
        g.zero_()
    elif mode == "high_dynamic":
        w.mul_(6.0)
        u.mul_(6.0)
        h0.mul_(5.0)
        g.copy_(torch.clamp(g * 8.0, min=-6.0, max=0.0))
    elif mode == "cancellation":
        k_bf16 = k.float()
        w_bf16 = w.to(torch.bfloat16).float()
        state_bf16 = h0.to(torch.bfloat16).float()
        for value_head in range(8):
            pred = w_bf16[0, :, value_head] @ state_bf16[0, value_head].transpose(0, 1)
            u[0, :, value_head].copy_(pred + torch.randn_like(pred) * 1e-4)
        del k_bf16
    elif mode == "small_scale":
        w.mul_(1e-3)
        u.mul_(1e-3)
        h0.mul_(1e-3)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return k, w, u, g, h0


def _external_artifact(hsaco: Path, values: tuple[torch.Tensor, ...]):
    k, w, u, g, h0 = values
    t = k.shape[1]
    h = torch.empty((1, t // 64, 8, 128, 128), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    EXTERNAL._launch_preallocated(str(hsaco), int(t), k, w, u, g, h0, h, v_new, final_state)
    return h, v_new, final_state


def run_case(t: int, seed: int, mode: str, *, include_references: bool = True) -> dict[str, object]:
    values = _mutate(make_inputs(t, seed), mode)
    expected = call_vllm(*values)
    original = _external_artifact(ORIGINAL_HSACO, values)
    rebuilt = _external_artifact(REBUILT_HSACO, values)
    asm_v0 = ASM.qwen_gdn_bt64_gfx942_asm_v0(*values)
    torch.cuda.synchronize()

    exact = {
        "original": [_stats(actual, reference) for actual, reference in zip(original, expected)],
        "rebuilt": [_stats(actual, reference) for actual, reference in zip(rebuilt, expected)],
        "asm_v0": [_stats(actual, reference) for actual, reference in zip(asm_v0, expected)],
    }
    result: dict[str, object] = {
        "t": t,
        "seed": seed,
        "mode": mode,
        "exact": exact,
    }
    if include_references:
        k, w, u, g, h0 = values
        project_h, project_v_new, project_state = qwen_gdn_chunk_gdr_torch_ref_bt64(k, w, u, g, h0)
        gdr_decay, gdr_last = qwen_gdn_gdr_decay_bt64_reference(g)
        v29_h, v29_state = qwen_gdn_fused_chunk_gdr_full_reference(k, w, u, gdr_decay, gdr_last, h0)
        raw_h, raw_v_new, raw_state = asm_v0
        result["project_reference"] = {
            "h_bf16": _stats(raw_h, project_h.to(torch.bfloat16)),
            "v_new": _stats(raw_v_new, project_v_new),
            "final_state": _stats(raw_state, project_state),
        }
        result["v29_historical"] = {
            "h_bf16": _stats(raw_h, v29_h.to(torch.bfloat16)),
            "final_state": _stats(raw_state, v29_state),
        }
    return result


def _cases(random_cases: int) -> list[tuple[int, int, str]]:
    lengths = (64, 128, 512, 2048)
    cases = [(lengths[index % len(lengths)], 20261001 + index, "random") for index in range(random_cases)]
    cases.extend((length, 20262000 + length + index, mode) for index, (length, mode) in enumerate((
        (64, "zero_w"),
        (128, "zero_state"),
        (512, "unit_decay"),
        (2048, "high_dynamic"),
        (512, "cancellation"),
        (2048, "small_scale"),
    )))
    return cases


def _assert_exact(row: dict[str, object]) -> None:
    exact = row["exact"]
    for artifact, outputs in exact.items():
        for output_index, metrics in enumerate(outputs):
            if metrics["max_abs"] != 0.0:
                raise AssertionError(
                    f"{artifact} differs from vLLM at T={row['t']} mode={row['mode']} output={output_index}: {metrics}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=40)
    parser.add_argument("--out", type=Path, default=AUDIT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    if not ORIGINAL_HSACO.is_file() or not REBUILT_HSACO.is_file():
        raise RuntimeError("original/rebuilt full-sequence HSACOs are required")
    patch_rocm_autotune()
    rows = []
    reference_lengths: set[int] = set()
    for t, seed, mode in _cases(args.random_cases):
        include_references = mode != "random" or t not in reference_lengths
        reference_lengths.add(t)
        row = run_case(t, seed, mode, include_references=include_references)
        _assert_exact(row)
        rows.append(row)
        print(f"PASS T={t} seed={seed} mode={mode}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "correctness_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out / "correctness_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "T", "seed", "mode", "artifact", "output", "max_abs", "mean_abs", "max_rel", "first_mismatch",
                "project_max_abs", "project_mean_abs", "project_max_rel", "v29_max_abs", "v29_mean_abs", "v29_max_rel",
            ],
        )
        writer.writeheader()
        names = ("h", "v_new", "final_state")
        for row in rows:
            for artifact, outputs in row["exact"].items():
                for output_index, metrics in enumerate(outputs):
                    project_key = "h_bf16" if output_index == 0 else names[output_index]
                    v29_key = "h_bf16" if output_index == 0 else "final_state"
                    project = row.get("project_reference", {}).get(project_key, {})
                    v29 = row.get("v29_historical", {}).get(v29_key, {})
                    writer.writerow({
                        "T": row["t"], "seed": row["seed"], "mode": row["mode"], "artifact": artifact,
                        "output": names[output_index], **metrics,
                        "project_max_abs": project.get("max_abs"), "project_mean_abs": project.get("mean_abs"), "project_max_rel": project.get("max_rel"),
                        "v29_max_abs": v29.get("max_abs"), "v29_mean_abs": v29.get("mean_abs"), "v29_max_rel": v29.get("max_rel"),
                    })
    print(json.dumps({"cases": len(rows), "status": "pass", "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

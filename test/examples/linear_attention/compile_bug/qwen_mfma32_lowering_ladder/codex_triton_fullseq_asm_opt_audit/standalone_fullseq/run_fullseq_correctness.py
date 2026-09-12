#!/usr/bin/env python3
"""Compare real vLLM long-sequence chunk_delta_h with original/rebuilt HSACOs.

The torch calculation mirrors the documented chunk recurrence and is an
independent numerical reference. The extracted and rebuilt objects are held to
bit-exact vLLM output; the reference comparison reports observed errors rather
than silently weakening that stronger ABI gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
VLLM_COMPARE = AUDIT.parents[4] / "vllm_compare"
sys.path.insert(0, str(AUDIT)); sys.path.insert(0, str(VLLM_COMPARE))
from capture_long_sequence import BT, HK, HV, KDIM, VDIM, call, make_inputs, patch_rocm_autotune  # noqa: E402


def tensor_to_file(value: torch.Tensor, path: Path) -> None:
    if value.dtype == torch.bfloat16:
        value.view(torch.uint16).cpu().numpy().tofile(path)
    else:
        value.cpu().numpy().tofile(path)


def file_to_tensor(path: Path, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    array = np.fromfile(path, dtype=np.uint16 if dtype == torch.bfloat16 else np.float32).reshape(shape).copy()
    value = torch.from_numpy(array)
    return value.view(torch.bfloat16) if dtype == torch.bfloat16 else value


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    maximum = delta.max()
    index = (delta == maximum).nonzero()
    return {"max_abs": float(maximum), "mean_abs": float(delta.mean()), "max_rel": float((delta / expected.float().abs().clamp_min(1e-7)).max()), "first_max_index": index[0].tolist() if index.numel() else None}


def apply_mode(tensors: tuple[torch.Tensor, ...], mode: str) -> tuple[torch.Tensor, ...]:
    k, w, u, g, h0 = tensors
    if mode == "w_zero": w.zero_()
    elif mode == "zero_state": h0.zero_()
    elif mode == "unit_decay": g.zero_()
    elif mode == "high_dynamic": w.mul_(5); u.mul_(8); h0.mul_(6); g.copy_(torch.linspace(-8, 0, k.shape[1], device="cuda").view(1, -1, 1).expand_as(g))
    elif mode == "cancellation":
        w.mul_(0.2); h0.mul_(0.2)
        for head in range(HV): u[0, :, head].copy_(w[0, :, head].to(torch.bfloat16).float() @ h0[0, head].to(torch.bfloat16).float().t() + torch.randn_like(u[0, :, head]) * 1e-5)
    elif mode == "small_scale": w.mul_(1e-3); u.mul_(1e-3); h0.mul_(1e-3)
    return k, w, u, g, h0


def torch_reference(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, h0: torch.Tensor):
    """Direct BF16-input/FP32-accumulation recurrence matching the Triton source."""
    t = k.shape[1]; chunks = t // BT; state = h0.clone(); h = torch.empty((1, chunks, HV, VDIM, KDIM), device="cuda", dtype=torch.bfloat16); v_new = torch.empty_like(u)
    for chunk in range(chunks):
        start = chunk * BT; h[:, chunk].copy_(state.to(torch.bfloat16))
        for head in range(HV):
            kh = head // 2
            # The Triton source loads W as FP32 and casts the state operand
            # to W's dtype, so this pred dot consumes FP32 W and FP32 state.
            corrected = u[0, start:start + BT, head] - w[0, start:start + BT, head] @ state[0, head].t()
            v_new[0, start:start + BT, head] = corrected
            decay = torch.exp(g[0, start + BT - 1, head] - g[0, start:start + BT, head])
            update = (corrected * decay[:, None]).to(torch.bfloat16).float().t() @ k[0, start:start + BT, kh].float()
            state[0, head] = state[0, head] * torch.exp(g[0, start + BT - 1, head]) + update
    return h, v_new, state


def write_inputs(root: Path, values: tuple[torch.Tensor, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, value in zip(("k_bf16.bin", "w_fp32.bin", "v_fp32.bin", "g_fp32.bin", "h0_fp32.bin"), (values[0], values[1], values[2], values[3], values[4])):
        tensor_to_file(value, root / name)


def run_harness(harness: Path, hsaco: Path, t: int, in_dir: Path, out_dir: Path) -> tuple[torch.Tensor, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(harness), str(hsaco), str(t), str(in_dir), str(out_dir), "0", "0"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    chunks = t // BT
    return (
        file_to_tensor(out_dir / "h_bf16.bin", torch.bfloat16, (1, chunks, HV, VDIM, KDIM)),
        file_to_tensor(out_dir / "v_new_fp32.bin", torch.float32, (1, t, HV, VDIM)),
        file_to_tensor(out_dir / "ht_fp32.bin", torch.float32, (1, HV, VDIM, KDIM)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--random-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--harness", type=Path, default=HERE / "fullseq_harness")
    parser.add_argument("--original", type=Path, default=AUDIT / "golden_fullseq/shared/original.hsaco")
    parser.add_argument("--rebuilt", type=Path, default=AUDIT / "golden_fullseq/shared/rebuilt.hsaco")
    parser.add_argument("--out", type=Path, default=AUDIT)
    args = parser.parse_args()
    if any(t < BT or t % BT for t in args.T): raise ValueError("T must be divisible by 64")
    patch_rocm_autotune()
    cases = [(args.T[index % len(args.T)], f"random_{index:02d}", "random") for index in range(args.random_cases)]
    modes = ["w_zero", "zero_state", "unit_decay", "high_dynamic", "cancellation", "small_scale"]
    cases.extend((args.T[index % len(args.T)], mode, mode) for index, mode in enumerate(modes))
    rows: list[dict[str, object]] = []
    case_root = args.out / "standalone_fullseq/cases"
    for index, (t, name, mode) in enumerate(cases):
        tensors = apply_mode(make_inputs(t, args.seed + index), mode)
        expected = call(*tensors); reference = torch_reference(*tensors); torch.cuda.synchronize()
        in_dir = case_root / name / "in"; write_inputs(in_dir, tensors)
        original = run_harness(args.harness, args.original, t, in_dir, case_root / name / "original")
        rebuilt = run_harness(args.harness, args.rebuilt, t, in_dir, case_root / name / "rebuilt")
        names = ("h", "v_new", "final_state")
        row = {"case": name, "mode": mode, "T": t}
        for label, actual in (("original", original), ("rebuilt", rebuilt)):
            for output_name, got, want in zip(names, actual, expected): row[f"{label}_vs_vllm_{output_name}"] = metrics(got, want.cpu())
        for output_name, got, want in zip(names, expected, reference): row[f"vllm_vs_reference_{output_name}"] = metrics(got, want)
        rows.append(row)
    for row in rows:
        for key, value in row.items():
            if key.endswith(("_h", "_v_new", "_final_state")) and isinstance(value, dict) and key.startswith(("original", "rebuilt")):
                if value["max_abs"] != 0: raise AssertionError(f"{row['case']} {key} is not bit-exact: {value}")
    result = {"random_cases": args.random_cases, "case_count": len(rows), "rows": rows}
    (args.out / "fullseq_reference_manifest.json").write_text(json.dumps({"reference": "direct torch BF16-input FP32-accumulation chunk recurrence", "tolerance": "reported; standalone/rebuilt must be bit-exact vs vLLM"}, indent=2) + "\n")
    with (args.out / "fullseq_correctness_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["case", "mode", "T", "original_h", "original_vnew", "original_state", "rebuilt_h", "rebuilt_vnew", "rebuilt_state", "vllm_ref_h", "vllm_ref_vnew", "vllm_ref_state"])
        for row in rows:
            writer.writerow([row["case"], row["mode"], row["T"], *[row[f"original_vs_vllm_{name}"]["max_abs"] for name in ("h", "v_new", "final_state")], *[row[f"rebuilt_vs_vllm_{name}"]["max_abs"] for name in ("h", "v_new", "final_state")], *[row[f"vllm_vs_reference_{name}"]["max_abs"] for name in ("h", "v_new", "final_state")]])
    (args.out / "fullseq_correctness_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case_count": len(rows), "standalone_bit_exact": True, "reference_max_abs": {name: max(row[f"vllm_vs_reference_{name}"]["max_abs"] for row in rows) for name in names}}, indent=2))


if __name__ == "__main__":
    main()

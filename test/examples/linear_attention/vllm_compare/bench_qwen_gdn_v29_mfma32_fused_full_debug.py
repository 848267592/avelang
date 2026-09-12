#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug import (
    MODE_DECAY_OFF,
    MODE_FEEDBACK_DISABLED,
    MODE_NORMAL,
    _make_debug_inputs,
    compare_debug,
    run_debug_kernel,
    torch_debug_reference,
)


MODES = {
    "normal": MODE_NORMAL,
    "decay_off": MODE_DECAY_OFF,
    "feedback_disabled": MODE_FEEDBACK_DISABLED,
}


def run_case(t: int, mode_name: str) -> None:
    mode = MODES[mode_name]
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = _make_debug_inputs(
        t,
        decay_off=mode_name == "decay_off",
        seed=29000 + t,
    )
    actual = run_debug_kernel(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, mode=mode)
    torch.cuda.synchronize()
    ref = torch_debug_reference(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, mode=mode)
    torch.cuda.synchronize()
    for name, (max_abs, mean_abs) in compare_debug(actual, ref).items():
        print(f"T={t} mode={mode_name} tensor={name} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--mode", choices=sorted(MODES), nargs="*", default=["normal", "decay_off", "feedback_disabled"])
    args = parser.parse_args()
    for t in args.T:
        for mode_name in args.mode:
            run_case(t, mode_name)


if __name__ == "__main__":
    main()

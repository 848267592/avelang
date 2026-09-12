"""GPU pytest coverage for the opt-in BT64 gfx942 asm v0 runtime route."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
FULLSEQ_AUDIT = AUDIT.parent / "codex_triton_fullseq_asm_opt_audit"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("asm_v0_runner", HERE / "run_correctness.py")
asm = runner.ASM


@pytest.fixture(autouse=True)
def setup():
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    if not (AUDIT / "assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco").is_file():
        pytest.skip("build asm v0 first")
    runner.patch_rocm_autotune()


@pytest.mark.parametrize("t", [64, 128, 512, 2048])
def test_asm_v0_matches_vllm_original_and_rebuilt(t: int):
    row = runner.run_case(t, 20261100 + t, "random")
    runner._assert_exact(row)


def test_asm_v0_adapter_cache_guard_hash_and_current_stream(monkeypatch):
    values = runner._mutate(runner.make_inputs(512, 20261177), "random")
    expected = runner.call_vllm(*values)
    before = asm.module_load_count()
    actual = asm.qwen_gdn_bt64_gfx942_asm_v0(*values)
    torch.cuda.synchronize()
    loaded = asm.module_load_count()
    repeated = asm.qwen_gdn_bt64_gfx942_asm_v0(*values)
    torch.cuda.synchronize()
    assert loaded in (before, before + 1)
    assert asm.module_load_count() == loaded
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))
    assert all(torch.equal(got, want) for got, want in zip(repeated, expected))

    h_fp32, v_new, final_state = asm.qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0(*values)
    assert h_fp32.dtype == torch.float32
    assert torch.equal(h_fp32, actual[0].float())
    assert torch.equal(v_new, actual[1])
    assert torch.equal(final_state, actual[2])

    sentinel = (torch.empty(0, device="cuda"),) * 3
    result = asm.qwen_gdn_bt64_gfx942_asm_v0(values[0][:, :128], *values[1:], fallback=lambda: sentinel)
    assert all(got is want for got, want in zip(result, sentinel))
    monkeypatch.setenv("QWEN_GDN_BT64_GFX942_ASM_V0_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        asm.qwen_gdn_bt64_gfx942_asm_v0(*values)
    monkeypatch.delenv("QWEN_GDN_BT64_GFX942_ASM_V0_SHA256")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_actual = asm.qwen_gdn_bt64_gfx942_asm_v0(*values)
    stream.synchronize()
    assert all(torch.equal(got, want) for got, want in zip(stream_actual, expected))

    monkeypatch.setattr(asm.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx1100"))
    with pytest.raises(ValueError, match="gfx942"):
        asm.qwen_gdn_bt64_gfx942_asm_v0(*values)

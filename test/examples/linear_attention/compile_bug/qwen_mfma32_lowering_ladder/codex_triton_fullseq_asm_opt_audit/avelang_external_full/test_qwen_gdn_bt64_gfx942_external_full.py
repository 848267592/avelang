"""Integration coverage for the audit-only T%64 external HSACO bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
VLLM_COMPARE = AUDIT.parents[4] / "vllm_compare"
sys.path.insert(0, str(AUDIT)); sys.path.insert(0, str(VLLM_COMPARE))
from capture_long_sequence import call, make_inputs, patch_rocm_autotune  # noqa: E402

spec = importlib.util.spec_from_file_location("qwen_full_external", HERE / "qwen_gdn_bt64_gfx942_external_full.py")
assert spec and spec.loader
external = importlib.util.module_from_spec(spec); spec.loader.exec_module(external)


HSACO = AUDIT / "golden_fullseq/shared/original.hsaco"
BRIDGE = HERE / "libqwen_triton_external_full_bridge.so"


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not torch.cuda.is_available(): pytest.skip("requires HIP GPU")
    if not HSACO.is_file() or not BRIDGE.is_file(): pytest.skip("build full-sequence bridge and golden artifact")
    patch_rocm_autotune()
    monkeypatch.setenv("QWEN_TRITON_FULLSEQ_HSACO_PATH", str(HSACO))
    monkeypatch.setenv("QWEN_TRITON_FULLSEQ_EXTERNAL_BRIDGE", str(BRIDGE))


@pytest.mark.parametrize("t", [512, 2048])
def test_external_full_matches_vllm_and_reuses_module(t: int):
    values = make_inputs(t, 20260715 + t); expected = call(*values)
    before = external.module_load_count(); actual = external.qwen_gdn_bt64_gfx942_external_full(*values); torch.cuda.synchronize(); loaded = external.module_load_count()
    repeated = external.qwen_gdn_bt64_gfx942_external_full(*values); torch.cuda.synchronize()
    assert loaded in (before, before + 1) and external.module_load_count() == loaded
    for got, want in zip(actual, expected): assert torch.equal(got, want)
    for got, want in zip(repeated, expected): assert torch.equal(got, want)


def test_guard_fallback_hash_and_stream(monkeypatch):
    values = make_inputs(512, 20260716); sentinel = (torch.empty(0, device="cuda"),) * 3
    result = external.qwen_gdn_bt64_gfx942_external_full(values[0][:, :128], *values[1:], fallback=lambda: sentinel)
    assert all(got is want for got, want in zip(result, sentinel))
    monkeypatch.setenv("QWEN_TRITON_FULLSEQ_HSACO_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"): external.qwen_gdn_bt64_gfx942_external_full(*values)
    monkeypatch.delenv("QWEN_TRITON_FULLSEQ_HSACO_SHA256")
    expected = call(*values); stream = torch.cuda.Stream()
    with torch.cuda.stream(stream): actual = external.qwen_gdn_bt64_gfx942_external_full(*values)
    stream.synchronize()
    for got, want in zip(actual, expected): assert torch.equal(got, want)
    monkeypatch.setattr(external.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx1100"))
    with pytest.raises(ValueError, match="gfx942"): external.qwen_gdn_bt64_gfx942_external_full(*values)

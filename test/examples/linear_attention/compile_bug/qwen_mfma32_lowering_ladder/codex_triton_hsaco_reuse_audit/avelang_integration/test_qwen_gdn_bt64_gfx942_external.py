"""GPU integration checks for the audit-only fixed Triton HSACO bridge."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
VLLM_COMPARE = AUDIT.parents[4] / "vllm_compare"
sys.path.insert(0, str(VLLM_COMPARE))
spec = importlib.util.spec_from_file_location("qwen_external", HERE / "qwen_gdn_bt64_gfx942_external.py")
assert spec and spec.loader
external = importlib.util.module_from_spec(spec)
spec.loader.exec_module(external)

from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402


HSACO = AUDIT / "extracted/original_triton_kernel.hsaco"
BRIDGE = HERE / "libqwen_triton_external_bridge.so"


def _inputs():
    torch.manual_seed(20260713)
    k = (torch.randn((1, 64, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w = (torch.randn((1, 64, 8, 128), device="cuda") * 0.04).contiguous()
    u = (torch.randn((1, 64, 8, 128), device="cuda") * 0.04).contiguous()
    g = (torch.randn((1, 64, 8), device="cuda") * 0.02).contiguous()
    h0 = (torch.randn((1, 8, 128, 128), device="cuda") * 0.03).contiguous()
    return k, w, u, g, h0


@pytest.fixture(autouse=True)
def _audit_env(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    if not HSACO.is_file() or not BRIDGE.is_file():
        pytest.skip("build the external bridge and materialize the local audit HSACO first")
    monkeypatch.setenv("QWEN_TRITON_HSACO_PATH", str(HSACO))
    monkeypatch.setenv("QWEN_TRITON_EXTERNAL_BRIDGE", str(BRIDGE))


def _wrapper(k, w, u, g, h0):
    return chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, gk=None, initial_state=h0,
        output_final_state=True, chunk_size=64, save_new_value=True, cu_seqlens=None,
    )


def test_external_fixed_specialization_matches_vllm_and_caches_module():
    k, w, u, g, h0 = _inputs()
    expected = _wrapper(k, w, u, g, h0)
    before = external.module_load_count()
    actual = external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0)
    torch.cuda.synchronize()
    after_first = external.module_load_count()
    repeated = external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0)
    torch.cuda.synchronize()
    assert after_first in (before, before + 1)
    assert external.module_load_count() == after_first
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
    for got, want in zip(repeated, expected):
        assert torch.equal(got, want)


def test_guard_and_missing_artifact_use_explicit_fallback(monkeypatch):
    k, w, u, g, h0 = _inputs()
    sentinel = (torch.empty(0, device="cuda"),) * 3
    result = external.qwen_gdn_bt64_gfx942_external(k[:, :32], w, u, g, h0, fallback=lambda: sentinel)
    assert all(got is want for got, want in zip(result, sentinel))
    monkeypatch.setenv("QWEN_TRITON_HSACO_PATH", str(HERE / "does-not-exist.hsaco"))
    result = external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0, fallback=lambda: sentinel)
    assert all(got is want for got, want in zip(result, sentinel))


def test_hash_mismatch_is_rejected_without_fallback(monkeypatch):
    k, w, u, g, h0 = _inputs()
    monkeypatch.setenv("QWEN_TRITON_HSACO_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0)


def test_wrong_gpu_is_rejected(monkeypatch):
    k, w, u, g, h0 = _inputs()
    monkeypatch.setattr(external.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx1100"))
    with pytest.raises(ValueError, match="gfx942"):
        external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0)


def test_current_nondefault_stream_is_forwarded():
    k, w, u, g, h0 = _inputs()
    expected = _wrapper(k, w, u, g, h0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = external.qwen_gdn_bt64_gfx942_external(k, w, u, g, h0)
    stream.synchronize()
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)

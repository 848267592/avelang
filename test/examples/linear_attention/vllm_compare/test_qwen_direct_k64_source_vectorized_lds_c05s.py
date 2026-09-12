"""Regression coverage for the C0.5S source-only LDS-store control."""

from __future__ import annotations

import pytest

import repro_qwen_direct_k64_source_vectorized_lds_c05s as repro


@pytest.mark.parametrize("tokens", [64, 512, 2048])
def test_source_scalar_and_packed_lds_staging_are_bit_exact(tokens: int) -> None:
    rows = repro.run_case(tokens, seed=20260903 + tokens, warmup=0, repeat=1)
    scalar = rows["scalar"]
    packed = rows["packed"]
    print("source_vectorized_lds_c05s", tokens, rows)
    assert scalar["finite"]
    assert packed["finite"]
    assert scalar["input_equal"]
    assert packed["input_equal"]
    assert scalar["cross_arm_equal"]
    assert scalar["sha256"] == packed["sha256"]

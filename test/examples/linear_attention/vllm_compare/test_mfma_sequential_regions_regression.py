#!/usr/bin/env python3
"""Regression coverage for sequential BF16 MFMA regions in one kernel.

The bug this covers is not Qwen-specific: a first MFMA region that computes a
pred tile can corrupt a later MFMA region that computes

    init * scale + v_decay.float().T @ k_chunk.float()

The update-only and no-op-pred cases are control cases.  The
pred_one_mfma_then_update case failed on MI300 before the backend fix.
"""

import unittest

import torch

from avelang.testing import has_rocm
from prototype_qwen_gdn_mfma_delta_staged import (
    BT,
    BV,
    delta_state_isolation_noop_no_mfma,
    delta_state_isolation_one_mfma,
    delta_state_staged,
)


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


@unittest.skipUnless(has_rocm(), "Requires ROCm/HIP with an AMD GPU.")
class TestMFMASequentialRegionsRegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260616)
        self.v_decay = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16)
        self.k_chunk = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16)
        self.init = (
            torch.randn((BV, 128), device="cuda", dtype=torch.float32) * 0.01
        )
        self.scale = 0.73
        self.expected = (
            self.init * self.scale
            + self.v_decay.float().T @ self.k_chunk.float()
        )

    def assertMatchesExpected(self, name: str, actual: torch.Tensor):
        torch.cuda.synchronize()
        max_abs = _max_abs(actual, self.expected)
        self.assertTrue(
            torch.allclose(actual.float(), self.expected.float(), atol=1e-4, rtol=1e-4),
            msg=f"{name} max_abs={max_abs}",
        )

    def test_update_only_baseline(self):
        actual = delta_state_staged(
            self.v_decay, self.k_chunk, self.init, self.scale
        )
        self.assertMatchesExpected("update_only_baseline", actual)

    def test_no_op_pred_no_mfma(self):
        actual = delta_state_isolation_noop_no_mfma(
            self.v_decay, self.k_chunk, self.init, self.scale
        )
        self.assertMatchesExpected("no_op_pred_no_mfma", actual)

    def test_pred_one_mfma_then_update(self):
        actual = delta_state_isolation_one_mfma(
            self.v_decay, self.k_chunk, self.init, self.scale
        )
        self.assertMatchesExpected("pred_one_mfma_then_update", actual)


if __name__ == "__main__":
    unittest.main()

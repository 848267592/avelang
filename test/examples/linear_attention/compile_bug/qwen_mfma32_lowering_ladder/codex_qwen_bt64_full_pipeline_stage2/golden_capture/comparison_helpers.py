"""Comparison helper re-exported for external audit/notebook use."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_runner import metrics

__all__ = ["metrics"]

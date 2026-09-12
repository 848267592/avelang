"""Stable entry point for the Stage 2 vLLM golden capture."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_runner import main


if __name__ == "__main__":
    main()

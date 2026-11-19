#!/usr/bin/env python
"""Thin launcher: `python scripts/run_inference.py ...` without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_vision.inference.predict import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

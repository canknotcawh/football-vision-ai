"""Single source of truth for project directories.

Every script imports paths from here instead of hardcoding relative strings,
so the project works no matter which directory a command runs from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_WEIGHTS = PRETRAINED_DIR / "yolo11n.pt"
DEFAULT_SAMPLE_IMAGE = SAMPLES_DIR / "bus.jpg"


def resolve(path: str | Path) -> Path:
    """Resolve a user path against the project root (absolute paths pass through)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)

#!/usr/bin/env python
"""Record the exact environment (Python, GPU, CUDA, library versions) to reports/.

Writes reports/environment.md (human readable) and reports/requirements.lock.txt
(full freeze, reproducible reference).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_vision.paths import PROJECT_ROOT, REPORTS_DIR  # noqa: E402

KEY_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "ultralytics-thop",
    "numpy",
    "opencv-python",
    "pillow",
    "matplotlib",
    "pandas",
    "scipy",
    "scikit-learn",
    "polars",
    "psutil",
    "jupyterlab",
    "pytest",
]


def version_of(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "—"


def nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip().replace(", ", " | ")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({exc})"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import torch

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "—"
    gpu_cap = torch.cuda.get_device_capability(0) if cuda_available else None

    frozen = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    )
    (REPORTS_DIR / "requirements.lock.txt").write_text("\n".join(frozen) + "\n", encoding="utf-8")

    rows = "\n".join(f"| `{name}` | {version_of(name)} |" for name in KEY_PACKAGES)
    lines = f"""# Environment Report — Phase 0

Generated: {datetime.now().isoformat(timespec="seconds")}

## System

| Item | Value |
| --- | --- |
| OS | {platform.platform()} |
| Kernel | {platform.release()} |
| CPU | {platform.processor() or "unknown"} |
| Python | {platform.python_version()} ({sys.executable}) |
| Project root | `{PROJECT_ROOT}` |

## GPU

| Item | Value |
| --- | --- |
| GPU | {gpu_name} |
| nvidia-smi | {nvidia_smi("name,memory.total,driver_version")} |
| Compute capability | {".".join(map(str, gpu_cap)) if gpu_cap else "—"} |
| CUDA available (torch) | {cuda_available} |
| CUDA runtime in torch | {torch.version.cuda or "—"} |
| cuDNN | {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "—"} |

## Key packages

| Package | Version |
| --- | --- |
{rows}

Full freeze ({len(frozen)} distributions): [`reports/requirements.lock.txt`](requirements.lock.txt)

## Reproduce

```bash
uv venv --python 3.12
uv sync                      # uses the pytorch-cu128 index declared in pyproject.toml
uv run python scripts/capture_env.py
```
"""
    (REPORTS_DIR / "environment.md").write_text(lines, encoding="utf-8")
    print(f"wrote {REPORTS_DIR / 'environment.md'}")
    print(f"wrote {REPORTS_DIR / 'requirements.lock.txt'} ({len(frozen)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

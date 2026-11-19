# Environment Report — Phase 0

Generated: 2026-11-19T23:21:34

## System

| Item | Value |
| --- | --- |
| OS | Linux-7.0.0-43-generic-x86_64-with-glibc2.43 |
| Kernel | 7.0.0-43-generic |
| CPU | unknown |
| Python | 3.12.16 (/home/cank/Desktop/football-vision-ai/.venv/bin/python) |
| Project root | `/home/cank/Desktop/football-vision-ai` |

## GPU

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| nvidia-smi | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 6144 MiB | 595.99.02 |
| Compute capability | 8.6 |
| CUDA available (torch) | True |
| CUDA runtime in torch | 12.8 |
| cuDNN | 91900 |

## Key packages

| Package | Version |
| --- | --- |
| `torch` | 2.11.0+cu128 |
| `torchvision` | 0.26.0+cu128 |
| `torchaudio` | 2.11.0+cu128 |
| `ultralytics` | 8.4.215 |
| `ultralytics-thop` | 2.2.5 |
| `numpy` | 2.5.5 |
| `opencv-python` | 5.1.0.95 |
| `pillow` | 13.0.0 |
| `matplotlib` | 3.11.2 |
| `pandas` | 3.1.1 |
| `scipy` | 1.18.1 |
| `scikit-learn` | 1.9.1 |
| `polars` | 2.1.0 |
| `psutil` | 7.2.2 |
| `jupyterlab` | 4.6.6 |
| `pytest` | 9.2.0 |

Full freeze (151 distributions): [`reports/requirements.lock.txt`](requirements.lock.txt)

## Reproduce

```bash
uv venv --python 3.12
uv sync                      # uses the pytorch-cu128 index declared in pyproject.toml
uv run python scripts/capture_env.py
```

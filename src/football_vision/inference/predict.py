"""Run YOLO11 inference with Ultralytics and report what the model detected.

Usage:
    uv run fva-infer --source data/samples/bus.jpg
    uv run fva-infer --source clip.mp4 --device 0 --conf 0.3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

from football_vision.paths import DEFAULT_SAMPLE_IMAGE, DEFAULT_WEIGHTS, RUNS_DIR, resolve


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 inference on the project GPU")
    parser.add_argument("--model", default=str(DEFAULT_WEIGHTS), help="path to .pt weights")
    parser.add_argument("--source", default=str(DEFAULT_SAMPLE_IMAGE), help="image, video or directory")
    parser.add_argument("--device", default="auto", help='"0" for GPU 0, "cpu", or "auto"')
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="phase0_inference", help="run name under runs/detect")
    return parser.parse_args(argv)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "0" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = resolve(args.model)
    source = resolve(args.source)
    if not model_path.is_file():
        raise SystemExit(f"weights not found: {model_path}")
    if not source.exists():
        raise SystemExit(f"source not found: {source}")

    device = pick_device(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu to run on CPU")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(RUNS_DIR / "detect"),
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    counts: Counter[str] = Counter()
    for result in results:
        for cls_id in result.boxes.cls.tolist():
            counts[result.names[int(cls_id)]] += 1

    print("\n=== Summary ===")
    print(f"device      : {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")
    print(f"model       : {model_path.name}")
    print(f"source      : {source}")
    print(f"images      : {len(results)}")
    print(f"detections  : {sum(counts.values())}")
    for label, count in counts.most_common():
        print(f"  - {label:<15} {count}")
    speed = results[0].speed if results else {}
    if speed:
        print(f"speed (ms)  : preprocess {speed.get('preprocess', 0):.1f} | "
              f"inference {speed.get('inference', 0):.1f} | postprocess {speed.get('postprocess', 0):.1f}")
    print(f"output      : {Path(results[0].save_dir) if results else RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

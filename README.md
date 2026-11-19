# Football Vision AI

Phân tích video bóng đá bằng **YOLO11** (Ultralytics), huấn luyện và suy luận trên GPU
**NVIDIA RTX 3050 6GB** (CUDA 12.8).

Trạng thái: **Phase 0 — Environment & Project Foundation: hoàn thành.**

## Kết quả Phase 0

| Hạng mục | Kết quả |
| --- | --- |
| Python | 3.12.16 (uv, `.venv`) |
| PyTorch | 2.11.0+cu128, `torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver 595.99.02, compute capability 8.6 |
| Ultralytics | 8.4.215 |
| Weights | `models/pretrained/yolo11n.pt` |
| Inference thực tế | `data/samples/bus.jpg` → **4 person, 1 bus**, inference **4.2 ms/ảnh** trên GPU |
| Kết quả lưu tại | `runs/detect/phase0_inference/bus.jpg` |
| Báo cáo môi trường | `reports/environment.md`, freeze đầy đủ `reports/requirements.lock.txt` |

## Cấu trúc dự án

```text
football-vision-ai/
├── pyproject.toml            # Metadata, dependencies, index PyTorch CUDA 12.8
├── configs/                  # Cấu hình YOLO / huấn luyện (yaml)
├── data/
│   ├── raw/                  # Dữ liệu gốc tải về (không commit)
│   ├── processed/            # Dữ liệu đã xử lý
│   ├── samples/              # Ảnh/video mẫu để test nhanh (bus.jpg)
│   └── splits/               # File chia train/val/test
├── models/
│   ├── pretrained/           # Weights gốc (yolo11n.pt)
│   └── checkpoints/          # Weights sau huấn luyện
├── notebooks/                # Jupyter notebook thử nghiệm
├── reports/                  # Báo cáo môi trường, kết quả đo
├── runs/                     # Output Ultralytics (detect/train/val)
├── scripts/                  # Entry point CLI (run_inference.py, capture_env.py)
├── src/football_vision/      # Mã nguồn chính
│   ├── paths.py              # Đường dẫn chuẩn dùng chung
│   └── inference/predict.py  # CLI inference
└── tests/                    # Unit test
```

`runs/`, weights và dữ liệu thô nằm trong `.gitignore`; các thư mục rỗng được giữ bằng `.gitkeep`.

## Cài đặt

```bash
uv venv --python 3.12
uv sync          # torch/torchvision lấy từ index pytorch-cu128 trong pyproject.toml
```

Kiểm tra GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Chạy inference

```bash
# Ảnh mẫu trên GPU 0
uv run fva-infer --source data/samples/bus.jpg --device 0

# Không cài package (PYTHONPATH tự thêm)
.venv/bin/python scripts/run_inference.py --source data/samples/bus.jpg --device 0

# Video / thư mục ảnh, đổi ngưỡng tin cậy
.venv/bin/python scripts/run_inference.py --source data/raw/match.mp4 --conf 0.3
```

Tuỳ chọn: `--model`, `--source`, `--device` (`0`, `cpu`, `auto`), `--imgsz`, `--conf`, `--name`.
Kết quả (ảnh/video đã vẽ box) ghi vào `runs/detect/<name>/`.

> Weights ngoài project: thay `--model` bằng đường dẫn `.pt` bất kỳ; Ultralytics sẽ tự tải
> `yolo11n.pt` nếu chưa có.

## Ghi lại môi trường

```bash
uv run python scripts/capture_env.py
```

Sinh lại `reports/environment.md` (phiên bản Python, GPU, CUDA, thư viện chính) và
`reports/requirements.lock.txt` (151 distribution).

## Bước tiếp theo (Phase 1)

- Chuẩn bị dataset bóng đá (person/ball/referee) và cấu hình `data/*.yaml`.
- Baseline detect + tracking (ByteTrack) trên video trận đấu.
- Đo FPS end-to-end trên RTX 3050 và tối ưu (imgsz, half precision, batch).

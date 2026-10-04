# Fine-tunes YOLOv8-nano on your Roboflow dataset.
# Uses the Roboflow export (format: "YOLOv8") in
# laptop/dataset/data.yaml. Weights end up in
# runs/detect/green_minifig/weights/best.pt
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
if not DATA_YAML.exists():
    raise SystemExit(f"Missing {DATA_YAML} - put the Roboflow export in laptop/dataset/")

# data.yaml uses paths relative to the dataset folder; give Ultralytics the
# absolute folder so this works wherever the repo is cloned.
data = yaml.safe_load(DATA_YAML.read_text())
data["path"] = str(DATA_YAML.parent.resolve())
RUN_YAML = Path(__file__).parent / "runs" / "data.yaml"
RUN_YAML.parent.mkdir(parents=True, exist_ok=True)
RUN_YAML.write_text(yaml.safe_dump(data))

if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Training on {device!r}")

model = YOLO("yolov8n.pt")  # COCO-pretrained, fine-tune on ~40-50 images
model.train(
    data=str(RUN_YAML),
    epochs=100,       # small dataset -> more epochs
    imgsz=640,
    batch=8,
    patience=30,      # stop early if val mAP stops improving
    project=str(Path(__file__).parent / "runs" / "detect"),  # keep output next to this script
    name="green_minifig",
    exist_ok=True,
    device=device,
)
print("Done -> laptop/runs/detect/green_minifig/weights/best.pt")

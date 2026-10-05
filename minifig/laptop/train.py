# Fine-tunes YOLOv8-nano on one minifig's Roboflow dataset.
#   python train.py --color green
#   python train.py --color blue
# Uses laptop/datasets/<color>/data.yaml (Roboflow "YOLOv8" export).
# Weights end up in runs/detect/<color>_minifig/weights/best.pt
import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).parent


# The __main__ guard is required on Windows: the GPU data loader starts worker
# processes that re-import this file, and without it each worker starts training again.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", required=True, help="dataset folder name under datasets/, e.g. green or blue")
    parser.add_argument("--epochs", type=int, default=100)  # small dataset -> more epochs
    args = parser.parse_args()

    DATA_YAML = HERE / "datasets" / args.color / "data.yaml"
    if not DATA_YAML.exists():
        raise SystemExit(f"Missing {DATA_YAML} - put the Roboflow export in laptop/datasets/{args.color}/")

    # data.yaml uses paths relative to the dataset folder; give Ultralytics the
    # absolute folder so this works wherever the repo is cloned.
    data = yaml.safe_load(DATA_YAML.read_text())
    data["path"] = str(DATA_YAML.parent.resolve())
    RUN_YAML = HERE / "runs" / f"{args.color}_data.yaml"
    RUN_YAML.parent.mkdir(parents=True, exist_ok=True)
    RUN_YAML.write_text(yaml.safe_dump(data))

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training {args.color} on {device!r}")

    model = YOLO("yolov8n.pt")  # COCO-pretrained, fine-tune on ~40-50 images
    model.train(
        data=str(RUN_YAML),
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        patience=30,      # stop early if val mAP stops improving
        project=str(HERE / "runs" / "detect"),  # keep output next to this script
        name=f"{args.color}_minifig",
        exist_ok=True,
        device=device,
    )
    print(f"Done -> laptop/runs/detect/{args.color}_minifig/weights/best.pt")


if __name__ == "__main__":
    main()

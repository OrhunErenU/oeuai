"""Asker/Sivil siniflandirici egitim scripti.

Kullanim:
    python scripts/train_human_cls.py --data datasets/human_crops

Veri seti yapisi (ImageFolder):
    human_crops/
      train/
        soldier/ civilian/
      val/
        soldier/ civilian/
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO


def _resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def train(
    data: str = "datasets/human_crops",
    model_size: str = "yolo11s-cls.pt",
    epochs: int = 80,
    imgsz: int = 224,
    batch: int = 64,
    device: str = "0",
    project: str = "runs/classify",
    name: str = "human_cls_v1",
):
    print("=" * 60)
    print("Tank Vision AI - Asker/Sivil Siniflandirici Egitimi")
    print(f"  Siniflar: soldier, civilian")
    print(f"  Veri: {data}")
    print("=" * 60)

    data = _resolve_path(data)
    model = YOLO(model_size)

    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=15,
        save=True,
        plots=True,
        project=project,
        name=name,
        exist_ok=False,
        verbose=True,
    )

    best_path = f"{project}/{name}/weights/best.pt"
    print(f"\n[+] Egitim tamamlandi! -> {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/human_crops")
    parser.add_argument("--model", default="yolo11s-cls.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="human_cls_v1")
    args = parser.parse_args()

    train(
        data=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

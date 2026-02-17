"""Silah turu siniflandirici egitim scripti.

Kullanim:
    python scripts/train_weapon_cls.py --data datasets/weapon_crops

Veri seti yapisi (ImageFolder):
    weapon_crops/
      train/
        rpg/ rifle/ pistol/ sniper/ grenade/ machine_gun/
      val/
        rpg/ rifle/ pistol/ sniper/ grenade/ machine_gun/
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
    data: str = "datasets/weapon_crops",
    model_size: str = "yolo11m-cls.pt",
    epochs: int = 100,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "0",
    project: str = "runs/classify",
    name: str = "weapon_cls_v1",
):
    print("=" * 60)
    print("Tank Vision AI - Silah Siniflandirici Egitimi")
    print(f"  Siniflar: rpg, rifle, pistol, sniper, grenade, machine_gun")
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
        patience=20,
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
    parser.add_argument("--data", default="datasets/weapon_crops")
    parser.add_argument("--model", default="yolo11m-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="weapon_cls_v1")
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

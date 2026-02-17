"""Tank marka/model siniflandirici egitim scripti.

Kullanim:
    python scripts/train_tank_cls.py --data datasets/tank_crops

Veri seti yapisi (ImageFolder):
    tank_crops/
      train/
        m1_abrams/ leopard_2/ t72/ t90/ challenger_2/ merkava_4/
        altay/ type_99/ k2_black_panther/ unknown/
      val/
        (ayni klasorler)
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
    data: str = "datasets/tank_crops",
    model_size: str = "yolo11m-cls.pt",
    epochs: int = 100,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "0",
    project: str = "runs/classify",
    name: str = "tank_cls_v1",
):
    print("=" * 60)
    print("Tank Vision AI - Tank Model Siniflandirici Egitimi")
    print(f"  Siniflar: M1 Abrams, Leopard 2, T-72, T-90, Challenger 2,")
    print(f"            Merkava 4, Altay, Type 99, K2 Black Panther, Unknown")
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
    parser.add_argument("--data", default="datasets/tank_crops")
    parser.add_argument("--model", default="yolo11m-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="tank_cls_v1")
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

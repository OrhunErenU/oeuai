"""Dost/Dusman (IFF) siniflandirici egitim scripti.

Kullanim:
    python scripts/train_foe_cls.py --data datasets/foe_crops

Veri seti yapisi (ImageFolder):
    foe_crops/
      train/
        friend/ foe/ unknown/
      val/
        friend/ foe/ unknown/
"""

import argparse

from ultralytics import YOLO


def train(
    data: str = "datasets/foe_crops",
    model_size: str = "yolo11m-cls.pt",
    epochs: int = 100,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "0",
    project: str = "runs/classify",
    name: str = "foe_cls_v1",
):
    print("=" * 60)
    print("Tank Vision AI - Dost/Dusman Siniflandirici Egitimi")
    print(f"  Siniflar: friend, foe, unknown")
    print(f"  Veri: {data}")
    print("=" * 60)

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
    parser.add_argument("--data", default="datasets/foe_crops")
    parser.add_argument("--model", default="yolo11m-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="foe_cls_v1")
    args = parser.parse_args()

    train(**vars(args))

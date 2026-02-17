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

from ultralytics import YOLO


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

    train(**vars(args))

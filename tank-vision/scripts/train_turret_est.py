"""Taret yon tahmincisi egitim scripti.

Taret bolgesi kirpilmis tank goruntulerinden taret yonunu
siniflandirir. 4 sinif: front (bize dogru), back (bizden uzak),
left (sol), right (sag).

Veri seti yapisi (ImageFolder):
    turret_crops/
      train/
        front/ back/ left/ right/
      val/
        front/ back/ left/ right/

front = namlu kameraya dogru (TEHLIKE)
back = namlu arka tarafa
left/right = namlu yandan gorunuyor

Kullanim:
    python scripts/train_turret_est.py --data datasets/turret_crops
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO


TURRET_CLASSES = {
    0: "front",   # Namlu bize dogru
    1: "back",    # Namlu arkaya donuk
    2: "left",    # Namlu sol tarafa
    3: "right",   # Namlu sag tarafa
}


def _resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def train(
    data: str = "datasets/turret_crops",
    model_size: str = "yolo11m-cls.pt",
    epochs: int = 150,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "0",
    patience: int = 25,
    project: str = "runs/classify",
    name: str = "turret_est_v1",
):
    """Taret yon tahmincisini egit.

    Args:
        data: Turret kirpimlerinin bulundugu ImageFolder dizini.
        model_size: Onceden egitilmis siniflandirma modeli.
        epochs: Egitim epoch sayisi.
        imgsz: Girdi boyutu.
        batch: Batch boyutu.
        device: GPU cihaz ID'si.
        patience: Early stopping sabirlilik.
        project: Cikti proje dizini.
        name: Calisma adi.
    """
    print("=" * 60)
    print("Tank Vision AI - Taret Yon Tahmincisi Egitimi")
    print("=" * 60)
    print(f"  Siniflar: front (bize dogru), back, left, right")
    print(f"  Model:   {model_size}")
    print(f"  Veri:    {data}")
    print(f"  Epochs:  {epochs}")
    print(f"  ImgSz:   {imgsz}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {device}")
    print("=" * 60)

    data = _resolve_path(data)
    model = YOLO(model_size)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10,
        plots=True,
        # Augmentasyon - taret icin ozel ayarlar
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15.0,        # Rotasyon (taret yonu icin sinirli)
        translate=0.1,
        scale=0.3,
        fliplr=0.0,          # Yatay flip KAPALI (yon bilgisini bozar)
        flipud=0.0,          # Dikey flip KAPALI
        erasing=0.1,
        # Diger
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5,
        workers=4,
        project=project,
        name=name,
        exist_ok=False,
        verbose=True,
        seed=42,
    )

    best_path = f"{project}/{name}/weights/best.pt"
    print(f"\n[+] Egitim tamamlandi!")
    print(f"    En iyi agirliklar: {best_path}")
    print(f"\n    Kullanim: config/default.yaml icinde")
    print(f"    turret_estimator: {best_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taret yon tahmincisi egitimi")
    parser.add_argument("--data", default="datasets/turret_crops")
    parser.add_argument("--model", default="yolo11m-cls.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="turret_est_v1")
    args = parser.parse_args()

    train(
        data=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

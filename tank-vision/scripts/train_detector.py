"""Ana YOLOv11 7-sinif detektor egitim scripti.

Kullanim:
    python scripts/train_detector.py --data config/dataset.yaml --epochs 200
"""

import argparse
import sys
from pathlib import Path

# Proje kokunu sys.path'e ekle (import sorunlarini onler)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO


def _resolve_data_path(data: str) -> str:
    """Veri seti YAML yolunu mutlak yola cevir."""
    p = Path(data)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p.resolve())


def train(
    data: str = "config/dataset.yaml",
    model_size: str = "yolo11m.pt",
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    patience: int = 30,
    project: str = "runs/detect",
    name: str = "tank_vision_v1",
):
    """Ana dedektoru egit.

    Args:
        data: Veri seti YAML dosyasi.
        model_size: Onceden egitilmis model (yolo11n/s/m/l/x.pt).
        epochs: Egitim epoch sayisi.
        imgsz: Girdi goruntu boyutu.
        batch: Batch boyutu (GPU bellegine gore ayarla).
        device: GPU cihaz ID'si.
        patience: Early stopping sabirlilik (epoch).
        project: Cikti proje dizini.
        name: Calisma adi.
    """
    print("=" * 60)
    print("Tank Vision AI - Ana Detektor Egitimi")
    print("=" * 60)
    print(f"  Model:   {model_size}")
    print(f"  Veri:    {data}")
    print(f"  Epochs:  {epochs}")
    print(f"  ImgSz:   {imgsz}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {device}")
    print("=" * 60)

    data = _resolve_data_path(data)

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
        val=True,
        plots=True,
        # Augmentasyon
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Diger
        rect=False,
        close_mosaic=15,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        momentum=0.937,
        workers=8,
        cache=False,
        project=project,
        name=name,
        exist_ok=False,
        verbose=True,
        seed=42,
    )

    # En iyi agirliklari kopyala
    best_path = f"{project}/{name}/weights/best.pt"
    print(f"\n[+] Egitim tamamlandi!")
    print(f"    En iyi agirleklar: {best_path}")

    # Dogrulama
    print("\n[*] Dogrulama metrikleri:")
    model_best = YOLO(best_path)
    metrics = model_best.val(data=data, device=device)
    print(f"    mAP50:    {metrics.box.map50:.4f}")
    print(f"    mAP50-95: {metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ana detektor egitimi")
    parser.add_argument("--data", default="config/dataset.yaml")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="tank_vision_v1")
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

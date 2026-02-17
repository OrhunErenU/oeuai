"""Model degerlendirme scripti.

Tum modellerin dogrulama metrikleri: mAP, precision, recall, confusion matrix.

Kullanim:
    python scripts/evaluate.py --model runs/detect/tank_vision_v1/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.logger import setup_logger

logger = setup_logger("evaluate")


def evaluate_detector(
    model_path: str,
    data_yaml: str = "config/dataset.yaml",
    device: str = "0",
    imgsz: int = 640,
):
    """Ana dedektoru degerlendir."""
    print("=" * 60)
    print("DETEKTOR DEGERLENDIRME")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, imgsz=imgsz)

    print(f"\n  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    # Sinif bazinda metrikler
    if hasattr(metrics.box, "ap_class_index"):
        print("\nSinif Bazinda AP50:")
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names.get(cls_idx, str(cls_idx))
            ap50 = metrics.box.ap50[i]
            print(f"  {cls_name:12s}: {ap50:.4f}")

    return metrics


def evaluate_classifier(
    model_path: str,
    data_dir: str,
    device: str = "0",
):
    """Siniflandirici modeli degerlendir."""
    if not Path(model_path).exists():
        print(f"[!] Model bulunamadi: {model_path}")
        return None

    model = YOLO(model_path)
    metrics = model.val(data=data_dir, device=device)

    print(f"\n  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model degerlendirme")
    parser.add_argument("--model", required=True, help="Model .pt dosyasi")
    parser.add_argument("--data", default="config/dataset.yaml", help="Veri YAML veya dizin")
    parser.add_argument("--device", default="0")
    parser.add_argument("--type", choices=["detect", "classify"], default="detect")
    args = parser.parse_args()

    if args.type == "detect":
        evaluate_detector(args.model, args.data, args.device)
    else:
        evaluate_classifier(args.model, args.data, args.device)

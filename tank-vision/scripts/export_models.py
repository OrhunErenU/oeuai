"""Model export araci.

Egitilmis modelleri TensorRT (FP16), ONNX ve diger formatlara export eder.
RTX GPU uzerinde 2-4x hizlanma saglar.

Kullanim:
    python scripts/export_models.py --format engine
    python scripts/export_models.py --format onnx
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics import YOLO

from utils.logger import setup_logger

logger = setup_logger("export")

# Varsayilan model yollari
DEFAULT_MODELS = {
    "detector": "runs/detect/tank_vision_v1/weights/best.pt",
    "weapon_cls": "runs/classify/weapon_cls_v1/weights/best.pt",
    "tank_cls": "runs/classify/tank_cls_v1/weights/best.pt",
    "human_cls": "runs/classify/human_cls_v1/weights/best.pt",
    "foe_cls": "runs/classify/foe_cls_v1/weights/best.pt",
}


def export_model(
    model_path: str,
    export_format: str = "engine",
    half: bool = True,
    imgsz: int = 640,
    device: int = 0,
) -> str | None:
    """Tek bir modeli export et.

    Args:
        model_path: .pt agirliklari dosya yolu.
        export_format: Cikti formati (engine, onnx, torchscript, openvino).
        half: FP16 kullan (TensorRT icin onemli).
        imgsz: Girdi boyutu.
        device: GPU cihaz ID.

    Returns:
        Export edilen dosya yolu veya None.
    """
    if not Path(model_path).exists():
        logger.warning(f"Model bulunamadi: {model_path}")
        return None

    logger.info(f"Export: {model_path} -> {export_format}")

    model = YOLO(model_path)
    result = model.export(
        format=export_format,
        half=half,
        imgsz=imgsz,
        device=device,
    )

    logger.info(f"  Tamamlandi: {result}")
    return str(result)


def export_all(
    export_format: str = "engine",
    half: bool = True,
    device: int = 0,
    models: dict | None = None,
):
    """Tum modelleri export et.

    Args:
        export_format: Cikti formati.
        half: FP16.
        device: GPU cihaz ID.
        models: Model yollari dict'i (None ise varsayilanlari kullan).
    """
    if models is None:
        models = DEFAULT_MODELS

    print("=" * 60)
    print(f"Tank Vision AI - Model Export ({export_format.upper()})")
    print("=" * 60)

    results = {}
    for name, path in models.items():
        # Siniflandiricilar icin daha kucuk imgsz
        imgsz = 224 if "cls" in name else 640
        result = export_model(path, export_format, half, imgsz, device)
        results[name] = result

    print("\n" + "=" * 60)
    print("EXPORT SONUCLARI")
    print("=" * 60)
    for name, result in results.items():
        status = result if result else "ATLANMIS"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model export")
    parser.add_argument(
        "--format",
        default="engine",
        choices=["engine", "onnx", "torchscript", "openvino"],
        help="Export formati",
    )
    parser.add_argument("--no-half", action="store_true", help="FP16 kullanma")
    parser.add_argument("--device", type=int, default=0, help="GPU cihaz ID")
    parser.add_argument("--model", default=None, help="Tek model export (yol)")
    args = parser.parse_args()

    if args.model:
        export_model(args.model, args.format, not args.no_half, device=args.device)
    else:
        export_all(args.format, not args.no_half, args.device)

#!/usr/bin/env python3
"""
Tank Vision AI — Model Export Script
Eğitilmiş YOLOv11m modelini ONNX ve TensorRT formatlarına export eder.
C++ inference pipeline için gerekli.

Kullanım:
  python scripts/export_model.py                    # Varsayılan v3 best.pt
  python scripts/export_model.py --weights best.pt  # Özel model
  python scripts/export_model.py --format engine    # Direkt TensorRT
"""

import argparse
import cv2

# OpenCV imshow fix
_native = cv2.imshow
from ultralytics import YOLO
cv2.imshow = _native

def main():
    parser = argparse.ArgumentParser(description="Tank Vision AI - Model Export")
    parser.add_argument("--weights", type=str,
                        default="C:/tv_data/v3/runs/tank_vision_v3m_r5/weights/best.pt",
                        help="Model weights path")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "engine", "both"],
                        help="Export format")
    parser.add_argument("--half", action="store_true", default=True,
                        help="FP16 half precision (default: True)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    args = parser.parse_args()

    print("=" * 50)
    print("  TANK VISION AI — MODEL EXPORT")
    print("=" * 50)

    model = YOLO(args.weights)
    print(f"Model loaded: {args.weights}")

    if args.format in ["onnx", "both"]:
        print("\n[1] Exporting to ONNX...")
        onnx_path = model.export(
            format="onnx",
            simplify=True,
            opset=17,
            imgsz=args.imgsz,
            half=False,  # ONNX doesn't support FP16 directly
        )
        print(f"  ONNX saved: {onnx_path}")

    if args.format in ["engine", "both"]:
        print("\n[2] Exporting to TensorRT Engine...")
        engine_path = model.export(
            format="engine",
            half=args.half,
            device=0,
            imgsz=args.imgsz,
            workspace=4,  # 4GB workspace
        )
        print(f"  TensorRT Engine saved: {engine_path}")

    print("\n" + "=" * 50)
    print("  EXPORT COMPLETE!")
    print("=" * 50)

    if args.format in ["onnx", "both"]:
        print(f"\nC++ ile kullanmak için:")
        print(f"  tank_vision --onnx {onnx_path} --source 0")
    if args.format in ["engine", "both"]:
        print(f"  tank_vision --engine {engine_path} --source 0")

if __name__ == "__main__":
    main()

"""Pipeline performans benchmark araci.

Farkli bilesenlerin ve tam pipeline'in FPS ve gecikme
olcumlerini yapar.

Kullanim:
    python scripts/benchmark.py --config config/default.yaml
"""

import argparse
import time

import cv2
import numpy as np
import torch

from utils.logger import setup_logger

logger = setup_logger("benchmark")


def benchmark_detection_only(
    model_path: str,
    device: str = "0",
    imgsz: int = 640,
    num_frames: int = 200,
    warmup: int = 20,
):
    """Sadece dedeksiyon FPS'ini olc."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model.predict(source=dummy, device=device, verbose=False)

    # Benchmark
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_frames):
        model.predict(source=dummy, device=device, verbose=False)

    if device != "cpu":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    fps = num_frames / elapsed
    ms_per_frame = elapsed / num_frames * 1000

    return fps, ms_per_frame


def benchmark_tracking(
    model_path: str,
    device: str = "0",
    imgsz: int = 640,
    num_frames: int = 200,
    warmup: int = 20,
):
    """Dedeksiyon + takip FPS'ini olc."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model.track(source=dummy, device=device, persist=True, verbose=False)

    # Benchmark
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_frames):
        model.track(source=dummy, device=device, persist=True, verbose=False)

    if device != "cpu":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    fps = num_frames / elapsed
    ms_per_frame = elapsed / num_frames * 1000

    return fps, ms_per_frame


def benchmark_full_pipeline(
    config_path: str,
    imgsz: int = 640,
    num_frames: int = 100,
):
    """Tam pipeline FPS'ini olc."""
    from inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(config_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        pipeline.process_frame(dummy)

    # Benchmark
    t0 = time.time()
    for _ in range(num_frames):
        pipeline.process_frame(dummy)

    elapsed = time.time() - t0
    fps = num_frames / elapsed
    ms_per_frame = elapsed / num_frames * 1000

    return fps, ms_per_frame


def run_benchmark(config_path: str, model_path: str, device: str = "0", imgsz: int = 640):
    """Tum benchmark'lari calistir."""
    print("=" * 60)
    print("TANK VISION AI - PERFORMANS BENCHMARK")
    print("=" * 60)

    # GPU bilgisi
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("GPU: Yok (CPU modu)")

    print(f"Girdi boyutu: {imgsz}x{imgsz}")
    print()

    # 1. Sadece dedeksiyon
    print("[1/3] Sadece dedeksiyon...")
    fps, ms = benchmark_detection_only(model_path, device, imgsz)
    print(f"      FPS: {fps:.1f} | {ms:.1f}ms/frame")

    # 2. Dedeksiyon + takip
    print("[2/3] Dedeksiyon + takip (BoT-SORT)...")
    fps_t, ms_t = benchmark_tracking(model_path, device, imgsz)
    print(f"      FPS: {fps_t:.1f} | {ms_t:.1f}ms/frame")

    # 3. Tam pipeline
    print("[3/3] Tam pipeline (tespit+takip+mesafe+hiz+tehdit+HUD)...")
    try:
        fps_p, ms_p = benchmark_full_pipeline(config_path, imgsz, num_frames=50)
        print(f"      FPS: {fps_p:.1f} | {ms_p:.1f}ms/frame")
    except Exception as e:
        print(f"      Atlanmis: {e}")
        fps_p, ms_p = 0, 0

    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"  Sadece tespit:    {fps:.1f} FPS")
    print(f"  Tespit + takip:   {fps_t:.1f} FPS")
    if fps_p > 0:
        print(f"  Tam pipeline:     {fps_p:.1f} FPS")
    print(f"\n  Hedef: 30+ FPS (gercek zamanli)")
    if fps_t >= 30:
        print("  Durum: BASARILI - Gercek zamanli calisabilir")
    else:
        print("  Durum: Model boyutunu kuculterek (n/s) hizlandirmayi deneyin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="runs/detect/tank_vision_v1/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_benchmark(args.config, args.model, args.device, args.imgsz)

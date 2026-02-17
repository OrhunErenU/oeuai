"""Toplu video isleme modulu.

Kayitli video dosyalarini toplu olarak isler ve
sonuclari JSON/video olarak kaydeder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from inference.pipeline import InferencePipeline
from utils.logger import setup_logger
from utils.video_io import VideoWriter

logger = setup_logger("batch")


def process_video(
    pipeline: InferencePipeline,
    input_path: str,
    output_video: str | None = None,
    output_json: str | None = None,
):
    """Tek bir video dosyasini isle.

    Args:
        pipeline: InferencePipeline instance.
        input_path: Girdi video dosyasi.
        output_video: Cikti video (None ise video kaydetme).
        output_json: Cikti JSON (None ise JSON kaydetme).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Video acilamadi: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_video:
        writer = VideoWriter(output_video, fps=fps, width=width, height=height)

    all_results = []
    frame_idx = 0
    t_start = time.time()

    logger.info(f"Isleniyor: {input_path} ({total} frame, {width}x{height})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        frame_idx += 1

        if writer:
            annotated = pipeline.display.render(frame, result)
            writer.write(annotated)

        if output_json:
            frame_data = {
                "frame_id": result.frame_id,
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "bbox": [round(v, 1) for v in d.bbox],
                        "track_id": d.track_id,
                        "distance_m": round(d.distance_m, 1) if d.distance_m else None,
                        "speed_kmh": round(d.speed_kmh, 1),
                        "altitude_m": round(d.altitude_m, 1) if d.altitude_m else None,
                        "time_to_reach": d.time_to_reach,
                        "threat_level": d.threat_label,
                        "foe_status": d.foe_status,
                        "weapon_type": d.weapon_type,
                        "tank_model": d.tank_model,
                        "human_type": d.human_type,
                        "is_targeting": d.is_targeting_us,
                    }
                    for d in result.detections
                ],
            }
            all_results.append(frame_data)

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            avg_fps = frame_idx / elapsed
            pct = (frame_idx / total * 100) if total > 0 else 0
            logger.info(f"  {frame_idx}/{total} ({pct:.0f}%) - {avg_fps:.1f} FPS")

    cap.release()
    if writer:
        writer.release()

    if output_json and all_results:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON kaydedildi: {output_json}")

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    logger.info(f"Tamamlandi: {frame_idx} frame, {elapsed:.1f}sn, {avg_fps:.1f} FPS")


def process_batch(
    config_path: str,
    input_dir: str,
    output_dir: str,
    save_video: bool = True,
    save_json: bool = True,
):
    """Bir dizindeki tum video dosyalarini toplu isle.

    Args:
        config_path: Pipeline konfigurasyon dosyasi.
        input_dir: Girdi video dizini.
        output_dir: Cikti dizini.
        save_video: Islennis video kaydet.
        save_json: JSON sonuc kaydet.
    """
    pipeline = InferencePipeline(config_path)
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    videos = [f for f in in_path.iterdir() if f.suffix.lower() in video_exts]

    logger.info(f"Toplu isleme: {len(videos)} video bulundu")

    for i, video_path in enumerate(videos):
        logger.info(f"\n[{i + 1}/{len(videos)}] {video_path.name}")

        out_video = str(out_path / f"{video_path.stem}_processed.mp4") if save_video else None
        out_json = str(out_path / f"{video_path.stem}_results.json") if save_json else None

        process_video(pipeline, str(video_path), out_video, out_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toplu video isleme")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", required=True, help="Girdi dizini/dosyasi")
    parser.add_argument("--output", required=True, help="Cikti dizini")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        pipeline = InferencePipeline(args.config)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        process_video(
            pipeline,
            str(inp),
            str(out_dir / f"{inp.stem}_processed.mp4") if not args.no_video else None,
            str(out_dir / f"{inp.stem}_results.json") if not args.no_json else None,
        )
    else:
        process_batch(args.config, args.input, args.output, not args.no_video, not args.no_json)

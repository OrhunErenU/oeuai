"""Video okuma/yazma yardimci fonksiyonlari."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoWriter:
    """Video dosyasi yazici."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        codec: str = "mp4v",
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.output_path = output_path

    def write(self, frame):
        """Frame yaz."""
        self.writer.write(frame)

    def release(self):
        """Yaziciyi kapat."""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def get_video_info(source) -> dict:
    """Video kaynagindan bilgi al."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {}

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    total_frames = info["frame_count"]
    fps = info["fps"] if info["fps"] > 0 else 30
    info["duration_sec"] = total_frames / fps if total_frames > 0 else 0

    cap.release()
    return info

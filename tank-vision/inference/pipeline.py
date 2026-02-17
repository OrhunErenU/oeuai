"""Ana gercek zamanli cikarim pipeline'i.

Tum sistemi birlestiren giris noktasi:
1. Frame yakala (kamera/video/RTSP)
2. Tespit + alt-siniflandirma
3. Takip guncelle
4. Mesafe / hiz / irtifa hesapla
5. Tehdit degerlendir
6. HUD ciz
7. Goruntule / kaydet

Kullanim:
    python inference/pipeline.py --source video.mp4
    python inference/pipeline.py --source 0          # webcam
    python inference/pipeline.py --source rtsp://...  # RTSP
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from inference.result import DetectionResult, FrameResult
from inference.stream import VideoStream
from modules.altitude import AltitudeEstimator
from modules.detection import DetectionPipeline
from modules.display import HUDDisplay
from modules.distance import DistanceEstimator
from modules.speed import SpeedCalculator
from modules.threat import ThreatAssessor
from modules.time_to_reach import TimeToReachCalculator
from modules.tracking import TrackManager
from utils.config_loader import load_config
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger
from utils.video_io import VideoWriter

logger = setup_logger("pipeline")


class InferencePipeline:
    """Ana gercek zamanli cikarim pipeline'i."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Args:
            config_path: Ana konfigurasyon dosyasi yolu.
        """
        self.config = load_config(config_path)
        system = self.config.get("system", {})
        dist_cfg = self.config.get("distance", {})
        tracker_cfg = self.config.get("tracker", {})

        # Tespit pipeline
        self.detection = DetectionPipeline(self.config)

        # Takip yoneticisi
        self.tracker = TrackManager(
            buffer_size=tracker_cfg.get("track_buffer", 60) * 3,  # 3x buffer
            max_lost_frames=tracker_cfg.get("track_buffer", 60),
        )

        # Mesafe tahmincisi
        obj_dims_path = "config/camera/object_dimensions.yaml"
        try:
            obj_dims = load_config(obj_dims_path)
        except FileNotFoundError:
            obj_dims = {"dimensions": {}, "tank_models": {}}

        focal_length = dist_cfg.get("camera_focal_length")
        if focal_length is None:
            # FOV'dan hesapla
            hfov = dist_cfg.get("camera_hfov", 60)
            img_w = system.get("imgsz", 640)
            focal_length = DistanceEstimator.focal_from_fov(hfov, img_w)

        all_dims = obj_dims.get("dimensions", {})
        all_dims["tank_models"] = obj_dims.get("tank_models", {})

        self.distance_est = DistanceEstimator(
            focal_length_px=focal_length,
            object_dimensions=all_dims,
        )

        # Hiz hesaplayici
        self.speed_calc = SpeedCalculator(
            camera_hfov_deg=dist_cfg.get("camera_hfov", 60),
            image_width_px=system.get("imgsz", 640),
        )

        # Irtifa tahmincisi
        self.altitude_est = AltitudeEstimator(
            camera_vfov_deg=dist_cfg.get("camera_vfov", 34),
            camera_tilt_deg=dist_cfg.get("camera_tilt", 0),
            camera_height_m=dist_cfg.get("camera_height", 2.5),
            image_height_px=system.get("imgsz", 640),
        )

        # Ulasma suresi
        self.ttr_calc = TimeToReachCalculator()

        # Tehdit motoru
        self.threat = ThreatAssessor(self.config.get("threat", {}))

        # HUD
        self.display = HUDDisplay(self.config.get("display", {}))

        # FPS
        self.fps_counter = FPSCounter()

        # Frame sayaci
        self._frame_id = 0

        logger.info("Pipeline baslatildi")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Tek bir frame'i tamamen isle.

        Args:
            frame: BGR goruntu.

        Returns:
            FrameResult (tum tespitler + meta veri).
        """
        t0 = time.time()
        self._frame_id += 1

        # 1. Tespit + alt-siniflandirma
        detections = self.detection.process_frame(frame, use_tracking=True)

        # 2. Takip guncelle
        self.tracker.update(detections)

        # 3. Her tespit icin mekansal analiz
        for det in detections:
            self._enrich_detection(det)

        # 4. Tehdit degerlendirme
        self.threat.assess_frame(detections)

        # 5. FPS guncelle
        self.fps_counter.tick()

        return FrameResult(
            frame_id=self._frame_id,
            timestamp=time.time(),
            detections=detections,
            fps=self.fps_counter.get(),
            processing_time_ms=(time.time() - t0) * 1000,
        )

    def _enrich_detection(self, det: DetectionResult):
        """Tespiti mesafe/hiz/irtifa/ulasma suresi ile zenginlestir."""
        # Mesafe tahmini
        det.distance_m = self.distance_est.estimate(
            det.class_name,
            det.bbox_width,
            det.bbox_height,
            sub_class=det.tank_model,
        )

        # Hiz hesabi (takip gecmisi gerektirir)
        if det.track_id is not None:
            positions, timestamps = self.tracker.get_history(det.track_id)
            dist_history = self.tracker.get_distance_history(det.track_id)

            if len(positions) >= 2:
                speed_info = self.speed_calc.calculate_speed(
                    positions, timestamps, dist_history
                )
                det.speed_ms = speed_info["speed_ms"]
                det.speed_kmh = speed_info["speed_kmh"]
                det.heading_deg = speed_info["heading_deg"]
                det.approaching = speed_info["approaching"]

            # Mesafeyi takip gecmisine ekle
            if det.distance_m is not None:
                self.tracker.distances[det.track_id].append(det.distance_m)

        # Irtifa (hava nesneleri icin)
        if det.class_name in ("drone", "aircraft", "bird"):
            det.altitude_m = self.altitude_est.estimate(
                det.center[1], det.distance_m
            )

        # Tanka ulasma suresi
        det.time_to_reach = self.ttr_calc.calculate(
            det.distance_m, det.speed_ms, det.approaching
        )

    def run(
        self,
        source=0,
        save_path: str | None = None,
        show: bool = True,
        max_frames: int | None = None,
    ):
        """Pipeline'i video kaynagi uzerinde calistir.

        Args:
            source: Video kaynagi (dosya yolu, RTSP URL, kamera indeksi).
            save_path: Cikti video dosyasi (None ise kaydetme).
            show: True ise pencerede goster.
            max_frames: Maksimum frame sayisi (None = sinir yok).
        """
        logger.info(f"Kaynak: {source}")

        stream = VideoStream(source).start()
        writer = None

        if save_path:
            writer = VideoWriter(
                save_path, fps=stream.fps, width=stream.width, height=stream.height
            )
            logger.info(f"Cikti: {save_path}")

        try:
            while stream.is_alive:
                frame = stream.read()
                if frame is None:
                    break

                # Frame isle
                result = self.process_frame(frame)

                # HUD ciz
                annotated = self.display.render(frame, result)

                # Kaydet
                if writer:
                    writer.write(annotated)

                # Goster
                if show:
                    cv2.imshow("Tank Vision AI", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Kullanici tarafindan durduruldu")
                        break
                    elif key == ord("s"):
                        # Ekran goruntusu
                        ss_path = f"screenshot_{self._frame_id}.jpg"
                        cv2.imwrite(ss_path, annotated)
                        logger.info(f"Ekran goruntusu: {ss_path}")

                # Kritik tehdit loglama
                for det in result.critical_threats:
                    logger.warning(
                        f"KRITIK TEHDIT: {det.summary()}"
                    )

                # Frame limiti
                if max_frames and self._frame_id >= max_frames:
                    break

        finally:
            stream.stop()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(f"Bitis. Toplam {self._frame_id} frame islendi.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tank Vision AI - Gercek Zamanli Cikarim")
    parser.add_argument("--source", default=0, help="Video kaynagi (dosya/kamera/RTSP)")
    parser.add_argument("--config", default="config/default.yaml", help="Konfigurasyon")
    parser.add_argument("--save", default=None, help="Cikti video dosyasi")
    parser.add_argument("--no-show", action="store_true", help="Pencere gosterme")
    parser.add_argument("--max-frames", type=int, default=None, help="Maks frame")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    pipeline = InferencePipeline(args.config)
    pipeline.run(
        source=source,
        save_path=args.save,
        show=not args.no_show,
        max_frames=args.max_frames,
    )

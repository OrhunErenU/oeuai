"""Ana YOLOv11 nesne tespit modeli.

7 sinif tespiti yapar: drone, tank, human, weapon, vehicle, aircraft, bird.
Hem tek-frame tespit hem de takipli tespit (BoT-SORT) destekler.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from data.class_mapping import PRIMARY_CLASSES
from inference.result import DetectionResult
from utils.bbox_utils import bbox_center


class PrimaryDetector:
    """Ana YOLOv11 detektor wrapper sinifi."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        """
        Args:
            weights_path: .pt agirliklari dosya yolu.
            device: Cihaz ("cuda:0", "cpu").
            conf: Minimum guven esigi.
            iou: NMS IoU esigi.
            imgsz: Girdi goruntu boyutu.
        """
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Tek frame uzerinde nesne tespiti yap.

        Args:
            frame: BGR goruntu (numpy array).

        Returns:
            DetectionResult listesi.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        return self._parse_results(results[0])

    def track(
        self,
        frame: np.ndarray,
        tracker_config: str = "botsort.yaml",
        persist: bool = True,
    ) -> list[DetectionResult]:
        """Frame uzerinde tespit + takip yap (BoT-SORT/ByteTrack).

        Args:
            frame: BGR goruntu.
            tracker_config: Tracker YAML dosya yolu.
            persist: Takip ID'lerini frame'ler arasi koru.

        Returns:
            Track ID'li DetectionResult listesi.
        """
        results = self.model.track(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=tracker_config,
            persist=persist,
            verbose=False,
        )
        return self._parse_results(results[0], with_tracking=True)

    def _parse_results(
        self, result, with_tracking: bool = False
    ) -> list[DetectionResult]:
        """Ultralytics sonucunu DetectionResult listesine donustur."""
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = PRIMARY_CLASSES.get(cls_id, f"unknown_{cls_id}")

            track_id = None
            if with_tracking and boxes.id is not None:
                track_id = int(boxes.id[i].item())

            center = bbox_center((x1, y1, x2, y2))

            det = DetectionResult(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                track_id=track_id,
                center=center,
            )
            detections.append(det)

        return detections

    def warmup(self):
        """Modeli isindir (ilk cikarimdaki gecikmeyi azalt)."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(dummy)

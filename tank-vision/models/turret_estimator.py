"""Taret yon tahmincisi.

Tespit edilen tank bbox'i icinde taret/namlu konumunu analiz ederek
taretin bize dogru hedef alip almadigini tahmin eder.

Yaklasim: Namlu keypoint tespiti + geometrik aci hesabi.
Basit mod: Bbox en-boy orani ve namlu gorunurluk analizi.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class TurretEstimator:
    """Tank taret yonu tahmincisi."""

    def __init__(self, weights_path: str | None = None, device: str = "cuda:0"):
        """
        Args:
            weights_path: Taret keypoint model agirliklari (None ise geometrik analiz kullanilir).
            device: Cihaz.
        """
        self.model = None
        self.device = device

        if weights_path:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)

    def estimate_direction(self, tank_crop: np.ndarray) -> dict:
        """Taret yonunu tahmin et.

        Args:
            tank_crop: BGR numpy array (tank bbox kirpimi).

        Returns:
            {
                "angle_deg": float,       # 0-360 derece
                "is_targeting_us": bool,  # Bize hedef aliyor mu
                "confidence": float,
                "method": str,            # "model" veya "geometric"
            }
        """
        if tank_crop.size == 0:
            return {
                "angle_deg": 0.0,
                "is_targeting_us": False,
                "confidence": 0.0,
                "method": "none",
            }

        if self.model is not None:
            return self._model_based_estimate(tank_crop)

        return self._geometric_estimate(tank_crop)

    # Siniflandirma modeli yon eslemesi
    DIRECTION_ANGLES = {
        "front": 180.0,   # Bize dogru -> 180 derece
        "back": 0.0,      # Bizden uzak -> 0 derece
        "left": 270.0,    # Sol tarafa -> 270 derece
        "right": 90.0,    # Sag tarafa -> 90 derece
    }
    DIRECTION_TARGETING = {"front"}  # Bize hedef alan yonler

    def _model_based_estimate(self, tank_crop: np.ndarray) -> dict:
        """Model tabanli taret yonu tahmini (siniflandirma modeli).

        Egitilen model 4 sinif cikarir: front, back, left, right.
        front = namlu bize dogru (TEHLIKE).
        """
        try:
            results = self.model.predict(
                source=tank_crop,
                device=self.device,
                verbose=False,
            )

            probs = results[0].probs
            if probs is None:
                return self._geometric_estimate(tank_crop)

            class_id = probs.top1
            confidence = float(probs.top1conf.item())

            # Sinif ID -> yon esleme
            direction_map = {0: "front", 1: "back", 2: "left", 3: "right"}
            direction = direction_map.get(class_id, "unknown")

            angle = self.DIRECTION_ANGLES.get(direction, 0.0)
            is_targeting = direction in self.DIRECTION_TARGETING and confidence > 0.5

            return {
                "angle_deg": angle,
                "is_targeting_us": is_targeting,
                "confidence": confidence,
                "method": "model",
            }
        except Exception:
            return self._geometric_estimate(tank_crop)

    def _geometric_estimate(self, tank_crop: np.ndarray) -> dict:
        """Geometrik analiz ile taret yonu tahmini.

        Namlu benzeri uzun, ince yatay/dikey hatlari kenar tespiti ile bulur.
        Namlu kameraya dogru bakiyorsa: daire seklinde gorunur (kisa).
        Yandan bakiyorsa: uzun bir hat olarak gorunur.
        """
        h, w = tank_crop.shape[:2]
        if h < 10 or w < 10:
            return {
                "angle_deg": 0.0,
                "is_targeting_us": False,
                "confidence": 0.0,
                "method": "geometric",
            }

        # Ust yariyi al (taret genellikle ustte)
        turret_region = tank_crop[: h // 2, :]

        # Kenar tespiti
        gray = cv2.cvtColor(turret_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Hough hat tespiti
        lines = cv2.HoughLinesP(
            edges, 1, math.pi / 180, threshold=30, minLineLength=w // 6, maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return {
                "angle_deg": 0.0,
                "is_targeting_us": False,
                "confidence": 0.2,
                "method": "geometric",
            }

        # En uzun hatti bul (muhtemelen namlu)
        best_line = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > best_length:
                best_length = length
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return {
                "angle_deg": 0.0,
                "is_targeting_us": False,
                "confidence": 0.2,
                "method": "geometric",
            }

        x1, y1, x2, y2 = best_line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = angle % 360

        # Namlu kameraya bakiyor mu analizi:
        # Namlu dogrudan bize bakiyorsa, bbox orani genis olur ve
        # namlu cok kisa gorunur (perspektif kisalmasi)
        barrel_ratio = best_length / w
        is_targeting = barrel_ratio < 0.15 and abs(angle - 90) < 30

        # Eger namlu yandan gorunuyorsa (uzun hat) hedef aliyor degil
        confidence = 0.5
        if barrel_ratio < 0.2:
            confidence = 0.7  # Namlu kisa gorunuyor, muhtemelen bize bakiyor
        elif barrel_ratio > 0.4:
            confidence = 0.6  # Namlu acikca yandan gorunuyor

        return {
            "angle_deg": angle,
            "is_targeting_us": is_targeting,
            "confidence": confidence,
            "method": "geometric",
        }

    @staticmethod
    def is_targeting(angle_deg: float, threshold_deg: float = 20.0) -> bool:
        """Taret bize dogru mu bakiyor?

        Args:
            angle_deg: Taret acisi (derece).
            threshold_deg: Tolerans esigi.
        """
        # 0 ve 180 derece: namlu kameraya dogru
        return angle_deg <= threshold_deg or abs(angle_deg - 180) <= threshold_deg

"""Tank marka/model alt-siniflandirici.

Tespit edilen tank bbox'inin kirpilmis goruntusunu alip
tank modelini siniflandirir: M1 Abrams, Leopard 2, T-72, T-90, vb.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from data.class_mapping import TANK_CLASSES


class TankClassifier:
    """YOLOv11-cls tabanli tank model siniflandirici."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.model = YOLO(weights_path)
        self.device = device
        self.classes = TANK_CLASSES

    def classify(self, cropped_image: np.ndarray) -> tuple[str, float]:
        """Kirpilmis tank goruntusunu siniflandir.

        Args:
            cropped_image: BGR numpy array (tank bbox kirpimi).

        Returns:
            (tank_modeli, guven_skoru): Ornek: ("leopard_2", 0.87)
        """
        if cropped_image.size == 0:
            return "unknown", 0.0

        results = self.model.predict(
            source=cropped_image,
            device=self.device,
            verbose=False,
        )

        probs = results[0].probs
        class_id = probs.top1
        confidence = float(probs.top1conf.item())
        class_name = self.classes.get(class_id, "unknown")

        return class_name, confidence

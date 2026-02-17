"""Insan alt-siniflandirici: Asker vs Sivil.

Tespit edilen insan bbox'inin kirpilmis goruntusunu alip
asker mi sivil mi siniflandirir.

Gorsel ipuclari: uniformalik, kamuflaj, kask, yelek, taktik ekipman.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from data.class_mapping import HUMAN_CLASSES


class HumanClassifier:
    """YOLOv11-cls tabanli asker/sivil siniflandirici."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.model = YOLO(weights_path)
        self.device = device
        self.classes = HUMAN_CLASSES

    def classify(self, cropped_image: np.ndarray) -> tuple[str, float]:
        """Kirpilmis insan goruntusunu siniflandir.

        Args:
            cropped_image: BGR numpy array (insan bbox kirpimi).

        Returns:
            (insan_turu, guven_skoru): Ornek: ("soldier", 0.92)
        """
        if cropped_image.size == 0:
            return "civilian", 0.0

        results = self.model.predict(
            source=cropped_image,
            device=self.device,
            verbose=False,
        )

        probs = results[0].probs
        class_id = probs.top1
        confidence = float(probs.top1conf.item())
        class_name = self.classes.get(class_id, "civilian")

        return class_name, confidence

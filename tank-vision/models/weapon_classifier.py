"""Silah turu alt-siniflandirici.

Tespit edilen silah bbox'inin kirpilmis goruntusunu alip
silah turunu siniflandirir: RPG, rifle, pistol, sniper, grenade, machine_gun.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from data.class_mapping import WEAPON_CLASSES


class WeaponClassifier:
    """YOLOv11-cls tabanli silah turu siniflandirici."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.model = YOLO(weights_path)
        self.device = device
        self.classes = WEAPON_CLASSES

    def classify(self, cropped_image: np.ndarray) -> tuple[str, float]:
        """Kirpilmis silah goruntusunu siniflandir.

        Args:
            cropped_image: BGR numpy array (silah bbox kirpimi).

        Returns:
            (silah_turu, guven_skoru): Ornek: ("rpg", 0.95)
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

    def classify_top_k(
        self, cropped_image: np.ndarray, k: int = 3
    ) -> list[tuple[str, float]]:
        """Ilk k tahmini dondur.

        Returns:
            [(sinif_adi, guven_skoru), ...] listesi.
        """
        if cropped_image.size == 0:
            return [("unknown", 0.0)]

        results = self.model.predict(
            source=cropped_image,
            device=self.device,
            verbose=False,
        )

        probs = results[0].probs
        top_k_ids = probs.top5[:k]
        top_k_confs = probs.top5conf[:k]

        return [
            (self.classes.get(int(cid), "unknown"), float(conf))
            for cid, conf in zip(top_k_ids, top_k_confs)
        ]

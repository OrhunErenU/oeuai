"""Dost/Dusman (IFF) siniflandirici.

Gorsel model + kural-tabanli mantik birlestirerek
dost, dusman veya bilinmeyen siniflandirmasi yapar.

Gorsel ipuclari: Arac/tank isaretleri, boya desenleri, kamuflaj aileleri.
Kural-tabanli: Bilinen dost tank modelleri YAML konfigurasyonundan okunur.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from data.class_mapping import FOE_CLASSES


class FoeClassifier:
    """Dost/Dusman siniflandirici (gorsel + kural tabanli)."""

    def __init__(
        self,
        weights_path: str,
        friendly_config: dict | None = None,
        device: str = "cuda:0",
    ):
        """
        Args:
            weights_path: Gorsel siniflandirici agirliklari.
            friendly_config: Dost konfigurasyonu:
                {
                    "friendly_tank_models": ["m1_abrams", "leopard_2", ...],
                    "friendly_markings": ["nato_star", ...]
                }
            device: Cihaz.
        """
        self.model = YOLO(weights_path)
        self.device = device
        self.classes = FOE_CLASSES
        self.friendly_tank_models = set()
        self.friendly_markings = set()

        if friendly_config:
            self.friendly_tank_models = set(
                friendly_config.get("friendly_tank_models", [])
            )
            self.friendly_markings = set(
                friendly_config.get("friendly_markings", [])
            )

    def classify(
        self,
        cropped_image: np.ndarray,
        object_class: str,
        sub_class: str | None = None,
    ) -> tuple[str, float]:
        """Dost/Dusman siniflandirmasi yap.

        Oncelikle kural-tabanli kontrol yapar (bilinen dost tank modelleri),
        sonra gorsel modelden tahmin alir ve sonuclari birlestirir.

        Args:
            cropped_image: BGR numpy array.
            object_class: Ana sinif ("tank", "drone", "aircraft", vb.).
            sub_class: Alt sinif bilgisi (ornek: "m1_abrams").

        Returns:
            (durum, guven): ("friend", 0.95) veya ("foe", 0.88) veya ("unknown", 0.5)
        """
        # Kural 1: Bilinen dost tank modeli mi?
        if object_class == "tank" and sub_class in self.friendly_tank_models:
            return "friend", 0.90

        # Kural 2: Kuslar her zaman dost degil, bilinmeyen
        if object_class == "bird":
            return "unknown", 1.0

        # Gorsel siniflandirma
        if cropped_image.size == 0:
            return "unknown", 0.0

        try:
            results = self.model.predict(
                source=cropped_image,
                device=self.device,
                verbose=False,
            )

            probs = results[0].probs
            class_id = probs.top1
            confidence = float(probs.top1conf.item())
            class_name = self.classes.get(class_id, "unknown")

            # Dusuk guvenli dusman tahminlerini "unknown" olarak isaretle
            # (yanlis dusman tespiti tehlikeli, yuksek esik)
            if class_name == "foe" and confidence < 0.65:
                return "unknown", confidence

            # Dusuk guvenli dost tahminlerini de "unknown" olarak isaretle
            if class_name == "friend" and confidence < 0.55:
                return "unknown", confidence

            return class_name, confidence
        except (RuntimeError, Exception):
            return "unknown", 0.0

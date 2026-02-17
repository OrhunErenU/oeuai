"""Tespit pipeline orkestratoru.

Her frame icin:
1. Ana YOLOv11 dedektoru calistir (+ takip)
2. Her tespit icin uygun alt-siniflandiriciyi calistir
3. Sonuclari DetectionResult olarak paketleyip dondur
"""

from __future__ import annotations

import numpy as np

from inference.result import DetectionResult
from models.detector import PrimaryDetector
from utils.bbox_utils import crop_image


class DetectionPipeline:
    """Tespit + alt-siniflandirma orkestratoru."""

    def __init__(self, config: dict):
        """
        Args:
            config: Ana konfigurasyon dict'i (default.yaml).
        """
        system = config.get("system", {})
        models_cfg = config.get("models", {})
        device = system.get("device", "cuda:0")

        # Ana detektor
        self.detector = PrimaryDetector(
            weights_path=models_cfg["detector"],
            device=device,
            conf=system.get("conf_thresh", 0.25),
            iou=system.get("iou_thresh", 0.45),
            imgsz=system.get("imgsz", 640),
        )

        # Alt-siniflandiriclar (lazy load - sadece agirliklari varsa yukle)
        self.weapon_cls = None
        self.tank_cls = None
        self.human_cls = None
        self.foe_cls = None
        self.turret_est = None

        self._load_subclassifiers(models_cfg, config, device)

        # Tracker config
        tracker_cfg = config.get("tracker", {})
        self.tracker_config = tracker_cfg.get("config", "botsort.yaml")

    def _load_subclassifiers(self, models_cfg: dict, config: dict, device: str):
        """Alt-siniflandirici modellerini yukle (varsa)."""
        from pathlib import Path

        if Path(models_cfg.get("weapon_classifier", "")).exists():
            from models.weapon_classifier import WeaponClassifier
            self.weapon_cls = WeaponClassifier(models_cfg["weapon_classifier"], device)

        if Path(models_cfg.get("tank_classifier", "")).exists():
            from models.tank_classifier import TankClassifier
            self.tank_cls = TankClassifier(models_cfg["tank_classifier"], device)

        if Path(models_cfg.get("human_classifier", "")).exists():
            from models.human_classifier import HumanClassifier
            self.human_cls = HumanClassifier(models_cfg["human_classifier"], device)

        if Path(models_cfg.get("foe_classifier", "")).exists():
            from models.foe_classifier import FoeClassifier
            friendly_cfg = config.get("friend_foe", {})
            self.foe_cls = FoeClassifier(
                models_cfg["foe_classifier"], friendly_cfg, device
            )

        if Path(models_cfg.get("turret_estimator", "")).exists():
            from models.turret_estimator import TurretEstimator
            self.turret_est = TurretEstimator(models_cfg["turret_estimator"], device)
        else:
            # Geometrik analiz (model olmadan)
            from models.turret_estimator import TurretEstimator
            self.turret_est = TurretEstimator(weights_path=None, device=device)

    def process_frame(
        self,
        frame: np.ndarray,
        use_tracking: bool = True,
    ) -> list[DetectionResult]:
        """Tek frame icin tespit + alt-siniflandirma.

        Args:
            frame: BGR goruntu.
            use_tracking: True ise BoT-SORT takip kullan.

        Returns:
            DetectionResult listesi (alt-siniflandirma bilgileri ile).
        """
        # 1. Ana tespit/takip
        if use_tracking:
            detections = self.detector.track(
                frame, tracker_config=self.tracker_config
            )
        else:
            detections = self.detector.detect(frame)

        # 2. Her tespit icin alt-siniflandirma
        for det in detections:
            crop = crop_image(frame, det.bbox, padding=0.05)
            if crop.size == 0:
                continue

            self._run_subclassification(crop, det)

        return detections

    def _run_subclassification(self, crop: np.ndarray, det: DetectionResult):
        """Sinifa uygun alt-siniflandiriciyi calistir."""
        # Silah turu
        if det.class_name == "weapon" and self.weapon_cls:
            det.weapon_type, det.weapon_conf = self.weapon_cls.classify(crop)

        # Tank modeli + taret + dost/dusman
        elif det.class_name == "tank":
            if self.tank_cls:
                det.tank_model, det.tank_conf = self.tank_cls.classify(crop)

            if self.turret_est:
                turret_info = self.turret_est.estimate_direction(crop)
                det.turret_angle = turret_info.get("angle_deg")
                det.is_targeting_us = turret_info.get("is_targeting_us", False)

            if self.foe_cls:
                det.foe_status, det.foe_conf = self.foe_cls.classify(
                    crop, "tank", det.tank_model
                )

        # Insan siniflandirma
        elif det.class_name == "human" and self.human_cls:
            det.human_type, det.human_conf = self.human_cls.classify(crop)

        # Dron/ucak dost-dusman
        elif det.class_name in ("drone", "aircraft") and self.foe_cls:
            det.foe_status, det.foe_conf = self.foe_cls.classify(
                crop, det.class_name
            )

    def warmup(self):
        """Tum modelleri isindir."""
        self.detector.warmup()

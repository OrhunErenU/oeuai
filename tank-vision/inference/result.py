"""Tespit sonucu veri yapilari.

Tum moduller arasi veri akisi bu dataclass'lar uzerinden gerceklesir.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """Tek bir tespit icin tum bilgileri iceren veri yapisi."""

    # Ana tespit
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (piksel)
    class_id: int = -1
    class_name: str = ""
    confidence: float = 0.0
    track_id: int | None = None
    center: tuple[float, float] = (0.0, 0.0)

    # Alt-siniflandirma
    weapon_type: str | None = None
    weapon_conf: float = 0.0
    tank_model: str | None = None
    tank_conf: float = 0.0
    human_type: str | None = None  # "soldier" veya "civilian"
    human_conf: float = 0.0
    foe_status: str | None = None  # "friend", "foe", "unknown"
    foe_conf: float = 0.0

    # Mekansal analiz
    distance_m: float | None = None
    speed_ms: float = 0.0
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    approaching: bool = False
    altitude_m: float | None = None
    time_to_reach: float | None = None

    # Taret analizi (sadece tanklar)
    turret_angle: float | None = None
    is_targeting_us: bool = False

    # Tehdit degerlendirme
    threat_level: int = 0
    threat_label: str = "none"
    threat_reasons: list[str] = field(default_factory=list)
    priority_score: float = 0.0

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def summary(self) -> str:
        """Kisa ozet metni dondur."""
        parts = [f"{self.class_name}"]
        if self.track_id is not None:
            parts.append(f"ID:{self.track_id}")
        if self.confidence > 0:
            parts.append(f"{self.confidence:.0%}")
        if self.distance_m is not None:
            parts.append(f"{self.distance_m:.0f}m")
        if self.speed_kmh > 1:
            parts.append(f"{self.speed_kmh:.0f}km/h")
        if self.threat_label != "none":
            parts.append(f"[{self.threat_label.upper()}]")
        if self.foe_status:
            parts.append(f"({self.foe_status})")
        return " | ".join(parts)


@dataclass
class FrameResult:
    """Tek bir frame icin tum tespit sonuclarini iceren veri yapisi."""

    frame_id: int = 0
    timestamp: float = 0.0
    detections: list[DetectionResult] = field(default_factory=list)
    fps: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def critical_threats(self) -> list[DetectionResult]:
        """Kritik tehdit seviyesindeki tespitler."""
        return [d for d in self.detections if d.threat_level >= 4]

    @property
    def high_threats(self) -> list[DetectionResult]:
        """Yuksek ve uzeri tehdit seviyesindeki tespitler."""
        return [d for d in self.detections if d.threat_level >= 3]

    def sorted_by_threat(self) -> list[DetectionResult]:
        """Tehdit puanina gore sirali tespitler (en yuksek once)."""
        return sorted(self.detections, key=lambda d: d.priority_score, reverse=True)

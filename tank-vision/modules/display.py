"""HUD (Heads-Up Display) goruntusu modulu.

Video frame uzerine tespit sonuclarini, tehdit bilgilerini
ve diger analiz verilerini gorsel olarak cizer.
"""

from __future__ import annotations

import cv2
import numpy as np

from inference.result import DetectionResult, FrameResult
from utils.color_palette import CLASS_COLORS, FOE_COLORS, THREAT_COLORS


class HUDDisplay:
    """Askeri HUD goruntusu olusturucu."""

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Goruntuleme konfigurasyonu (default.yaml "display" bolumu).
        """
        cfg = config or {}
        self.show_distance = cfg.get("show_distance", True)
        self.show_speed = cfg.get("show_speed", True)
        self.show_threat = cfg.get("show_threat_level", True)
        self.show_class = cfg.get("show_classification", True)
        self.show_track = cfg.get("show_track_id", True)
        self.show_altitude = cfg.get("show_altitude", True)
        self.show_ttr = cfg.get("show_time_to_reach", True)
        self.show_foe = cfg.get("show_foe_status", True)
        self.show_turret = cfg.get("show_turret_warning", True)
        self.opacity = cfg.get("hud_opacity", 0.7)

    def render(
        self,
        frame: np.ndarray,
        frame_result: FrameResult,
    ) -> np.ndarray:
        """Frame uzerine tum HUD bilgilerini ciz.

        Args:
            frame: BGR goruntu.
            frame_result: Bu frame'in tum sonuclari.

        Returns:
            Uzerine HUD cizilmis frame.
        """
        overlay = frame.copy()

        for det in frame_result.detections:
            self._draw_detection(overlay, det)

        # Overlay transparanlik
        output = cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0)

        # Sabit HUD elemanlari (transparansizdan bagimsiz)
        self._draw_dashboard(output, frame_result)
        self._draw_fps(output, frame_result.fps)

        return output

    def _draw_detection(self, frame: np.ndarray, det: DetectionResult):
        """Tek bir tespit icin HUD ciz."""
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = THREAT_COLORS.get(det.threat_level, (200, 200, 200))
        cls_color = CLASS_COLORS.get(det.class_name, (255, 255, 255))

        # Bbox ciz
        thickness = 3 if det.threat_level >= 3 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Sinif etiketi
        y_offset = y1 - 10
        if y_offset < 20:
            y_offset = y2 + 20

        # Etiket arka plan
        label_parts = []
        if self.show_class:
            label_parts.append(det.class_name.upper())
        if self.show_track and det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        label_parts.append(f"{det.confidence:.0%}")

        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y_offset - th - 4), (x1 + tw + 4, y_offset), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y_offset - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # Alt bilgiler
        info_y = y2 + 15
        info_lines = []

        if self.show_distance and det.distance_m is not None:
            info_lines.append(f"Mesafe: {det.distance_m:.0f}m")

        if self.show_speed and det.speed_kmh > 1:
            arrow = ">>>" if det.approaching else "---"
            info_lines.append(f"Hiz: {det.speed_kmh:.0f}km/h {arrow}")

        if self.show_altitude and det.altitude_m is not None:
            info_lines.append(f"Irtifa: {det.altitude_m:.0f}m")

        if self.show_ttr and det.time_to_reach is not None:
            info_lines.append(f"Ulasma: {det.time_to_reach:.1f}sn")

        if self.show_foe and det.foe_status:
            foe_color = FOE_COLORS.get(det.foe_status, (255, 255, 255))
            status_text = det.foe_status.upper()
            cv2.putText(
                frame, status_text, (x2 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, foe_color, 2,
            )

        if self.show_threat and det.threat_label != "none":
            info_lines.append(f"Tehdit: {det.threat_label.upper()}")

        # Alt sinif bilgileri
        if det.weapon_type:
            info_lines.append(f"Silah: {det.weapon_type}")
        if det.tank_model:
            info_lines.append(f"Model: {det.tank_model}")
        if det.human_type:
            info_lines.append(f"Tip: {det.human_type}")

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame, line, (x1, info_y + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color, 1,
            )

        # Taret uyarisi
        if self.show_turret and det.is_targeting_us:
            self._draw_targeting_warning(frame, det)

    def _draw_targeting_warning(self, frame: np.ndarray, det: DetectionResult):
        """HEDEF ALINIYORSUNUZ uyarisi ciz."""
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Yanip sonen kirmizi cerceve
        cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 3)

        # Nisan isareti
        size = 20
        cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), size // 2, (0, 0, 255), 2)

        # Uyari metni
        cv2.putText(
            frame, "!! HEDEF ALINIYOR !!", (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    def _draw_dashboard(self, frame: np.ndarray, result: FrameResult):
        """Ekranin sag ustune tehdit ozet paneli ciz."""
        h, w = frame.shape[:2]
        panel_w = 280
        panel_h = 40 + len(result.high_threats) * 25 + 30
        panel_h = min(panel_h, h - 20)

        x1 = w - panel_w - 10
        y1 = 10

        # Panel arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Baslik
        cv2.putText(
            frame, f"TEHDIT PANELI ({result.num_detections} nesne)",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        # Yuksek tehdit listesi
        high = result.sorted_by_threat()[:8]
        for i, det in enumerate(high):
            if det.threat_level < 1:
                continue
            ty = y1 + 40 + i * 25
            color = THREAT_COLORS.get(det.threat_level, (200, 200, 200))

            text = f"{det.class_name}"
            if det.track_id is not None:
                text += f" #{det.track_id}"
            if det.distance_m is not None:
                text += f" {det.distance_m:.0f}m"
            text += f" [{det.threat_label.upper()}]"

            cv2.putText(
                frame, text, (x1 + 10, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
            )

    def _draw_fps(self, frame: np.ndarray, fps: float):
        """FPS gostergesi."""
        color = (0, 255, 0) if fps >= 25 else (0, 255, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

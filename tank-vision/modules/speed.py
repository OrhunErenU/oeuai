"""Hiz ve yorunge hesaplama modulu.

Takip edilen nesnelerin frame-arasi piksel yer degisimini
gercek dunya hizina donusturur. Nesnenin yaklasip yaklasmadigini
ve hareket yonunu belirler.
"""

from __future__ import annotations

import math

import numpy as np


class SpeedCalculator:
    """Gercek dunya hizi hesaplayici."""

    def __init__(self, camera_hfov_deg: float, image_width_px: int):
        """
        Args:
            camera_hfov_deg: Kamera yatay goris acisi (derece).
            image_width_px: Goruntu genisligi (piksel).
        """
        self.hfov = camera_hfov_deg
        self.image_width = image_width_px
        # Piksel basina aci (derece/piksel)
        self.deg_per_pixel = camera_hfov_deg / image_width_px

    def calculate_speed(
        self,
        positions: list[tuple[float, float]],
        timestamps: list[float],
        distances: list[float],
        n_frames: int = 10,
    ) -> dict:
        """Gercek dunya hizini hesapla.

        Piksel yer degisimini acisal yer degisimine donusturur,
        mesafe ile birlestirip gercek dunya hizini bulur.

        Args:
            positions: (cx, cy) piksel pozisyonlari gecmisi.
            timestamps: Zaman damgalari (saniye).
            distances: Mesafe tahminleri (metre) gecmisi.
            n_frames: Hesaplamada kullanilacak son frame sayisi.

        Returns:
            {
                "speed_ms": float,     # m/s
                "speed_kmh": float,    # km/h
                "heading_deg": float,  # Hareket yonu (0=sag, 90=yukari)
                "approaching": bool,   # Yaklasma durumu
            }
        """
        result = {
            "speed_ms": 0.0,
            "speed_kmh": 0.0,
            "heading_deg": 0.0,
            "approaching": False,
        }

        if len(positions) < 2 or len(timestamps) < 2:
            return result

        # Son n frame'i al
        n = min(n_frames, len(positions), len(timestamps))
        pos = positions[-n:]
        times = timestamps[-n:]

        dt = times[-1] - times[0]
        if dt < 1e-6:
            return result

        # Piksel yer degisimi
        dx_px = pos[-1][0] - pos[0][0]
        dy_px = pos[-1][1] - pos[0][1]
        pixel_speed = math.sqrt(dx_px**2 + dy_px**2) / dt

        # Hareket yonu (derece, 0=sag, 90=yukari)
        heading = math.degrees(math.atan2(-dy_px, dx_px)) % 360
        result["heading_deg"] = heading

        # Mesafe bilgisi varsa gercek dunya hizini hesapla
        if distances and len(distances) >= 2:
            # Yaklasma kontrolu
            result["approaching"] = self._is_approaching(distances)

            # Ortalama mesafe
            avg_distance = np.mean(distances[-n:]) if len(distances) >= n else np.mean(distances)

            if avg_distance > 0:
                # Acisal hizi gercek dunya hizina donustur
                # gercek_hiz = acisal_hiz * mesafe
                angular_speed_rad = math.radians(pixel_speed * self.deg_per_pixel)
                tangential_speed = angular_speed_rad * avg_distance

                # Radyal hiz (yaklasma/uzaklasma)
                radial_speed = 0.0
                if len(distances) >= 2:
                    n_dist = min(n, len(distances))
                    dist_recent = distances[-n_dist:]
                    dist_times = timestamps[-n_dist:]
                    if len(dist_recent) >= 2:
                        dist_dt = dist_times[-1] - dist_times[0]
                        if dist_dt > 0:
                            radial_speed = abs(
                                (dist_recent[0] - dist_recent[-1]) / dist_dt
                            )

                # Toplam hiz (teget + radyal)
                total_speed = math.sqrt(tangential_speed**2 + radial_speed**2)
                result["speed_ms"] = round(total_speed, 2)
                result["speed_kmh"] = round(total_speed * 3.6, 1)
        else:
            # Mesafe bilgisi yoksa acisal hiz uzerinden tahmin yap.
            # Varsayilan tahmini mesafe: sinifa gore ortalama beklenen mesafe kullan.
            # Bu tam dogruluk saglamaz ama 0.1 carpanından cok daha iyidir.
            angular_speed_deg = pixel_speed * self.deg_per_pixel
            angular_speed_rad = math.radians(angular_speed_deg)
            # Varsayilan mesafe: 100m (orta menzil tahmini)
            # Gercek mesafe bilinmediginde kaba ama tutarli bir tahmin saglar.
            default_distance = 100.0
            estimated_speed = angular_speed_rad * default_distance
            result["speed_ms"] = round(estimated_speed, 2)
            result["speed_kmh"] = round(estimated_speed * 3.6, 1)

        return result

    def _is_approaching(self, distances: list[float], min_samples: int = 3) -> bool:
        """Nesne yaklasma durumunu belirle.

        Son birkac mesafe olcumunun trendine bakar.
        """
        if len(distances) < min_samples:
            return False

        recent = distances[-min(10, len(distances)):]

        # Basit lineer trend: son mesafe ilk mesafeden kucuk mu?
        if recent[-1] < recent[0] * 0.95:  # %5 azalma esigi
            return True

        return False

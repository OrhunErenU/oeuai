"""Irtifa tahmini modulu (hava nesneleri icin).

Dron, ucak ve kuslarin irtifasini tahmin eder.
Nesnenin frame'deki dikey konumunu, kamera parametrelerini
ve mesafe tahminini birlestirerek hesaplar.
"""

from __future__ import annotations

import math


class AltitudeEstimator:
    """Hava nesneleri icin irtifa tahmincisi."""

    def __init__(
        self,
        camera_vfov_deg: float,
        camera_tilt_deg: float,
        camera_height_m: float,
        image_height_px: int,
    ):
        """
        Args:
            camera_vfov_deg: Kamera dikey goris acisi (derece).
            camera_tilt_deg: Kamera dikey egim acisi (derece, yukari pozitif).
            camera_height_m: Kameranin yerden yuksekligi (metre).
            image_height_px: Goruntu yuksekligi (piksel).
        """
        self.vfov = camera_vfov_deg
        self.tilt = camera_tilt_deg
        self.camera_height = camera_height_m
        self.image_height = image_height_px
        self.deg_per_pixel = camera_vfov_deg / image_height_px

    def estimate(self, bbox_center_y: float, distance_m: float) -> float | None:
        """Irtifayi metre cinsinden tahmin et.

        Args:
            bbox_center_y: Bbox merkez y koordinati (piksel, yukari 0).
            distance_m: Nesneye tahmini mesafe (metre).

        Returns:
            Tahmini irtifa (metre, yer seviyesinden) veya None.
        """
        if distance_m is None or distance_m <= 0:
            return None

        # Goruntunun ortasindan ofset (piksel)
        # Negatif = yukarida, pozitif = asagida
        pixel_offset = bbox_center_y - (self.image_height / 2)

        # Acisal ofset (derece)
        # Ust pikseller negatif ofset -> positif aci (yukariya bakmak)
        angle_offset_deg = -pixel_offset * self.deg_per_pixel

        # Toplam yükseklik acisi = kamera egimi + piksel ofseti
        elevation_deg = self.tilt + angle_offset_deg

        # Irtifa hesabi
        # altitude = camera_height + distance * sin(elevation_angle)
        elevation_rad = math.radians(elevation_deg)
        altitude = self.camera_height + distance_m * math.sin(elevation_rad)

        # Negatif irtifa olamaz (yer altinda degil)
        altitude = max(0.0, altitude)

        return round(altitude, 1)

    def estimate_relative(self, bbox_center_y: float, distance_m: float) -> float | None:
        """Tanka gore goreceli irtifa (tank = 0m).

        Returns:
            Goreceli irtifa (metre). Pozitif = yukarda.
        """
        abs_altitude = self.estimate(bbox_center_y, distance_m)
        if abs_altitude is None:
            return None
        return round(abs_altitude - self.camera_height, 1)

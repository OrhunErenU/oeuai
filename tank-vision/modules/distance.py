"""Mesafe tahmini modulu.

Pinhole kamera modeli kullanarak nesne mesafesini tahmin eder.
Formul: mesafe = (gercek_boyut * odak_uzakligi_piksel) / bbox_boyutu_piksel

Alt sinif bilinen boyutlari kullanir (ornegin T-72 icin 2.23m yukseklik).
"""

from __future__ import annotations

import math


class DistanceEstimator:
    """Pinhole kamera modeli tabanli mesafe tahmincisi."""

    def __init__(self, focal_length_px: float, object_dimensions: dict):
        """
        Args:
            focal_length_px: Kamera odak uzakligi (piksel). Kalibrasyon ile belirlenir.
            object_dimensions: Bilinen nesne boyutlari dict'i.
                {
                    "drone": {"width": 0.5, "height": 0.3},
                    "tank": {"width": 3.7, "height": 2.4},
                    ...
                }
        """
        self.focal_length = focal_length_px
        self.dimensions = object_dimensions

    def estimate(
        self,
        class_name: str,
        bbox_width_px: float,
        bbox_height_px: float,
        sub_class: str | None = None,
    ) -> float | None:
        """Mesafeyi metre cinsinden tahmin et.

        Daha kararli sonuc icin yukseklik kullanilir (yonelimden daha az etkilenir).
        Alt sinif biliniyorsa (ornegin tank modeli), daha hassas boyutlar kullanilir.

        Args:
            class_name: Ana sinif adi ("drone", "tank", vb.).
            bbox_width_px: Bounding box genisligi (piksel).
            bbox_height_px: Bounding box yuksekligi (piksel).
            sub_class: Opsiyonel alt sinif (ornek: "t72").

        Returns:
            Tahmini mesafe (metre) veya hesaplanamazsa None.
        """
        if self.focal_length is None or self.focal_length <= 0:
            return None

        if bbox_height_px < 1 or bbox_width_px < 1:
            return None

        # Alt sinif boyutlarini ara
        known = None
        tank_models = self.dimensions.get("tank_models", {})
        if sub_class and sub_class in tank_models:
            known = tank_models[sub_class]
        elif class_name in self.dimensions:
            known = self.dimensions[class_name]

        if known is None:
            return None

        # Yukseklik tabanli mesafe tahmini (daha kararli)
        known_height = known.get("height", 1.0)
        distance_h = (known_height * self.focal_length) / bbox_height_px

        # Genislik tabanli mesafe tahmini
        known_width = known.get("width", 1.0)
        distance_w = (known_width * self.focal_length) / bbox_width_px

        # Ortalama al (iki tahminin ortalamasi daha kararli)
        distance = (distance_h + distance_w) / 2

        # Makul aralik kontrolu (1m - 20km)
        distance = max(1.0, min(20000.0, distance))

        return distance

    @staticmethod
    def calibrate_focal_length(
        known_distance_m: float,
        known_real_height_m: float,
        perceived_height_px: float,
    ) -> float:
        """Bilinen bir referans olcumunden odak uzakligini hesapla.

        Bir nesneyi bilinen mesafede gorup, piksel yuksekligini olcun:
        focal_length = (piksel_yukseklik * bilinen_mesafe) / gercek_yukseklik

        Args:
            known_distance_m: Bilinen mesafe (metre).
            known_real_height_m: Nesnenin gercek yuksekligi (metre).
            perceived_height_px: Nesnenin goruntudeki piksel yuksekligi.

        Returns:
            Hesaplanan odak uzakligi (piksel).
        """
        if known_real_height_m <= 0 or perceived_height_px <= 0:
            raise ValueError("Boyutlar sifirdan buyuk olmali")

        return (perceived_height_px * known_distance_m) / known_real_height_m

    @staticmethod
    def focal_from_fov(fov_deg: float, image_size_px: int) -> float:
        """FOV (goris acisi) ve goruntu boyutundan odak uzakligi hesapla.

        Args:
            fov_deg: Goris acisi (derece).
            image_size_px: Goruntu boyutu (piksel, genislik veya yukseklik).

        Returns:
            Odak uzakligi (piksel).
        """
        return image_size_px / (2 * math.tan(math.radians(fov_deg / 2)))

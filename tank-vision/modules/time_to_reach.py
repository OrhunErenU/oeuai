"""Tanka ulasma suresi hesaplama modulu.

Nesnenin mevcut mesafesi ve yaklasma hizina gore
tanka ulasma suresini saniye cinsinden hesaplar.
"""

from __future__ import annotations


class TimeToReachCalculator:
    """Tanka ulasma suresi hesaplayici."""

    def __init__(self, min_speed_threshold: float = 0.5):
        """
        Args:
            min_speed_threshold: Hesaplama icin minimum hiz (m/s).
                Bundan yavas nesneler icin None dondurulur.
        """
        self.min_speed = min_speed_threshold

    def calculate(
        self,
        distance_m: float | None,
        speed_ms: float,
        approaching: bool,
    ) -> float | None:
        """Tanka ulasma suresini hesapla.

        Args:
            distance_m: Nesneye mesafe (metre).
            speed_ms: Nesne hizi (m/s).
            approaching: Nesne yaklasiyorsa True.

        Returns:
            Tahmini ulasma suresi (saniye) veya None (yaklasmiyorsa/hesaplanamazsa).
        """
        if distance_m is None or distance_m <= 0:
            return None

        if not approaching:
            return None

        if speed_ms < self.min_speed:
            return None

        time_seconds = distance_m / speed_ms

        # Makul aralik: 0.1s - 3600s (1 saat)
        time_seconds = max(0.1, min(3600.0, time_seconds))

        return round(time_seconds, 1)

    def calculate_with_acceleration(
        self,
        distance_m: float | None,
        speed_ms: float,
        acceleration_ms2: float,
        approaching: bool,
    ) -> float | None:
        """Ivme dahil ulasma suresi hesabi.

        d = v*t + 0.5*a*t^2 denklemini cozer.

        Args:
            distance_m: Mesafe (metre).
            speed_ms: Mevcut hiz (m/s).
            acceleration_ms2: Ivme (m/s^2, pozitif = hizlaniyor).
            approaching: Yaklasma durumu.

        Returns:
            Tahmini ulasma suresi (saniye) veya None.
        """
        if distance_m is None or distance_m <= 0 or not approaching:
            return None

        if speed_ms < self.min_speed and acceleration_ms2 <= 0:
            return None

        # Sabit hiz (ivme ~0)
        if abs(acceleration_ms2) < 0.01:
            return self.calculate(distance_m, speed_ms, approaching)

        # Ikinci derece denklem: 0.5*a*t^2 + v*t - d = 0
        a = 0.5 * acceleration_ms2
        b = speed_ms
        c = -distance_m

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None

        import math
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        # Pozitif ve en kucuk zamani sec
        candidates = [t for t in (t1, t2) if t > 0]
        if not candidates:
            return None

        return round(min(candidates), 1)

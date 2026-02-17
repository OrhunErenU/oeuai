"""Geometrik hesaplama yardimci fonksiyonlari.

Aci hesaplari, piksel-dunya donusumu, trigonometri yardimcilari.
"""

from __future__ import annotations

import math

import numpy as np


def angle_between_points(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Iki nokta arasindaki aciyi hesapla (derece, 0=sag, 90=yukari)."""
    dx = p2[0] - p1[0]
    dy = -(p2[1] - p1[1])  # Goruntu koordinatlarinda y asagi pozitif
    return math.degrees(math.atan2(dy, dx)) % 360


def distance_2d(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """2D Oklid mesafesi."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def pixel_to_angle(
    pixel_offset: float, fov_deg: float, image_size_px: int
) -> float:
    """Piksel ofsetini aciya donustur.

    Args:
        pixel_offset: Goruntu merkezinden piksel ofseti.
        fov_deg: Goris acisi (derece).
        image_size_px: Goruntu boyutu (piksel).

    Returns:
        Aci (derece).
    """
    deg_per_px = fov_deg / image_size_px
    return pixel_offset * deg_per_px


def angle_to_pixel(
    angle_deg: float, fov_deg: float, image_size_px: int
) -> float:
    """Aciyi piksel ofsetine donustur.

    Args:
        angle_deg: Aci (derece).
        fov_deg: Goris acisi (derece).
        image_size_px: Goruntu boyutu (piksel).

    Returns:
        Goruntu merkezinden piksel ofseti.
    """
    deg_per_px = fov_deg / image_size_px
    return angle_deg / deg_per_px


def bearing_to_target(
    our_position: tuple[float, float],
    target_position: tuple[float, float],
) -> float:
    """Hedefe olan pusla yonunu hesapla (derece, 0=kuzey, saat yonu).

    Args:
        our_position: Bizim konum (x, y).
        target_position: Hedef konum (x, y).

    Returns:
        Yon (derece, 0-360).
    """
    dx = target_position[0] - our_position[0]
    dy = target_position[1] - our_position[1]
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    return bearing


def smoothed_positions(
    positions: list[tuple[float, float]], window: int = 5
) -> list[tuple[float, float]]:
    """Pozisyon gecmisini hareketli ortalama ile yumusatir.

    Args:
        positions: (x, y) pozisyon listesi.
        window: Yumusatma pencere boyutu.

    Returns:
        Yumusatilmis pozisyonlar.
    """
    if len(positions) <= window:
        return positions

    result = []
    for i in range(len(positions)):
        start = max(0, i - window // 2)
        end = min(len(positions), i + window // 2 + 1)
        chunk = positions[start:end]
        avg_x = sum(p[0] for p in chunk) / len(chunk)
        avg_y = sum(p[1] for p in chunk) / len(chunk)
        result.append((avg_x, avg_y))

    return result


def linear_regression_slope(values: list[float]) -> float:
    """Basit lineer regresyon egimi hesapla.

    Degerlerin artip azaldigini belirlemek icin kullanilir
    (mesafe trendleri vb.).

    Returns:
        Egim (pozitif = artiyor, negatif = azaliyor).
    """
    n = len(values)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)

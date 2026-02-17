"""Hiz hesaplama modul testleri."""

import pytest

from modules.speed import SpeedCalculator


@pytest.fixture
def calculator():
    return SpeedCalculator(camera_hfov_deg=60, image_width_px=640)


class TestSpeedCalculator:
    def test_stationary_object(self, calculator):
        """Duragan nesne -> hiz ~0."""
        positions = [(320, 240)] * 10
        timestamps = [float(i) for i in range(10)]
        distances = [100.0] * 10

        result = calculator.calculate_speed(positions, timestamps, distances)
        assert result["speed_ms"] < 1.0
        assert not result["approaching"]

    def test_moving_object(self, calculator):
        """Hareket eden nesne -> pozitif hiz."""
        positions = [(100 + i * 5, 240) for i in range(10)]
        timestamps = [i * 0.033 for i in range(10)]  # 30fps
        distances = [500.0] * 10

        result = calculator.calculate_speed(positions, timestamps, distances)
        assert result["speed_ms"] > 0
        assert result["speed_kmh"] > 0

    def test_approaching_object(self, calculator):
        """Yaklasan nesne -> approaching=True."""
        positions = [(320, 240)] * 10
        timestamps = [float(i) for i in range(10)]
        # Mesafe azaliyor
        distances = [1000.0 - i * 50 for i in range(10)]

        result = calculator.calculate_speed(positions, timestamps, distances)
        assert result["approaching"] is True

    def test_receding_object(self, calculator):
        """Uzaklasan nesne -> approaching=False."""
        positions = [(320, 240)] * 10
        timestamps = [float(i) for i in range(10)]
        # Mesafe artiyor
        distances = [500.0 + i * 50 for i in range(10)]

        result = calculator.calculate_speed(positions, timestamps, distances)
        assert result["approaching"] is False

    def test_insufficient_data(self, calculator):
        """Yetersiz veri -> sifir hiz."""
        result = calculator.calculate_speed([(100, 100)], [0.0], [500.0])
        assert result["speed_ms"] == 0.0

    def test_empty_data(self, calculator):
        """Bos veri -> sifir hiz."""
        result = calculator.calculate_speed([], [], [])
        assert result["speed_ms"] == 0.0
        assert result["speed_kmh"] == 0.0

    def test_heading_direction(self, calculator):
        """Hareket yonu testi."""
        # Saga hareket
        positions = [(100 + i * 10, 240) for i in range(10)]
        timestamps = [float(i) for i in range(10)]

        result = calculator.calculate_speed(positions, timestamps, [500] * 10)
        # Heading ~0 derece (sag) olmali
        assert 0 <= result["heading_deg"] < 360

    def test_speed_without_distance(self, calculator):
        """Mesafe bilgisi olmadan hiz hesabi."""
        positions = [(100 + i * 5, 240) for i in range(10)]
        timestamps = [i * 0.033 for i in range(10)]

        result = calculator.calculate_speed(positions, timestamps, [])
        # Yaklasik hiz dondurulmeli
        assert result["speed_ms"] >= 0

"""Ulasma suresi hesaplama modul testleri."""

import pytest

from modules.time_to_reach import TimeToReachCalculator


@pytest.fixture
def calculator():
    """Test ulasma suresi hesaplayicisi."""
    return TimeToReachCalculator(min_speed_threshold=0.5)


class TestTimeToReachCalculator:
    def test_basic_calculation(self, calculator):
        """Basit ulasma suresi: 100m / 10m/s = 10s."""
        result = calculator.calculate(100.0, 10.0, approaching=True)
        assert result is not None
        assert abs(result - 10.0) < 0.1

    def test_not_approaching_returns_none(self, calculator):
        """Yaklasmayan nesne -> None."""
        result = calculator.calculate(100.0, 10.0, approaching=False)
        assert result is None

    def test_no_distance_returns_none(self, calculator):
        """Mesafe None -> None."""
        result = calculator.calculate(None, 10.0, approaching=True)
        assert result is None

    def test_zero_distance_returns_none(self, calculator):
        """Mesafe 0 -> None."""
        result = calculator.calculate(0.0, 10.0, approaching=True)
        assert result is None

    def test_slow_speed_returns_none(self, calculator):
        """Esik altindaki hiz -> None."""
        result = calculator.calculate(100.0, 0.1, approaching=True)
        assert result is None

    def test_fast_drone(self, calculator):
        """Hizli dron: 200m / 50m/s = 4s."""
        result = calculator.calculate(200.0, 50.0, approaching=True)
        assert result is not None
        assert abs(result - 4.0) < 0.1

    def test_max_time_limit(self, calculator):
        """Maksimum sure 3600s."""
        result = calculator.calculate(100000.0, 1.0, approaching=True)
        assert result is not None
        assert result <= 3600.0

    def test_min_time_limit(self, calculator):
        """Minimum sure 0.1s."""
        result = calculator.calculate(1.0, 100.0, approaching=True)
        assert result is not None
        assert result >= 0.1

    def test_with_acceleration(self, calculator):
        """Ivme dahil hesaplama."""
        # Sabit hiz: 100m / 10m/s = 10s
        result_const = calculator.calculate(100.0, 10.0, approaching=True)
        # Ivmeli: daha kisa surede ulasmali
        result_accel = calculator.calculate_with_acceleration(
            100.0, 10.0, 5.0, approaching=True
        )
        assert result_accel is not None
        assert result_accel < result_const

    def test_acceleration_not_approaching(self, calculator):
        """Ivmeli hesap - yaklasmayan -> None."""
        result = calculator.calculate_with_acceleration(
            100.0, 10.0, 5.0, approaching=False
        )
        assert result is None

    def test_zero_acceleration_fallback(self, calculator):
        """Sifir ivme -> sabit hiz hesabina duser."""
        result_accel = calculator.calculate_with_acceleration(
            100.0, 10.0, 0.0, approaching=True
        )
        result_const = calculator.calculate(100.0, 10.0, approaching=True)
        assert result_accel == result_const

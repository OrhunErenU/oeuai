"""Irtifa tahmini modul testleri."""

import pytest

from modules.altitude import AltitudeEstimator


@pytest.fixture
def estimator():
    """Test irtifa tahmincisi."""
    return AltitudeEstimator(
        camera_vfov_deg=34,
        camera_tilt_deg=0,
        camera_height_m=2.5,
        image_height_px=640,
    )


class TestAltitudeEstimator:
    def test_above_center_positive_altitude(self, estimator):
        """Goruntu merkezinin ustundeki nesne -> pozitif irtifa."""
        # y=100 -> merkezin ustu (320'den kucuk)
        alt = estimator.estimate(100.0, 500.0)
        assert alt is not None
        assert alt > 0

    def test_below_center_low_altitude(self, estimator):
        """Goruntu merkezinin altindaki nesne -> dusuk/sifir irtifa."""
        # y=600 -> merkezin alti
        alt = estimator.estimate(600.0, 100.0)
        assert alt is not None
        assert alt >= 0  # Negatif olamaz

    def test_center_equals_camera_height(self, estimator):
        """Goruntu ortasindaki nesne -> kamera yuksekligi civarinda."""
        alt = estimator.estimate(320.0, 100.0)
        assert alt is not None
        # Kamera egimi 0 ise, ortadaki nesne kamera yuksekligine yakin olmali
        assert abs(alt - 2.5) < 5.0

    def test_none_distance_returns_none(self, estimator):
        """Mesafe None ise -> None."""
        assert estimator.estimate(100.0, None) is None

    def test_zero_distance_returns_none(self, estimator):
        """Mesafe 0 ise -> None."""
        assert estimator.estimate(100.0, 0.0) is None

    def test_farther_object_higher_altitude(self, estimator):
        """Ayni piksel konumunda uzak nesne -> daha yuksek irtifa."""
        alt_near = estimator.estimate(100.0, 100.0)
        alt_far = estimator.estimate(100.0, 1000.0)
        assert alt_near is not None
        assert alt_far is not None
        assert alt_far > alt_near

    def test_altitude_never_negative(self, estimator):
        """Irtifa asla negatif olmamali."""
        for y in range(0, 640, 50):
            alt = estimator.estimate(float(y), 500.0)
            assert alt is not None
            assert alt >= 0

    def test_relative_altitude(self, estimator):
        """Goreceli irtifa (tanka gore)."""
        rel_alt = estimator.estimate_relative(100.0, 500.0)
        abs_alt = estimator.estimate(100.0, 500.0)
        assert rel_alt is not None
        assert abs_alt is not None
        assert abs(rel_alt - (abs_alt - 2.5)) < 0.01

    def test_tilted_camera(self):
        """Yukari bakan kamera -> daha yuksek irtifalar."""
        est_flat = AltitudeEstimator(
            camera_vfov_deg=34, camera_tilt_deg=0,
            camera_height_m=2.5, image_height_px=640,
        )
        est_up = AltitudeEstimator(
            camera_vfov_deg=34, camera_tilt_deg=15,
            camera_height_m=2.5, image_height_px=640,
        )
        alt_flat = est_flat.estimate(320.0, 500.0)
        alt_up = est_up.estimate(320.0, 500.0)
        assert alt_up > alt_flat

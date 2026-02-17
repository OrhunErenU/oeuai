"""Mesafe tahmini modul testleri."""

import math

import pytest

from modules.distance import DistanceEstimator


@pytest.fixture
def estimator():
    """Test mesafe tahmincisi."""
    dims = {
        "drone": {"width": 0.5, "height": 0.3},
        "tank": {"width": 3.7, "height": 2.4},
        "human": {"width": 0.5, "height": 1.75},
        "bird": {"width": 0.6, "height": 0.3},
        "tank_models": {
            "t72": {"width": 3.59, "height": 2.23},
            "m1_abrams": {"width": 3.66, "height": 2.44},
        },
    }
    # 60 derece FOV, 640px genislik -> focal ~554px
    focal = DistanceEstimator.focal_from_fov(60, 640)
    return DistanceEstimator(focal_length_px=focal, object_dimensions=dims)


class TestDistanceEstimator:
    def test_basic_distance(self, estimator):
        """Basit mesafe tahmini testi."""
        # Insan 100px yuksekliginde -> belirli bir mesafe
        result = estimator.estimate("human", 30, 100)
        assert result is not None
        assert result > 0

    def test_closer_objects_larger_bbox(self, estimator):
        """Yakin nesneler daha buyuk bbox'a sahip -> daha kisa mesafe."""
        far = estimator.estimate("human", 20, 50)
        near = estimator.estimate("human", 60, 150)
        assert near < far

    def test_known_tank_model_distance(self, estimator):
        """Alt sinif (T-72) ile daha hassas mesafe."""
        generic = estimator.estimate("tank", 200, 150)
        specific = estimator.estimate("tank", 200, 150, sub_class="t72")
        assert generic is not None
        assert specific is not None
        # Her ikisi de makul mesafe olmali
        assert 1 < generic < 20000
        assert 1 < specific < 20000

    def test_zero_bbox_returns_none(self, estimator):
        """Sifir boyutlu bbox -> None."""
        assert estimator.estimate("human", 0, 0) is None
        assert estimator.estimate("human", 100, 0) is None

    def test_unknown_class_returns_none(self, estimator):
        """Bilinmeyen sinif -> None."""
        assert estimator.estimate("unknown_thing", 100, 100) is None

    def test_distance_range(self, estimator):
        """Mesafe 1m-20km araliginda olmali."""
        result = estimator.estimate("drone", 500, 300)
        assert result is None or (1 <= result <= 20000)

    def test_calibrate_focal_length(self):
        """Odak uzakligi kalibrasyon testi."""
        # Bilinen: 1.75m insan, 10m mesafe, 97px yukseklik
        focal = DistanceEstimator.calibrate_focal_length(10.0, 1.75, 97.0)
        assert focal > 0
        # focal = 97 * 10 / 1.75 = 554.28
        assert abs(focal - 554.28) < 1.0

    def test_focal_from_fov(self):
        """FOV'dan odak uzakligi hesabi."""
        # 60 derece FOV, 640px
        focal = DistanceEstimator.focal_from_fov(60, 640)
        expected = 640 / (2 * math.tan(math.radians(30)))
        assert abs(focal - expected) < 0.01

    def test_none_focal_length(self):
        """Odak uzakligi None -> her zaman None."""
        est = DistanceEstimator(focal_length_px=None, object_dimensions={"human": {"height": 1.75}})
        assert est.estimate("human", 100, 100) is None

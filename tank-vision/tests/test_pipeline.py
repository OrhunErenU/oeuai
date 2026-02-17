"""Pipeline entegrasyon testleri."""

import pytest

from inference.result import DetectionResult, FrameResult
from modules.altitude import AltitudeEstimator
from modules.time_to_reach import TimeToReachCalculator


class TestTimeToReach:
    def test_approaching_object(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate(1000.0, 50.0, approaching=True)
        assert ttr is not None
        assert abs(ttr - 20.0) < 0.5  # 1000m / 50m/s = 20s

    def test_not_approaching(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate(1000.0, 50.0, approaching=False)
        assert ttr is None

    def test_zero_speed(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate(1000.0, 0.0, approaching=True)
        assert ttr is None

    def test_no_distance(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate(None, 50.0, approaching=True)
        assert ttr is None

    def test_very_close(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate(5.0, 10.0, approaching=True)
        assert ttr is not None
        assert ttr < 1.0

    def test_with_acceleration(self):
        calc = TimeToReachCalculator()
        ttr = calc.calculate_with_acceleration(
            distance_m=500.0,
            speed_ms=20.0,
            acceleration_ms2=5.0,
            approaching=True,
        )
        assert ttr is not None
        assert ttr > 0


class TestAltitudeEstimator:
    @pytest.fixture
    def estimator(self):
        return AltitudeEstimator(
            camera_vfov_deg=34.0,
            camera_tilt_deg=0.0,
            camera_height_m=2.5,
            image_height_px=640,
        )

    def test_object_above(self, estimator):
        """Goruntunun ust yarisindaki nesne -> pozitif irtifa."""
        # y=100 (ortanin ustunde)
        alt = estimator.estimate(100.0, 1000.0)
        assert alt is not None
        assert alt > 2.5  # Kameradan daha yuksek

    def test_object_at_center(self, estimator):
        """Goruntu merkezindeki nesne -> kamera yuksekligine yakin."""
        alt = estimator.estimate(320.0, 100.0)
        assert alt is not None
        assert abs(alt - 2.5) < 10  # Kamera yuksekligine yakin

    def test_no_distance(self, estimator):
        """Mesafe yok -> None."""
        assert estimator.estimate(100.0, None) is None

    def test_negative_altitude_clamped(self, estimator):
        """Negatif irtifa -> 0."""
        # Cok asagidaki nesne
        alt = estimator.estimate(600.0, 50.0)
        assert alt is not None
        assert alt >= 0

    def test_relative_altitude(self, estimator):
        """Goreceli irtifa testi."""
        rel = estimator.estimate_relative(100.0, 1000.0)
        assert rel is not None
        # Yukaridaki nesne icin goreceli irtifa pozitif olmali


class TestFrameResultProperties:
    def test_critical_threats(self):
        dets = [
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=4, class_name="drone"),
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=2, class_name="vehicle"),
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=0, class_name="bird"),
        ]
        fr = FrameResult(detections=dets)
        assert len(fr.critical_threats) == 1
        assert fr.critical_threats[0].class_name == "drone"

    def test_high_threats(self):
        dets = [
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=4, class_name="drone"),
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=3, class_name="tank"),
            DetectionResult(bbox=(0, 0, 10, 10), threat_level=1, class_name="vehicle"),
        ]
        fr = FrameResult(detections=dets)
        assert len(fr.high_threats) == 2

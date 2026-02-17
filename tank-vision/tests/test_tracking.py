"""Nesne takip modul testleri."""

import time

import pytest

from inference.result import DetectionResult
from modules.tracking import TrackManager


@pytest.fixture
def tracker():
    return TrackManager(buffer_size=30, max_lost_frames=10)


class TestTrackManager:
    def test_update_single_detection(self, tracker):
        """Tek tespit takip gecmisine eklenmeli."""
        det = DetectionResult(
            bbox=(100, 100, 200, 200),
            track_id=1,
            center=(150.0, 150.0),
        )
        tracker.update([det])
        assert tracker.get_track_length(1) == 1

    def test_multi_frame_history(self, tracker):
        """Birden fazla frame'de pozisyon gecmisi buyumeli."""
        for i in range(5):
            det = DetectionResult(
                bbox=(100 + i, 100, 200 + i, 200),
                track_id=1,
                center=(150.0 + i, 150.0),
            )
            tracker.update([det])

        positions, timestamps = tracker.get_history(1)
        assert len(positions) == 5

    def test_velocity_calculation(self, tracker):
        """Piksel hizi hesaplanabilmeli."""
        for i in range(10):
            det = DetectionResult(
                bbox=(100 + i * 10, 100, 200 + i * 10, 200),
                track_id=1,
                center=(150.0 + i * 10, 150.0),
            )
            tracker.update([det])
            time.sleep(0.01)

        vx, vy = tracker.get_velocity_pixels(1, n_frames=5)
        # x yonunde hareket var
        assert vx > 0
        # y yonunde hareket yok
        assert abs(vy) < abs(vx)

    def test_multiple_tracks(self, tracker):
        """Birden fazla track es zamanli izlenebilmeli."""
        dets = [
            DetectionResult(bbox=(100, 100, 200, 200), track_id=1, center=(150, 150)),
            DetectionResult(bbox=(300, 300, 400, 400), track_id=2, center=(350, 350)),
        ]
        tracker.update(dets)

        assert tracker.get_track_length(1) == 1
        assert tracker.get_track_length(2) == 1

    def test_lost_track_cleanup(self, tracker):
        """Gorunmeyen track'ler temizlenmeli."""
        det = DetectionResult(
            bbox=(100, 100, 200, 200), track_id=1, center=(150, 150)
        )
        tracker.update([det])
        assert tracker.is_active(1)

        # 15 bos frame (max_lost_frames=10)
        for _ in range(15):
            tracker.update([])

        assert not tracker.is_active(1)

    def test_distance_history(self, tracker):
        """Mesafe gecmisi saklanmali."""
        for i in range(5):
            det = DetectionResult(
                bbox=(100, 100, 200, 200),
                track_id=1,
                center=(150, 150),
                distance_m=1000.0 - i * 50,
            )
            tracker.update([det])

        dist_history = tracker.get_distance_history(1)
        assert len(dist_history) == 5
        assert dist_history[0] > dist_history[-1]

    def test_none_track_id_ignored(self, tracker):
        """track_id=None olan tespitler takip edilmemeli."""
        det = DetectionResult(
            bbox=(100, 100, 200, 200), track_id=None, center=(150, 150)
        )
        tracker.update([det])
        assert tracker.get_track_length(None) == 0

    def test_reset(self, tracker):
        """Reset tum verileri temizlemeli."""
        det = DetectionResult(
            bbox=(100, 100, 200, 200), track_id=1, center=(150, 150)
        )
        tracker.update([det])
        tracker.reset()
        assert tracker.get_track_length(1) == 0

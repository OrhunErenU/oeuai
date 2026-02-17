"""Nesne takip yonetici modulu.

Her track ID icin pozisyon ve zaman gecmisini tutar.
Hiz ve yörünge hesaplamalari icin veri saglar.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from inference.result import DetectionResult


class TrackManager:
    """Nesne takip gecmisi yoneticisi."""

    def __init__(self, buffer_size: int = 90, max_lost_frames: int = 60):
        """
        Args:
            buffer_size: Her track icin saklanacak pozisyon sayisi.
            max_lost_frames: Gorulmeyen track'lerin temizlenmeden once bekleme suresi.
        """
        self.buffer_size = buffer_size
        self.max_lost_frames = max_lost_frames

        # track_id -> deque of (cx, cy)
        self.positions: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        # track_id -> deque of float (timestamp)
        self.timestamps: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        # track_id -> deque of float (mesafe)
        self.distances: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        # track_id -> son gorulen frame
        self._last_seen: dict[int, int] = {}
        self._frame_count = 0

    def update(self, detections: list[DetectionResult]):
        """Yeni frame tespitleri ile takip gecmisini guncelle.

        Args:
            detections: Bu frame'deki DetectionResult listesi.
        """
        self._frame_count += 1
        current_time = time.time()
        seen_ids = set()

        for det in detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            self.positions[tid].append(det.center)
            self.timestamps[tid].append(current_time)
            if det.distance_m is not None:
                self.distances[tid].append(det.distance_m)
            self._last_seen[tid] = self._frame_count
            seen_ids.add(tid)

        # Eski track'leri temizle
        lost_ids = [
            tid
            for tid, last in self._last_seen.items()
            if self._frame_count - last > self.max_lost_frames
        ]
        for tid in lost_ids:
            self._remove_track(tid)

    def get_history(self, track_id: int) -> tuple[list, list]:
        """Bir track'in pozisyon ve zaman gecmisini dondur.

        Returns:
            (positions, timestamps): Pozisyon listesi ve zaman damgalari.
        """
        return list(self.positions[track_id]), list(self.timestamps[track_id])

    def get_distance_history(self, track_id: int) -> list[float]:
        """Bir track'in mesafe gecmisini dondur."""
        return list(self.distances[track_id])

    def get_velocity_pixels(
        self, track_id: int, n_frames: int = 5
    ) -> tuple[float, float]:
        """Son n frame uzerinden piksel hizini hesapla.

        Returns:
            (vx, vy): Piksel/saniye cinsinden hiz.
        """
        positions = list(self.positions[track_id])
        times = list(self.timestamps[track_id])

        if len(positions) < 2:
            return 0.0, 0.0

        n = min(n_frames, len(positions))
        recent_pos = positions[-n:]
        recent_times = times[-n:]

        dt = recent_times[-1] - recent_times[0]
        if dt < 1e-6:
            return 0.0, 0.0

        dx = recent_pos[-1][0] - recent_pos[0][0]
        dy = recent_pos[-1][1] - recent_pos[0][1]

        return dx / dt, dy / dt

    def get_track_length(self, track_id: int) -> int:
        """Bir track'in gecmis boyutunu dondur."""
        return len(self.positions[track_id])

    def is_active(self, track_id: int) -> bool:
        """Track hala aktif mi?"""
        if track_id not in self._last_seen:
            return False
        return self._frame_count - self._last_seen[track_id] <= self.max_lost_frames

    def _remove_track(self, track_id: int):
        """Track'i tamamen kaldir."""
        self.positions.pop(track_id, None)
        self.timestamps.pop(track_id, None)
        self.distances.pop(track_id, None)
        self._last_seen.pop(track_id, None)

    def reset(self):
        """Tum track gecmisini sifirla."""
        self.positions.clear()
        self.timestamps.clear()
        self.distances.clear()
        self._last_seen.clear()
        self._frame_count = 0

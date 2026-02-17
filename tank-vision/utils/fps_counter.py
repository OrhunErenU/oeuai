"""FPS (Frame Per Second) sayaci.

Rolling average ile kararli FPS olcumu yapar.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Rolling average FPS sayaci."""

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Ortalama pencere boyutu (frame sayisi).
        """
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._last_time = time.time()

    def tick(self):
        """Yeni frame islendi olarak isaretle."""
        now = time.time()
        self._timestamps.append(now)
        self._last_time = now

    def get(self) -> float:
        """Mevcut FPS degerini dondur."""
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt

    def get_ms(self) -> float:
        """Son frame islem suresi (milisaniye)."""
        if len(self._timestamps) < 2:
            return 0.0
        return (self._timestamps[-1] - self._timestamps[-2]) * 1000

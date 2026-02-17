"""Thread'li video stream handler.

Non-blocking frame okuma ile ana islem dongusu gecikme
yasamadan frame alabilir.
"""

from __future__ import annotations

from queue import Empty, Queue
from threading import Thread

import cv2


class VideoStream:
    """Thread'li video kaynak okuyucu."""

    def __init__(self, source, queue_size: int = 3):
        """
        Args:
            source: Video kaynagi (dosya yolu, RTSP URL, kamera indeksi).
            queue_size: Frame kuyrugu boyutu.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def start(self) -> "VideoStream":
        """Arka plan frame okuma thread'ini baslat."""
        thread = Thread(target=self._read_frames, daemon=True)
        thread.start()
        return self

    def _read_frames(self):
        """Arka planda frame oku ve kuyruga ekle."""
        while not self.stopped:
            if self.queue.full():
                # Eski frame'i at, yenisini koy (gercek zamanli icin)
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break

            self.queue.put(frame)

    def read(self, timeout: float = 2.0):
        """Kuyruktan frame oku.

        Args:
            timeout: Bekleme suresi (saniye).

        Returns:
            Frame (numpy array) veya None (stream bittiyse).
        """
        if self.stopped and self.queue.empty():
            return None
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        """Stream'i durdur."""
        self.stopped = True
        self.cap.release()

    @property
    def is_alive(self) -> bool:
        return not self.stopped or not self.queue.empty()

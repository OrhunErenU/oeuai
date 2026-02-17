"""Bounding box yardimci fonksiyonlari.

YOLO ve diger formatlar arasi donusum, kirpma, normalizasyon islemleri.
"""

from __future__ import annotations

import numpy as np


def xyxy_to_xywh(bbox: tuple) -> tuple:
    """(x1,y1,x2,y2) -> (x_center, y_center, width, height)"""
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1,
    )


def xywh_to_xyxy(bbox: tuple) -> tuple:
    """(x_center, y_center, width, height) -> (x1,y1,x2,y2)"""
    cx, cy, w, h = bbox
    return (
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    )


def normalize_bbox(bbox: tuple, img_width: int, img_height: int) -> tuple:
    """Piksel bbox'i normalize et (0-1 araligi).

    Girdi: (x1, y1, x2, y2) piksel koordinatlari
    Cikti: (x_center, y_center, norm_w, norm_h) normalize YOLO formati
    """
    x1, y1, x2, y2 = bbox
    return (
        ((x1 + x2) / 2) / img_width,
        ((y1 + y2) / 2) / img_height,
        (x2 - x1) / img_width,
        (y2 - y1) / img_height,
    )


def denormalize_bbox(bbox: tuple, img_width: int, img_height: int) -> tuple:
    """Normalize bbox'i piksel koordinatlarina donustur.

    Girdi: (x_center, y_center, norm_w, norm_h) YOLO formati
    Cikti: (x1, y1, x2, y2) piksel koordinatlari
    """
    cx, cy, nw, nh = bbox
    w = nw * img_width
    h = nh * img_height
    x1 = cx * img_width - w / 2
    y1 = cy * img_height - h / 2
    return (x1, y1, x1 + w, y1 + h)


def crop_image(image: np.ndarray, bbox: tuple, padding: float = 0.0) -> np.ndarray:
    """Goruntuden bbox bolgesini kirp.

    Args:
        image: BGR goruntu (numpy array).
        bbox: (x1, y1, x2, y2) piksel koordinatlari.
        padding: Bbox cevresine eklenen dolgu orani (0.0-1.0).

    Returns:
        Kirpilmis goruntu.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if padding > 0:
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * padding
        pad_y = bh * padding
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return image[y1:y2, x1:x2].copy()


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """Bbox merkezini hesapla. Girdi: (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_iou(box1: tuple, box2: tuple) -> float:
    """Iki bbox arasindaki IoU (Intersection over Union) hesapla.

    Girdi: (x1, y1, x2, y2) formati.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union

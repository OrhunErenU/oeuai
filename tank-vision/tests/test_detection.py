"""Tespit modulu testleri."""

import pytest

from inference.result import DetectionResult, FrameResult
from utils.bbox_utils import (
    bbox_center,
    bbox_iou,
    crop_image,
    denormalize_bbox,
    normalize_bbox,
    xyxy_to_xywh,
    xywh_to_xyxy,
)


class TestBboxUtils:
    def test_xyxy_to_xywh(self):
        result = xyxy_to_xywh((100, 200, 300, 400))
        assert result == (200, 300, 200, 200)

    def test_xywh_to_xyxy(self):
        result = xywh_to_xyxy((200, 300, 200, 200))
        assert result == (100, 200, 300, 400)

    def test_roundtrip_conversion(self):
        bbox = (50.0, 100.0, 250.0, 350.0)
        xywh = xyxy_to_xywh(bbox)
        back = xywh_to_xyxy(xywh)
        for a, b in zip(bbox, back):
            assert abs(a - b) < 0.001

    def test_normalize_denormalize(self):
        bbox = (100, 200, 300, 400)
        norm = normalize_bbox(bbox, 640, 480)
        denorm = denormalize_bbox(norm, 640, 480)
        for a, b in zip(bbox, denorm):
            assert abs(a - b) < 0.5

    def test_bbox_center(self):
        result = bbox_center((100, 200, 300, 400))
        assert result == (200, 300)

    def test_bbox_iou_perfect_overlap(self):
        box = (100, 100, 200, 200)
        assert abs(bbox_iou(box, box) - 1.0) < 0.001

    def test_bbox_iou_no_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)
        assert bbox_iou(box1, box2) == 0.0

    def test_bbox_iou_partial_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = bbox_iou(box1, box2)
        assert 0 < iou < 1
        # Hesaplama: kesisim = 50*50=2500, birlesim = 10000+10000-2500=17500
        expected = 2500 / 17500
        assert abs(iou - expected) < 0.001

    def test_crop_image(self, sample_frame):
        crop = crop_image(sample_frame, (100, 100, 300, 300))
        assert crop.shape == (200, 200, 3)

    def test_crop_image_with_padding(self, sample_frame):
        crop = crop_image(sample_frame, (100, 100, 300, 300), padding=0.1)
        # Padding ile daha buyuk olmali
        assert crop.shape[0] >= 200
        assert crop.shape[1] >= 200

    def test_crop_image_edge_case(self, sample_frame):
        """Goruntu sinirini asan bbox."""
        crop = crop_image(sample_frame, (600, 600, 700, 700))
        assert crop.size > 0


class TestDetectionResult:
    def test_bbox_dimensions(self, sample_detection_drone):
        assert sample_detection_drone.bbox_width == 50.0
        assert sample_detection_drone.bbox_height == 40.0

    def test_summary(self, sample_detection_drone):
        s = sample_detection_drone.summary()
        assert "drone" in s
        assert "#1" in s  # track_id

    def test_default_values(self):
        det = DetectionResult(bbox=(0, 0, 10, 10))
        assert det.class_name == ""
        assert det.threat_level == 0
        assert det.distance_m is None
        assert det.weapon_type is None


class TestFrameResult:
    def test_num_detections(self, sample_frame_result):
        assert sample_frame_result.num_detections == 3

    def test_sorted_by_threat(self, sample_frame_result):
        # Once tehdit degerlendir
        from modules.threat import ThreatAssessor
        config = {
            "drone_danger_distance": 500,
            "tank_danger_distance": 2000,
            "human_danger_distance": 200,
            "weapon_danger_distance": 300,
            "vehicle_danger_distance": 1000,
            "aircraft_danger_distance": 3000,
            "bird_danger_distance": 0,
        }
        assessor = ThreatAssessor(config)
        for det in sample_frame_result.detections:
            result = assessor.assess(det)
            det.priority_score = result["priority_score"]

        sorted_dets = sample_frame_result.sorted_by_threat()
        # En yuksek puan once
        for i in range(len(sorted_dets) - 1):
            assert sorted_dets[i].priority_score >= sorted_dets[i + 1].priority_score

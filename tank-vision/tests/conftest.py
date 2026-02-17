"""Ortak test fixture'lari."""

import numpy as np
import pytest

from inference.result import DetectionResult, FrameResult


@pytest.fixture
def sample_frame():
    """640x640 ornek BGR goruntu."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection_drone():
    """Ornek dron tespiti."""
    return DetectionResult(
        bbox=(100.0, 50.0, 150.0, 90.0),
        class_id=0,
        class_name="drone",
        confidence=0.85,
        track_id=1,
        center=(125.0, 70.0),
        foe_status="foe",
        distance_m=300.0,
        speed_ms=15.0,
        speed_kmh=54.0,
        approaching=True,
        altitude_m=120.0,
        time_to_reach=20.0,
    )


@pytest.fixture
def sample_detection_tank():
    """Ornek dusman tank tespiti."""
    return DetectionResult(
        bbox=(200.0, 300.0, 500.0, 500.0),
        class_id=1,
        class_name="tank",
        confidence=0.92,
        track_id=2,
        center=(350.0, 400.0),
        tank_model="t72",
        tank_conf=0.88,
        foe_status="foe",
        foe_conf=0.85,
        distance_m=1500.0,
        speed_ms=8.0,
        speed_kmh=28.8,
        approaching=True,
        is_targeting_us=True,
        turret_angle=5.0,
    )


@pytest.fixture
def sample_detection_human_soldier():
    """Ornek asker tespiti."""
    return DetectionResult(
        bbox=(300.0, 200.0, 340.0, 350.0),
        class_id=2,
        class_name="human",
        confidence=0.78,
        track_id=3,
        center=(320.0, 275.0),
        human_type="soldier",
        human_conf=0.82,
        distance_m=150.0,
    )


@pytest.fixture
def sample_detection_bird():
    """Ornek kus tespiti."""
    return DetectionResult(
        bbox=(400.0, 30.0, 420.0, 50.0),
        class_id=6,
        class_name="bird",
        confidence=0.65,
        track_id=4,
        center=(410.0, 40.0),
    )


@pytest.fixture
def sample_detection_weapon():
    """Ornek silah tespiti."""
    return DetectionResult(
        bbox=(310.0, 270.0, 370.0, 290.0),
        class_id=3,
        class_name="weapon",
        confidence=0.80,
        weapon_type="rpg",
        weapon_conf=0.75,
        distance_m=150.0,
    )


@pytest.fixture
def sample_frame_result(
    sample_detection_drone,
    sample_detection_tank,
    sample_detection_bird,
):
    """Ornek FrameResult."""
    return FrameResult(
        frame_id=100,
        timestamp=1000.0,
        detections=[
            sample_detection_drone,
            sample_detection_tank,
            sample_detection_bird,
        ],
        fps=30.0,
        processing_time_ms=33.3,
    )


@pytest.fixture
def default_config():
    """Varsayilan test konfigurasyonu."""
    return {
        "system": {
            "device": "cpu",
            "fp16": False,
            "imgsz": 640,
            "conf_thresh": 0.25,
            "iou_thresh": 0.45,
            "max_det": 100,
        },
        "threat": {
            "drone_danger_distance": 500,
            "tank_danger_distance": 2000,
            "human_danger_distance": 200,
            "weapon_danger_distance": 300,
            "vehicle_danger_distance": 1000,
            "aircraft_danger_distance": 3000,
            "bird_danger_distance": 0,
            "speed_threat_multiplier": 1.5,
        },
        "distance": {
            "camera_hfov": 60,
            "camera_vfov": 34,
            "camera_tilt": 0,
            "camera_height": 2.5,
        },
    }

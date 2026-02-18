"""Tehdit degerlendirme modul testleri."""

import pytest

from inference.result import DetectionResult
from modules.threat import ThreatAssessor


@pytest.fixture
def assessor(default_config):
    return ThreatAssessor(default_config["threat"])


class TestThreatAssessor:
    def test_bird_is_no_threat(self, assessor, sample_detection_bird):
        """Kus her zaman tehdit degil."""
        result = assessor.assess(sample_detection_bird)
        assert result["threat_level"] == 0
        assert result["threat_label"] == "none"

    def test_close_enemy_tank_targeting_us(self, assessor):
        """Yakin dusman tank bize hedef aliyor -> CRITICAL."""
        det = DetectionResult(
            bbox=(200, 300, 500, 500),
            class_id=1,
            class_name="tank",
            confidence=0.92,
            foe_status="foe",
            distance_m=400.0,  # 2000m tehlike mesafesinin %20'si
            speed_ms=10.0,
            speed_kmh=36.0,
            approaching=True,
            is_targeting_us=True,
            tank_model="t72",
        )
        result = assessor.assess(det)
        assert result["threat_level"] == 4  # CRITICAL
        assert result["threat_label"] == "critical"
        assert result["priority_score"] >= 80

    def test_friendly_tank(self, assessor):
        """Dost tank -> dusuk tehdit."""
        det = DetectionResult(
            bbox=(200, 300, 500, 500),
            class_id=1,
            class_name="tank",
            confidence=0.90,
            foe_status="friend",
            distance_m=500.0,
        )
        result = assessor.assess(det)
        # Dost tank dusuk veya sifir tehdit olmali
        assert result["threat_level"] <= 1

    def test_approaching_drone(self, assessor):
        """Yaklasan dusman dron -> yuksek tehdit."""
        det = DetectionResult(
            bbox=(100, 50, 150, 90),
            class_id=0,
            class_name="drone",
            confidence=0.85,
            foe_status="foe",
            distance_m=100.0,
            speed_ms=20.0,
            speed_kmh=72.0,
            approaching=True,
            time_to_reach=5.0,
        )
        result = assessor.assess(det)
        assert result["threat_level"] >= 3  # HIGH veya CRITICAL

    def test_rpg_weapon(self, assessor):
        """RPG tespiti -> yuksek tehdit."""
        det = DetectionResult(
            bbox=(310, 270, 370, 290),
            class_id=3,
            class_name="weapon",
            confidence=0.80,
            weapon_type="rpg",
            distance_m=100.0,
        )
        result = assessor.assess(det)
        assert result["threat_level"] >= 2  # MEDIUM+
        assert any("RPG" in r for r in result["reasons"])

    def test_distant_civilian(self, assessor):
        """Uzaktaki sivil -> dusuk/sifir tehdit."""
        det = DetectionResult(
            bbox=(300, 200, 340, 350),
            class_id=2,
            class_name="human",
            confidence=0.78,
            human_type="civilian",
            distance_m=500.0,
        )
        result = assessor.assess(det)
        assert result["threat_level"] <= 1

    def test_close_soldier(self, assessor):
        """Yakin asker -> orta tehdit."""
        det = DetectionResult(
            bbox=(300, 200, 340, 350),
            class_id=2,
            class_name="human",
            confidence=0.78,
            human_type="soldier",
            distance_m=50.0,  # 200m tehlike mesafesinin %25'i
            foe_status="unknown",
        )
        result = assessor.assess(det)
        assert result["threat_level"] >= 2  # MEDIUM+

    def test_unknown_identity(self, assessor):
        """Kimlik bilinmiyor -> orta seviye skor eklenir."""
        det = DetectionResult(
            bbox=(200, 300, 500, 500),
            class_id=4,
            class_name="vehicle",
            confidence=0.75,
            foe_status="unknown",
            distance_m=800.0,
        )
        result = assessor.assess(det)
        assert result["priority_score"] >= 20  # "unknown" = +20

    def test_assess_frame(self, assessor, sample_detection_drone, sample_detection_bird):
        """Frame degerlendirme - tum tespitleri isler."""
        detections = [sample_detection_drone, sample_detection_bird]
        result = assessor.assess_frame(detections)

        assert len(result) == 2
        assert result[0].threat_label != ""
        assert result[1].threat_label == "none"

    def test_score_bounds(self, assessor):
        """Puan her zaman >= 0."""
        det = DetectionResult(
            bbox=(100, 100, 200, 200),
            class_id=4,
            class_name="vehicle",
            confidence=0.5,
            foe_status="friend",
        )
        result = assessor.assess(det)
        assert result["priority_score"] >= 0

    def test_drone_low_altitude_fpv_risk(self, assessor):
        """Alcak ucuslu dron -> FPV/kamikaze riski."""
        det = DetectionResult(
            bbox=(100, 50, 150, 90),
            class_id=0,
            class_name="drone",
            confidence=0.85,
            altitude_m=10.0,
            distance_m=200.0,
        )
        result = assessor.assess(det)
        assert any("alcak" in r.lower() or "fpv" in r.lower() for r in result["reasons"])

    def test_drone_high_speed_approaching(self, assessor):
        """Yuksek hizli yaklasan dron -> ekstra tehdit."""
        det = DetectionResult(
            bbox=(100, 50, 150, 90),
            class_id=0,
            class_name="drone",
            confidence=0.85,
            foe_status="foe",
            speed_ms=25.0,
            speed_kmh=90.0,
            approaching=True,
            distance_m=300.0,
        )
        result = assessor.assess(det)
        assert result["threat_level"] >= 3

    def test_machine_gun_weapon(self, assessor):
        """Makineli tufek -> ek tehdit puani."""
        det = DetectionResult(
            bbox=(310, 270, 370, 290),
            class_id=3,
            class_name="weapon",
            confidence=0.80,
            weapon_type="machine_gun",
            distance_m=100.0,
        )
        result = assessor.assess(det)
        assert any("Makineli" in r for r in result["reasons"])

    def test_approaching_enemy_vehicle(self, assessor):
        """Yaklasan dusman araci -> tehdit."""
        det = DetectionResult(
            bbox=(100, 200, 300, 350),
            class_id=4,
            class_name="vehicle",
            confidence=0.75,
            foe_status="foe",
            approaching=True,
            speed_ms=15.0,
            speed_kmh=54.0,
            distance_m=500.0,
        )
        result = assessor.assess(det)
        assert result["threat_level"] >= 2

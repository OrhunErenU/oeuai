"""Tehdit degerlendirme motoru.

Tum analiz modullerinden gelen verileri (tespit, alt-siniflandirma,
mesafe, hiz, yorunge, dost/dusman, taret yonu) birlestirerek
her nesne icin tehdit seviyesi ve oncelik puani hesaplar.

Tehdit seviyeleri:
  0 = NONE     (0-9 puan)
  1 = LOW      (10-29 puan)
  2 = MEDIUM   (30-54 puan)
  3 = HIGH     (55-79 puan)
  4 = CRITICAL (80+ puan)
"""

from __future__ import annotations

from data.class_mapping import THREAT_LEVELS
from inference.result import DetectionResult


class ThreatAssessor:
    """Cok faktorlu tehdit degerlendirme motoru."""

    def __init__(self, config: dict):
        """
        Args:
            config: Tehdit konfigurasyonu (default.yaml "threat" bolumu).
                {
                    "drone_danger_distance": 500,
                    "tank_danger_distance": 2000,
                    ...
                }
        """
        self.danger_distances = {
            "drone": config.get("drone_danger_distance", 500),
            "tank": config.get("tank_danger_distance", 2000),
            "human": config.get("human_danger_distance", 200),
            "weapon": config.get("weapon_danger_distance", 300),
            "vehicle": config.get("vehicle_danger_distance", 1000),
            "aircraft": config.get("aircraft_danger_distance", 3000),
            "bird": config.get("bird_danger_distance", 0),
        }
        self.speed_multiplier = config.get("speed_threat_multiplier", 1.5)

    def assess(self, detection: DetectionResult) -> dict:
        """Tek bir tespit icin tehdit degerlendirmesi yap.

        Args:
            detection: Tum bilgileri doldurulmus DetectionResult.

        Returns:
            {
                "threat_level": int,       # 0-4
                "threat_label": str,       # "none", "low", "medium", "high", "critical"
                "reasons": list[str],      # Insanca okunabilir sebepler
                "priority_score": float,   # Sayisal puan (siralama icin)
            }
        """
        score = 0.0
        reasons = []
        class_name = detection.class_name

        # Kus = tehdit degil
        if class_name == "bird":
            return {
                "threat_level": 0,
                "threat_label": "none",
                "reasons": ["Kus - tehdit degil"],
                "priority_score": 0.0,
            }

        # === Kural 1: Dost/Dusman durumu ===
        if detection.foe_status == "foe":
            score += 50
            reasons.append("DUSMAN olarak tanimlandi")
        elif detection.foe_status == "unknown":
            score += 20
            reasons.append("Kimlik bilinmiyor")
        elif detection.foe_status == "friend":
            score -= 30  # Dost = tehdit azaltici
            reasons.append("DOST olarak tanimlandi")

        # === Kural 2: Mesafe tabanli tehdit ===
        danger_dist = self.danger_distances.get(class_name, 1000)
        if detection.distance_m is not None and danger_dist > 0:
            dist = detection.distance_m
            if dist < danger_dist * 0.25:
                score += 40
                reasons.append(f"Cok yakin: {dist:.0f}m")
            elif dist < danger_dist * 0.5:
                score += 25
                reasons.append(f"Yakin: {dist:.0f}m")
            elif dist < danger_dist:
                score += 10
                reasons.append(f"Tehlike menziline girdi: {dist:.0f}m")

        # === Kural 3: Yaklasma hizi ===
        if detection.approaching and detection.speed_ms > 1.0:
            speed_score = min(30, detection.speed_ms * 2)
            score += speed_score
            reasons.append(f"Yaklasma hizi: {detection.speed_kmh:.0f} km/h")

        # === Kural 4: Tank taret yonu ===
        if class_name == "tank" and detection.is_targeting_us:
            score += 40
            reasons.append("TARET BIZE HEDEF ALIYOR")

        # === Kural 5: Silah tespiti ===
        if class_name == "weapon":
            score += 30
            weapon_info = detection.weapon_type or "bilinmeyen"
            reasons.append(f"Silah tespit: {weapon_info}")

            # RPG ozellikle tehlikeli
            if detection.weapon_type == "rpg":
                score += 15
                reasons.append("RPG - yuksek tehdit")
            elif detection.weapon_type == "sniper":
                score += 10
                reasons.append("Keskin nisanci - yuksek menzil")
            elif detection.weapon_type == "machine_gun":
                score += 10
                reasons.append("Makineli tufek - yuksek ates gucu")

        # === Kural 6: Asker tespiti ===
        if class_name == "human":
            if detection.human_type == "soldier":
                score += 15
                reasons.append("Asker tespit edildi")
            elif detection.human_type == "civilian":
                score -= 10
                reasons.append("Sivil - dusuk tehdit")

        # === Kural 7: Dron kapsamli tehdit analizi ===
        if class_name == "drone":
            # 7a: Dron ulasma suresi
            if detection.time_to_reach is not None:
                if detection.time_to_reach < 3:
                    score += 45
                    reasons.append(f"ACIL! DRON {detection.time_to_reach:.1f}sn icinde ulasacak!")
                elif detection.time_to_reach < 5:
                    score += 40
                    reasons.append(f"DRON {detection.time_to_reach:.1f}sn icinde ulasacak!")
                elif detection.time_to_reach < 10:
                    score += 35
                    reasons.append(f"Dron {detection.time_to_reach:.1f}sn icinde ulasacak")
                elif detection.time_to_reach < 30:
                    score += 20
                    reasons.append(f"Dron {detection.time_to_reach:.0f}sn icinde ulasacak")

            # 7b: Dron alcak ucus (kamikaze/FPV dron gostergesi)
            if detection.altitude_m is not None:
                if detection.altitude_m < 20:
                    score += 25
                    reasons.append(f"Dron cok alcak: {detection.altitude_m:.0f}m (FPV/kamikaze riski)")
                elif detection.altitude_m < 50:
                    score += 15
                    reasons.append(f"Dron alcak ucuyor: {detection.altitude_m:.0f}m")

            # 7c: Yuksek hizli dron (FPV dron gostergesi: >60km/h)
            if detection.speed_kmh > 60 and detection.approaching:
                score += 20
                reasons.append(f"Yuksek hizli dron: {detection.speed_kmh:.0f}km/h (FPV riski)")

            # 7d: Dron dusman ise ek tehdit
            if detection.foe_status == "foe":
                score += 10
                reasons.append("Dusman dron tespit edildi")

        # === Kural 8: Ucak tehdit ===
        if class_name == "aircraft":
            score += 15  # Ucaklar genelde yuksek tehdit
            reasons.append("Hava araci tespit edildi")
            if detection.time_to_reach is not None and detection.time_to_reach < 30:
                score += 20
                reasons.append(f"Ucak {detection.time_to_reach:.0f}sn icinde ulasacak")

        # === Kural 9: Arac yaklasma tehdidi ===
        if class_name == "vehicle" and detection.approaching:
            if detection.foe_status == "foe":
                score += 15
                reasons.append("Dusman araci yaklasıyor")
            elif detection.foe_status == "unknown" and detection.speed_kmh > 50:
                score += 10
                reasons.append(f"Kimliksiz arac hizla yaklasıyor: {detection.speed_kmh:.0f}km/h")

        # === Kural 10: Dusman tank ===
        if class_name == "tank" and detection.foe_status == "foe":
            score += 10  # Ek bonus - dusman tank her zaman kritik
            if detection.tank_model:
                reasons.append(f"Tank modeli: {detection.tank_model}")

        # Minimum 0
        score = max(0.0, score)

        # Seviye hesapla
        threat_level = self._score_to_level(score)
        threat_label = THREAT_LEVELS.get(threat_level, "none")

        return {
            "threat_level": threat_level,
            "threat_label": threat_label,
            "reasons": reasons,
            "priority_score": score,
        }

    def _score_to_level(self, score: float) -> int:
        """Sayisal puani tehdit seviyesine donustur."""
        if score >= 80:
            return 4  # CRITICAL
        if score >= 55:
            return 3  # HIGH
        if score >= 30:
            return 2  # MEDIUM
        if score >= 10:
            return 1  # LOW
        return 0  # NONE

    def assess_frame(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """Tum frame tespitlerini degerlendir ve sonuclari DetectionResult'a yaz.

        Args:
            detections: Bu frame'deki DetectionResult listesi.

        Returns:
            Ayni liste (threat bilgileri doldurulmus olarak).
        """
        for det in detections:
            result = self.assess(det)
            det.threat_level = result["threat_level"]
            det.threat_label = result["threat_label"]
            det.threat_reasons = result["reasons"]
            det.priority_score = result["priority_score"]

        return detections

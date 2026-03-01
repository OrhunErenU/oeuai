"""Tank Vision AI - Gelismis HUD Arayuzu ve Inference Sistemi.

Ozellikler:
- Gercek zamanli nesne tespiti (YOLOv11n)
- Tank marka/model siniflandirma
- Sivil/Asker ayrimi
- Silah turu tespiti
- Duman/Ates/Patlama tespiti
- Namlu yonu tespiti -> HEDEF uyarisi
- Drone mesafe/hiz/yukseklik tahmini
- Ani hareket / tehdit algilama
- Askeri HUD arayuzu

Kullanim:
    python scripts/tank_vision_hud.py                    # Webcam
    python scripts/tank_vision_hud.py --source video.mp4 # Video
    python scripts/tank_vision_hud.py --source resim.jpg # Resim
"""

import argparse
import sys
import time
import math
import os
from pathlib import Path
from collections import deque

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
_original_imshow = cv2.imshow

from ultralytics import YOLO
import numpy as np

cv2.imshow = _original_imshow

# ============================================================
# SINIF TANIMLARI
# ============================================================

# Ana dedektör siniflari (v1 - mevcut model)
CLASSES_V1 = {
    0: "Drone", 1: "Tank", 2: "Human", 3: "Weapon",
    4: "Vehicle", 5: "Aircraft", 6: "Bird",
}

# Genisletilmis siniflar (v2 - yeni model)
CLASSES_V2 = {
    0: "Drone", 1: "Tank", 2: "Human", 3: "Weapon", 4: "Vehicle",
    5: "Aircraft", 6: "Bird", 7: "Smoke", 8: "Fire", 9: "Explosion",
    10: "Soldier", 11: "Civilian", 12: "Rifle", 13: "Pistol", 14: "Barrel",
}

# Tehdit seviyeleri
THREAT_LEVELS = {
    "Tank": 5, "Weapon": 4, "Rifle": 4, "Pistol": 3, "Soldier": 3,
    "Drone": 3, "Aircraft": 4, "Explosion": 5, "Fire": 4, "Smoke": 2,
    "Barrel": 5, "Vehicle": 2, "Human": 1, "Civilian": 0, "Bird": 0,
}

# Renk paleti (BGR)
COLORS = {
    "Drone": (0, 165, 255),     # Turuncu
    "Tank": (0, 0, 200),        # Kirmizi
    "Human": (200, 200, 0),     # Cyan
    "Weapon": (0, 0, 255),      # Kirmizi
    "Vehicle": (200, 200, 200), # Gri
    "Aircraft": (255, 100, 0),  # Mavi
    "Bird": (0, 200, 0),        # Yesil
    "Smoke": (180, 180, 180),   # Acik gri
    "Fire": (0, 100, 255),      # Turuncu-kirmizi
    "Explosion": (0, 0, 255),   # Kirmizi
    "Soldier": (0, 140, 255),   # Turuncu
    "Civilian": (0, 255, 0),    # Yesil
    "Rifle": (0, 50, 255),      # Kirmizi
    "Pistol": (0, 80, 200),     # Koyu turuncu
    "Barrel": (0, 0, 255),      # Kirmizi
}

# Tank modelleri (alt siniflandirici icin)
TANK_MODELS = [
    "M60", "Altay", "Leopard 2A4", "M1 Abrams", "T-72",
    "T-90", "Merkava", "Challenger 2", "Leclerc", "K2 Black Panther"
]


# ============================================================
# TRACKER - Nesne takibi ve hiz/mesafe tahmini
# ============================================================

class ObjectTracker:
    """Basit nesne takipcisi - IoU tabanli."""

    def __init__(self, max_history=30):
        self.tracks = {}
        self.next_id = 0
        self.max_history = max_history

    def update(self, detections, frame_time):
        """Yeni tespitlerle track'leri guncelle."""
        new_tracks = {}
        used_dets = set()

        # Mevcut track'leri eslestirilmeye calis
        for tid, track in self.tracks.items():
            best_iou = 0.3  # Minimum IoU esigi
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                if det["cls_name"] != track["cls_name"]:
                    continue

                iou = self._calc_iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx >= 0:
                det = detections[best_det_idx]
                used_dets.add(best_det_idx)

                # Pozisyon gecmisini guncelle
                cx = (det["box"][0] + det["box"][2]) / 2
                cy = (det["box"][1] + det["box"][3]) / 2
                track["positions"].append((cx, cy, frame_time))
                if len(track["positions"]) > self.max_history:
                    track["positions"].popleft()

                track["box"] = det["box"]
                track["conf"] = det["conf"]
                track["last_seen"] = frame_time
                track["age"] += 1
                new_tracks[tid] = track
            elif frame_time - track["last_seen"] < 0.5:
                # 0.5 saniye grace period
                new_tracks[tid] = track

        # Eslesmeyen tespitler icin yeni track olustur
        for i, det in enumerate(detections):
            if i not in used_dets:
                cx = (det["box"][0] + det["box"][2]) / 2
                cy = (det["box"][1] + det["box"][3]) / 2
                new_tracks[self.next_id] = {
                    "box": det["box"],
                    "cls_name": det["cls_name"],
                    "conf": det["conf"],
                    "positions": deque([(cx, cy, frame_time)]),
                    "last_seen": frame_time,
                    "age": 0,
                }
                self.next_id += 1

        self.tracks = new_tracks
        return self.tracks

    def get_speed(self, track_id, pixels_per_meter=50):
        """Track'in piksel hizini hesapla (m/s tahmini)."""
        track = self.tracks.get(track_id)
        if not track or len(track["positions"]) < 2:
            return 0

        p1 = track["positions"][-2]
        p2 = track["positions"][-1]
        dt = p2[2] - p1[2]
        if dt <= 0:
            return 0

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pixel_speed = math.sqrt(dx * dx + dy * dy) / dt

        return pixel_speed / pixels_per_meter

    def is_sudden_movement(self, track_id, threshold=5.0):
        """Ani hareket tespiti (m/s)."""
        speed = self.get_speed(track_id)
        return speed > threshold

    @staticmethod
    def _calc_iou(box1, box2):
        """IoU hesapla."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / max(area1 + area2 - inter, 1e-6)


# ============================================================
# DRONE MESAFE/YUKSEKLIK TAHMINI
# ============================================================

class DroneEstimator:
    """Drone mesafe ve yukseklik tahmincisi (bbox boyutuna dayali)."""

    # Tipik drone boyutlari (metre)
    TYPICAL_DRONE_SIZE = 0.4  # DJI Mavic benzeri
    FOCAL_LENGTH = 800  # Yaklasik focal length (piksel)

    @classmethod
    def estimate(cls, box, frame_h, frame_w):
        """Drone bbox'indan mesafe/yukseklik tahmin et."""
        w = box[2] - box[0]
        h = box[3] - box[1]
        bbox_size = max(w, h)

        if bbox_size < 5:
            return {"distance": 999, "height": 999, "size": "?"}

        # Mesafe tahmini (basit pinhole model)
        distance = (cls.TYPICAL_DRONE_SIZE * cls.FOCAL_LENGTH) / bbox_size

        # Yukseklik tahmini (frame'deki dikey pozisyona gore)
        cy = (box[1] + box[3]) / 2
        # Ust kisimda = yuksek, alt kisimda = alcak
        height_ratio = 1.0 - (cy / frame_h)
        height = distance * height_ratio * 0.7  # Kaba tahmin

        size_label = "Buyuk" if bbox_size > 150 else "Orta" if bbox_size > 50 else "Kucuk"

        return {
            "distance": round(distance, 1),
            "height": round(max(height, 0.5), 1),
            "size": size_label,
        }


# ============================================================
# NAMLU YON TESPITI
# ============================================================

def estimate_barrel_direction(tank_box, all_detections, frame_center_x):
    """Tank namlusunun bize donuk olup olmadigini tahmin et.

    Basit yontem: Tank bbox'inin en-boy oranina ve
    frame merkeze gore konumuna bakar.
    """
    x1, y1, x2, y2 = tank_box
    tank_cx = (x1 + x2) / 2
    tank_w = x2 - x1
    tank_h = y2 - y1

    # Barrel tespiti varsa onu kullan
    for det in all_detections:
        if det["cls_name"] == "Barrel":
            bx1, by1, bx2, by2 = det["box"]
            barrel_cx = (bx1 + bx2) / 2
            barrel_cy = (by1 + by2) / 2

            # Barrel tank icinde mi?
            if x1 < barrel_cx < x2 and y1 < barrel_cy < y2:
                # Barrel kameraya dogru = frame merkezine yakin
                dist_to_center = abs(barrel_cx - frame_center_x)
                if dist_to_center < tank_w * 0.3:
                    return "HEDEF", 0.9
                return "YAN", 0.7

    # Barrel tespiti yoksa bbox oranina bak
    aspect = tank_w / max(tank_h, 1)

    if aspect > 1.8:
        # Yandan gorunum (genis)
        return "YAN", 0.5
    elif aspect < 0.8:
        # Onden/arkadan (dar, uzun)
        return "HEDEF", 0.6
    else:
        # Belirsiz
        return "BELIRSIZ", 0.3


# ============================================================
# TEHDIT DEGERLENDIRME
# ============================================================

def assess_threat(detections, tracker, frame_center_x):
    """Genel tehdit seviyesini degerlendir."""
    max_threat = 0
    threat_sources = []

    for det in detections:
        cls = det["cls_name"]
        base_threat = THREAT_LEVELS.get(cls, 0)

        # Tank namlusu bize donukse tehdit artir
        if cls == "Tank":
            direction, conf = estimate_barrel_direction(
                det["box"], detections, frame_center_x
            )
            if direction == "HEDEF":
                base_threat = 5
                threat_sources.append(f"TANK NAMLUSU -> HEDEF! ({conf:.0%})")

        # Yakindaki silahlı insan
        if cls in ("Rifle", "Pistol", "Weapon"):
            base_threat = max(base_threat, 4)
            threat_sources.append(f"SILAHLI HEDEF: {cls}")

        # Patlama/ates
        if cls in ("Explosion", "Fire"):
            threat_sources.append(f"TEHLIKE: {cls}")

        # Drone yaklasiyorsa
        if cls == "Drone":
            threat_sources.append("DRONE TESPIT")

        max_threat = max(max_threat, base_threat)

    # Ani hareket kontrolu
    for tid, track in tracker.tracks.items():
        if tracker.is_sudden_movement(tid):
            max_threat = max(max_threat, 3)
            threat_sources.append(f"ANI HAREKET: {track['cls_name']}")

    return max_threat, threat_sources


# ============================================================
# HUD CIZIM FONKSIYONLARI
# ============================================================

def draw_crosshair(frame):
    """Nisan artisi ciz."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = 20
    color = (0, 255, 0)

    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
    cv2.circle(frame, (cx, cy), size // 2, color, 1)


def draw_compass(frame, angle=0):
    """Basit pusula gostergesi."""
    h, w = frame.shape[:2]
    cx, cy = w - 50, h - 50
    r = 30

    cv2.circle(frame, (cx, cy), r, (0, 255, 0), 1)
    # N isareti
    nx = int(cx + r * 0.8 * math.sin(math.radians(angle)))
    ny = int(cy - r * 0.8 * math.cos(math.radians(angle)))
    cv2.line(frame, (cx, cy), (nx, ny), (0, 0, 255), 2)
    cv2.putText(frame, "N", (cx - 5, cy - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def draw_hud_frame(frame):
    """HUD cercevesi ciz."""
    h, w = frame.shape[:2]
    color = (0, 255, 0)

    # Kose cizgileri
    corner_len = 30
    thickness = 2

    # Sol ust
    cv2.line(frame, (10, 10), (10 + corner_len, 10), color, thickness)
    cv2.line(frame, (10, 10), (10, 10 + corner_len), color, thickness)
    # Sag ust
    cv2.line(frame, (w - 10, 10), (w - 10 - corner_len, 10), color, thickness)
    cv2.line(frame, (w - 10, 10), (w - 10, 10 + corner_len), color, thickness)
    # Sol alt
    cv2.line(frame, (10, h - 10), (10 + corner_len, h - 10), color, thickness)
    cv2.line(frame, (10, h - 10), (10, h - 10 - corner_len), color, thickness)
    # Sag alt
    cv2.line(frame, (w - 10, h - 10), (w - 10 - corner_len, h - 10), color, thickness)
    cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - corner_len), color, thickness)


def draw_threat_bar(frame, threat_level, threat_sources):
    """Tehdit seviyesi gostergesi."""
    h, w = frame.shape[:2]

    # Tehdit renkleri: 0=Yesil, 1-2=Sari, 3-4=Turuncu, 5=Kirmizi
    if threat_level == 0:
        bar_color = (0, 180, 0)
        label = "GUVENLI"
    elif threat_level <= 2:
        bar_color = (0, 200, 200)
        label = "DIKKAT"
    elif threat_level <= 4:
        bar_color = (0, 140, 255)
        label = "TEHLIKE"
    else:
        bar_color = (0, 0, 255)
        label = "KRITIK TEHDIT"

    # Ust bant
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Tehdit bar
    bar_w = int((threat_level / 5.0) * (w - 20))
    cv2.rectangle(frame, (10, 8), (10 + bar_w, 32), bar_color, -1)
    cv2.rectangle(frame, (10, 8), (w - 10, 32), bar_color, 1)

    # Label
    cv2.putText(frame, f"TEHDIT: {label} [{threat_level}/5]",
                (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tehdit kaynaklari
    y_offset = 55
    for src in threat_sources[:3]:
        cv2.putText(frame, f"! {src}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        y_offset += 18


def draw_detection_box(frame, det, tracker, frame_h, frame_w):
    """Tek tespit kutusunu detayli ciz."""
    x1, y1, x2, y2 = det["box"]
    cls = det["cls_name"]
    conf = det["conf"]
    color = COLORS.get(cls, (255, 255, 255))

    # Kutu
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Kose isareti
    corner = 10
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 3)

    # Etiket arkaplan
    label = f"{cls} {conf:.0%}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Drone ise mesafe/yukseklik goster
    if cls == "Drone":
        est = DroneEstimator.estimate(det["box"], frame_h, frame_w)
        info = f"~{est['distance']}m | H:{est['height']}m | {est['size']}"
        cv2.putText(frame, info, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Tank ise namlu yonu goster
    if cls == "Tank":
        direction, dir_conf = estimate_barrel_direction(
            det["box"], [], frame_w // 2
        )
        dir_color = (0, 0, 255) if direction == "HEDEF" else (0, 200, 200)
        cv2.putText(frame, f"Namlu: {direction}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dir_color, 1)


def draw_info_panel(frame, fps, detection_count, frame_count):
    """Sag alt bilgi paneli."""
    h, w = frame.shape[:2]

    # Panel arkaplan
    panel_x = w - 180
    panel_y = h - 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Bilgiler
    color = (0, 255, 0)
    cv2.putText(frame, f"FPS: {fps:.0f}", (panel_x + 5, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Tespit: {detection_count}", (panel_x + 5, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Frame: {frame_count}", (panel_x + 5, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, "Q: Cikis", (panel_x + 5, panel_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_detection_list(frame, detections):
    """Sol alt - tespit edilen nesneler listesi."""
    h, w = frame.shape[:2]

    # Panel
    panel_h = min(len(detections) * 20 + 30, 200)
    panel_y = h - panel_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, panel_y), (220, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "TESPITLER:", (15, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Sinif bazinda say
    class_counts = {}
    for det in detections:
        cls = det["cls_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    y = panel_y + 35
    for cls, count in sorted(class_counts.items(), key=lambda x: -THREAT_LEVELS.get(x[0], 0)):
        color = COLORS.get(cls, (255, 255, 255))
        threat = THREAT_LEVELS.get(cls, 0)
        marker = "!" * min(threat, 3)
        cv2.putText(frame, f"{marker} {cls}: {count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
        if y > h - 20:
            break


# ============================================================
# ANA INFERENCE DONGUSU
# ============================================================

def find_best_model():
    """En iyi mevcut modeli bul."""
    search_paths = [
        _PROJECT_ROOT / "runs" / "detect" / "tank_vision_v11n" / "weights" / "best.pt",
        _PROJECT_ROOT / "runs" / "detect" / "tank_vision_v2" / "weights" / "best.pt",
        _PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "tank_vision_v12" / "weights" / "best.pt",
    ]

    for p in search_paths:
        if p.exists():
            return str(p)

    return None


def run_inference(args):
    """Ana inference dongusu."""
    # Model yukle
    model_path = args.model
    if not model_path:
        model_path = find_best_model()
        if not model_path:
            print("HATA: Model bulunamadi! --model parametresi ile belirtin.")
            return

    print(f"Model yukleniyor: {model_path}")
    model = YOLO(model_path)

    # Sinif isimlerini modelden al
    class_names = {}
    if hasattr(model, 'names'):
        class_names = model.names
    print(f"Siniflar: {class_names}")

    # Source ayarla
    source = args.source
    if source == "0" or source is None:
        source = 0

    if source != 0 and not Path(str(source)).exists():
        print(f"HATA: '{source}' bulunamadi!")
        return

    # Tracker
    tracker = ObjectTracker()

    # Video/Webcam
    cap = None
    is_video = isinstance(source, int) or str(source).endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    )

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("HATA: Video/webcam acilamadi!")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) if not isinstance(source, int) else 0
        if video_fps <= 0:
            video_fps = 30
        frame_delay = int(1000 / video_fps) if not isinstance(source, int) else 1

        # Her N frame'de tespit yap
        detect_every = args.skip_frames

        print(f"Video FPS: {video_fps:.0f} | Tespit her {detect_every} frame")
        print("Baslatiliyor... Cikmak icin 'q' bas.")

        frame_count = 0
        last_detections = []
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    continue
                break

            frame_count += 1
            curr_time = time.time()
            h, w = frame.shape[:2]

            # Tespit
            if frame_count % detect_every == 0:
                results = model(frame, conf=args.conf, verbose=False)
                last_detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                        # Ilk harfi buyuk yap
                        cls_name = cls_name.capitalize() if cls_name else cls_name

                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "cls": cls_id,
                            "cls_name": cls_name,
                            "conf": float(box.conf[0]),
                        })

                # Tracker guncelle
                tracker.update(last_detections, curr_time)

            # Tehdit degerlendirme
            threat_level, threat_sources = assess_threat(
                last_detections, tracker, w // 2
            )

            # --- HUD CIZIM ---
            # Tespit kutulari
            for det in last_detections:
                draw_detection_box(frame, det, tracker, h, w)

            # HUD elemanlari
            draw_hud_frame(frame)
            draw_crosshair(frame)
            draw_threat_bar(frame, threat_level, threat_sources)
            draw_compass(frame)

            # FPS
            fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            draw_info_panel(frame, fps_display, len(last_detections), frame_count)
            draw_detection_list(frame, last_detections)

            # TANK VISION yazisi
            cv2.putText(frame, "TANK VISION AI", (w // 2 - 80, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Tank Vision AI - HUD", frame)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Tek resim
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"HATA: Resim okunamadi: {source}")
            return

        h, w = frame.shape[:2]
        results = model(frame, conf=args.conf, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                cls_name = cls_name.capitalize() if cls_name else cls_name
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "cls": cls_id,
                    "cls_name": cls_name,
                    "conf": float(box.conf[0]),
                })

        # Tracker tek seferlik
        tracker.update(detections, time.time())
        threat_level, threat_sources = assess_threat(detections, tracker, w // 2)

        for det in detections:
            draw_detection_box(frame, det, tracker, h, w)

        draw_hud_frame(frame)
        draw_crosshair(frame)
        draw_threat_bar(frame, threat_level, threat_sources)
        draw_compass(frame)
        draw_info_panel(frame, 0, len(detections), 1)
        draw_detection_list(frame, detections)
        cv2.putText(frame, "TANK VISION AI", (w // 2 - 80, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Kaydet
        out_dir = _PROJECT_ROOT / "test_outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"hud_{Path(str(source)).stem}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"Sonuc kaydedildi: {out_path}")

        cv2.imshow("Tank Vision AI - HUD", frame)
        print("Kapatmak icin herhangi bir tusa bas...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Tank Vision AI kapandi.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tank Vision AI - HUD")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosyasi (.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="Video/resim/webcam (0=webcam)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Guven esigi (default: 0.35)")
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Her N frame'de tespit yap (default: 2)")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()

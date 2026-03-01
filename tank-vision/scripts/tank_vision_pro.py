"""Tank Vision AI PRO - Gelismis Askeri Goruntu Analiz Sistemi.

Moduller:
1. YOLOv11n Ana Dedektor (15 sinif)
2. ByteTrack Obje Takibi (ID, hiz, yon)
3. Depth Anything Mesafe Tahmini
4. Tank/Drone Alt-Siniflandirici (marka/model)
5. SAHI Kucuk Obje Tespiti
6. YOLOv11-pose Tehdit Analizi
7. Askeri HUD Arayuzu

Kullanim:
    python scripts/tank_vision_pro.py                      # Webcam
    python scripts/tank_vision_pro.py --source video.mp4   # Video
    python scripts/tank_vision_pro.py --source resim.jpg   # Resim
    python scripts/tank_vision_pro.py --sahi               # SAHI modu (kucuk objeler)
"""

import argparse
import sys
import time
import math
import os
from pathlib import Path
from collections import deque, defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
_native_imshow = cv2.imshow  # Ultralytics override'ini engelle
import numpy as np
from ultralytics import YOLO
cv2.imshow = _native_imshow  # Orijinal imshow'u geri yukle

# ============================================================
# SINIF TANIMLARI
# ============================================================

CLASSES_V2 = {
    0: "Drone", 1: "Tank", 2: "Human", 3: "Weapon", 4: "Vehicle",
    5: "Aircraft", 6: "Bird", 7: "Smoke", 8: "Fire", 9: "Explosion",
    10: "Soldier", 11: "Civilian", 12: "Rifle", 13: "Pistol", 14: "Barrel",
}

THREAT_LEVELS = {
    "Tank": 5, "Weapon": 4, "Rifle": 4, "Pistol": 3, "Soldier": 3,
    "Drone": 3, "Aircraft": 4, "Explosion": 5, "Fire": 4, "Smoke": 2,
    "Barrel": 5, "Vehicle": 2, "Human": 1, "Civilian": 0, "Bird": 0,
}

COLORS = {
    "Drone": (0, 165, 255), "Tank": (0, 0, 200), "Human": (200, 200, 0),
    "Weapon": (0, 0, 255), "Vehicle": (200, 200, 200), "Aircraft": (255, 100, 0),
    "Bird": (0, 200, 0), "Smoke": (180, 180, 180), "Fire": (0, 100, 255),
    "Explosion": (0, 0, 255), "Soldier": (0, 140, 255), "Civilian": (0, 255, 0),
    "Rifle": (0, 50, 255), "Pistol": (0, 80, 200), "Barrel": (0, 0, 255),
}

# Tank marka/model veritabani
TANK_MODELS = {
    "TR": ["Altay", "M60T", "Leopard 2A4 TR"],
    "RU": ["T-72B3", "T-90M", "T-14 Armata", "T-80BVM"],
    "US": ["M1A2 Abrams", "M1A1", "M60A3"],
    "DE": ["Leopard 2A7", "Leopard 2A6", "Leopard 2A4"],
    "UK": ["Challenger 2", "Challenger 3"],
    "IL": ["Merkava Mk4", "Merkava Mk3"],
    "FR": ["Leclerc"],
    "KR": ["K2 Black Panther"],
    "CN": ["Type 99A", "Type 96B"],
}

DRONE_MODELS = {
    "TR": ["Bayraktar TB2", "Bayraktar TB3", "Bayraktar Akinci", "Anka-S"],
    "US": ["MQ-9 Reaper", "MQ-1 Predator", "RQ-4 Global Hawk", "MQ-1C Gray Eagle"],
    "CN": ["Wing Loong II", "CH-5", "CH-4"],
    "IL": ["Heron TP", "Hermes 900", "Harop"],
    "IR": ["Shahed-136", "Mohajer-6"],
    "RU": ["Orion", "Forpost"],
    "DJI": ["Mavic 3", "Matrice 300", "Mini 3 Pro"],
}


# ============================================================
# MODUL 1: BYTETRACK OBJE TAKIBI
# ============================================================

class ByteTrackTracker:
    """ByteTrack tabanli obje takipcisi.

    Ozellikleri:
    - Her objeye benzersiz ID atar
    - Hiz ve yon hesaplar
    - Kaybolan objeleri hatirlar
    - Yuksek/dusuk guvenli tespitleri ayri isler
    """

    def __init__(self, max_age=90, min_hits=2, iou_threshold=0.15,
                 high_thresh=0.5, low_thresh=0.1):
        self.max_age = max_age          # Kayip frame limiti (90 = 3sn @30fps)
        self.min_hits = min_hits        # Min gorunme sayisi
        self.iou_threshold = iou_threshold  # Cok dusuk IoU -> hizli hareket eden objeleri yakala
        self.high_thresh = high_thresh  # Yuksek guven esigi
        self.low_thresh = low_thresh    # Dusuk guven esigi
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections, frame_time):
        """Tespitlerle track'leri guncelle.

        ByteTrack stratejisi:
        1. Yuksek guvenli tespitleri mevcut track'lere esle
        2. Eslesmeyen track'leri dusuk guvenli tespitlerle esle
        3. Kalan tespitlerden yeni track olustur
        """
        self.frame_count += 1

        # Tespitleri guven seviyesine gore ayir
        high_dets = [d for d in detections if d["conf"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["conf"] < self.high_thresh]

        # --- 1. Asama: Yuksek guvenli tespitler ---
        matched_tracks = set()
        matched_dets = set()

        for tid, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_idx = -1
            best_score = 0

            # Pozisyon tahmini: objenin hareket yonune gore predicted box
            pred_box = self._predict_box(track)

            for i, det in enumerate(high_dets):
                if i in matched_dets:
                    continue
                # Hem mevcut box hem predicted box ile IoU hesapla, buyugunu al
                iou_current = self._calc_iou(track["box"], det["box"])
                iou_predicted = self._calc_iou(pred_box, det["box"])
                iou = max(iou_current, iou_predicted)

                # Merkez mesafesi bonusu - yakin merkezler daha yuksek skor
                center_bonus = self._center_distance_bonus(track["box"], det["box"])

                # Ayni sinifsa bonus ver (ID degismesini onle)
                cls_bonus = 0.15 if det.get("cls_name") == track.get("cls_name") else 0
                score = iou + cls_bonus + center_bonus
                if score > best_score and (iou > self.iou_threshold * 0.3 or center_bonus > 0.1):
                    best_score = score
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                self._update_track(tid, high_dets[best_idx], frame_time)
                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        # --- 2. Asama: Eslesmeyen track'leri dusuk guvenli ile esle ---
        unmatched_tracks = {tid for tid in self.tracks if tid not in matched_tracks}
        matched_low = set()

        for tid in unmatched_tracks:
            track = self.tracks[tid]
            best_iou = self.iou_threshold * 0.5  # Dusuk guven icin esik daha dusuk
            best_idx = -1

            for i, det in enumerate(low_dets):
                if i in matched_low:
                    continue
                iou = self._calc_iou(track["box"], det["box"])
                # Ayni sinif bonusu
                cls_bonus = 0.1 if det.get("cls_name") == track.get("cls_name") else 0
                if iou + cls_bonus > best_iou:
                    best_iou = iou + cls_bonus
                    best_idx = i

            if best_idx >= 0:
                self._update_track(tid, low_dets[best_idx], frame_time)
                matched_tracks.add(tid)
                matched_low.add(best_idx)

        # --- 3. Asama: Yeni track'ler olustur ---
        for i, det in enumerate(high_dets):
            if i not in matched_dets:
                self._create_track(det, frame_time)

        # --- 4. Asama: Eski track'leri temizle ---
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track["missed"] += 1
                if track["missed"] > self.max_age:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks

    # Class onay icin gereken minimum frame sayisi
    CONFIRM_FRAMES = 3      # 3 frame ayni class -> onaylanir, HUD'da gosterilir
    LOCK_FRAMES = 15        # 15 frame ayni class -> kilitlenir, ASLA degismez

    def _create_track(self, det, frame_time):
        """Yeni track olustur — henuz onaylanmamis (pending)."""
        cx = (det["box"][0] + det["box"][2]) / 2
        cy = (det["box"][1] + det["box"][3]) / 2

        self.tracks[self.next_id] = {
            "id": self.next_id,
            "box": det["box"],
            "cls_name": det["cls_name"],
            "cls_id": det.get("cls", -1),
            "conf": det["conf"],
            "positions": deque(maxlen=60),
            "timestamps": deque(maxlen=60),
            "boxes": deque(maxlen=60),
            "last_seen": frame_time,
            "first_seen": frame_time,
            "hits": 1,
            "missed": 0,
            "speed_mps": 0,
            "heading": 0,
            "approaching": False,
            "sub_class": None,
            "distance_m": None,
            "height_m": None,
            "threat_score": 0,
            "pose_info": None,
            # --- CLASS OYLAMA SISTEMI ---
            "cls_votes": defaultdict(int),  # Her class icin oy sayisi
            "cls_locked": False,            # True ise class asla degismez
            "cls_confirmed": False,         # True ise class onaylandi (2+ frame)
        }
        self.tracks[self.next_id]["cls_votes"][det["cls_name"]] = 1
        self.tracks[self.next_id]["positions"].append((cx, cy))
        self.tracks[self.next_id]["timestamps"].append(frame_time)
        self.tracks[self.next_id]["boxes"].append(det["box"])
        self.next_id += 1

    def _update_track(self, tid, det, frame_time):
        """Mevcut track'i guncelle — class oylama + kilitleme sistemi."""
        track = self.tracks[tid]
        cx = (det["box"][0] + det["box"][2]) / 2
        cy = (det["box"][1] + det["box"][3]) / 2

        track["box"] = det["box"]
        track["last_seen"] = frame_time
        track["hits"] += 1
        track["missed"] = 0

        new_cls = det["cls_name"]

        # --- CLASS OYLAMA ---
        # cls_votes yoksa (eski track) olustur
        if "cls_votes" not in track:
            track["cls_votes"] = defaultdict(int)
            track["cls_votes"][track["cls_name"]] = track["hits"]
            track["cls_locked"] = False
            track["cls_confirmed"] = False

        track["cls_votes"][new_cls] += 1

        # --- KILITLI MI? ---
        if track.get("cls_locked"):
            # Kilitli: class ASLA degismez, sadece conf guncelle
            track["conf"] = max(det["conf"], track["conf"])
        else:
            # En cok oy alan class'i bul
            best_cls = max(track["cls_votes"], key=track["cls_votes"].get)
            best_votes = track["cls_votes"][best_cls]

            # 2+ frame ayni class -> onayla
            if best_votes >= self.CONFIRM_FRAMES and not track.get("cls_confirmed"):
                track["cls_confirmed"] = True
                track["cls_name"] = best_cls
                track["conf"] = det["conf"]

            # 12+ frame ayni class -> kilitle (bir daha degismez)
            if best_votes >= self.LOCK_FRAMES:
                track["cls_locked"] = True
                track["cls_name"] = best_cls
                track["conf"] = det["conf"]

            # Henuz onaylanmadiysa en cok oy alani goster
            if not track.get("cls_confirmed"):
                track["cls_name"] = best_cls
                track["conf"] = det["conf"]
            elif track.get("cls_confirmed") and not track.get("cls_locked"):
                # Onaylandi ama kilitlenmedi: sadece cok yuksek farkla degisir
                if best_cls != track["cls_name"] and best_votes > track["cls_votes"][track["cls_name"]] * 2:
                    track["cls_name"] = best_cls
                track["conf"] = det["conf"]

        track["positions"].append((cx, cy))
        track["timestamps"].append(frame_time)
        track["boxes"].append(det["box"])

        # Hiz ve yon hesapla
        if len(track["positions"]) >= 2:
            self._calc_motion(track)

    def _calc_motion(self, track):
        """Hiz, yon ve yaklasma durumunu hesapla."""
        positions = track["positions"]
        timestamps = track["timestamps"]
        boxes = track["boxes"]

        if len(positions) < 2:
            return

        # Son 10 frame'in ortalamasi (gurultu azaltma)
        n = min(10, len(positions))

        # Piksel hizi
        dx = positions[-1][0] - positions[-n][0]
        dy = positions[-1][1] - positions[-n][1]
        dt = timestamps[-1] - timestamps[-n]

        if dt > 0:
            pixel_speed = math.sqrt(dx*dx + dy*dy) / dt
            track["pixel_speed"] = pixel_speed

            # Mesafe bilgisi varsa gercek hiz hesapla
            dist = track.get("distance_m")
            if dist and dist > 0 and dist < 500:
                # Uzak objelerde piksel hizi daha az = gercek hiz daha fazla
                # Kaba formul: speed_real = pixel_speed * distance / focal_length
                track["speed_mps"] = pixel_speed * dist / 800.0
            else:
                track["speed_mps"] = pixel_speed / 50.0

            # Yon (derece, 0=yukari, saat yonunde)
            track["heading"] = math.degrees(math.atan2(dx, -dy)) % 360

        # Yaklasma tespiti: bbox buyuyorsa yaklasiyordur
        if len(boxes) >= 3:
            old_area = (boxes[-n][2] - boxes[-n][0]) * (boxes[-n][3] - boxes[-n][1])
            new_area = (boxes[-1][2] - boxes[-1][0]) * (boxes[-1][3] - boxes[-1][1])
            if old_area > 0:
                growth = new_area / old_area
                track["approaching"] = growth > 1.03  # %3 buyume = yaklasma

        # ETA hesapla (bize kac saniyede gelir)
        dist = track.get("distance_m")
        speed = track.get("speed_mps", 0)
        if track.get("approaching") and dist and speed > 0.5:
            track["eta_seconds"] = dist / speed
        else:
            track["eta_seconds"] = None

    def get_confirmed_tracks(self):
        """Onaylanmis track'leri dondur (class onaylanmis + min_hits gecmis)."""
        return {tid: t for tid, t in self.tracks.items()
                if t["hits"] >= self.min_hits and t.get("cls_confirmed", True)}

    def get_track_trail(self, tid, max_points=20):
        """Track'in iz noktalarini dondur (HUD'da cizim icin)."""
        track = self.tracks.get(tid)
        if not track:
            return []
        positions = list(track["positions"])
        step = max(1, len(positions) // max_points)
        return positions[::step]

    def _predict_box(self, track):
        """Objenin hareket yonune gore sonraki konumunu tahmin et."""
        positions = track.get("positions")
        if not positions or len(positions) < 2:
            return track["box"]

        # Son 2 pozisyondan hareket vektoru
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]

        # Mevcut box'u hareket yonune kaydir
        x1, y1, x2, y2 = track["box"]
        return (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))

    @staticmethod
    def _center_distance_bonus(box1, box2):
        """Iki box'un merkezleri ne kadar yakinsa o kadar bonus ver."""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2

        # Box boyutuna gore normalize et
        size1 = max(box1[2] - box1[0], box1[3] - box1[1], 1)
        size2 = max(box2[2] - box2[0], box2[3] - box2[1], 1)
        avg_size = (size1 + size2) / 2

        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        ratio = dist / avg_size

        # Merkez mesafesi box boyutunun 2 katindan azsa bonus ver
        if ratio < 2.0:
            return max(0, 0.2 * (1.0 - ratio / 2.0))
        return 0

    @staticmethod
    def _calc_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / max(a1 + a2 - inter, 1e-6)


# ============================================================
# MODUL 2: DEPTH ANYTHING - MESAFE TAHMINI
# ============================================================

class DepthEstimator:
    """Depth Anything v2 ile monokuler derinlik tahmini.

    Tek kameradan her pikselin yaklasik mesafesini tahmin eder.
    Bu sayede:
    - Objenin kameradan uzakligi
    - Drone yuksekligi
    - Tehdit mesafesi
    hesaplanabilir.
    """

    def __init__(self, model_size="small"):
        """
        model_size: 'small' (hizli), 'base' (dengeli), 'large' (dogruluk)
        """
        self.model = None
        self.transform = None
        self.device = None
        self.available = False
        self.model_size = model_size
        self._last_depth_map = None
        self._frame_count = 0
        self._update_every = 5  # Her 5 frame'de bir depth hesapla (performans)

        self._try_load()

    def _try_load(self):
        """Depth Anything modelini yukle."""
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Depth Anything v2 - transformers ile
            try:
                from transformers import pipeline
                self.pipe = pipeline(
                    task="depth-estimation",
                    model=f"depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1,
                )
                self.available = True
                self._use_pipeline = True
                print(f"[DEPTH] Depth Anything v2 yuklendi ({self.model_size})")
            except Exception:
                # Fallback: basit pinhole model
                self.available = True
                self._use_pipeline = False
                print("[DEPTH] Depth Anything yuklenemedi, pinhole model kullanilacak")

        except ImportError:
            print("[DEPTH] torch bulunamadi, mesafe tahmini devre disi")
            self.available = False

    def estimate_depth_map(self, frame):
        """Frame'in derinlik haritasini hesapla."""
        self._frame_count += 1

        if not self.available:
            return None

        # Performans: her N frame'de bir guncelle
        if self._frame_count % self._update_every != 0 and self._last_depth_map is not None:
            return self._last_depth_map

        if hasattr(self, '_use_pipeline') and self._use_pipeline:
            try:
                from PIL import Image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = self.pipe(img)
                depth_map = np.array(result["depth"])
                # Normalize 0-1
                depth_map = depth_map.astype(np.float32)
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
                self._last_depth_map = depth_map
                return depth_map
            except Exception:
                pass

        return None

    def get_object_distance(self, depth_map, box, frame_shape):
        """Objenin tahmini mesafesini dondur (metre)."""
        if depth_map is None:
            # Fallback: bbox boyutuna gore kaba tahmin
            return self._pinhole_estimate(box, frame_shape)

        x1, y1, x2, y2 = box
        h, w = depth_map.shape[:2]
        fh, fw = frame_shape[:2]

        # Depth map frame'den farkli boyutta olabilir
        sx, sy = w / fw, h / fh
        rx1 = int(x1 * sx)
        ry1 = int(y1 * sy)
        rx2 = int(x2 * sx)
        ry2 = int(y2 * sy)

        # ROI'nin merkez bolgesini al (kenarlar guvenilmez)
        cx1 = rx1 + (rx2 - rx1) // 4
        cy1 = ry1 + (ry2 - ry1) // 4
        cx2 = rx2 - (rx2 - rx1) // 4
        cy2 = ry2 - (ry2 - ry1) // 4

        roi = depth_map[max(0,cy1):max(1,cy2), max(0,cx1):max(1,cx2)]
        if roi.size == 0:
            return self._pinhole_estimate(box, frame_shape)

        # Depth degeri: 0=yakin, 1=uzak (Depth Anything convention)
        mean_depth = np.median(roi)

        # Kaba metreye cevir (kalibrasyon gerekir, su an tahmini)
        # 0 -> 1m, 1 -> 200m araliginda log scale
        distance = 1.0 + mean_depth * 199.0

        return round(distance, 1)

    def _pinhole_estimate(self, box, frame_shape):
        """iPhone 15 kalibrasyonu ile mesafe tahmini.

        iPhone 15 ana kamera:
        - Sensor: 1/1.65", 48MP (12MP cikti)
        - Focal length: 26mm (35mm eq), gercek ~6.86mm
        - Sensor boyutu: ~9.8mm x 7.3mm
        - FOV: ~78 derece (yatay)
        """
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        bbox_size = max(bw, bh)
        frame_w = frame_shape[1]

        if bbox_size < 5:
            return 999

        # iPhone 15 focal length (piksel cinsinden)
        # f_pixel = f_mm * image_width / sensor_width_mm
        # 1080p: f_pixel = 6.86 * 1920 / 9.8 = ~1343
        # 640px (YOLO): f_pixel = 6.86 * 640 / 9.8 = ~448
        focal_pixel = 6.86 * frame_w / 9.8

        # Obje tipine gore gercek boyut tahmini (metre)
        REAL_SIZES = {
            "Drone": 0.5,       # Kucuk drone ~50cm
            "Tank": 7.0,        # Tank ~7m
            "Human": 1.7,       # Insan ~1.7m
            "Soldier": 1.7,
            "Civilian": 1.7,
            "Vehicle": 4.5,     # Arac ~4.5m
            "Aircraft": 15.0,   # Ucak ~15m
            "Bird": 0.3,        # Kus ~30cm
            "Rifle": 1.0,       # Tufek ~1m
            "Pistol": 0.3,
            "Barrel": 5.0,      # Namlu ~5m
        }

        # Track'ten sinif bilgisi alamiyoruz burda, varsayilan boyut
        typical_size = 2.0
        distance = (typical_size * focal_pixel) / bbox_size

        return round(min(distance, 999), 1)

    def get_object_distance_calibrated(self, depth_map, box, frame_shape, cls_name):
        """Sinif bilgisi ile kalibreli mesafe."""
        REAL_SIZES = {
            "Drone": 0.5, "Tank": 7.0, "Human": 1.7, "Soldier": 1.7,
            "Civilian": 1.7, "Vehicle": 4.5, "Aircraft": 15.0,
            "Bird": 0.3, "Rifle": 1.0, "Pistol": 0.3, "Barrel": 5.0,
            "Weapon": 0.8, "Smoke": 5.0, "Fire": 3.0, "Explosion": 5.0,
        }

        if depth_map is not None:
            return self.get_object_distance(depth_map, box, frame_shape)

        bw = box[2] - box[0]
        bh = box[3] - box[1]
        bbox_size = max(bw, bh)
        frame_w = frame_shape[1]

        if bbox_size < 5:
            return 999

        focal_pixel = 6.86 * frame_w / 9.8
        real_size = REAL_SIZES.get(cls_name, 2.0)
        distance = (real_size * focal_pixel) / bbox_size

        return round(min(distance, 999), 1)

    def get_drone_height(self, depth_map, box, frame_shape):
        """Drone yuksekligini tahmin et."""
        distance = self.get_object_distance(depth_map, box, frame_shape)

        # Drone frame'in ust kismindaysa daha yuksek
        cy = (box[1] + box[3]) / 2
        fh = frame_shape[0]
        height_ratio = max(0.1, 1.0 - (cy / fh))
        height = distance * height_ratio * 0.5

        return round(max(height, 0.5), 1)


# ============================================================
# MODUL 3: ALT-SINIFLANDIRICI (Tank/Drone Model Tanima)
# ============================================================

class SubClassifier:
    """Ana tespitin icinden crop alip marka/model siniflandirmasi yapar.

    Ornek:
    - Tank tespiti -> crop -> "Leopard 2A4"
    - Drone tespiti -> crop -> "Bayraktar TB2"
    - Arac tespiti -> crop -> "Zirhlı Arac" / "Kamyon"
    """

    def __init__(self):
        self.tank_classifier = None
        self.drone_classifier = None
        self.cache = {}  # Track ID -> son siniflandirma
        self._classify_every = 10  # Her 10 frame'de bir siniflandir
        self._frame_count = 0

        self._try_load_models()

    def _try_load_models(self):
        """Alt siniflandirici modelleri yukle."""
        # Tank siniflandirici
        tank_model_paths = [
            _PROJECT_ROOT / "runs" / "classify" / "tank_classifier" / "weights" / "best.pt",
            _PROJECT_ROOT / "models" / "tank_classifier.pt",
        ]
        for p in tank_model_paths:
            if p.exists():
                try:
                    self.tank_classifier = YOLO(str(p))
                    print(f"[SUB-CLS] Tank siniflandirici yuklendi: {p}")
                except Exception as e:
                    print(f"[SUB-CLS] Tank model hatasi: {e}")

        # Drone siniflandirici
        drone_model_paths = [
            _PROJECT_ROOT / "runs" / "classify" / "drone_classifier" / "weights" / "best.pt",
            _PROJECT_ROOT / "models" / "drone_classifier.pt",
        ]
        for p in drone_model_paths:
            if p.exists():
                try:
                    self.drone_classifier = YOLO(str(p))
                    print(f"[SUB-CLS] Drone siniflandirici yuklendi: {p}")
                except Exception as e:
                    print(f"[SUB-CLS] Drone model hatasi: {e}")

        if not self.tank_classifier and not self.drone_classifier:
            print("[SUB-CLS] Alt siniflandirici modeli bulunamadi (opsiyonel)")

    def classify(self, frame, track_id, box, cls_name):
        """Objeyi alt-siniflandir."""
        self._frame_count += 1

        # Cache kontrol
        if track_id in self.cache and self._frame_count % self._classify_every != 0:
            return self.cache[track_id]

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # Crop al (biraz margin ekle)
        margin = 10
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(w, x2 + margin)
        cy2 = min(h, y2 + margin)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return None

        result = None

        if cls_name == "Tank" and self.tank_classifier:
            try:
                res = self.tank_classifier(crop, verbose=False)
                if res and res[0].probs is not None:
                    top1_idx = res[0].probs.top1
                    top1_conf = float(res[0].probs.top1conf)
                    if top1_conf > 0.5:
                        result = {
                            "model": res[0].names[top1_idx],
                            "conf": top1_conf,
                        }
            except Exception:
                pass

        elif cls_name == "Drone" and self.drone_classifier:
            try:
                res = self.drone_classifier(crop, verbose=False)
                if res and res[0].probs is not None:
                    top1_idx = res[0].probs.top1
                    top1_conf = float(res[0].probs.top1conf)
                    if top1_conf > 0.5:
                        result = {
                            "model": res[0].names[top1_idx],
                            "conf": top1_conf,
                        }
            except Exception:
                pass

        if result:
            self.cache[track_id] = result

        return result


# ============================================================
# MODUL 4: SAHI - KUCUK OBJE TESPITI
# ============================================================

class SAHIDetector:
    """Sliced Aided Hyper Inference - Kucuk obje tespiti.

    Goruntyu parcalara boler, her parcada ayri tespit yapar,
    sonuclari birlestirip NMS uygular.
    Ozellikle uzaktaki drone ve insan tespiti icin kritik.
    """

    def __init__(self, model, slice_size=320, overlap_ratio=0.2, conf=0.25):
        self.model = model
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf = conf

    def detect(self, frame, class_names):
        """SAHI ile tespit yap."""
        h, w = frame.shape[:2]
        stride = int(self.slice_size * (1 - self.overlap_ratio))

        all_dets = []

        # Normal boyut tespiti
        normal_results = self.model(frame, conf=self.conf, device=0, verbose=False)
        for r in normal_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                cls_name = cls_name.capitalize() if cls_name else cls_name
                all_dets.append({
                    "box": (x1, y1, x2, y2),
                    "cls": cls_id,
                    "cls_name": cls_name,
                    "conf": float(box.conf[0]),
                })

        # Parcali tespit
        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                x_end = min(x_start + self.slice_size, w)
                y_end = min(y_start + self.slice_size, h)

                # Cok kucuk parcalari atla
                if (x_end - x_start) < self.slice_size * 0.5:
                    continue
                if (y_end - y_start) < self.slice_size * 0.5:
                    continue

                slice_img = frame[y_start:y_end, x_start:x_end]

                results = self.model(slice_img, conf=self.conf, device=0, verbose=False)

                for r in results:
                    for box in r.boxes:
                        sx1, sy1, sx2, sy2 = map(int, box.xyxy[0].tolist())
                        # Global koordinatlara cevir
                        gx1 = sx1 + x_start
                        gy1 = sy1 + y_start
                        gx2 = sx2 + x_start
                        gy2 = sy2 + y_start

                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                        cls_name = cls_name.capitalize() if cls_name else cls_name

                        all_dets.append({
                            "box": (gx1, gy1, gx2, gy2),
                            "cls": cls_id,
                            "cls_name": cls_name,
                            "conf": float(box.conf[0]),
                        })

        # NMS - ayni sinifta cakisan tespitleri birlestir
        all_dets = self._nms(all_dets, iou_threshold=0.5)

        return all_dets

    def _nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression."""
        if not detections:
            return []

        # Sinif bazinda grupla
        by_class = defaultdict(list)
        for det in detections:
            by_class[det["cls"]].append(det)

        result = []
        for cls_id, dets in by_class.items():
            # Guvene gore sirala (yuksekten dusuge)
            dets.sort(key=lambda x: -x["conf"])

            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)

                remaining = []
                for d in dets:
                    iou = ByteTrackTracker._calc_iou(best["box"], d["box"])
                    if iou < iou_threshold:
                        remaining.append(d)
                dets = remaining

            result.extend(keep)

        return result


# ============================================================
# MODUL 5: POSE ESTIMATION - TEHDIT ANALIZI
# ============================================================

class PoseAnalyzer:
    """YOLOv11-pose ile insan pozu analizi.

    Tespit edilen insanlarin pozunu analiz ederek:
    - Silah tutma pozu -> TEHDIT
    - Eller yukari -> TESLIM
    - Yerde -> YARALI/ETKISIZ
    - Kosu -> KACIS/SALDIRI
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self):
        self.pose_model = None
        self.available = False
        self._try_load()

    def _try_load(self):
        """Pose modelini yukle."""
        try:
            self.pose_model = YOLO("yolo11n-pose.pt")
            self.available = True
            print("[POSE] YOLOv11n-pose yuklendi")
        except Exception as e:
            print(f"[POSE] Pose modeli yuklenemedi: {e}")
            self.available = False

    def analyze(self, frame, person_box):
        """Kisi kutusunda pose analizi yap."""
        if not self.available:
            return {"pose": "BILINMIYOR", "threat": False}

        x1, y1, x2, y2 = person_box
        h, w = frame.shape[:2]

        # Crop
        margin = 20
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return {"pose": "BILINMIYOR", "threat": False}

        try:
            results = self.pose_model(crop, verbose=False)
            if not results or not results[0].keypoints:
                return {"pose": "BILINMIYOR", "threat": False}

            kpts = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
            confs = results[0].keypoints.conf[0].cpu().numpy()  # (17,)

            return self._analyze_pose(kpts, confs, crop.shape)

        except Exception:
            return {"pose": "BILINMIYOR", "threat": False}

    def _analyze_pose(self, kpts, confs, crop_shape):
        """Keypoint'lerden poz analizi."""
        h, w = crop_shape[:2]

        # Keypoint indeksleri
        L_SHOULDER, R_SHOULDER = 5, 6
        L_ELBOW, R_ELBOW = 7, 8
        L_WRIST, R_WRIST = 9, 10
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16

        min_conf = 0.3

        # Eller yukari mi? (teslim)
        if (confs[L_WRIST] > min_conf and confs[R_WRIST] > min_conf and
            confs[L_SHOULDER] > min_conf and confs[R_SHOULDER] > min_conf):

            l_hand_up = kpts[L_WRIST][1] < kpts[L_SHOULDER][1] - 30
            r_hand_up = kpts[R_WRIST][1] < kpts[R_SHOULDER][1] - 30

            if l_hand_up and r_hand_up:
                return {"pose": "TESLIM", "threat": False, "detail": "Eller havada"}

        # Yerde mi? (yarali/etkisiz)
        if (confs[L_HIP] > min_conf and confs[R_HIP] > min_conf):
            hip_y = (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2
            if hip_y > h * 0.7:  # Kalca alt kisimda
                if (confs[L_SHOULDER] > min_conf and confs[R_SHOULDER] > min_conf):
                    shoulder_y = (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2
                    if abs(shoulder_y - hip_y) < h * 0.15:  # Yatay
                        return {"pose": "YERDE", "threat": False, "detail": "Yerde yatiyor"}

        # Silah tutma pozu (kollar one uzanmis)
        if (confs[L_WRIST] > min_conf and confs[R_WRIST] > min_conf and
            confs[L_ELBOW] > min_conf and confs[R_ELBOW] > min_conf):

            # Her iki el de one uzanmis mi?
            l_arm_extended = abs(kpts[L_WRIST][1] - kpts[L_ELBOW][1]) < h * 0.1
            r_arm_extended = abs(kpts[R_WRIST][1] - kpts[R_ELBOW][1]) < h * 0.1

            # Eller yakin mi? (silah tutma)
            hand_dist = math.sqrt(
                (kpts[L_WRIST][0] - kpts[R_WRIST][0])**2 +
                (kpts[L_WRIST][1] - kpts[R_WRIST][1])**2
            )
            hands_close = hand_dist < w * 0.3

            if (l_arm_extended or r_arm_extended) and hands_close:
                return {"pose": "SILAHLI", "threat": True, "detail": "Silah tutma pozu"}

        # Normal ayakta
        return {"pose": "AYAKTA", "threat": False, "detail": "Normal durus"}


# ============================================================
# MODUL 6: TEHDIT DEGERLENDIRME (GELISMIS)
# ============================================================

def assess_threat_advanced(tracks, depth_estimator, depth_map, frame_shape, frame_center_x):
    """Gelismis tehdit degerlendirmesi.

    Faktörler:
    - Obje tipi ve sayisi
    - Mesafe (yakin = daha tehlikeli)
    - Yaklasma hizi
    - Namlu yonu
    - Silahli insan pozu
    - Patlama/ates
    """
    max_threat = 0
    threat_sources = []

    for tid, track in tracks.items():
        cls = track["cls_name"]
        base_threat = THREAT_LEVELS.get(cls, 0)

        # Mesafe faktoru: yakin objeler daha tehlikeli
        dist = track.get("distance_m")
        if dist and dist < 50:
            base_threat = min(5, base_threat + 1)
            if dist < 20:
                base_threat = min(5, base_threat + 1)

        # Yaklasma faktoru
        if track.get("approaching"):
            base_threat = min(5, base_threat + 1)
            if cls == "Drone":
                threat_sources.append(f"DRONE YAKLASIYOR! #{tid} ~{track.get('speed_mps', 0):.0f} m/s")
            elif cls == "Tank":
                threat_sources.append(f"TANK YAKLASIYOR! #{tid}")
            elif cls in ("Soldier", "Human"):
                threat_sources.append(f"SILAHLI KISI YAKLASIYOR! #{tid}")

        # Tank namlusu kontrolu
        if cls == "Tank":
            direction = _check_barrel_direction(track["box"], tracks, frame_center_x)
            if direction == "HEDEF":
                base_threat = 5
                threat_sources.append(f"TANK NAMLUSU -> HEDEF! #{tid}")

        # Silahli insan
        if cls in ("Rifle", "Pistol", "Weapon"):
            threat_sources.append(f"SILAHLI HEDEF: {cls} #{tid}")

        # Patlama/ates
        if cls == "Explosion":
            threat_sources.append(f"PATLAMA TESPIT! #{tid}")
        elif cls == "Fire":
            threat_sources.append(f"ATES TESPIT #{tid}")

        # Pose bilgisi
        if track.get("pose_info") and track["pose_info"].get("threat"):
            base_threat = min(5, base_threat + 1)
            threat_sources.append(f"TEHDIT POZU: {track['pose_info']['detail']} #{tid}")

        # Drone ozellikleri
        if cls == "Drone":
            speed = track.get("speed_mps", 0)
            if speed > 10:
                base_threat = min(5, base_threat + 1)
                threat_sources.append(f"HIZLI DRONE: {speed:.0f} m/s #{tid}")

        track["threat_score"] = base_threat
        max_threat = max(max_threat, base_threat)

    return max_threat, threat_sources[:5]  # Max 5 kaynak


def _check_barrel_direction(tank_box, tracks, frame_center_x):
    """Tank namlu yonunu kontrol et."""
    x1, y1, x2, y2 = tank_box
    tank_cx = (x1 + x2) / 2
    tank_w = x2 - x1
    tank_h = y2 - y1

    # Barrel tespiti varsa
    for tid, track in tracks.items():
        if track["cls_name"] == "Barrel":
            bx1, by1, bx2, by2 = track["box"]
            barrel_cx = (bx1 + bx2) / 2
            barrel_cy = (by1 + by2) / 2

            if x1 < barrel_cx < x2 and y1 < barrel_cy < y2:
                dist_to_center = abs(barrel_cx - frame_center_x)
                if dist_to_center < tank_w * 0.3:
                    return "HEDEF"
                return "YAN"

    # Bbox aspect ratio
    aspect = tank_w / max(tank_h, 1)
    if aspect > 1.8:
        return "YAN"
    elif aspect < 0.8:
        return "HEDEF"
    return "BELIRSIZ"


# ============================================================
# MODUL 7: GELISMIS HUD CIZIM
# ============================================================

class MilitaryHUD:
    """Askeri HUD arayuzu - tum bilgileri gorsel olarak sunar."""

    def __init__(self):
        self.flash_counter = 0
        self.alert_sound_played = False

    def draw_all(self, frame, tracks, threat_level, threat_sources, fps,
                 frame_count, depth_map=None):
        """Tum HUD elemanlarini ciz."""
        h, w = frame.shape[:2]

        # Arkaplan overlay
        self._draw_hud_frame(frame)
        self._draw_crosshair(frame)

        # Tespit kutulari ve bilgileri
        for tid, track in tracks.items():
            self._draw_track_box(frame, tid, track, h, w)

        # Tehdit gostergesi
        self._draw_threat_bar(frame, threat_level, threat_sources)

        # Bilgi panelleri
        self._draw_info_panel(frame, fps, len(tracks), frame_count)
        self._draw_detection_list(frame, tracks)
        self._draw_minimap(frame, tracks)
        self._draw_compass(frame)

        # KRITIK TEHDIT flash efekti
        if threat_level >= 5:
            self._draw_critical_alert(frame)

        # Marka
        cv2.putText(frame, "TANK VISION AI PRO", (w // 2 - 100, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_hud_frame(self, frame):
        """HUD cercevesi."""
        h, w = frame.shape[:2]
        color = (0, 255, 0)
        corner_len = 40
        t = 2

        # 4 kose
        for (cx, cy, dx, dy) in [
            (10, 10, 1, 1), (w-10, 10, -1, 1),
            (10, h-10, 1, -1), (w-10, h-10, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), color, t)
            cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), color, t)

        # Ust/alt cizgi (kesik)
        for x in range(50, w-50, 20):
            cv2.line(frame, (x, 5), (x+10, 5), (0, 100, 0), 1)
            cv2.line(frame, (x, h-5), (x+10, h-5), (0, 100, 0), 1)

    def _draw_crosshair(self, frame):
        """Kucuk askeri nisan."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0)

        # Kucuk dis cember
        cv2.circle(frame, (cx, cy), 15, color, 1)
        # Nokta
        cv2.circle(frame, (cx, cy), 2, color, -1)
        # Kisa cizgiler
        gap = 5
        length = 12
        cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, 1)
        cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, 1)
        cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, 1)

    # Gosterilmeyecek siniflar
    HIDDEN_CLASSES = {"Fire", "Smoke", "Bird", "fire", "smoke", "bird"}

    def _draw_track_box(self, frame, tid, track, frame_h, frame_w):
        """Tek track kutusunu ciz (ID, hiz, mesafe, iz)."""
        x1, y1, x2, y2 = track["box"]
        cls = track["cls_name"]

        # Gizli siniflar icin kutu cizme
        if cls in self.HIDDEN_CLASSES:
            return
        conf = track["conf"]
        color = COLORS.get(cls, (255, 255, 255))

        # Tehdit seviyesine gore kutu kalinligi
        thickness = 3 if track.get("threat_score", 0) >= 4 else 2

        # Ana kutu
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Kose isareti
        corner = 12
        for (px, py, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (px, py), (px + dx*corner, py), color, 3)
            cv2.line(frame, (px, py), (px, py + dy*corner), color, 3)

        # --- Ust etiket ---
        # ID + Sinif + Guven
        label = f"#{tid} {cls} {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # --- Alt bilgi ---
        info_y = y2 + 14

        # Mesafe
        dist = track.get("distance_m")
        if dist and dist < 500:
            cv2.putText(frame, f"{dist:.0f}m", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            info_y += 14

        # Hiz (m/s)
        speed = track.get("speed_mps", 0)
        if speed > 0.5:
            cv2.putText(frame, f"{speed:.1f} m/s", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            info_y += 14

        # ETA (kac saniyede gelir)
        eta = track.get("eta_seconds")
        if eta and eta < 300:
            eta_color = (0, 0, 255) if eta < 10 else (0, 140, 255) if eta < 30 else color
            cv2.putText(frame, f"ETA: {eta:.0f}sn", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, eta_color, 1)
            info_y += 14

        # Alt sinif (tank modeli vs)
        sub = track.get("sub_class")
        if sub:
            cv2.putText(frame, f"{sub['model']} ({sub['conf']:.0%})", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 14

        # Drone spesifik bilgiler
        if cls == "Drone":
            height = track.get("height_m")
            if height:
                cv2.putText(frame, f"H:{height:.0f}m", (x1, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                info_y += 14

            if track.get("approaching"):
                cv2.putText(frame, ">>> YAKLASIYOR <<<", (x1, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # Tank namlu yonu
        if cls == "Tank":
            direction = _check_barrel_direction(track["box"], {}, frame_w // 2)
            dir_color = (0, 0, 255) if direction == "HEDEF" else (0, 200, 200)
            cv2.putText(frame, f"Namlu: {direction}", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dir_color, 1)

        # Pose bilgisi
        if cls in ("Human", "Soldier", "Civilian") and track.get("pose_info"):
            pose = track["pose_info"]
            pose_color = (0, 0, 255) if pose.get("threat") else (0, 255, 0)
            cv2.putText(frame, f"Poz: {pose['pose']}", (x1, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, pose_color, 1)

        # Iz ciz (trail)
        positions = list(track.get("positions", []))
        if len(positions) > 2:
            for i in range(1, len(positions)):
                alpha = i / len(positions)
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, trail_color, 1)

    def _draw_threat_bar(self, frame, threat_level, threat_sources):
        """Tehdit gostergesi."""
        h, w = frame.shape[:2]

        threat_config = {
            0: ((0, 180, 0), "GUVENLI"),
            1: ((0, 200, 200), "DIKKAT"),
            2: ((0, 200, 200), "DIKKAT"),
            3: ((0, 140, 255), "TEHLIKE"),
            4: ((0, 140, 255), "YUKSEK TEHLIKE"),
            5: ((0, 0, 255), "KRITIK TEHDIT"),
        }
        bar_color, label = threat_config.get(threat_level, ((0, 180, 0), "GUVENLI"))

        # Ust bant
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Progress bar
        bar_w = int((threat_level / 5.0) * (w - 20))
        cv2.rectangle(frame, (10, 8), (10 + bar_w, 35), bar_color, -1)
        cv2.rectangle(frame, (10, 8), (w - 10, 35), bar_color, 1)

        # Segment cizgileri
        for i in range(1, 5):
            seg_x = 10 + int(i * (w - 20) / 5)
            cv2.line(frame, (seg_x, 8), (seg_x, 35), (50, 50, 50), 1)

        # Label
        cv2.putText(frame, f"TEHDIT: {label} [{threat_level}/5]",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Tehdit kaynaklari
        y = 58
        for src in threat_sources[:4]:
            self.flash_counter += 1
            if threat_level >= 4 and self.flash_counter % 30 < 15:
                txt_color = (0, 0, 255)
            else:
                txt_color = (0, 140, 255)
            cv2.putText(frame, f">> {src}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)
            y += 16

    def _draw_info_panel(self, frame, fps, track_count, frame_count):
        """Sag alt bilgi paneli."""
        h, w = frame.shape[:2]
        panel_x = w - 200
        panel_y = h - 110

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        color = (0, 255, 0)
        y = panel_y + 18
        for label, value in [
            ("FPS", f"{fps:.0f}"),
            ("IZLENEN", f"{track_count}"),
            ("FRAME", f"{frame_count}"),
            ("MODEL", "YOLOv11m"),
        ]:
            cv2.putText(frame, f"{label}: {value}", (panel_x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 18

        cv2.putText(frame, "Q: Cikis | S: Screenshot", (panel_x + 8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    def _draw_detection_list(self, frame, tracks):
        """Sol alt - tespit listesi."""
        h, w = frame.shape[:2]

        class_counts = defaultdict(int)
        for tid, track in tracks.items():
            class_counts[track["cls_name"]] += 1

        if not class_counts:
            return

        panel_h = min(len(class_counts) * 18 + 30, 200)
        panel_y = h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (230, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "TESPITLER:", (15, panel_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

        y = panel_y + 33
        for cls, count in sorted(class_counts.items(),
                                  key=lambda x: -THREAT_LEVELS.get(x[0], 0)):
            color = COLORS.get(cls, (255, 255, 255))
            threat = THREAT_LEVELS.get(cls, 0)
            marker = "!" * min(threat, 3)
            cv2.putText(frame, f"{marker} {cls}: {count}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += 17
            if y > h - 20:
                break

    def _draw_minimap(self, frame, tracks):
        """Sag ust kose - minimap (kus bakisi pozisyon gostergesi)."""
        h, w = frame.shape[:2]

        map_w, map_h = 150, 120
        map_x, map_y = w - map_w - 10, 50

        # Minimap arkaplan
        overlay = frame.copy()
        cv2.rectangle(overlay, (map_x, map_y), (map_x + map_w, map_y + map_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h), (0, 100, 0), 1)

        # "BIZ" gostergesi (merkez)
        bcx = map_x + map_w // 2
        bcy = map_y + map_h - 15
        cv2.drawMarker(frame, (bcx, bcy), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 10, 2)

        # Objeleri minimap'e yerlestir
        for tid, track in tracks.items():
            cx = (track["box"][0] + track["box"][2]) / 2
            cy = (track["box"][1] + track["box"][3]) / 2

            # Frame koordinatlarini minimap'e cevir
            mx = int(map_x + (cx / w) * map_w)
            my = int(map_y + (cy / h) * map_h * 0.8)

            color = COLORS.get(track["cls_name"], (255, 255, 255))
            threat = track.get("threat_score", 0)

            if threat >= 4:
                cv2.circle(frame, (mx, my), 4, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (mx, my), 3, color, -1)

            # ID
            cv2.putText(frame, str(tid), (mx + 5, my - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

    def _draw_compass(self, frame, angle=0):
        """Pusula."""
        h, w = frame.shape[:2]
        cx, cy = w - 50, h - 130
        r = 25

        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 1)
        nx = int(cx + r * 0.7 * math.sin(math.radians(angle)))
        ny = int(cy - r * 0.7 * math.cos(math.radians(angle)))
        cv2.line(frame, (cx, cy), (nx, ny), (0, 0, 255), 2)

        for direction, deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            dx = int(cx + (r + 8) * math.sin(math.radians(deg)))
            dy = int(cy - (r + 8) * math.cos(math.radians(deg)))
            cv2.putText(frame, direction, (dx - 4, dy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

    def _draw_critical_alert(self, frame):
        """Kritik tehdit flash efekti."""
        h, w = frame.shape[:2]
        self.flash_counter += 1

        if self.flash_counter % 20 < 10:
            # Kirmizi cerceve flash
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)
            cv2.putText(frame, "!!! KRITIK TEHDIT !!!", (w//2 - 120, h//2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# ============================================================
# ANA INFERENCE DONGUSU
# ============================================================

# Sinif bazli minimum confidence esikleri
# Karistirilmasi kolay siniflar icin yuksek esik
CLASS_CONF_THRESHOLDS = {
    "Drone": 0.45,
    "Tank": 0.40,
    "Aircraft": 0.45,
    "Bird": 0.50,
    "Human": 0.35,
    "Soldier": 0.35,
    "Civilian": 0.35,
    "Vehicle": 0.35,
    "Weapon": 0.40,
    "Rifle": 0.40,
    "Pistol": 0.45,
    "Barrel": 0.40,
    "Smoke": 0.40,
    "Fire": 0.40,
    "Explosion": 0.45,
}

# Fiziksel boyut kurallari: (min_area, max_area, min_aspect, max_aspect)
# aspect = width / height
CLASS_SIZE_RULES = {
    "Drone":    (200,   50000,  0.3,  5.0),   # Kucuk-orta, kare-yatay
    "Tank":     (5000,  500000, 1.0,  5.0),   # Buyuk, yatay
    "Human":    (1000,  200000, 0.2,  0.9),   # Dikey (uzun, dar)
    "Soldier":  (1000,  200000, 0.2,  0.9),   # Dikey
    "Civilian": (1000,  200000, 0.2,  0.9),   # Dikey
    "Vehicle":  (3000,  500000, 0.8,  5.0),   # Buyuk, yatay
    "Aircraft": (8000,  800000, 0.5,  5.0),   # Cok buyuk
    "Bird":     (50,    5000,   0.3,  4.0),   # Cok kucuk
    "Rifle":    (500,   30000,  1.5,  10.0),  # Yatay (uzun, ince)
    "Pistol":   (200,   15000,  0.5,  3.0),   # Kucuk
}


def fix_drone_confusion(detections):
    """Tum siniflar icin fiziksel kural kontrolu + confidence filtresi.

    1. Sinif bazli minimum confidence kontrolu
    2. Boyut + en-boy orani fiziksel kontrol
    3. Drone/Bird/Aircraft ozel karisiklik duzeltmesi
    4. Cakisan tespitlerde en iyi conf kalir
    """
    if not detections:
        return detections

    fixed = []

    for det in detections:
        cls = det["cls_name"]
        conf = det["conf"]
        x1, y1, x2, y2 = det["box"]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect = w / max(h, 1)

        # --- 1. Confidence filtresi ---
        min_conf = CLASS_CONF_THRESHOLDS.get(cls, 0.35)
        if conf < min_conf:
            continue  # Dusuk confidence -> atla

        # --- 2. Boyut + oran kontrolu ---
        if cls in CLASS_SIZE_RULES:
            min_area, max_area, min_asp, max_asp = CLASS_SIZE_RULES[cls]
            if area < min_area or area > max_area:
                # Boyut uyumsuz -> sinifi duzelt veya atla
                det = _try_fix_class(det, area, aspect)
                if det is None:
                    continue
            elif aspect < min_asp or aspect > max_asp:
                # Oran uyumsuz -> sinifi duzelt veya atla
                det = _try_fix_class(det, area, aspect)
                if det is None:
                    continue

        # --- 3. Drone/Bird/Aircraft ozel ---
        cls = det["cls_name"]  # _try_fix_class degistirmis olabilir
        if cls == "Bird" and area > 8000:
            det["cls_name"] = "Drone"
            det["cls"] = 0
        elif cls == "Aircraft" and area < 5000:
            det["cls_name"] = "Drone"
            det["cls"] = 0
        elif cls == "Bird" and conf < 0.55 and area > 3000:
            det["cls_name"] = "Drone"
            det["cls"] = 0

        fixed.append(det)

    # --- 4. Cakisan tespitlerde en iyi conf kalsin ---
    result = []
    used = set()
    for i, d1 in enumerate(fixed):
        if i in used:
            continue
        best = d1
        for j, d2 in enumerate(fixed):
            if j <= i or j in used:
                continue
            iou = ByteTrackTracker._calc_iou(d1["box"], d2["box"])
            if iou > 0.5:
                used.add(j)
                if d2["conf"] > best["conf"]:
                    best = d2
        result.append(best)

    return result


def _try_fix_class(det, area, aspect):
    """Boyut/oran uyumsuz tespitin sinifini duzeltmeye calis.

    Ornegin: insana drone denmis -> boyut buyukse ve dikey ise Human yap
    Veya: arabaya tank denmis -> boyut kucukse Vehicle yap
    Duzeltilemiyrsa None dondur (tespit atilir).
    """
    cls = det["cls_name"]
    conf = det["conf"]

    # Drone denmis ama cok buyuk ve dikey -> muhtemelen Human
    if cls == "Drone" and area > 50000 and aspect < 0.9:
        det["cls_name"] = "Human"
        det["cls"] = 2  # human cls_id
        return det

    # Drone denmis ama orta buyuklukte -> muhtemelen Bird
    if cls == "Drone" and area < 200:
        return None  # Cok kucuk, gurultu

    # Tank denmis ama cok kucuk -> muhtemelen Vehicle
    if cls == "Tank" and area < 5000:
        det["cls_name"] = "Vehicle"
        det["cls"] = 4  # vehicle cls_id
        return det

    # Human denmis ama yatay -> muhtemelen Vehicle veya yanlis
    if cls in ("Human", "Soldier", "Civilian") and aspect > 2.0 and area > 10000:
        det["cls_name"] = "Vehicle"
        det["cls"] = 4
        return det

    # Duzeltilemediyse dusuk conf ise atla
    if conf < 0.5:
        return None

    return det  # Yuksek conf, olduğu gibi birak


def find_best_model():
    """En iyi modeli bul."""
    search_paths = [
        Path("C:/tv_data/v3/runs/tank_vision_v3m_r5/weights/best.engine"),
        Path("C:/tv_data/v3/runs/tank_vision_v3m_r5/weights/best.pt"),
        Path("C:/tv_data/v3/runs/tank_vision_v3m_r4/weights/best.pt"),
        Path("C:/tv_data/v3/runs/tank_vision_v3m_r3/weights/best.pt"),
        Path("C:/tv_data/v3/runs/tank_vision_v3m/weights/best.pt"),
        Path("C:/tv_data/v3/runs/tank_vision_v3/weights/best.pt"),
        _PROJECT_ROOT / "runs" / "detect" / "tank_vision_v2_full" / "weights" / "best.pt",
        _PROJECT_ROOT / "models" / "best.pt",
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    return None


def run_inference(args):
    """Ana inference dongusu - tum moduller entegre."""

    # ---- Model Yukle ----
    model_path = args.model or find_best_model()
    if not model_path:
        print("HATA: Model bulunamadi! --model ile belirtin.")
        return

    print(f"[MAIN] Model yukleniyor: {model_path}")
    model = YOLO(model_path)

    # GPU'ya kalici olarak tasi
    import torch
    _use_gpu = torch.cuda.is_available()
    if _use_gpu:
        # Warmup - modeli GPU'ya yukle ve isit
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, device=0, verbose=False)  # ilk cagri GPU'ya tasiyor
        model(dummy, device=0, verbose=False)  # warmup
        print(f"[MAIN] GPU: {torch.cuda.get_device_name(0)} ✓ (warmup OK)")
    else:
        print("[MAIN] UYARI: GPU bulunamadi, CPU kullanilacak (YAVAS!)")

    # GPU device numarasi
    _device = 0 if _use_gpu else "cpu"

    class_names = model.names if hasattr(model, 'names') else {}
    print(f"[MAIN] Siniflar: {class_names}")

    # ---- Modulleri Baslat ----
    print("\n--- MODULLER BASLATILIYOR ---")

    # 1. ByteTrack Tracker
    tracker = ByteTrackTracker(
        max_age=30, min_hits=2,
        iou_threshold=0.3,
        high_thresh=0.5, low_thresh=0.1
    )
    print("[OK] ByteTrack Tracker")

    # 2. Depth Estimator
    depth_estimator = DepthEstimator(model_size="small")

    # 3. Sub-classifier
    sub_classifier = SubClassifier()

    # 4. SAHI (opsiyonel)
    sahi_detector = None
    if args.sahi:
        sahi_detector = SAHIDetector(model, slice_size=args.sahi_slice, conf=args.conf)
        print(f"[OK] SAHI Detector (slice={args.sahi_slice})")

    # 5. Pose Analyzer
    pose_analyzer = PoseAnalyzer() if args.pose else None

    # 6. HUD
    hud = MilitaryHUD()
    print("[OK] Military HUD")

    print("--- TUM MODULLER HAZIR ---\n")

    # ---- Source ----
    source = args.source
    if source == "0" or source is None:
        source = 0

    # Ekran yakalama modu
    if str(source).lower() == "screen":
        _run_screen(model, class_names, tracker, depth_estimator,
                    sub_classifier, sahi_detector, pose_analyzer, hud, args)
        return

    if source != 0 and not Path(str(source)).exists():
        print(f"HATA: '{source}' bulunamadi!")
        return

    is_video = isinstance(source, int) or str(source).endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    )

    if is_video:
        _run_video(source, model, class_names, tracker, depth_estimator,
                   sub_classifier, sahi_detector, pose_analyzer, hud, args)
    else:
        _run_image(source, model, class_names, tracker, depth_estimator,
                   sub_classifier, sahi_detector, pose_analyzer, hud, args)


def _run_screen(model, class_names, tracker, depth_estimator,
                sub_classifier, sahi_detector, pose_analyzer, hud, args):
    """Ekran yakalama modu - ekrani canli izleyip tespit yapar."""
    import mss

    sct = mss.mss()
    monitor = sct.monitors[1]  # Ana ekran

    print(f"[SCREEN] Ekran yakalama: {monitor['width']}x{monitor['height']}")
    print("Baslatiliyor... Q=Cikis, S=Screenshot, D=Depth toggle, P=Pose toggle")

    frame_count = 0
    last_detections = []
    prev_time = time.time()
    show_depth = False
    detect_every = args.skip_frames

    while True:
        curr_time = time.time()

        # Ekrani yakala
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame_count += 1
        h, w = frame.shape[:2]

        # ---- TESPIT ----
        if frame_count % detect_every == 0:
            if sahi_detector:
                last_detections = sahi_detector.detect(frame, class_names)
            else:
                results = model(frame, conf=args.conf, device=0, verbose=False)
                last_detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                        cls_name = cls_name.capitalize() if cls_name else cls_name
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "cls": cls_id,
                            "cls_name": cls_name,
                            "conf": float(box.conf[0]),
                        })

            # Drone karisikligi duzelt
            last_detections = fix_drone_confusion(last_detections)

            # Tracker guncelle
            tracker.update(last_detections, curr_time)

        # ---- DEPTH ----
        depth_map = None
        if depth_estimator.available:
            depth_map = depth_estimator.estimate_depth_map(frame)

        # ---- TRACK'LERI ZENGINLESTIR ----
        confirmed = tracker.get_confirmed_tracks()

        for tid, track in confirmed.items():
            if hasattr(depth_estimator, 'get_object_distance_calibrated'):
                track["distance_m"] = depth_estimator.get_object_distance_calibrated(
                    depth_map, track["box"], frame.shape, track["cls_name"]
                )
            else:
                track["distance_m"] = depth_estimator.get_object_distance(
                    depth_map, track["box"], frame.shape
                )

            if track["cls_name"] == "Drone":
                track["height_m"] = depth_estimator.get_drone_height(
                    depth_map, track["box"], frame.shape
                )

            sub = sub_classifier.classify(frame, tid, track["box"], track["cls_name"])
            if sub:
                track["sub_class"] = sub

            if pose_analyzer and track["cls_name"] in ("Human", "Soldier", "Civilian"):
                if frame_count % 5 == 0:
                    pose = pose_analyzer.analyze(frame, track["box"])
                    track["pose_info"] = pose

        # ---- TEHDIT DEGERLENDIRME ----
        threat_level, threat_sources = assess_threat_advanced(
            confirmed, depth_estimator, depth_map, frame.shape, w // 2
        )

        # ---- FPS ----
        fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ---- HUD CIZ ----
        if show_depth and depth_map is not None:
            depth_vis = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            depth_vis = cv2.resize(depth_vis, (w, h))
            frame = cv2.addWeighted(frame, 0.6, depth_vis, 0.4, 0)

        hud.draw_all(frame, confirmed, threat_level, threat_sources,
                     fps_display, frame_count, depth_map)

        cv2.imshow("Tank Vision AI - SCREEN", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            fname = f"screen_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVE] {fname}")
        elif key == ord('d'):
            show_depth = not show_depth
        elif key == ord('p'):
            args.pose = not args.pose

    cv2.destroyAllWindows()
    sct.close()
    print("[SCREEN] Ekran yakalama durduruldu.")


def _run_video(source, model, class_names, tracker, depth_estimator,
               sub_classifier, sahi_detector, pose_analyzer, hud, args):
    """Video/webcam inference."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("HATA: Video/webcam acilamadi!")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_delay = 1  # Minimum bekleme - GPU hizinda oynat, yavaslatma
    detect_every = args.skip_frames

    print(f"Video FPS: {video_fps:.0f} | Tespit her {detect_every} frame")
    print("Baslatiliyor... Q=Cikis, S=Screenshot, D=Depth toggle, P=Pose toggle")

    frame_count = 0
    last_detections = []
    prev_time = time.time()
    show_depth = False

    # Video kayit
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ret, test_frame = cap.read()
        if ret:
            fh, fw = test_frame.shape[:2]
            writer = cv2.VideoWriter(args.output, fourcc, video_fps, (fw, fh))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, int):
                continue
            break

        frame_count += 1
        curr_time = time.time()
        h, w = frame.shape[:2]

        # ---- TESPIT ----
        if frame_count % detect_every == 0:
            if sahi_detector:
                last_detections = sahi_detector.detect(frame, class_names)
            else:
                results = model(frame, conf=args.conf, device=0, verbose=False)
                last_detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                        cls_name = cls_name.capitalize() if cls_name else cls_name
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "cls": cls_id,
                            "cls_name": cls_name,
                            "conf": float(box.conf[0]),
                        })

            # ---- DRONE/BIRD/AIRCRAFT DUZELTME ----
            last_detections = fix_drone_confusion(last_detections)

            # ---- TRACKER GUNCELLE ----
            tracker.update(last_detections, curr_time)

        # ---- DEPTH ----
        depth_map = None
        if depth_estimator.available:
            depth_map = depth_estimator.estimate_depth_map(frame)

        # ---- TRACK'LERI ZENGINLESTIR ----
        confirmed = tracker.get_confirmed_tracks()

        for tid, track in confirmed.items():
            # Kalibreli mesafe (sinif bilgisi ile)
            if hasattr(depth_estimator, 'get_object_distance_calibrated'):
                track["distance_m"] = depth_estimator.get_object_distance_calibrated(
                    depth_map, track["box"], frame.shape, track["cls_name"]
                )
            else:
                track["distance_m"] = depth_estimator.get_object_distance(
                    depth_map, track["box"], frame.shape
                )

            # Drone yukseklik
            if track["cls_name"] == "Drone":
                track["height_m"] = depth_estimator.get_drone_height(
                    depth_map, track["box"], frame.shape
                )

            # Alt siniflandirma
            sub = sub_classifier.classify(frame, tid, track["box"], track["cls_name"])
            if sub:
                track["sub_class"] = sub

            # Pose analizi
            if pose_analyzer and track["cls_name"] in ("Human", "Soldier", "Civilian"):
                if frame_count % 5 == 0:  # Her 5 frame'de bir
                    pose = pose_analyzer.analyze(frame, track["box"])
                    track["pose_info"] = pose

        # ---- TEHDIT DEGERLENDIRME ----
        threat_level, threat_sources = assess_threat_advanced(
            confirmed, depth_estimator, depth_map, frame.shape, w // 2
        )

        # ---- FPS ----
        fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ---- HUD CIZ ----
        if show_depth and depth_map is not None:
            # Depth overlay
            depth_vis = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )
            depth_vis = cv2.resize(depth_vis, (w, h))
            frame = cv2.addWeighted(frame, 0.6, depth_vis, 0.4, 0)

        hud.draw_all(frame, confirmed, threat_level, threat_sources,
                     fps_display, frame_count, depth_map)

        # Video kayit
        if writer:
            writer.write(frame)

        cv2.imshow("Tank Vision AI PRO", frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Screenshot
            out_dir = _PROJECT_ROOT / "screenshots"
            out_dir.mkdir(exist_ok=True)
            fname = out_dir / f"screenshot_{frame_count}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"Screenshot: {fname}")
        elif key == ord('d'):
            show_depth = not show_depth
            print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Tank Vision AI PRO kapandi.")


def _run_image(source, model, class_names, tracker, depth_estimator,
               sub_classifier, sahi_detector, pose_analyzer, hud, args):
    """Tek resim inference."""
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"HATA: Resim okunamadi: {source}")
        return

    h, w = frame.shape[:2]
    curr_time = time.time()

    # Tespit
    if sahi_detector:
        detections = sahi_detector.detect(frame, class_names)
    else:
        results = model(frame, conf=args.conf, device=0, verbose=False)
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

    # Drone/Bird/Aircraft duzeltme
    detections = fix_drone_confusion(detections)

    # Tracker
    tracker.update(detections, curr_time)
    confirmed = tracker.get_confirmed_tracks()
    # Resimde min_hits=1 olsun
    confirmed = tracker.tracks

    # Depth
    depth_map = depth_estimator.estimate_depth_map(frame) if depth_estimator.available else None

    for tid, track in confirmed.items():
        if hasattr(depth_estimator, 'get_object_distance_calibrated'):
            track["distance_m"] = depth_estimator.get_object_distance_calibrated(
                depth_map, track["box"], frame.shape, track["cls_name"]
            )
        else:
            track["distance_m"] = depth_estimator.get_object_distance(
                depth_map, track["box"], frame.shape
            )
        if track["cls_name"] == "Drone":
            track["height_m"] = depth_estimator.get_drone_height(
                depth_map, track["box"], frame.shape
            )
        sub = sub_classifier.classify(frame, tid, track["box"], track["cls_name"])
        if sub:
            track["sub_class"] = sub

    # Tehdit
    threat_level, threat_sources = assess_threat_advanced(
        confirmed, depth_estimator, depth_map, frame.shape, w // 2
    )

    # HUD
    hud.draw_all(frame, confirmed, threat_level, threat_sources, 0, 1, depth_map)

    # Kaydet
    out_dir = _PROJECT_ROOT / "test_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"pro_{Path(str(source)).stem}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"Sonuc kaydedildi: {out_path}")

    cv2.imshow("Tank Vision AI PRO", frame)
    print("Kapatmak icin herhangi bir tusa bas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tank Vision AI PRO")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosyasi (.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="Video/resim/webcam (0=webcam)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Guven esigi (default: 0.35)")
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Her N frame'de tespit (default: 2)")
    parser.add_argument("--sahi", action="store_true",
                        help="SAHI modu (kucuk objeler icin)")
    parser.add_argument("--sahi-slice", type=int, default=320,
                        help="SAHI dilim boyutu (default: 320)")
    parser.add_argument("--pose", action="store_true",
                        help="Pose estimation aktif")
    parser.add_argument("--output", type=str, default=None,
                        help="Video kayit dosyasi (.mp4)")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()

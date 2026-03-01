"""Hizli model test scripti - webcam, video veya resim uzerinde calistir.

Kullanim:
    # Webcam ile test (varsayilan):
    python scripts/quick_test.py

    # Resim ile test:
    python scripts/quick_test.py --source resim.jpg

    # Video ile test:
    python scripts/quick_test.py --source video.mp4

    # Belirli bir model ile:
    python scripts/quick_test.py --model runs/detect/runs/detect/tank_vision_v12/weights/best.pt

    # Sadece kaydet (GUI olmadan):
    python scripts/quick_test.py --source resim.jpg --save-only

Cikis icin 'q' tusuna bas.
"""

import argparse
import sys
import time
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
# Ultralytics cv2.imshow'u override ediyor, orijinali saklayalim
_original_imshow = cv2.imshow

from ultralytics import YOLO

# Orijinal imshow'u geri yukle
cv2.imshow = _original_imshow

# Sinif isimleri ve renkleri
CLASS_NAMES = {
    0: "Drone",
    1: "Tank",
    2: "Human",
    3: "Weapon",
    4: "Vehicle",
    5: "Aircraft",
    6: "Bird",
}

CLASS_COLORS = {
    0: (0, 0, 255),      # Drone - Kirmizi
    1: (0, 100, 0),      # Tank - Koyu Yesil
    2: (255, 200, 0),    # Human - Cyan
    3: (0, 0, 200),      # Weapon - Koyu Kirmizi
    4: (255, 150, 0),    # Vehicle - Turuncu
    5: (200, 200, 0),    # Aircraft - Acik Mavi
    6: (0, 255, 255),    # Bird - Sari
}

THREAT_CLASSES = {0, 1, 3}  # Drone, Tank, Weapon -> tehdit


def _has_gui():
    """GUI destegi var mi kontrol et."""
    try:
        # Kucuk bir test penceresi ac/kapat
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


# GUI destegini bir kere kontrol et
GUI_AVAILABLE = _has_gui()


def draw_hud(frame, detections):
    """Basit HUD overlay cizer."""
    h, w = frame.shape[:2]

    # Ust bar
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, "TANK VISION AI - LIVE", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tespit sayisi
    n = len(detections)
    threat_count = sum(1 for d in detections if d['cls'] in THREAT_CLASSES)
    status_color = (0, 0, 255) if threat_count > 0 else (0, 255, 0)
    cv2.putText(frame, f"Targets: {n}  Threats: {threat_count}", (w - 350, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Detections
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls = det['cls']
        conf = det['conf']
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        name = CLASS_NAMES.get(cls, f"cls_{cls}")

        # Kutu
        thickness = 3 if cls in THREAT_CLASSES else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Etiket
        label = f"{name} {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Tehdit ikonu
        if cls in THREAT_CLASSES:
            cv2.putText(frame, "!", (x2 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Alt bar - sinif ozeti
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    class_counts = {}
    for d in detections:
        name = CLASS_NAMES.get(d['cls'], '?')
        class_counts[name] = class_counts.get(name, 0) + 1
    summary = "  |  ".join(f"{k}: {v}" for k, v in class_counts.items())
    cv2.putText(frame, summary if summary else "No detections",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def run_inference(model_path, source, conf_threshold=0.25, save_only=False):
    """Model ile inference calistir."""
    print(f"\n{'='*50}")
    print(f"  Tank Vision AI - Test Mode")
    print(f"  Model: {model_path}")
    print(f"  Source: {source}")
    print(f"  Conf: {conf_threshold}")
    print(f"  GUI: {'Var' if GUI_AVAILABLE and not save_only else 'Yok (dosyaya kaydedilecek)'}")
    print(f"{'='*50}\n")

    model = YOLO(str(model_path))
    use_gui = GUI_AVAILABLE and not save_only

    # Cikis klasoru
    output_dir = _PROJECT_ROOT / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Webcam mi?
    if source == "0" or source == 0:
        source = 0
        print("Webcam aciliyor... (cikmak icin 'q')")
    elif source != 0 and not Path(str(source)).exists():
        print(f"HATA: '{source}' bulunamadi!")
        return

    cap = None

    if isinstance(source, int) or str(source).endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("HATA: Video/webcam acilamadi!")
            return

        # Video FPS'ini al (webcam degilse orijinal hizda oynat)
        video_fps = cap.get(cv2.CAP_PROP_FPS) if not isinstance(source, int) else 0
        if video_fps <= 0:
            video_fps = 30  # varsayilan
        frame_delay = int(1000 / video_fps) if not isinstance(source, int) else 1

        # Her N frame'de bir tespit yap (arada son tespiti goster)
        detect_every = 3 if isinstance(source, int) else 2
        frame_count = 0
        last_detections = []

        # Canli gosterim - q'ya basana kadar devam eder
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    continue
                break

            frame_count += 1

            # Her N frame'de model calistir, arada onceki sonucu kullan
            if frame_count % detect_every == 0:
                results = model(frame, conf=conf_threshold, verbose=False)
                last_detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        last_detections.append({
                            'box': (x1, y1, x2, y2),
                            'cls': int(box.cls[0]),
                            'conf': float(box.conf[0]),
                        })

            frame = draw_hud(frame, last_detections)

            # FPS
            curr_time = time.time()
            fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps_display:.0f}", (frame.shape[1] - 130, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Tank Vision AI", frame)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Test tamamlandi.")
        return

    # Tek resim
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"HATA: Resim okunamadi: {source}")
        return

    results = model(frame, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'box': (x1, y1, x2, y2),
                'cls': int(box.cls[0]),
                'conf': float(box.conf[0]),
            })

    frame = draw_hud(frame, detections)
    print(f"\n{len(detections)} nesne tespit edildi:")
    for d in detections:
        name = CLASS_NAMES.get(d['cls'], '?')
        print(f"  - {name}: {d['conf']:.1%}")

    # Sonucu kaydet
    stem = Path(str(source)).stem
    out_path = output_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"\nSonuc kaydedildi: {out_path}")

    if use_gui:
        cv2.imshow("Tank Vision AI", frame)
        print("Devam etmek icin bir tusa bas...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"(GUI yok - sonucu gormek icin dosyayi ac: {out_path})")

    # Ayrica Windows'ta otomatik ac
    if not use_gui and sys.platform == 'win32':
        os.startfile(str(out_path))
        print("Resim aciliyor...")


def main():
    parser = argparse.ArgumentParser(description="Tank Vision AI - Quick Test")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dosyasi (.pt)")
    parser.add_argument("--source", type=str, default="0",
                        help="Kaynak: 0=webcam, dosya.jpg, video.mp4")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (varsayilan: 0.25)")
    parser.add_argument("--save-only", action="store_true",
                        help="Sadece dosyaya kaydet, GUI gosterme")
    args = parser.parse_args()

    # Model yolu
    if args.model:
        model_path = Path(args.model)
    else:
        # Otomatik en iyi modeli bul
        best = _PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "tank_vision_v12" / "weights" / "best.pt"
        if best.exists():
            model_path = best
        else:
            # Herhangi bir best.pt ara
            candidates = list((_PROJECT_ROOT / "runs").rglob("best.pt"))
            if candidates:
                model_path = candidates[-1]  # En son
            else:
                print("HATA: Model bulunamadi! --model ile belirtin.")
                return

    if not model_path.exists():
        print(f"HATA: Model bulunamadi: {model_path}")
        return

    run_inference(model_path, args.source, args.conf, args.save_only)


if __name__ == "__main__":
    main()

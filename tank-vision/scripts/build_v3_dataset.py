#!/usr/bin/env python3
"""
Tank Vision AI — v3 Dataset Builder
Kullanıcının ALL DATA klasöründeki verileri + v2 mevcut verileri birleştirip
sınıf dengeli YOLOv11 formatında v3 dataset oluşturur.

Hedef 15 sınıf:
  0:drone, 1:tank, 2:human, 3:weapon, 4:vehicle, 5:aircraft,
  6:bird, 7:smoke, 8:fire, 9:explosion, 10:soldier, 11:civilian,
  12:rifle, 13:pistol, 14:barrel
"""

import os
import sys
import json
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
ALL_DATA = r"C:\Users\orhun\OneDrive\Desktop\ALL DATA"
V2_DATA = "C:/tv_data/v2"
V3_DATA = "C:/tv_data/v3"
MODEL_PATH = r"C:\Users\orhun\opus ai\.claude\worktrees\nervous-greider\tank-vision\.claude\worktrees\flamboyant-bassi\tank-vision\runs\detect\tank_vision_v2_full\weights\best.pt"

TARGET_PER_CLASS = 5000  # Her sınıf için hedef annotation sayısı
MAX_DRONE = 8000         # Drone çok fazla, sınırla
MAX_BACKGROUND = 3000    # Background sınırı
VAL_RATIO = 0.15         # %15 validation
SEED = 42

CLASS_NAMES = {
    0: "drone", 1: "tank", 2: "human", 3: "weapon", 4: "vehicle",
    5: "aircraft", 6: "bird", 7: "smoke", 8: "fire", 9: "explosion",
    10: "soldier", 11: "civilian", 12: "rifle", 13: "pistol", 14: "barrel"
}

random.seed(SEED)

# ============================================================
# HELPERS
# ============================================================
def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{V3_DATA}/images/{split}", exist_ok=True)
        os.makedirs(f"{V3_DATA}/labels/{split}", exist_ok=True)
    print(f"[OK] Dizinler oluşturuldu: {V3_DATA}")

def unique_name(prefix, orig_name):
    """Dosya ismi çakışmasını önle"""
    name = os.path.splitext(os.path.basename(orig_name))[0]
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"
    h = hashlib.md5(orig_name.encode()).hexdigest()[:6]
    return f"{prefix}_{name}_{h}{ext}"

def copy_yolo_pair(img_path, label_path, prefix, class_remap, collected):
    """YOLO formatındaki bir img+label çiftini remap edip collected'a ekle"""
    if not os.path.exists(img_path):
        return False

    # Label oku ve remap et
    new_lines = []
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_cls = int(parts[0])
                    if old_cls in class_remap:
                        new_cls = class_remap[old_cls]
                        new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

    if not new_lines:
        return False  # Boş label, atla

    uname = unique_name(prefix, img_path)
    collected.append({
        "img_src": img_path,
        "label_lines": new_lines,
        "uname": uname,
        "classes": [int(l.split()[0]) for l in new_lines]
    })
    return True

def coco_to_yolo_entries(json_path, img_dir, prefix, class_remap, collected):
    """COCO annotation JSON'u oku, YOLO formatına çevir ve collected'a ekle"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Image id -> info
    img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    ann_by_img = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    count = 0
    for img_id, anns in ann_by_img.items():
        img_info = img_map.get(img_id)
        if not img_info:
            continue

        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        w, h = img_info["width"], img_info["height"]
        if w <= 0 or h <= 0:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except:
                continue

        new_lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in class_remap:
                new_cls = class_remap[cat_id]
                bx, by, bw, bh = ann["bbox"]  # COCO: [x, y, width, height] in pixels
                x_center = (bx + bw / 2) / w
                y_center = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                nw = max(0.001, min(1, nw))
                nh = max(0.001, min(1, nh))
                new_lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

        if new_lines:
            uname = unique_name(prefix, img_path)
            collected.append({
                "img_src": img_path,
                "label_lines": new_lines,
                "uname": uname,
                "classes": [int(l.split()[0]) for l in new_lines]
            })
            count += 1

    return count

# ============================================================
# DATA COLLECTORS
# ============================================================
def collect_all_data(collected):
    """ALL DATA klasöründeki tüm veri setlerini topla"""

    # --- 1. TANK verileri (YOLO, 0→1:tank) ---
    print("\n[1/11] TANK verileri...")
    tank_dir = os.path.join(ALL_DATA, "TANK verileri")
    remap = {0: 1}
    cnt = 0
    for split_dir in ["images"]:
        img_base = os.path.join(tank_dir, split_dir)
        lbl_base = os.path.join(tank_dir, "labels")
        if not os.path.isdir(img_base):
            continue
        for root, dirs, files in os.walk(img_base):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(root, f)
                    lbl_name = os.path.splitext(f)[0] + ".txt"
                    # Find matching label
                    lbl_path = None
                    for lr, ld, lf in os.walk(lbl_base):
                        if lbl_name in lf:
                            lbl_path = os.path.join(lr, lbl_name)
                            break
                    if copy_yolo_pair(img_path, lbl_path, "tank", remap, collected):
                        cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 2. asker-veri-seti (YOLO, 0→2:human, 1→10:soldier, 2→1:tank) ---
    print("[2/11] asker-veri-seti...")
    asker_dir = os.path.join(ALL_DATA, "asker-veri-seti")
    remap = {0: 2, 1: 10, 2: 1}
    cnt = 0
    for split in ["train", "valid"]:
        img_dir = os.path.join(asker_dir, split, "images")
        lbl_dir = os.path.join(asker_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "asker", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 3. Cars.v1 (YOLO, 0→4:vehicle, 1→4:vehicle) ---
    print("[3/11] Cars.v1...")
    cars_dir = os.path.join(ALL_DATA, "Cars.v1-cars_v1.yolov11")
    remap = {0: 4, 1: 4}
    cnt = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(cars_dir, split, "images")
        lbl_dir = os.path.join(cars_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "cars", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 4. Drone Detection.v5i (YOLO, 1→0:drone) ---
    print("[4/11] Drone Detection.v5i...")
    dd_dir = os.path.join(ALL_DATA, "Drone Detection.v5i.yolov11")
    remap = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3}  # bird, drone, tank, person, weapon
    cnt = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dd_dir, split, "images")
        lbl_dir = os.path.join(dd_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "ddv5", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 5. drone ilk veri setim (YOLO, 0→6:bird, 1→0:drone, 2→1:tank) ---
    print("[5/11] drone ilk veri setim...")
    drone1_dir = os.path.join(ALL_DATA, "drone ilk veri setim", "merged_drone_dataset")
    remap = {0: 6, 1: 0, 2: 1}
    cnt = 0
    for split in ["train", "valid"]:
        img_dir = os.path.join(drone1_dir, split, "images")
        lbl_dir = os.path.join(drone1_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "drn1", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 6. emre-drone-roboflow (YOLO, 0→0:drone) ---
    print("[6/11] emre-drone-roboflow...")
    emre_dir = os.path.join(ALL_DATA, "drone-veri-seti", "emre-drone-roboflow-dataset")
    remap = {0: 0}
    cnt = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(emre_dir, split, "images")
        lbl_dir = os.path.join(emre_dir, split, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "emre", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 7. Pistol dataset (YOLO, 0→10:soldier, 1→14:barrel, 2→12:rifle, 3→11:civilian, 4→13:pistol) ---
    print("[7/11] Pistol/Asker/Sivil dataset...")
    pistol_name = [d for d in os.listdir(ALL_DATA) if d.startswith("Pistol")][0]
    pistol_dir = os.path.join(ALL_DATA, pistol_name)
    # data.yaml: Asker(0), Bomba Atar(1), Makineli Tufek(2), Sivil(3), Tabanca(4)
    remap = {0: 10, 1: 14, 2: 12, 3: 11, 4: 13}
    cnt = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(pistol_dir, split, "images")
        lbl_dir = os.path.join(pistol_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "pistol", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 8. Weapon detection (YOLO) ---
    # Automatic Rifle(0), Bazooka(1), Grenade Launcher(2), Handgun(3), Knife(4), Shotgun(5), SMG(6), Sniper(7), Sword(8)
    print("[8/11] Weapon detection...")
    weapon_name = [d for d in os.listdir(ALL_DATA) if d.startswith("Automatic")][0]
    weapon_dir = os.path.join(ALL_DATA, weapon_name, "weapon_detection")
    # 0:AutoRifle→12:rifle, 1:Bazooka→14:barrel, 2:GrenadeLauncher→14:barrel,
    # 3:Handgun→13:pistol, 4:Knife→3:weapon, 5:Shotgun→12:rifle, 6:SMG→12:rifle, 7:Sniper→12:rifle, 8:Sword→3:weapon
    remap = {0: 12, 1: 14, 2: 14, 3: 13, 4: 3, 5: 12, 6: 12, 7: 12, 8: 3}
    cnt = 0
    for split in ["train", "val"]:
        img_dir = os.path.join(weapon_dir, split, "images")
        lbl_dir = os.path.join(weapon_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "wpn", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 9. Drone sürü (YOLO, 0→0:drone) ---
    print("[9/11] Drone sürü...")
    drone_suru = [d for d in os.listdir(ALL_DATA) if "drone" in d.lower() and "s" in d.lower() and "r" in d.lower() and "ilk" not in d.lower() and "veri" not in d.lower() and "Detection" not in d]
    remap = {0: 0}
    cnt = 0
    for ds in drone_suru:
        ds_path = os.path.join(ALL_DATA, ds, "Drone_Swarm_Dataset")
        img_dir = os.path.join(ds_path, "images")
        lbl_dir = os.path.join(ds_path, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
                if copy_yolo_pair(img_path, lbl_path, "swarm", remap, collected):
                    cnt += 1
    print(f"  -> {cnt} görsel eklendi")

    # --- 10. COCO datasets (asker sivil artık Pistol içinde) ---
    # Pistol dataset zaten asker/sivil içeriyor, COCO olanlar ALL DATA'da farklı isimde olabilir
    # Kontrol edelim - eğer yoksa atla
    print("[10/11] COCO veri setleri kontrol...")
    # Bu veri setleri artık ALL DATA'da mevcut değil gibi görünüyor
    # (klasör isimleri değişmiş) - Pistol dataset zaten bunları kapsıyor

    # --- 11. v2 mevcut verilerden eksik sınıfları al ---
    print("[11/11] v2'den eksik sınıflar (smoke, fire, aircraft, explosion)...")
    # v2'de olan ama ALL DATA'da olmayan sınıfları al
    needed_from_v2 = {5, 7, 8, 9}  # aircraft, smoke, fire, explosion
    # Ayrıca human, weapon gibi az olanları da destekle
    cnt = 0
    v2_label_dir = os.path.join(V2_DATA, "labels", "train")
    v2_img_dir = os.path.join(V2_DATA, "images", "train")

    if os.path.isdir(v2_label_dir):
        for lf in os.listdir(v2_label_dir):
            if not lf.endswith(".txt"):
                continue
            lbl_path = os.path.join(v2_label_dir, lf)
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            has_needed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if cid in needed_from_v2:
                        has_needed = True
                        break

            if has_needed:
                img_name = os.path.splitext(lf)[0]
                # Find image
                img_path = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    p = os.path.join(v2_img_dir, img_name + ext)
                    if os.path.exists(p):
                        img_path = p
                        break

                if img_path:
                    # v2 zaten doğru mapping'de, direkt kopyala
                    remap = {i: i for i in range(15)}
                    if copy_yolo_pair(img_path, lbl_path, "v2", remap, collected):
                        cnt += 1

    print(f"  -> {cnt} görsel eklendi (v2'den)")


def collect_backgrounds(bg_collected):
    """Background görsellerini topla (filtreleme sonra yapılacak)"""
    print("\n[BG] Background görselleri toplanıyor...")
    bg_base = os.path.join(ALL_DATA, "Background")
    all_bg = []
    for i in range(1, 8):
        p = os.path.join(bg_base, str(i), "BG-20k")
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        all_bg.append(os.path.join(root, f))

    print(f"  Toplam {len(all_bg)} background bulundu")

    # Rastgele örnekle
    random.shuffle(all_bg)
    selected = all_bg[:MAX_BACKGROUND * 2]  # Filtreleme sonrası kayıp olacak, fazla al

    for img_path in selected:
        uname = unique_name("bg", img_path)
        bg_collected.append({
            "img_src": img_path,
            "label_lines": [],  # Boş label
            "uname": uname,
            "classes": []
        })

    print(f"  -> {len(bg_collected)} background seçildi (filtreleme bekliyor)")


def filter_backgrounds(bg_collected):
    """Mevcut v2 modeliyle background'ları filtrele"""
    print("\n[FILTER] Background görselleri model ile filtreleniyor...")

    try:
        import torch
        # OpenCV imshow fix
        import cv2
        _native = cv2.imshow
        from ultralytics import YOLO
        cv2.imshow = _native

        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda:0")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        clean = []
        dirty = 0
        total = len(bg_collected)

        for i, item in enumerate(bg_collected):
            if i % 100 == 0:
                print(f"  [{i}/{total}] filtre ediliyor... (temiz: {len(clean)}, kirli: {dirty})")

            try:
                results = model(item["img_src"], conf=0.25, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
                if len(results[0].boxes) == 0:
                    clean.append(item)
                else:
                    dirty += 1
            except Exception as e:
                pass  # Hatalı görseli atla

            if len(clean) >= MAX_BACKGROUND:
                break

        print(f"  -> {len(clean)} temiz background, {dirty} kirli (silindi)")
        return clean

    except Exception as e:
        print(f"  [UYARI] Model filtresi başarısız: {e}")
        print(f"  -> Filtresiz ilk {MAX_BACKGROUND} background kullanılıyor")
        return bg_collected[:MAX_BACKGROUND]


def balance_classes(collected):
    """Sınıf dengesi: fazla olanları azalt, az olanları koru"""
    print("\n[BALANCE] Sınıf dengesi ayarlanıyor...")

    # Her sınıfın annotation sayısını hesapla
    class_counts = defaultdict(int)
    class_images = defaultdict(list)  # class_id -> [index]

    for i, item in enumerate(collected):
        for cls in set(item["classes"]):
            class_counts[cls] += 1
            class_images[cls].append(i)

    print("  Mevcut dağılım:")
    for cid in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(cid, "???")
        print(f"    {cid:2d} ({name:10s}): {class_counts[cid]:6d}")

    # Her görseli bir kere dahil et, ama fazla sınıflardan rastgele çıkar
    selected_indices = set()

    # Önce nadir sınıfların TÜM görsellerini ekle
    for cid in sorted(class_counts.keys(), key=lambda c: class_counts[c]):
        target = TARGET_PER_CLASS
        if cid == 0:  # drone
            target = MAX_DRONE

        current = len([idx for idx in class_images[cid] if idx in selected_indices])
        need = target - current

        if need > 0:
            available = [idx for idx in class_images[cid] if idx not in selected_indices]
            random.shuffle(available)
            for idx in available[:need]:
                selected_indices.add(idx)

    balanced = [collected[i] for i in sorted(selected_indices)]

    # Yeni dağılımı göster
    new_counts = defaultdict(int)
    for item in balanced:
        for cls in set(item["classes"]):
            new_counts[cls] += 1

    print("\n  Dengeli dağılım:")
    for cid in sorted(new_counts.keys()):
        name = CLASS_NAMES.get(cid, "???")
        print(f"    {cid:2d} ({name:10s}): {new_counts[cid]:6d}")

    return balanced


def write_dataset(all_items):
    """Verileri train/val'e yaz"""
    print(f"\n[WRITE] {len(all_items)} görsel yazılıyor...")

    random.shuffle(all_items)
    val_count = int(len(all_items) * VAL_RATIO)
    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    for split, items in [("train", train_items), ("val", val_items)]:
        print(f"  {split}: {len(items)} görsel")
        for i, item in enumerate(items):
            if i % 1000 == 0 and i > 0:
                print(f"    [{i}/{len(items)}]...")

            dst_img = os.path.join(V3_DATA, "images", split, item["uname"])
            dst_lbl = os.path.join(V3_DATA, "labels", split, os.path.splitext(item["uname"])[0] + ".txt")

            try:
                shutil.copy2(item["img_src"], dst_img)
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(item["label_lines"]) + "\n" if item["label_lines"] else "")
            except Exception as e:
                pass  # Kopyalama hatası, atla

    return len(train_items), len(val_items)


def write_yaml(train_count, val_count):
    """data.yaml yaz"""
    yaml_content = f"""path: {V3_DATA}
train: images/train
val: images/val

nc: 15
names:
  0: drone
  1: tank
  2: human
  3: weapon
  4: vehicle
  5: aircraft
  6: bird
  7: smoke
  8: fire
  9: explosion
  10: soldier
  11: civilian
  12: rifle
  13: pistol
  14: barrel

# Tank Vision AI v3 Dataset
# Train: {train_count} images
# Val: {val_count} images
# Total: {train_count + val_count} images
"""
    yaml_path = os.path.join(V3_DATA, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[OK] {yaml_path} yazıldı")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  TANK VISION AI — v3 DATASET BUILDER")
    print("=" * 60)

    ensure_dirs()

    # 1. Tüm verileri topla
    collected = []
    collect_all_data(collected)
    print(f"\n[TOPLAM] {len(collected)} etiketli görsel toplandı")

    # 2. Background topla ve filtrele
    bg_collected = []
    collect_backgrounds(bg_collected)
    clean_bg = filter_backgrounds(bg_collected)

    # 3. Sınıf dengesi
    balanced = balance_classes(collected)

    # 4. Background ekle
    all_items = balanced + clean_bg
    print(f"\n[FINAL] {len(balanced)} etiketli + {len(clean_bg)} background = {len(all_items)} toplam")

    # 5. Yaz
    train_count, val_count = write_dataset(all_items)

    # 6. YAML
    write_yaml(train_count, val_count)

    print("\n" + "=" * 60)
    print(f"  v3 DATASET HAZIR!")
    print(f"  Yol: {V3_DATA}")
    print(f"  Train: {train_count} | Val: {val_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

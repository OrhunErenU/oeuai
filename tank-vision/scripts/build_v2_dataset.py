"""Tank Vision v2 - Genisletilmis veri seti olusturucu.

Mevcut tank_vision veri setini alir, ek veri kaynaklarini ekler,
yeni sinif ID'lerini gunceller ve v2 dataset olusturur.

v2 Sinif haritasi (15 sinif):
  0: drone, 1: tank, 2: human, 3: weapon, 4: vehicle,
  5: aircraft, 6: bird, 7: smoke, 8: fire, 9: explosion,
  10: soldier, 11: civilian, 12: rifle, 13: pistol, 14: barrel

Veri kaynaklari:
  - v1 (tank_vision): 7 sinif (drone, tank, human, weapon, vehicle, aircraft, bird)
  - D-Fire (Kaggle): fire(0->8), smoke(1->7)
  - Roboflow Soldier/Civilian: Civilian(0->11), Soldier(1->10)
  - Roboflow Weapons: Rifle(0->12), knife(1->3), pistol(2->13), shot-gun(3->3), submachine-gun(4->3)
  - Kaggle Military Assets (12 classes) - ek kaynak
"""

import os
import sys
import shutil
import random
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
V1_DIR = DATASETS_DIR / "tank_vision"
# Kisa yol kullan - Windows 260 karakter yol siniri sorunu
V2_DIR = Path("C:/tv_data/v2")

# Roboflow kaynaklari
SOLDIER_CIVILIAN_DIR = DATASETS_DIR / "roboflow_soldier_civilian"
WEAPONS_DIR = Path("C:/tv_data/weapons/data")
MILITARY_TANKS_DIR = DATASETS_DIR / "roboflow_military_tanks"

# Kaggle kaynaklari
KAGGLE_DFIRE_DIR = DATASETS_DIR / "kaggle_dfire"
KAGGLE_MILITARY_DIR = DATASETS_DIR / "kaggle_military"

# Sinif eslestirmeleri
# D-Fire: 0=fire, 1=smoke -> v2: 8=fire, 7=smoke
DFIRE_CLASS_MAP = {0: 8, 1: 7}

# Roboflow Soldier/Civilian: 0=Civilian, 1=Soldier -> v2: 11=civilian, 10=soldier
SOLDIER_CIV_MAP = {0: 11, 1: 10}

# Roboflow Weapons: 0=Rifle, 1=knife, 2=pistol, 3=shot-gun, 4=submachine-gun
# -> v2: Rifle->12, knife->3(weapon), pistol->13, shot-gun->3(weapon), submachine-gun->3(weapon)
WEAPONS_MAP = {0: 12, 1: 3, 2: 13, 3: 3, 4: 3}

# Kaggle Military Assets 12 classes
# Typical classes: Aircraft, Helicopter, Tank, Military Vehicle, Soldier, etc.
# Will be mapped after inspecting data.yaml
MILITARY_ASSETS_MAP = {}  # will be populated dynamically

V2_CLASS_NAMES = {
    0: "drone", 1: "tank", 2: "human", 3: "weapon", 4: "vehicle",
    5: "aircraft", 6: "bird", 7: "smoke", 8: "fire", 9: "explosion",
    10: "soldier", 11: "civilian", 12: "rifle", 13: "pistol", 14: "barrel"
}


def copy_v1_to_v2():
    """Mevcut v1 veri setini v2'ye kopyala."""
    print("[1/5] Mevcut v1 veri seti kopyalaniyor...")

    for split in ["train", "val"]:
        src_img = V1_DIR / "images" / split
        src_lbl = V1_DIR / "labels" / split
        dst_img = V2_DIR / "images" / split
        dst_lbl = V2_DIR / "labels" / split

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        if not src_img.exists():
            print(f"  UYARI: {src_img} bulunamadi, atlaniyor.")
            continue

        img_count = 0
        for img_file in src_img.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                shutil.copy2(img_file, dst_img / img_file.name)

                lbl_file = src_lbl / (img_file.stem + ".txt")
                if lbl_file.exists():
                    shutil.copy2(lbl_file, dst_lbl / lbl_file.name)
                else:
                    (dst_lbl / (img_file.stem + ".txt")).touch()

                img_count += 1

        print(f"  {split}: {img_count} goruntu kopyalandi.")


def integrate_roboflow_dataset(name, src_dir, class_map, prefix):
    """Genel Roboflow veri seti entegrasyon fonksiyonu."""
    if not src_dir.exists():
        print(f"  {name}: dizin bulunamadi ({src_dir}), atlaniyor.")
        return

    print(f"\n[{name}] entegre ediliyor: {src_dir}")

    for split_name, v2_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        img_dir = src_dir / split_name / "images"
        lbl_dir = src_dir / split_name / "labels"

        if not img_dir.exists():
            continue

        dst_img = V2_DIR / "images" / v2_split
        dst_lbl = V2_DIR / "labels" / v2_split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            new_name = f"{prefix}_{img_file.name}"
            shutil.copy2(img_file, dst_img / new_name)

            lbl_file = lbl_dir / (img_file.stem + ".txt")
            new_lbl = dst_lbl / f"{prefix}_{img_file.stem}.txt"

            if lbl_file.exists():
                with open(lbl_file) as f:
                    lines = f.readlines()
                with open(new_lbl, "w") as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_cls = int(parts[0])
                            new_cls = class_map.get(old_cls, -1)
                            if new_cls >= 0:
                                f.write(f"{new_cls} {' '.join(parts[1:])}\n")
            else:
                new_lbl.touch()

            count += 1

        if count > 0:
            print(f"  {split_name} -> v2/{v2_split}: {count} goruntu eklendi.")


def integrate_dfire():
    """D-Fire veri setini v2'ye entegre et (Kaggle zip veya acilmis)."""

    # Kaggle zip dosyasini kontrol et
    dfire_zip = KAGGLE_DFIRE_DIR / "smoke-fire-detection-yolo.zip"
    dfire_extracted = KAGGLE_DFIRE_DIR / "smoke-fire-detection-yolo"

    # Diger olasi konumlar
    possible_dirs = [
        dfire_extracted,
        KAGGLE_DFIRE_DIR,
        DATASETS_DIR / "smoke-fire-detection-yolo",
    ]

    # Zip varsa ve acilmamissa ac
    if dfire_zip.exists() and not dfire_extracted.exists():
        print(f"\n[D-Fire] Zip dosyasi aciliyor: {dfire_zip}")
        try:
            with zipfile.ZipFile(dfire_zip, 'r') as zf:
                zf.extractall(KAGGLE_DFIRE_DIR)
            print("  Zip acildi.")
        except Exception as e:
            print(f"  Zip acilamadi: {e}")

    dfire_dir = None
    for d in possible_dirs:
        if d.exists() and (d / "train").exists():
            dfire_dir = d
            break
        # Alt dizinleri kontrol et
        for sub in d.iterdir() if d.exists() else []:
            if sub.is_dir() and (sub / "train").exists():
                dfire_dir = sub
                break
        if dfire_dir:
            break

    if dfire_dir is None:
        print("\n[D-Fire] Veri seti hazir degil (indirme devam ediyor olabilir).")
        return

    print(f"\n[D-Fire] entegre ediliyor: {dfire_dir}")
    integrate_roboflow_dataset("D-Fire", dfire_dir, DFIRE_CLASS_MAP, "dfire")


def integrate_kaggle_military():
    """Kaggle Military Assets 12 sinif veri setini entegre et."""
    mil_zip = KAGGLE_MILITARY_DIR / "military-assets-dataset-12-classes-yolo8-format.zip"
    mil_extracted = KAGGLE_MILITARY_DIR / "data"

    if mil_zip.exists() and not mil_extracted.exists():
        print(f"\n[Military Assets] Zip dosyasi aciliyor: {mil_zip}")
        try:
            with zipfile.ZipFile(mil_zip, 'r') as zf:
                zf.extractall(KAGGLE_MILITARY_DIR)
            print("  Zip acildi.")
        except Exception as e:
            print(f"  Zip acilamadi: {e}")

    # data.yaml bul
    data_yaml = None
    for root, dirs, files in os.walk(KAGGLE_MILITARY_DIR):
        if "data.yaml" in files:
            data_yaml = Path(root) / "data.yaml"
            break

    if data_yaml is None:
        print("\n[Military Assets] Veri seti hazir degil.")
        return

    # data.yaml'i oku ve sinif haritasi olustur
    import yaml
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    # Sinif eslestirmesi
    mil_map = {}
    for idx, name in names.items():
        name_lower = name.lower().strip()
        if "tank" in name_lower:
            mil_map[int(idx)] = 1  # tank
        elif "aircraft" in name_lower or "jet" in name_lower or "plane" in name_lower:
            mil_map[int(idx)] = 5  # aircraft
        elif "helicopter" in name_lower:
            mil_map[int(idx)] = 5  # aircraft
        elif "drone" in name_lower or "uav" in name_lower:
            mil_map[int(idx)] = 0  # drone
        elif "soldier" in name_lower or "troop" in name_lower:
            mil_map[int(idx)] = 10  # soldier
        elif "vehicle" in name_lower or "truck" in name_lower or "apc" in name_lower:
            mil_map[int(idx)] = 4  # vehicle
        elif "weapon" in name_lower or "gun" in name_lower:
            mil_map[int(idx)] = 3  # weapon
        elif "rifle" in name_lower:
            mil_map[int(idx)] = 12  # rifle
        elif "missile" in name_lower or "rocket" in name_lower:
            mil_map[int(idx)] = 3  # weapon
        elif "ship" in name_lower or "boat" in name_lower:
            mil_map[int(idx)] = 4  # vehicle
        elif "explosion" in name_lower:
            mil_map[int(idx)] = 9  # explosion
        else:
            mil_map[int(idx)] = 4  # default: vehicle

    print(f"\n[Military Assets] Sinif eslestirmesi:")
    for idx, name in names.items():
        v2_cls = mil_map.get(int(idx), -1)
        v2_name = V2_CLASS_NAMES.get(v2_cls, "???")
        print(f"  {idx}: {name} -> {v2_cls} ({v2_name})")

    # Veri setinin koku (train/valid/test dizinlerini iceren yer)
    dataset_root = data_yaml.parent
    integrate_roboflow_dataset("Military Assets", dataset_root, mil_map, "mil")


def integrate_soldier_civilian():
    """Roboflow Soldier/Civilian veri setini entegre et."""
    integrate_roboflow_dataset(
        "Soldier/Civilian",
        SOLDIER_CIVILIAN_DIR,
        SOLDIER_CIV_MAP,
        "solciv"
    )


def integrate_weapons():
    """Roboflow Weapons veri setini entegre et."""
    integrate_roboflow_dataset(
        "Weapons",
        WEAPONS_DIR,
        WEAPONS_MAP,
        "wpn"
    )


def count_dataset():
    """v2 veri seti istatistiklerini goster."""
    print("\n" + "=" * 60)
    print("  TANK VISION v2 - VERI SETI ISTATISTIKLERI")
    print("=" * 60)

    total_images = 0
    total_annotations = 0

    for split in ["train", "val"]:
        img_dir = V2_DIR / "images" / split
        lbl_dir = V2_DIR / "labels" / split

        if not img_dir.exists():
            continue

        img_count = len(list(img_dir.glob("*.*")))
        total_images += img_count
        class_counts = {}

        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                            total_annotations += 1
                        except ValueError:
                            pass

        print(f"\n  {split.upper()}: {img_count} goruntu")
        if class_counts:
            for cls_id in sorted(class_counts.keys()):
                name = V2_CLASS_NAMES.get(cls_id, f"sinif_{cls_id}")
                count = class_counts[cls_id]
                bar = "#" * min(count // 50, 40)
                print(f"    [{cls_id:2d}] {name:12s}: {count:6d} adet  {bar}")
        else:
            print("    (bos)")

    print(f"\n  TOPLAM: {total_images} goruntu, {total_annotations} annotation")


def main():
    print("=" * 60)
    print("  Tank Vision v2 - Genisletilmis Veri Seti Olusturucu")
    print("=" * 60)

    # Temiz baslangic
    if V2_DIR.exists():
        print(f"  Mevcut v2 dizini siliniyor: {V2_DIR}")
        shutil.rmtree(V2_DIR)

    # 1. v1 kopyala (7 sinif: drone, tank, human, weapon, vehicle, aircraft, bird)
    copy_v1_to_v2()

    # 2. Soldier/Civilian ekle
    print("\n[2/5] Soldier/Civilian ekleniyor...")
    integrate_soldier_civilian()

    # 3. Weapons ekle
    print("\n[3/5] Weapons ekleniyor...")
    integrate_weapons()

    # 4. D-Fire ekle (smoke, fire)
    print("\n[4/5] D-Fire (smoke/fire) ekleniyor...")
    integrate_dfire()

    # 5. Military Assets ekle
    print("\n[5/5] Military Assets ekleniyor...")
    integrate_kaggle_military()

    # Istatistikler
    count_dataset()

    print(f"\n{'=' * 60}")
    print(f"  [TAMAM] v2 veri seti hazir!")
    print(f"  Konum: {V2_DIR}")
    print(f"  Config: config/dataset_v2.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""COCO gorsellerini indir, YOLO formatina donustur ve tum datasetleri birlestir.

Kullanim:
    python scripts/download_and_prepare.py
"""

import json
import random
import shutil
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from data.class_mapping import COCO_TO_PRIMARY

# ============================================================
# Yapilandirma
# ============================================================
DATASETS_DIR = _PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
COCO_ANN_FILE = RAW_DIR / "coco_subset" / "annotations" / "instances_train2017.json"
COCO_IMAGES_DIR = RAW_DIR / "coco_subset" / "images"
COCO_LABELS_DIR = RAW_DIR / "coco_subset" / "labels"

DRONE_IMAGES_DIR = RAW_DIR / "seraphim_drone" / "train" / "images"
DRONE_LABELS_DIR = RAW_DIR / "seraphim_drone" / "train" / "labels"

# Nihai birlesmis dataset
MERGED_DIR = DATASETS_DIR / "tank_vision"
MERGED_TRAIN_IMG = MERGED_DIR / "images" / "train"
MERGED_TRAIN_LBL = MERGED_DIR / "labels" / "train"
MERGED_VAL_IMG = MERGED_DIR / "images" / "val"
MERGED_VAL_LBL = MERGED_DIR / "labels" / "val"

COCO_BASE_URL = "http://images.cocodataset.org/train2017"

# Her COCO sinifi icin max goruntu sayisi
MAX_PER_CLASS = 3000
# Drone datasetinden max goruntu
MAX_DRONE = 20000
# Validation orani
VAL_RATIO = 0.1

SEED = 42


def step1_select_coco_images():
    """COCO annotation'larindan hedef siniflari sec."""
    print("\n[ADIM 1] COCO hedef goruntuleri seciliyor...")

    with open(COCO_ANN_FILE, "r") as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}

    # Sinif bazinda goruntu ID'leri
    class_images = defaultdict(set)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in COCO_TO_PRIMARY:
            class_images[cat_id].add(ann["image_id"])

    random.seed(SEED)
    selected_ids = set()
    for cat_id, img_ids in class_images.items():
        sample_size = min(MAX_PER_CLASS, len(img_ids))
        selected = random.sample(list(img_ids), sample_size)
        selected_ids.update(selected)
        cat_name = next(
            c["name"] for c in coco["categories"] if c["id"] == cat_id
        )
        print(f"  {cat_name} (COCO {cat_id}): {len(img_ids)} -> {sample_size}")

    print(f"  Toplam benzersiz goruntu: {len(selected_ids)}")
    return selected_ids, coco, img_info


def step2_download_coco_images(selected_ids, img_info):
    """Secilen COCO gorsellerini indir."""
    print("\n[ADIM 2] COCO gorselleri indiriliyor...")
    COCO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Onceden indirilmis gorselleri kontrol et
    existing = {f.name for f in COCO_IMAGES_DIR.glob("*.jpg")}
    to_download = []
    for img_id in selected_ids:
        fname = img_info[img_id]["file_name"]
        if fname not in existing:
            to_download.append((img_id, fname))

    print(f"  Mevcut: {len(existing)}, Indirilecek: {len(to_download)}")

    if not to_download:
        print("  Tum gorseller zaten indirilmis!")
        return

    def _download_one(item):
        _, fname = item
        url = f"{COCO_BASE_URL}/{fname}"
        dest = COCO_IMAGES_DIR / fname
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception:
            return False

    failed = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_download_one, item): item for item in to_download}
        with tqdm(total=len(to_download), desc="COCO indirme") as pbar:
            for future in as_completed(futures):
                if not future.result():
                    failed += 1
                pbar.update(1)

    print(f"  Tamamlandi! Basarisiz: {failed}")


def step3_convert_coco_to_yolo(selected_ids, coco, img_info):
    """COCO annotation'larini YOLO formatina donustur."""
    print("\n[ADIM 3] COCO -> YOLO donusumu...")
    COCO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Image ID -> annotations gruplama
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in COCO_TO_PRIMARY and ann["image_id"] in selected_ids:
            img_anns[ann["image_id"]].append(ann)

    converted = 0
    for img_id, anns in tqdm(img_anns.items(), desc="COCO->YOLO"):
        info = img_info[img_id]
        img_w, img_h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem

        lines = []
        for ann in anns:
            target_cls = COCO_TO_PRIMARY[ann["category_id"]]
            x, y, w, h = ann["bbox"]  # COCO: top-left x,y,w,h

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            lines.append(
                f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        if lines:
            label_path = COCO_LABELS_DIR / f"{stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            converted += 1

    print(f"  {converted} etiket dosyasi olusturuldu -> {COCO_LABELS_DIR}")
    return converted


def step4_merge_datasets():
    """Drone ve COCO datasetlerini birlestir."""
    print("\n[ADIM 4] Datasetler birlestiriliyor...")

    for d in [MERGED_TRAIN_IMG, MERGED_TRAIN_LBL, MERGED_VAL_IMG, MERGED_VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    all_pairs = []  # (img_path, lbl_path, prefix)

    # --- Drone dataseti ---
    drone_imgs = sorted(DRONE_IMAGES_DIR.glob("*.jpg"))
    drone_count = 0
    for img_path in drone_imgs:
        lbl_path = DRONE_LABELS_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            all_pairs.append((img_path, lbl_path, "drone"))
            drone_count += 1
            if drone_count >= MAX_DRONE:
                break
    print(f"  Drone: {drone_count} goruntu-etiket cifti")

    # --- COCO dataseti ---
    coco_imgs = sorted(COCO_IMAGES_DIR.glob("*.jpg"))
    coco_count = 0
    for img_path in coco_imgs:
        lbl_path = COCO_LABELS_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            all_pairs.append((img_path, lbl_path, "coco"))
            coco_count += 1
    print(f"  COCO: {coco_count} goruntu-etiket cifti")

    print(f"  Toplam: {len(all_pairs)} cift")

    # Train/val ayirma
    random.seed(SEED)
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * VAL_RATIO)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    print(f"  Train: {len(train_pairs)}, Val: {val_size}")

    # Dosyalari kopyala
    def copy_pairs(pairs, img_dir, lbl_dir, desc):
        for img_path, lbl_path, prefix in tqdm(pairs, desc=desc):
            # Benzersiz isim: prefix_stem
            new_name = f"{prefix}_{img_path.stem}"
            dst_img = img_dir / f"{new_name}{img_path.suffix}"
            dst_lbl = lbl_dir / f"{new_name}.txt"

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)

    copy_pairs(train_pairs, MERGED_TRAIN_IMG, MERGED_TRAIN_LBL, "Train kopyalama")
    copy_pairs(val_pairs, MERGED_VAL_IMG, MERGED_VAL_LBL, "Val kopyalama")


def step5_create_yaml():
    """dataset.yaml olustur."""
    print("\n[ADIM 5] dataset.yaml olusturuluyor...")

    yaml_content = f"""# Tank Vision AI Dataset
# Otomatik olusturuldu

path: {MERGED_DIR.as_posix()}
train: images/train
val: images/val

nc: 7
names:
  0: drone
  1: tank
  2: human
  3: weapon
  4: vehicle
  5: aircraft
  6: bird
"""

    yaml_path = MERGED_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  -> {yaml_path}")

    # Istatistikleri yazdir
    train_count = len(list(MERGED_TRAIN_IMG.glob("*.*")))
    val_count = len(list(MERGED_VAL_IMG.glob("*.*")))
    print(f"\n  Nihai dataset:")
    print(f"    Train: {train_count} goruntu")
    print(f"    Val:   {val_count} goruntu")

    # Sinif dagilimi kontrol
    from collections import Counter
    class_counter = Counter()
    for lbl_file in MERGED_TRAIN_LBL.glob("*.txt"):
        for line in lbl_file.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.strip().split()[0])
                class_counter[cls_id] += 1

    names = {0: "drone", 1: "tank", 2: "human", 3: "weapon", 4: "vehicle", 5: "aircraft", 6: "bird"}
    print("\n  Sinif dagilimi (train):")
    for cls_id in sorted(class_counter.keys()):
        print(f"    {names.get(cls_id, '?')}: {class_counter[cls_id]} annotation")


if __name__ == "__main__":
    print("=" * 60)
    print("TANK VISION AI - DATASET HAZIRLAMA")
    print("=" * 60)

    selected_ids, coco, img_info = step1_select_coco_images()
    step2_download_coco_images(selected_ids, img_info)
    step3_convert_coco_to_yolo(selected_ids, coco, img_info)
    step4_merge_datasets()
    step5_create_yaml()

    print("\n" + "=" * 60)
    print("DATASET HAZIRLAMA TAMAMLANDI!")
    print("=" * 60)

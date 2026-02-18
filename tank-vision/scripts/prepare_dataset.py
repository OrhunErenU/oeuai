"""Tank Vision AI - Veri Seti Hazirlama Pipeline'i.

Bu script tum veri seti hazirlama adimlarini otomatiklestirir:
1. Veri setlerini indir (drone, tank, silah, insan, ucak)
2. Formatlari YOLO'ya donustur
3. Siniflari yeniden esle (7-sinif semasi)
4. Veri setlerini birlestir
5. Augmentasyon uygula
6. train/val/test olarak bol
7. Alt-siniflandirici icin crop veri setleri olustur

Kullanim:
    # Tam pipeline (Roboflow API key ile):
    python scripts/prepare_dataset.py --roboflow-key YOUR_KEY

    # Sadece ucretsiz veri setleri (HuggingFace + COCO):
    python scripts/prepare_dataset.py

    # Sadece birlestirme (zaten indirilmis veri setleri):
    python scripts/prepare_dataset.py --skip-download --raw-dir datasets/raw

    # Sadece crop olusturma (zaten birlestirilmis veri seti):
    python scripts/prepare_dataset.py --only-crops --dataset-dir datasets/tank_vision
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def step_1_download(output_dir: str, roboflow_key: str | None = None):
    """Adim 1: Ham veri setlerini indir."""
    from data.download import download_all
    print("\n" + "=" * 60)
    print("ADIM 1: Veri Setlerini Indir")
    print("=" * 60)
    download_all(output_dir, roboflow_key)


def step_2_convert(raw_dir: str, converted_dir: str):
    """Adim 2: Tum formatlari YOLO'ya donustur."""
    from data.convert import coco_to_yolo, remap_classes

    print("\n" + "=" * 60)
    print("ADIM 2: Format Donusumu")
    print("=" * 60)

    raw = Path(raw_dir)
    out = Path(converted_dir)
    out.mkdir(parents=True, exist_ok=True)

    # COCO subset donusumu
    coco_ann = raw / "coco_subset" / "annotations" / "instances_train2017.json"
    if coco_ann.exists():
        coco_img = raw / "coco_subset" / "train2017"
        if coco_img.exists():
            print("[*] COCO -> YOLO donusumu...")
            coco_to_yolo(
                str(coco_ann),
                str(coco_img),
                str(out / "coco_labels"),
            )
        else:
            print("[!] COCO gorselleri bulunamadi. Manuel indirme gerekli.")
    else:
        print("[!] COCO annotations bulunamadi. Atlaniyor.")

    # Seraphim drone veri seti (zaten YOLO formatinda)
    seraphim = raw / "seraphim_drone"
    if seraphim.exists():
        print(f"[*] Seraphim drone veri seti bulundu: {seraphim}")
    else:
        print("[!] Seraphim drone veri seti bulunamadi.")

    # Roboflow veri setleri (zaten YOLO formatinda indirilir)
    for rf_name in ["roboflow_military-vehicles-bkfhm", "roboflow_weapons-detection-nkmht",
                     "roboflow_aircraft-detection-fyyob"]:
        rf_dir = raw / rf_name
        if rf_dir.exists():
            print(f"[*] Roboflow veri seti bulundu: {rf_name}")


def step_3_merge(raw_dir: str, output_dir: str, seed: int = 42):
    """Adim 3: Veri setlerini birlestir."""
    from data.merge import merge_datasets

    print("\n" + "=" * 60)
    print("ADIM 3: Veri Seti Birlestirme")
    print("=" * 60)

    raw = Path(raw_dir)
    sources = []

    # Seraphim drone
    seraphim = raw / "seraphim_drone"
    if seraphim.exists():
        for split in ("train", "valid", "test"):
            img_dir = seraphim / split / "images"
            lbl_dir = seraphim / split / "labels"
            if img_dir.exists() and lbl_dir.exists():
                sources.append({
                    "name": f"seraphim_{split}",
                    "images_dir": str(img_dir),
                    "labels_dir": str(lbl_dir),
                    "class_map": {0: 0},  # drone -> drone
                })

    # Roboflow veri setleri
    for rf_name, class_map in [
        ("roboflow_military-vehicles-bkfhm", None),   # Zaten uygun siniflar
        ("roboflow_weapons-detection-nkmht", None),
        ("roboflow_aircraft-detection-fyyob", None),
    ]:
        rf_dir = raw / rf_name
        if rf_dir.exists():
            for split in ("train", "valid", "test"):
                img_dir = rf_dir / split / "images"
                lbl_dir = rf_dir / split / "labels"
                if img_dir.exists() and lbl_dir.exists():
                    sources.append({
                        "name": f"{rf_name}_{split}",
                        "images_dir": str(img_dir),
                        "labels_dir": str(lbl_dir),
                        "class_map": class_map,
                    })

    if not sources:
        print("[!] Hicbir veri seti kaynagi bulunamadi!")
        print("    Oncelikle veri setlerini indirin:")
        print("    python scripts/prepare_dataset.py --roboflow-key YOUR_KEY")
        return

    merge_datasets(
        sources=sources,
        output_dir=output_dir,
        seed=seed,
    )


def step_4_create_crops(dataset_dir: str, output_base: str):
    """Adim 4: Alt-siniflandirici icin crop veri setleri olustur.

    Ana detektor veri setinden sinif bazli kirpimlar cikarir.
    Bu kirpimlar tank_classifier, weapon_classifier, human_classifier
    veri setleri icin temel olusturur.

    NOT: Gercek uretimde, her alt-siniflandirici icin
    ayri etiketleme gerekir (ornegin: tank kirpimini M1 Abrams olarak etiketle).
    Bu script sadece sinif bazli kirpim yapar.
    """
    import cv2
    import numpy as np

    print("\n" + "=" * 60)
    print("ADIM 4: Alt-siniflandirici Crop Veri Setleri Olustur")
    print("=" * 60)

    ds = Path(dataset_dir)
    out = Path(output_base)

    crop_configs = {
        "tank_crops": {"class_ids": [1], "min_size": 64},
        "weapon_crops": {"class_ids": [3], "min_size": 32},
        "human_crops": {"class_ids": [2], "min_size": 48},
    }

    for crop_name, cfg in crop_configs.items():
        crop_dir = out / crop_name
        for split in ("train", "val"):
            (crop_dir / split / "unclassified").mkdir(parents=True, exist_ok=True)

        for split in ("train", "val"):
            img_dir = ds / "images" / split
            lbl_dir = ds / "labels" / split

            if not img_dir.exists() or not lbl_dir.exists():
                continue

            crop_count = 0
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                for line in lbl_path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    cls_id = int(parts[0])

                    if cls_id not in cfg["class_ids"]:
                        continue

                    # YOLO -> piksel
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    # Padding
                    pad = int(max(x2 - x1, y2 - y1) * 0.05)
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)

                    crop = img[y1:y2, x1:x2]
                    if crop.shape[0] < cfg["min_size"] or crop.shape[1] < cfg["min_size"]:
                        continue

                    crop_path = crop_dir / split / "unclassified" / f"{img_path.stem}_crop{crop_count}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    crop_count += 1

            print(f"  {crop_name}/{split}: {crop_count} crop olusturuldu")

    print("\n[!] ONEMLI: Crop'lar 'unclassified' klasorunde olusturuldu.")
    print("    Alt-siniflandirma icin bu crop'lari elle etiketlemeniz gerekiyor:")
    print("    - tank_crops/ -> m1_abrams/, leopard_2/, t72/, t90/, ...")
    print("    - weapon_crops/ -> rpg/, rifle/, pistol/, sniper/, ...")
    print("    - human_crops/ -> soldier/, civilian/")


def step_5_augment(dataset_dir: str):
    """Adim 5: Augmentasyon uygula."""
    from data.augment import apply_augmentation

    print("\n" + "=" * 60)
    print("ADIM 5: Veri Artirma (Augmentasyon)")
    print("=" * 60)

    ds = Path(dataset_dir)
    train_img = ds / "images" / "train"
    train_lbl = ds / "labels" / "train"

    if not train_img.exists():
        print("[!] Egitim veri seti bulunamadi. Atlaniyor.")
        return

    print(f"[*] Train goruntu sayisi: {len(list(train_img.glob('*')))}")
    print("[!] Augmentasyon egitim sirasinda otomatik uygulanir (YOLO built-in).")
    print("    Ek augmentasyon icin: python -m data.augment --dir ", train_img)


def main():
    parser = argparse.ArgumentParser(
        description="Tank Vision AI - Veri Seti Hazirlama Pipeline'i",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
    # Tam pipeline:
    python scripts/prepare_dataset.py --roboflow-key RF_API_KEY

    # Sadece ucretsiz veri setleri:
    python scripts/prepare_dataset.py

    # Sadece birlestirme (zaten indirilmis):
    python scripts/prepare_dataset.py --skip-download --raw-dir datasets/raw

    # Sadece crop olusturma:
    python scripts/prepare_dataset.py --only-crops --dataset-dir datasets/tank_vision
        """,
    )
    parser.add_argument("--roboflow-key", default=None, help="Roboflow API anahtari")
    parser.add_argument("--raw-dir", default="datasets/raw", help="Ham veri seti dizini")
    parser.add_argument("--dataset-dir", default="datasets/tank_vision", help="Birlestirilmis veri seti dizini")
    parser.add_argument("--crops-dir", default="datasets", help="Crop veri setleri ana dizini")
    parser.add_argument("--skip-download", action="store_true", help="Indirmeyi atla")
    parser.add_argument("--only-crops", action="store_true", help="Sadece crop olustur")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")

    args = parser.parse_args()

    raw_dir = str((_PROJECT_ROOT / args.raw_dir).resolve())
    dataset_dir = str((_PROJECT_ROOT / args.dataset_dir).resolve())
    crops_dir = str((_PROJECT_ROOT / args.crops_dir).resolve())

    print("=" * 60)
    print("  TANK VISION AI - VERI SETI HAZIRLAMA")
    print("=" * 60)
    print(f"  Ham veri: {raw_dir}")
    print(f"  Cikti:    {dataset_dir}")
    print(f"  Crop'lar: {crops_dir}")
    print("=" * 60)

    if args.only_crops:
        step_4_create_crops(dataset_dir, crops_dir)
        return

    if not args.skip_download:
        step_1_download(raw_dir, args.roboflow_key)

    step_2_convert(raw_dir, raw_dir)
    step_3_merge(raw_dir, dataset_dir, seed=args.seed)
    step_4_create_crops(dataset_dir, crops_dir)
    step_5_augment(dataset_dir)

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print(f"\nVeri seti: {dataset_dir}")
    print(f"Crop'lar:  {crops_dir}")
    print(f"\nSiradaki adimlar:")
    print(f"  1. Crop'lari siniflandirin (elle etiketleme)")
    print(f"  2. Ana detektoru egitin:")
    print(f"     python scripts/train_detector.py --data {dataset_dir}/dataset.yaml")
    print(f"  3. Alt-siniflandiriciları egitin:")
    print(f"     python scripts/train_tank_cls.py --data {crops_dir}/tank_crops")
    print(f"     python scripts/train_weapon_cls.py --data {crops_dir}/weapon_crops")
    print(f"     python scripts/train_human_cls.py --data {crops_dir}/human_crops")
    print(f"  4. Dost/Dusman siniflandirici icin ayri veri seti hazirlayin")
    print(f"  5. Taret tahmincisi icin ayri veri seti hazirlayin")


if __name__ == "__main__":
    main()

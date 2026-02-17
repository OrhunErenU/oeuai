"""Veri setini train/val/test olarak bolme araci.

Mevcut bir YOLO formatindaki veri setini stratified olarak boler.
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Veri setini train/val/test olarak bol.

    Args:
        images_dir: Kaynak gorseller dizini.
        labels_dir: Kaynak etiketler dizini.
        output_dir: Cikti dizini (images/ ve labels/ alt klasorleri olusturulur).
        train_ratio: Egitim orani.
        val_ratio: Dogrulama orani.
        test_ratio: Test orani.
        seed: Rastgelelik tohumu.

    Returns:
        Split basina goruntu sayilari dict'i.
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out = Path(output_dir)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Goruntu-etiket ciftlerini bul
    pairs = []
    for img in img_dir.iterdir():
        if img.suffix.lower() not in img_extensions:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))

    print(f"[*] {len(pairs)} goruntu-etiket cifti bulundu")

    # Stratified bolme: her goruntunun baskil sinifina gore grupla
    class_groups = defaultdict(list)
    for img, lbl in pairs:
        lines = lbl.read_text().strip().split("\n")
        class_counts = defaultdict(int)
        for line in lines:
            if line.strip():
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
        # Baskil sinif
        dominant = max(class_counts, key=class_counts.get) if class_counts else -1
        class_groups[dominant].append((img, lbl))

    # Her sinif grubunu bol
    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    for cls_id, group in class_groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    # Dosyalari kopyala
    result = {}
    for split_name, split_pairs in splits.items():
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img, lbl in tqdm(split_pairs, desc=split_name):
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)

        result[split_name] = len(split_pairs)
        print(f"  {split_name}: {len(split_pairs)}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Veri seti bolucu")
    parser.add_argument("--images", required=True, help="Gorseller dizini")
    parser.add_argument("--labels", required=True, help="Etiketler dizini")
    parser.add_argument("--output", required=True, help="Cikti dizini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.images, args.labels, args.output, seed=args.seed)

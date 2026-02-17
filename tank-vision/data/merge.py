"""Birden fazla veri setini birlestirme araci.

Farkli kaynaklardan gelen veri setlerini tek bir YOLO formatinda birlestirip
train/val/test olarak boler.
"""

import random
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from tqdm import tqdm

from data.class_mapping import PRIMARY_CLASSES


def merge_datasets(
    sources: list[dict],
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """Birden fazla veri setini birlestirir ve boler.

    Args:
        sources: Veri seti kaynaklari listesi. Her eleman:
            {
                "name": "kaynak_adi",
                "images_dir": "gorseller/dizin/yolu",
                "labels_dir": "etiketler/dizin/yolu",
                "class_map": {eski_id: yeni_id, ...} veya None
            }
        output_dir: Birlestirilmis veri seti cikti dizini.
        train_ratio: Egitim orani.
        val_ratio: Dogrulama orani.
        test_ratio: Test orani.
        seed: Rastgelelik tohumu.

    Returns:
        Istatistik dict'i: sinif basina goruntu sayilari.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    out = Path(output_dir)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Tum goruntu-etiket ciftlerini topla
    all_pairs = []

    for src in sources:
        name = src["name"]
        img_dir = Path(src["images_dir"])
        lbl_dir = Path(src["labels_dir"])
        class_map = src.get("class_map")

        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

        for img_path in tqdm(images, desc=f"Kaynak: {name}"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Sinif esleme uygula (gerekirse)
            if class_map:
                lines = label_path.read_text().strip().split("\n")
                new_lines = []
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    old_cls = int(parts[0])
                    if old_cls in class_map:
                        parts[0] = str(class_map[old_cls])
                        new_lines.append(" ".join(parts))
                if not new_lines:
                    continue
                label_content = "\n".join(new_lines) + "\n"
            else:
                label_content = label_path.read_text()

            # Benzersiz dosya adi (kaynak oneki ile catismayi onle)
            unique_name = f"{name}_{img_path.stem}"

            all_pairs.append({
                "img_src": img_path,
                "label_content": label_content,
                "unique_name": unique_name,
                "img_ext": img_path.suffix,
            })

    print(f"\n[*] Toplam {len(all_pairs)} goruntu-etiket cifti bulundu")

    # Karistir ve bol
    random.seed(seed)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train : n_train + n_val],
        "test": all_pairs[n_train + n_val :],
    }

    # Dosyalari kopyala
    stats = {"total": 0, "per_split": {}, "per_class": {}}

    for split_name, pairs in splits.items():
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name

        for pair in tqdm(pairs, desc=f"{split_name} yaziliyor"):
            dst_img = img_out / f"{pair['unique_name']}{pair['img_ext']}"
            dst_lbl = lbl_out / f"{pair['unique_name']}.txt"

            shutil.copy2(pair["img_src"], dst_img)
            dst_lbl.write_text(pair["label_content"])

        stats["per_split"][split_name] = len(pairs)
        stats["total"] += len(pairs)

    # Sinif dagilimi hesapla
    for split_name in ("train", "val", "test"):
        lbl_dir = out / "labels" / split_name
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                cls_id = int(line.strip().split()[0])
                cls_name = PRIMARY_CLASSES.get(cls_id, f"unknown_{cls_id}")
                stats["per_class"][cls_name] = stats["per_class"].get(cls_name, 0) + 1

    # dataset.yaml olustur
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(PRIMARY_CLASSES),
        "names": dict(PRIMARY_CLASSES),
    }

    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)

    # Rapor
    print("\n" + "=" * 50)
    print("Birlestirme Raporu")
    print("=" * 50)
    print(f"Toplam: {stats['total']} goruntu")
    for split, count in stats["per_split"].items():
        print(f"  {split}: {count}")
    print("\nSinif dagilimi:")
    for cls, count in sorted(stats["per_class"].items()):
        print(f"  {cls}: {count}")
    print(f"\ndataset.yaml -> {yaml_path}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Veri seti birlestirici")
    parser.add_argument("--config", required=True, help="Birlestirme konfigurasyon YAML")
    parser.add_argument("--output", required=True, help="Cikti dizini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    merge_datasets(
        sources=config["sources"],
        output_dir=args.output,
        train_ratio=config.get("train_ratio", 0.80),
        val_ratio=config.get("val_ratio", 0.15),
        test_ratio=config.get("test_ratio", 0.05),
        seed=args.seed,
    )

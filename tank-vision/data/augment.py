"""Veri cogaltma (augmentation) pipeline'i.

Albumentations kutuphanesi ile askeri goruntu verisi icin optimize
edilmis cogaltma stratejileri uygular.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def get_military_augmentation() -> A.Compose:
    """Askeri veri seti icin optimize edilmis augmentation pipeline'i."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomRain(slant_lower=-10, slant_upper=10, p=0.1),
            A.RandomSunFlare(p=0.08),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            A.MotionBlur(blur_limit=7, p=0.15),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.RandomScale(scale_limit=0.3, p=0.2),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            A.ToGray(p=0.05),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_night_augmentation() -> A.Compose:
    """Gece/dusuk isik kosullari icin augmentation."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, -0.2), contrast_limit=0.2, p=0.8
            ),
            A.GaussNoise(var_limit=(20.0, 80.0), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=-40, val_shift_limit=-30, p=0.6
            ),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def read_yolo_labels(label_path: Path) -> tuple[list, list]:
    """YOLO etiket dosyasini oku.

    Returns:
        (bboxes, class_labels): YOLO formatinda bbox listesi ve sinif ID listesi.
    """
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:5])
        bboxes.append([x_center, y_center, w, h])
        class_labels.append(cls_id)

    return bboxes, class_labels


def write_yolo_labels(label_path: Path, bboxes: list, class_labels: list):
    """YOLO etiket dosyasina yaz."""
    lines = []
    for bbox, cls_id in zip(bboxes, class_labels):
        x, y, w, h = bbox
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    num_augmentations: int = 3,
    include_night: bool = True,
) -> int:
    """Veri setini cogalt.

    Args:
        images_dir: Kaynak gorseller.
        labels_dir: Kaynak etiketler.
        output_images_dir: Cogaltilmis gorseller ciktisi.
        output_labels_dir: Cogaltilmis etiketler ciktisi.
        num_augmentations: Her goruntu basi cogaltma sayisi.
        include_night: Gece augmentasyonu dahil et.

    Returns:
        Olusturulan yeni goruntu sayisi.
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out_img = Path(output_images_dir)
    out_lbl = Path(output_labels_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    aug_day = get_military_augmentation()
    aug_night = get_night_augmentation() if include_night else None

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    created = 0
    for img_path in tqdm(images, desc="Augmentasyon"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = read_yolo_labels(lbl_path)

        if not bboxes:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for i in range(num_augmentations):
            # Son iterasyonda gece augmentasyonu uygula
            if include_night and i == num_augmentations - 1 and aug_night:
                transform = aug_night
            else:
                transform = aug_day

            try:
                result = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
            except Exception:
                continue

            if not result["bboxes"]:
                continue

            aug_name = f"{img_path.stem}_aug{i}"
            aug_img_path = out_img / f"{aug_name}.jpg"
            aug_lbl_path = out_lbl / f"{aug_name}.txt"

            cv2.imwrite(str(aug_img_path), result["image"])
            write_yolo_labels(
                aug_lbl_path, result["bboxes"], result["class_labels"]
            )
            created += 1

    print(f"[+] {created} yeni goruntu olusturuldu")
    return created


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Veri cogaltma")
    parser.add_argument("--images", required=True, help="Gorseller dizini")
    parser.add_argument("--labels", required=True, help="Etiketler dizini")
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--num-aug", type=int, default=3, help="Cogaltma sayisi")
    parser.add_argument("--no-night", action="store_true", help="Gece augmentasyonu devre disi")
    args = parser.parse_args()

    augment_dataset(
        args.images,
        args.labels,
        args.output_images,
        args.output_labels,
        num_augmentations=args.num_aug,
        include_night=not args.no_night,
    )

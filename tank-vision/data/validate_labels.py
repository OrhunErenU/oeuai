"""Etiket kalite kontrol araci.

YOLO formatindaki etiket dosyalarini dogrular:
- Eksik goruntu/etiket cifti
- Gecersiz koordinatlar (0-1 araliginda olmayan)
- Gecersiz sinif ID'leri
- Minimum bbox boyutu
- Sinif dagilimi raporu
- Bozuk goruntu tespiti
"""

from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from data.class_mapping import PRIMARY_CLASSES


def validate_dataset(
    images_dir: str,
    labels_dir: str,
    num_classes: int = 7,
    min_bbox_size: float = 0.001,
    check_images: bool = False,
) -> dict:
    """Veri setini dogrula ve rapor uret.

    Args:
        images_dir: Gorseller dizini.
        labels_dir: Etiketler dizini.
        num_classes: Beklenen sinif sayisi.
        min_bbox_size: Minimum bbox boyutu (normalize).
        check_images: Gorselleri OpenCV ile okumaya calis (yavas ama kapsamli).

    Returns:
        Dogrulama raporu dict'i.
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions}
    labels = {f.stem: f for f in lbl_dir.glob("*.txt")}

    report = {
        "total_images": len(images),
        "total_labels": len(labels),
        "missing_labels": [],
        "missing_images": [],
        "invalid_coords": [],
        "invalid_classes": [],
        "tiny_bboxes": [],
        "empty_labels": [],
        "corrupt_images": [],
        "class_distribution": Counter(),
        "bbox_count": 0,
        "errors": 0,
    }

    # Eksik etiket dosyalari
    for stem in images:
        if stem not in labels:
            report["missing_labels"].append(stem)

    # Eksik goruntu dosyalari
    for stem in labels:
        if stem not in images:
            report["missing_images"].append(stem)

    # Etiket dosyalarini dogrula
    for stem, lbl_path in tqdm(labels.items(), desc="Etiket dogrulama"):
        content = lbl_path.read_text().strip()
        if not content:
            report["empty_labels"].append(stem)
            continue

        for line_num, line in enumerate(content.split("\n"), 1):
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 5:
                report["invalid_coords"].append((stem, line_num, "yetersiz alan"))
                report["errors"] += 1
                continue

            try:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
            except ValueError:
                report["invalid_coords"].append((stem, line_num, "gecersiz sayi"))
                report["errors"] += 1
                continue

            # Sinif ID kontrolu
            if cls_id < 0 or cls_id >= num_classes:
                report["invalid_classes"].append((stem, line_num, cls_id))
                report["errors"] += 1

            # Koordinat aralik kontrolu
            for val, name in [
                (x_center, "x_center"), (y_center, "y_center"),
                (w, "width"), (h, "height"),
            ]:
                if val < 0.0 or val > 1.0:
                    report["invalid_coords"].append(
                        (stem, line_num, f"{name}={val:.4f} aralik disi")
                    )
                    report["errors"] += 1

            # Minimum bbox boyutu
            if w < min_bbox_size or h < min_bbox_size:
                report["tiny_bboxes"].append((stem, line_num, f"w={w:.6f}, h={h:.6f}"))

            report["class_distribution"][cls_id] += 1
            report["bbox_count"] += 1

    # Bozuk goruntu kontrolu (opsiyonel, yavas)
    if check_images:
        for stem, img_path in tqdm(images.items(), desc="Goruntu kontrolu"):
            img = cv2.imread(str(img_path))
            if img is None:
                report["corrupt_images"].append(stem)
                report["errors"] += 1

    # Rapor yazdir
    _print_report(report)
    return report


def _print_report(report: dict):
    """Dogrulama raporunu yazdir."""
    print("\n" + "=" * 60)
    print("VERI SETI DOGRULAMA RAPORU")
    print("=" * 60)

    print(f"\nToplam goruntu: {report['total_images']}")
    print(f"Toplam etiket:  {report['total_labels']}")
    print(f"Toplam bbox:    {report['bbox_count']}")
    print(f"Toplam hata:    {report['errors']}")

    if report["missing_labels"]:
        print(f"\n[!] Eksik etiketler: {len(report['missing_labels'])}")
        for s in report["missing_labels"][:5]:
            print(f"    - {s}")
        if len(report["missing_labels"]) > 5:
            print(f"    ... ve {len(report['missing_labels']) - 5} daha")

    if report["missing_images"]:
        print(f"\n[!] Eksik gorseller: {len(report['missing_images'])}")

    if report["invalid_coords"]:
        print(f"\n[!] Gecersiz koordinatlar: {len(report['invalid_coords'])}")
        for item in report["invalid_coords"][:5]:
            print(f"    - {item}")

    if report["invalid_classes"]:
        print(f"\n[!] Gecersiz siniflar: {len(report['invalid_classes'])}")

    if report["empty_labels"]:
        print(f"\n[!] Bos etiket dosyalari: {len(report['empty_labels'])}")

    if report["tiny_bboxes"]:
        print(f"\n[!] Cok kucuk bbox'lar: {len(report['tiny_bboxes'])}")

    if report["corrupt_images"]:
        print(f"\n[!] Bozuk gorseller: {len(report['corrupt_images'])}")

    print("\nSinif Dagilimi:")
    total = sum(report["class_distribution"].values())
    for cls_id in sorted(report["class_distribution"]):
        count = report["class_distribution"][cls_id]
        name = PRIMARY_CLASSES.get(cls_id, f"bilinmeyen_{cls_id}")
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  [{cls_id}] {name:12s}: {count:7d} ({pct:5.1f}%) {bar}")

    status = "BASARILI" if report["errors"] == 0 else "HATALI"
    print(f"\nSonuc: {status}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Etiket dogrulama")
    parser.add_argument("--images", required=True, help="Gorseller dizini")
    parser.add_argument("--labels", required=True, help="Etiketler dizini")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    validate_dataset(args.images, args.labels, args.num_classes, check_images=args.check_images)

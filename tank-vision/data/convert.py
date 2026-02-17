"""Veri seti format donusturucu.

COCO JSON ve Pascal VOC XML formatlarini YOLO TXT formatina donusturur.
Sinif ID'lerini ana 7-sinif semasina yeniden esler.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from data.class_mapping import COCO_TO_PRIMARY


def coco_to_yolo(
    json_path: str,
    images_dir: str,
    output_dir: str,
    class_mapping: dict | None = None,
) -> int:
    """COCO JSON formatini YOLO TXT formatina donustur.

    Args:
        json_path: COCO annotations JSON dosya yolu.
        images_dir: COCO gorselleri dizini.
        output_dir: YOLO etiket dosyalari cikti dizini.
        class_mapping: COCO category_id -> hedef class_id esleme.
            None ise COCO_TO_PRIMARY kullanilir.

    Returns:
        Donusturulen goruntu sayisi.
    """
    if class_mapping is None:
        class_mapping = COCO_TO_PRIMARY

    with open(json_path, "r") as f:
        coco = json.load(f)

    # COCO category_id -> bizim esleme
    cat_map = {}
    for cat in coco["categories"]:
        coco_cat_id = cat["id"]
        if coco_cat_id in class_mapping:
            cat_map[coco_cat_id] = class_mapping[coco_cat_id]

    # Image ID -> bilgi esleme
    img_info = {img["id"]: img for img in coco["images"]}

    # Image ID -> annotations gruplama
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_map:
            continue
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, annotations in tqdm(img_annotations.items(), desc="COCO -> YOLO"):
        info = img_info[img_id]
        img_w = info["width"]
        img_h = info["height"]
        filename = Path(info["file_name"]).stem

        lines = []
        for ann in annotations:
            target_cls = cat_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height

            # YOLO formatina donustur (normalize merkez koordinatlari)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Sinir kontrol
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if lines:
            label_path = out / f"{filename}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            converted += 1

    print(f"[+] {converted} goruntu donusturuldu -> {out}")
    return converted


def voc_to_yolo(
    xml_dir: str,
    output_dir: str,
    class_mapping: dict[str, int],
    img_width: int | None = None,
    img_height: int | None = None,
) -> int:
    """Pascal VOC XML formatini YOLO TXT formatina donustur.

    Args:
        xml_dir: VOC XML dosyalari dizini.
        output_dir: YOLO etiket dosyalari cikti dizini.
        class_mapping: VOC sinif adi -> hedef class_id esleme.
            Ornek: {"person": 2, "car": 4, "tank": 1}
        img_width: Sabit goruntu genisligi (None ise XML'den okunur).
        img_height: Sabit goruntu yuksekligi (None ise XML'den okunur).

    Returns:
        Donusturulen dosya sayisi.
    """
    xml_path = Path(xml_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for xml_file in tqdm(list(xml_path.glob("*.xml")), desc="VOC -> YOLO"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Goruntu boyutlari
        size = root.find("size")
        if size is not None and img_width is None:
            w = int(size.find("width").text)
            h = int(size.find("height").text)
        else:
            w = img_width or 640
            h = img_height or 640

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_mapping:
                continue

            target_cls = class_mapping[name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # YOLO formatina donustur
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            norm_w = (xmax - xmin) / w
            norm_h = (ymax - ymin) / h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if lines:
            label_path = out / f"{xml_file.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            converted += 1

    print(f"[+] {converted} dosya donusturuldu -> {out}")
    return converted


def remap_classes(label_dir: str, old_to_new: dict[int, int]) -> int:
    """Mevcut YOLO etiket dosyalarinda sinif ID'lerini yeniden esle.

    Args:
        label_dir: YOLO etiket dosyalari dizini.
        old_to_new: Eski class_id -> yeni class_id esleme.

    Returns:
        Guncellenen dosya sayisi.
    """
    label_path = Path(label_dir)
    updated = 0

    for txt_file in tqdm(list(label_path.glob("*.txt")), desc="Sinif yeniden esleme"):
        lines = txt_file.read_text().strip().split("\n")
        new_lines = []
        changed = False

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            old_cls = int(parts[0])
            if old_cls in old_to_new:
                parts[0] = str(old_to_new[old_cls])
                changed = True
                new_lines.append(" ".join(parts))
            elif old_cls not in old_to_new.values():
                # Eslenmeyen sinif - atla
                changed = True
                continue
            else:
                new_lines.append(line.strip())

        if changed:
            if new_lines:
                txt_file.write_text("\n".join(new_lines) + "\n")
            else:
                txt_file.unlink()
            updated += 1

    print(f"[+] {updated} dosya guncellendi")
    return updated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format donusturucu")
    sub = parser.add_subparsers(dest="command")

    # COCO donusumu
    coco_p = sub.add_parser("coco", help="COCO JSON -> YOLO TXT")
    coco_p.add_argument("--json", required=True, help="COCO annotations JSON")
    coco_p.add_argument("--images", required=True, help="COCO gorselleri dizini")
    coco_p.add_argument("--output", required=True, help="Cikti dizini")

    # VOC donusumu
    voc_p = sub.add_parser("voc", help="Pascal VOC XML -> YOLO TXT")
    voc_p.add_argument("--xml-dir", required=True, help="VOC XML dizini")
    voc_p.add_argument("--output", required=True, help="Cikti dizini")

    # Sinif yeniden esleme
    remap_p = sub.add_parser("remap", help="Sinif ID yeniden esleme")
    remap_p.add_argument("--label-dir", required=True, help="Etiket dizini")
    remap_p.add_argument("--mapping", required=True, help="Esleme JSON dosyasi")

    args = parser.parse_args()

    if args.command == "coco":
        coco_to_yolo(args.json, args.images, args.output)
    elif args.command == "voc":
        print("VOC donusumu icin class_mapping parametresi gerekli.")
    elif args.command == "remap":
        mapping = json.loads(Path(args.mapping).read_text())
        mapping = {int(k): int(v) for k, v in mapping.items()}
        remap_classes(args.label_dir, mapping)

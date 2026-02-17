"""Acik kaynak veri seti indirme araclari.

Desteklenen veri setleri:
- Seraphim Drone Detection (HuggingFace)
- COCO 2017 (person, bird, car, truck, bus, airplane siniflari)
- Roboflow askeri arac/tank veri setleri
- Roboflow silah veri setleri
- Roboflow ucak/helikopter veri setleri
"""

import json
import shutil
from pathlib import Path

from tqdm import tqdm


def download_seraphim_drone(output_dir: str) -> Path:
    """Seraphim Drone Detection veri setini HuggingFace'den indir.

    YOLO formatinda ~75K goruntu icerir.
    Kaynak: lgrzybowski/seraphim-drone-detection-dataset
    """
    from huggingface_hub import snapshot_download

    out = Path(output_dir) / "seraphim_drone"
    out.mkdir(parents=True, exist_ok=True)

    print("[*] Seraphim Drone veri seti indiriliyor...")
    snapshot_download(
        repo_id="lgrzybowski/seraphim-drone-detection-dataset",
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"[+] Seraphim Drone -> {out}")
    return out


def download_coco_subset(output_dir: str, year: str = "2017") -> Path:
    """COCO veri setinden belirli siniflari indir ve cikar.

    Indirilen siniflar: person(0), car(2), bus(5), truck(7), bird(14), airplane(4)
    """
    import urllib.request
    import zipfile

    out = Path(output_dir) / "coco_subset"
    out.mkdir(parents=True, exist_ok=True)

    base_url = f"http://images.cocodataset.org/zips/train{year}.zip"
    ann_url = f"http://images.cocodataset.org/annotations/annotations_trainval{year}.zip"

    # Annotations indir
    ann_zip = out / f"annotations_{year}.zip"
    if not ann_zip.exists():
        print(f"[*] COCO {year} annotations indiriliyor...")
        urllib.request.urlretrieve(ann_url, str(ann_zip))

    # Annotations cikar
    ann_dir = out / "annotations"
    if not ann_dir.exists():
        print("[*] Annotations cikartiliyor...")
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(str(out))

    print(f"[+] COCO annotations -> {ann_dir}")
    print("[!] COCO gorselleri cok buyuk (~18GB). Manuel indirme onerilir:")
    print(f"    {base_url}")
    print("    Indirdikten sonra data/convert.py ile YOLO formatina donusturun.")

    return out


def download_roboflow_dataset(
    output_dir: str,
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    dataset_format: str = "yolov11",
) -> Path:
    """Roboflow'dan veri seti indir.

    Args:
        output_dir: Cikti dizini.
        workspace: Roboflow workspace adi.
        project: Proje adi.
        version: Veri seti versiyonu.
        api_key: Roboflow API anahtari.
        dataset_format: Indirme formati.
    """
    from roboflow import Roboflow

    out = Path(output_dir) / f"roboflow_{project}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Roboflow veri seti indiriliyor: {workspace}/{project} v{version}...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(dataset_format, location=str(out))

    print(f"[+] Roboflow {project} -> {out}")
    return out


def download_military_tanks(output_dir: str, api_key: str) -> Path:
    """Askeri tank veri setlerini Roboflow'dan indir."""
    return download_roboflow_dataset(
        output_dir=output_dir,
        workspace="ds",
        project="military-vehicles-bkfhm",
        version=1,
        api_key=api_key,
    )


def download_weapons(output_dir: str, api_key: str) -> Path:
    """Silah tespit veri setlerini Roboflow'dan indir."""
    return download_roboflow_dataset(
        output_dir=output_dir,
        workspace="weapons-detection",
        project="weapons-detection-nkmht",
        version=1,
        api_key=api_key,
    )


def download_aircraft(output_dir: str, api_key: str) -> Path:
    """Ucak/helikopter veri setlerini Roboflow'dan indir."""
    return download_roboflow_dataset(
        output_dir=output_dir,
        workspace="aircraft",
        project="aircraft-detection-fyyob",
        version=1,
        api_key=api_key,
    )


def download_all(output_dir: str, roboflow_api_key: str | None = None):
    """Tum veri setlerini indir.

    Args:
        output_dir: Ana cikti dizini.
        roboflow_api_key: Roboflow API anahtari (None ise Roboflow veri setleri atlanir).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tank Vision AI - Veri Seti Indirme")
    print("=" * 60)

    # HuggingFace veri setleri (API anahtari gerektirmez)
    download_seraphim_drone(str(out))
    download_coco_subset(str(out))

    # Roboflow veri setleri
    if roboflow_api_key:
        download_military_tanks(str(out), roboflow_api_key)
        download_weapons(str(out), roboflow_api_key)
        download_aircraft(str(out), roboflow_api_key)
    else:
        print("[!] Roboflow API anahtari verilmedi. Roboflow veri setleri atlanacak.")
        print("    API anahtari almak icin: https://roboflow.com/settings")

    print("=" * 60)
    print("[+] Indirme tamamlandi!")
    print(f"    Cikti dizini: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Veri seti indirici")
    parser.add_argument("--output", default="datasets/raw", help="Cikti dizini")
    parser.add_argument("--roboflow-key", default=None, help="Roboflow API anahtari")
    args = parser.parse_args()

    download_all(args.output, args.roboflow_key)

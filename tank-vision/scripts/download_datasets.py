"""Genisletilmis veri seti indirme scripti.

Indirilen veri setleri:
1. D-Fire (duman/ates) - GitHub
2. Tank modelleri - Kaggle
3. Silah tespiti - Kaggle
4. Askeri veri seti - Kaggle

Roboflow veri setleri icin API key gerekli - bunlari kullanici indirecek.
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def run_cmd(cmd, cwd=None):
    """Komutu calistir ve ciktisini goster."""
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  HATA: {result.stderr[:500]}")
        return False
    return True


def download_dfire():
    """D-Fire Dataset - 21,000+ duman/ates goruntusu."""
    dest = DATASETS_DIR / "dfire"
    if dest.exists() and len(list((dest).rglob("*.jpg"))) > 1000:
        print("[OK] D-Fire zaten indirilmis.")
        return True

    print("\n[1/4] D-Fire Dataset indiriliyor (duman/ates)...")
    dest.mkdir(parents=True, exist_ok=True)

    # GitHub'dan clone et
    if not (dest / "DFireDataset").exists():
        success = run_cmd(
            f'git clone --depth 1 https://github.com/gaiasd/DFireDataset "{dest / "DFireDataset"}"'
        )
        if not success:
            print("  D-Fire indirilemedi. Manuel indirme gerekebilir.")
            print("  URL: https://github.com/gaiasd/DFireDataset")
            return False

    print("[OK] D-Fire indirildi!")
    return True


def download_kaggle_dataset(dataset_id, dest_name, description):
    """Kaggle veri seti indir."""
    dest = DATASETS_DIR / dest_name
    if dest.exists() and any(dest.rglob("*")):
        print(f"[OK] {description} zaten indirilmis.")
        return True

    print(f"\n{description} indiriliyor...")
    dest.mkdir(parents=True, exist_ok=True)

    # kaggle CLI kontrol
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  Kaggle CLI bulunamadi. Kuruluyor...")
        run_cmd(f'"{sys.executable}" -m pip install kaggle --quiet')

    # kaggle.json kontrol
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"  UYARI: {kaggle_json} bulunamadi!")
        print(f"  1. https://www.kaggle.com/settings adresinden API token indir")
        print(f"  2. {kaggle_json} olarak kaydet")
        print(f"  3. Bu scripti tekrar calistir")
        print(f"  VEYA manuel indir: https://www.kaggle.com/datasets/{dataset_id}")
        return False

    success = run_cmd(
        f'kaggle datasets download -d {dataset_id} -p "{dest}" --unzip',
        cwd=str(dest)
    )
    if not success:
        print(f"  {description} indirilemedi. Manuel indirme:")
        print(f"  URL: https://www.kaggle.com/datasets/{dataset_id}")
        return False

    print(f"[OK] {description} indirildi!")
    return True


def check_roboflow_datasets():
    """Roboflow veri setleri icin talimatlar goster."""
    print("\n" + "=" * 60)
    print("ROBOFLOW VERI SETLERI (Manuel indirme gerekli)")
    print("=" * 60)

    datasets = [
        {
            "name": "Tank Modelleri (100+ model)",
            "url": "https://universe.roboflow.com/garage-rjbkw/tanks-detection-d0ayl",
            "workspace": "garage-rjbkw",
            "project": "tanks-detection-d0ayl",
        },
        {
            "name": "Askeri Tanklar (5540 goruntu)",
            "url": "https://universe.roboflow.com/militarytanks-c2etq/military-tanks/dataset/1",
            "workspace": "militarytanks-c2etq",
            "project": "military-tanks",
        },
        {
            "name": "Sivil/Asker Tespiti",
            "url": "https://universe.roboflow.com/camouflage/soldier-civilian-detection",
            "workspace": "camouflage",
            "project": "soldier-civilian-detection",
        },
        {
            "name": "Silah Tespiti (22K goruntu)",
            "url": "https://universe.roboflow.com/xian-douglas/weapondetection-xx3lz",
            "workspace": "xian-douglas",
            "project": "weapondetection-xx3lz",
        },
        {
            "name": "Tank Parcalari Segmentasyon (namlu tespiti)",
            "url": "https://universe.roboflow.com/tank-project/tank-parts-segmentation-d78dw/dataset/10",
            "workspace": "tank-project",
            "project": "tank-parts-segmentation-d78dw",
        },
    ]

    print("\nRoboflow API key ile indirmek icin:")
    print("1. https://roboflow.com adresinden ucretsiz hesap ac")
    print("2. Settings > API Key kisminden key'i kopyala")
    print("3. Asagidaki kodu calistir:\n")
    print("```python")
    print("from roboflow import Roboflow")
    print("rf = Roboflow(api_key='SENIN_API_KEY')")

    for ds in datasets:
        print(f"\n# {ds['name']}")
        print(f"# URL: {ds['url']}")
        print(f"project = rf.workspace('{ds['workspace']}').project('{ds['project']}')")
        print(f"project.version(1).download('yolov8')")

    print("```")
    print("\nVEYA her veri setini yukardaki URL'lerden")
    print("'Download Dataset > YOLOv8' secenegiyle indirebilirsin.")


def main():
    print("=" * 60)
    print("  Tank Vision AI - Veri Seti Indirici")
    print("=" * 60)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. D-Fire (duman/ates) - GitHub'dan
    results["D-Fire"] = download_dfire()

    # 2. Tank modelleri - Kaggle
    results["Tank Modelleri"] = download_kaggle_dataset(
        "antoreepjana/military-tanks-dataset-images",
        "kaggle_tanks",
        "[2/4] Tank Modelleri (Kaggle)"
    )

    # 3. Silah tespiti - Kaggle
    results["Silah Tespiti"] = download_kaggle_dataset(
        "raghavnanjappan/weapon-dataset-for-yolov5",
        "kaggle_weapons",
        "[3/4] Silah Tespiti (Kaggle)"
    )

    # 4. Askeri veri seti - Kaggle
    results["Askeri Varliklar"] = download_kaggle_dataset(
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "kaggle_military",
        "[4/4] Askeri Varliklar 12 Sinif (Kaggle)"
    )

    # Roboflow talimatlar
    check_roboflow_datasets()

    # Ozet
    print("\n" + "=" * 60)
    print("  INDIRME OZETI")
    print("=" * 60)
    for name, ok in results.items():
        status = "BASARILI" if ok else "BEKLIYOR"
        print(f"  [{status}] {name}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  {len(failed)} veri seti indirilemedi - yukardaki talimatlara bak.")
    else:
        print("\n  Tum otomatik indirmeler tamamlandi!")


if __name__ == "__main__":
    main()

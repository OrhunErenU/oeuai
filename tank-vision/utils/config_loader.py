"""YAML konfigurasyon yukleyici."""

import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """YAML konfigurasyon dosyasini yukle.

    Ortam degiskenlerini ($VAR veya ${VAR}) otomatik genisletir.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return _expand_env_vars(config)


def _expand_env_vars(obj):
    """Dict/list icindeki string degerlerde ortam degiskenlerini genislet."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def get_project_root() -> Path:
    """Proje kok dizinini dondur (config/ klasorunun bulundugu yer)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir():
            return current
        current = current.parent
    raise RuntimeError("Proje kok dizini bulunamadi")

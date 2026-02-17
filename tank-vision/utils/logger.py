"""Loglama yapilandirmasi."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "tank_vision",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Logger olustur ve yapilandir.

    Args:
        name: Logger adi.
        level: Log seviyesi.
        log_file: Opsiyonel log dosya yolu.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konsol handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Dosya handler (opsiyonel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

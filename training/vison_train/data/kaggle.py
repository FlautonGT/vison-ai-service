"""Kaggle download helpers for Vast.ai dataset preparation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def kaggle_credentials_available() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    return creds_path.exists()


def require_kaggle_credentials() -> None:
    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials were not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place kaggle.json in ~/.kaggle."
        )


def _api():
    require_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(dataset_ref: str, output_dir: str | Path, unzip: bool = True, force: bool = False) -> dict[str, Any]:
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".download_complete.json"
    if marker.exists() and not force:
        return {
            "dataset_ref": dataset_ref,
            "output_dir": str(target_dir),
            "status": "skipped_existing",
        }

    api = _api()
    api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=unzip, quiet=False)
    marker.write_text(f'{{"dataset_ref": "{dataset_ref}", "status": "downloaded"}}\n', encoding="utf-8")
    return {
        "dataset_ref": dataset_ref,
        "output_dir": str(target_dir),
        "status": "downloaded",
    }

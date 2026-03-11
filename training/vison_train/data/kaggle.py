"""Kaggle download helpers for Vast.ai dataset preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any


def _legacy_creds_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _access_token_paths() -> list[Path]:
    base = Path.home() / ".kaggle"
    return [base / "access_token", base / "access_token.txt"]


def _has_legacy_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return _legacy_creds_path().exists()


def _has_token_credentials() -> bool:
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    return any(path.exists() for path in _access_token_paths())


def kaggle_credentials_available() -> bool:
    return _has_legacy_credentials() or _has_token_credentials()


def require_kaggle_credentials() -> None:
    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials were not found. Set KAGGLE_USERNAME/KAGGLE_KEY, set KAGGLE_API_TOKEN, "
            "place kaggle.json in ~/.kaggle, or place access_token in ~/.kaggle."
        )


def _api():
    require_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _download_with_kaggle_api(dataset_ref: str, target_dir: Path, unzip: bool) -> None:
    api = _api()
    api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=unzip, quiet=False)


def _download_with_kagglehub(dataset_ref: str, target_dir: Path, force: bool) -> None:
    import kagglehub

    cached_path = Path(kagglehub.dataset_download(dataset_ref, force_download=force)).resolve()
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if cached_path == target_dir:
        return
    if cached_path.is_dir():
        shutil.copytree(cached_path, target_dir, dirs_exist_ok=True)
        return
    shutil.copy2(cached_path, target_dir / cached_path.name)


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

    if _has_legacy_credentials():
        _download_with_kaggle_api(dataset_ref, target_dir, unzip=unzip)
        auth_mode = "kaggle_api"
    else:
        _download_with_kagglehub(dataset_ref, target_dir, force=force)
        auth_mode = "kagglehub_token"

    marker.write_text(
        json.dumps({"dataset_ref": dataset_ref, "status": "downloaded", "auth_mode": auth_mode}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_ref": dataset_ref,
        "output_dir": str(target_dir),
        "status": "downloaded",
        "auth_mode": auth_mode,
    }

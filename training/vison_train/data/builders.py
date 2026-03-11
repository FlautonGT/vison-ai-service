"""Dataset-specific manifest builders for Vast.ai training jobs."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .manifests import save_manifest

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_TOKENS = {"real", "live", "bona_fide", "bonafide", "genuine", "authentic", "none"}
FAKE_TOKENS = {"fake", "attack", "spoof", "synthetic", "ai", "deepfake", "replay", "print", "mask"}
UTKFACE_RE = re.compile(r"^(\d+)_(\d)_(\d)_")
CELEBA_ATTRS = {
    "Eyeglasses": "eyeglasses",
    "Wearing_Hat": "hat_cap",
}
CELEBAMASK_LABELS = {
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_g": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_r": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "neck_l": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}


def _list_images(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _source_id(path: Path) -> str:
    return _slug(path.name or path.parent.name)


def _detect_binary_label(path: Path) -> int | None:
    tokens = {_slug(part) for part in path.parts}
    if tokens & FAKE_TOKENS:
        return 1
    if tokens & REAL_TOKENS:
        return 0
    return None


def _detect_attack_type(path: Path) -> str:
    lowered = " ".join(_slug(part) for part in path.parts)
    for token in ["print", "replay", "mask", "paper", "screen", "photo", "video", "synthetic"]:
        if token in lowered:
            return token
    return "unknown"


def _quality_proxy_score(image_path: str) -> float:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        brightness_score = max(0.0, 100.0 - abs(brightness - 140.0) * 0.7)
        sharpness_score = min(100.0, sharpness / 12.0)
        return round(float(np.clip((brightness_score * 0.45) + (sharpness_score * 0.55), 0.0, 100.0)), 4)
    except Exception:
        return 0.0


def build_deepfake_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        for image_path in _list_images(dataset_dir):
            label = _detect_binary_label(image_path)
            if label is None:
                continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": int(label),
                    "source_dataset": dataset_dir.name,
                    "region_proxy": "unknown",
                }
            )
    return pd.DataFrame(rows)


def build_pad_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        for image_path in _list_images(dataset_dir):
            label = _detect_binary_label(image_path)
            if label is None:
                continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "is_attack": int(label),
                    "attack_type": _detect_attack_type(image_path) if int(label) == 1 else "bona_fide",
                    "source_dataset": dataset_dir.name,
                    "region_proxy": "unknown",
                }
            )
    return pd.DataFrame(rows)


def build_identity_manifest(dataset_dirs: Iterable[str | Path], min_images_per_subject: int = 2) -> pd.DataFrame:
    rows = []
    subject_counter = 0
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        for leaf_dir in [path for path in dataset_dir.rglob("*") if path.is_dir()]:
            images = [item for item in leaf_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS] if leaf_dir.exists() else []
            if len(images) < min_images_per_subject:
                continue
            subject_id = f"{_source_id(dataset_dir)}_{subject_counter:06d}"
            subject_counter += 1
            for image_path in sorted(images):
                capture_type = "document" if any(token in _slug(str(image_path.parent)) for token in ["id", "document", "card", "passport"]) else "selfie"
                rows.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "subject_id": subject_id,
                        "capture_type": capture_type,
                        "source_dataset": dataset_dir.name,
                        "region_proxy": "unknown",
                    }
                )
    return pd.DataFrame(rows)


def generate_verification_pairs(frame: pd.DataFrame, positives_per_subject: int = 2, negatives_per_subject: int = 2, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    grouped = {subject_id: group.reset_index(drop=True) for subject_id, group in frame.groupby("subject_id")}
    subject_ids = sorted(grouped.keys())
    for subject_id, group in grouped.items():
        if len(group) < 2:
            continue
        indices = list(range(len(group)))
        rng.shuffle(indices)
        positive_count = 0
        for left_idx in indices:
            for right_idx in indices:
                if right_idx <= left_idx:
                    continue
                rows.append(
                    {
                        "left_image_path": str(group.iloc[left_idx]["image_path"]),
                        "right_image_path": str(group.iloc[right_idx]["image_path"]),
                        "is_match": 1,
                        "left_subject_id": subject_id,
                        "right_subject_id": subject_id,
                    }
                )
                positive_count += 1
                if positive_count >= positives_per_subject:
                    break
            if positive_count >= positives_per_subject:
                break

        negative_subjects = [item for item in subject_ids if item != subject_id]
        rng.shuffle(negative_subjects)
        for other_subject in negative_subjects[:negatives_per_subject]:
            left_row = group.sample(n=1, random_state=seed).iloc[0]
            right_row = grouped[other_subject].sample(n=1, random_state=seed).iloc[0]
            rows.append(
                {
                    "left_image_path": str(left_row["image_path"]),
                    "right_image_path": str(right_row["image_path"]),
                    "is_match": 0,
                    "left_subject_id": subject_id,
                    "right_subject_id": other_subject,
                }
            )
    return pd.DataFrame(rows)


def build_age_gender_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        csv_candidates = list(dataset_dir.rglob("fairface_label_*.csv")) + list(dataset_dir.rglob("fairface_label_train.csv"))
        if csv_candidates:
            for csv_path in csv_candidates:
                frame = pd.read_csv(csv_path)
                image_col = "file" if "file" in frame.columns else frame.columns[0]
                for _, row in frame.iterrows():
                    image_path = dataset_dir / str(row[image_col])
                    gender_raw = str(row.get("gender", "")).strip().lower()
                    gender_id = 1 if gender_raw.startswith("female") else 0
                    age_group = str(row.get("age", "unknown"))
                    age_proxy = {
                        "0-2": 1,
                        "3-9": 6,
                        "10-19": 15,
                        "20-29": 25,
                        "30-39": 35,
                        "40-49": 45,
                        "50-59": 55,
                        "60-69": 65,
                        "more than 70": 75,
                    }.get(age_group.lower(), 30)
                    rows.append(
                        {
                            "image_path": str(image_path.resolve()),
                            "age": float(age_proxy),
                            "gender_id": int(gender_id),
                            "age_group": age_group,
                            "source_dataset": dataset_dir.name,
                            "region_proxy": "unknown",
                        }
                    )
            continue

        for image_path in _list_images(dataset_dir):
            match = UTKFACE_RE.match(image_path.name)
            if match:
                rows.append(
                    {
                        "image_path": str(image_path),
                        "age": float(match.group(1)),
                        "gender_id": int(match.group(2)),
                        "age_group": _age_group(float(match.group(1))),
                        "source_dataset": dataset_dir.name,
                        "region_proxy": "unknown",
                    }
                )
                continue

            parent_name = image_path.parent.name
            if parent_name.isdigit():
                age = float(parent_name)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "age": age,
                        "gender_id": -1,
                        "age_group": _age_group(age),
                        "source_dataset": dataset_dir.name,
                        "region_proxy": "unknown",
                    }
                )
    return pd.DataFrame(rows)


def _age_group(age: float) -> str:
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 35:
        return "young_adult"
    if age < 55:
        return "adult"
    return "senior"


def build_attribute_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        csv_candidates = list(dataset_dir.rglob("list_attr_celeba.csv"))
        if csv_candidates:
            for csv_path in csv_candidates:
                frame = pd.read_csv(csv_path)
                image_col = "image_id" if "image_id" in frame.columns else frame.columns[0]
                for _, row in frame.iterrows():
                    image_path = next((candidate for candidate in [dataset_dir / str(row[image_col]), dataset_dir / "img_align_celeba" / str(row[image_col]), dataset_dir / "img_align_celeba" / "img_align_celeba" / str(row[image_col])] if candidate.exists()), None)
                    if image_path is None:
                        continue
                    attrs = {name: -1.0 for name in ["eyeglasses", "sunglasses", "mask", "hat_cap", "major_occlusion"]}
                    for celeba_col, target_col in CELEBA_ATTRS.items():
                        if celeba_col in row:
                            attrs[target_col] = 1.0 if float(row[celeba_col]) > 0 else 0.0
                    rows.append(
                        {
                            "image_path": str(image_path.resolve()),
                            **attrs,
                            "source_dataset": dataset_dir.name,
                            "region_proxy": "unknown",
                        }
                    )
            continue

        for image_path in _list_images(dataset_dir):
            label_name = _slug(image_path.parent.name)
            attrs = {
                "eyeglasses": -1.0,
                "sunglasses": -1.0,
                "mask": -1.0,
                "hat_cap": -1.0,
                "major_occlusion": -1.0,
            }
            if label_name in {"glasses", "eyeglasses"}:
                attrs.update({"eyeglasses": 1.0, "sunglasses": 0.0, "mask": 0.0, "major_occlusion": 0.0})
            elif label_name == "sunglasses":
                attrs.update({"eyeglasses": 0.0, "sunglasses": 1.0, "mask": 0.0, "major_occlusion": 1.0})
            elif label_name in {"mask", "coverings"}:
                attrs.update({"eyeglasses": 0.0, "sunglasses": 0.0, "mask": 1.0, "major_occlusion": 1.0})
            elif label_name in {"none", "plain", "without_mask"}:
                attrs.update({"eyeglasses": 0.0, "sunglasses": 0.0, "mask": 0.0, "major_occlusion": 0.0})
            elif label_name in {"hand", "other"}:
                attrs.update({"major_occlusion": 1.0})
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    **attrs,
                    "source_dataset": dataset_dir.name,
                    "region_proxy": "unknown",
                }
            )
    return pd.DataFrame(rows)


def build_quality_manifest(source_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source_frame.iterrows():
        rows.append(
            {
                "image_path": str(row["image_path"]),
                "quality_score": _quality_proxy_score(str(row["image_path"])),
                "source_dataset": row.get("source_dataset", "unknown"),
                "region_proxy": row.get("region_proxy", "unknown"),
                "age_group": row.get("age_group", "unknown"),
                "gender": row.get("gender", row.get("gender_id", "unknown")),
            }
        )
    return pd.DataFrame(rows)


def build_celebamaskhq_parser_manifest(dataset_dirs: Iterable[str | Path], generated_mask_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(generated_mask_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        images = [path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".png"} and "img" in path.parent.name.lower()]
        if not images:
            continue
        for image_path in images:
            stem = image_path.stem
            mask_files = list(dataset_dir.rglob(f"{stem}_*.png"))
            if not mask_files:
                continue
            first_mask = Image.open(str(mask_files[0])).convert("L")
            combined = np.zeros((first_mask.height, first_mask.width), dtype=np.uint8)
            for mask_file in sorted(mask_files):
                label_key = mask_file.stem.split("_", 1)[1] if "_" in mask_file.stem else ""
                class_id = CELEBAMASK_LABELS.get(label_key)
                if class_id is None:
                    continue
                mask = np.array(Image.open(str(mask_file)).convert("L")) > 0
                combined[mask] = class_id
            if not np.any(combined):
                continue
            hash_id = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:12]
            output_path = output_dir / f"{hash_id}.png"
            Image.fromarray(combined).save(output_path)
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(output_path),
                    "source_dataset": dataset_dir.name,
                }
            )
    return pd.DataFrame(rows)


def write_manifest(frame: pd.DataFrame, output_path: str | Path) -> Path:
    target = Path(output_path).expanduser().resolve()
    save_manifest(frame, target)
    return target

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
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
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


def _existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _valid_images(paths: Iterable[Path]) -> list[Path]:
    return [path.resolve() for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTS and _is_valid_image(path)]


def _capture_type_for_path(path: Path) -> str:
    lowered = " ".join(_slug(part) for part in path.parts[-4:])
    if any(token in lowered for token in ["id", "document", "doc", "card", "passport", "license"]):
        return "document"
    return "selfie"


def _region_proxy_from_group(value: str) -> str:
    lowered = _slug(value)
    if any(token in lowered for token in ["indonesia", "indonesian"]):
        return "indonesia"
    if any(token in lowered for token in ["southeast_asia", "south_east_asia", "asean"]):
        return "southeast_asia"
    if any(token in lowered for token in ["south_asia", "east_asia", "middle_east", "asian", "asia"]):
        return "asia"
    return "global"


def _detect_binary_label(path: Path) -> int | None:
    context_parts = [_slug(part) for part in path.parts[-3:]]
    lowered = " ".join(context_parts)
    if any(token in lowered for token in FAKE_TOKENS):
        return 1
    if any(token in lowered for token in REAL_TOKENS):
        return 0
    return None


def _detect_attack_type(path: Path) -> str:
    lowered = " ".join(_slug(part) for part in path.parts[-4:])
    for token in ["print", "replay", "mask", "paper", "screen", "photo", "video", "synthetic"]:
        if token in lowered:
            return token
    return "unknown"


def _video_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS]


def _extract_video_frames(video_path: Path, cache_dir: Path, frames_per_video: int = 8) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_name = hashlib.md5(str(video_path.resolve()).encode("utf-8")).hexdigest()[:16]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_positions = [0]
        else:
            requested = max(1, min(frames_per_video, frame_count))
            frame_positions = sorted({int(position) for position in np.linspace(0, frame_count - 1, num=requested)})

        output_paths: list[Path] = []
        for index, frame_position in enumerate(frame_positions):
            output_path = cache_dir / f"{base_name}_{index:02d}.jpg"
            if output_path.exists():
                output_paths.append(output_path)
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            if cv2.imwrite(str(output_path), frame):
                output_paths.append(output_path)
        return output_paths
    finally:
        capture.release()


def _country_to_region(country_code: str) -> str:
    country = country_code.strip().upper()
    if country == "ID":
        return "indonesia"
    if country in {"BN", "KH", "LA", "MM", "MY", "PH", "SG", "TH", "TL", "VN"}:
        return "southeast_asia"
    if country:
        return "asia"
    return "unknown"


def _fairface_region_proxy(race_label: str) -> str:
    race = race_label.strip().lower()
    if "southeast asian" in race:
        return "southeast_asia"
    if any(token in race for token in ["east asian", "indian", "middle eastern"]):
        return "asia"
    return "global"


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
        frame_cache = dataset_dir / ".frame_cache"
        for image_path in _valid_images(_list_images(dataset_dir)):
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
        for video_path in _video_files(dataset_dir):
            label = _detect_binary_label(video_path)
            if label is None:
                continue
            for frame_path in _extract_video_frames(video_path.resolve(), frame_cache, frames_per_video=6):
                rows.append(
                    {
                        "image_path": str(frame_path.resolve()),
                        "label": int(label),
                        "source_dataset": dataset_dir.name,
                        "region_proxy": "unknown",
                    }
                )
    return pd.DataFrame(rows)


def build_pad_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        seen_paths: set[str] = set()
        frame_cache = dataset_dir / ".frame_cache"

        for csv_path in dataset_dir.rglob("asian_people.csv"):
            frame = pd.read_csv(csv_path)
            for _, row in frame.iterrows():
                region_proxy = _country_to_region(str(row.get("country", "")))
                selfie_rel = str(row.get("selfie_link", "")).lstrip("/")
                video_rel = str(row.get("video_link", "")).lstrip("/")
                selfie_path = next(
                    (
                        candidate
                        for candidate in [
                            csv_path.parent / "files" / selfie_rel,
                            dataset_dir / "files" / selfie_rel,
                            dataset_dir / selfie_rel,
                        ]
                        if candidate.exists()
                    ),
                    None,
                )
                if selfie_path is not None:
                    selfie_str = str(selfie_path.resolve())
                    if selfie_str not in seen_paths:
                        rows.append(
                            {
                                "image_path": selfie_str,
                                "is_attack": 0,
                                "attack_type": "bona_fide",
                                "source_dataset": dataset_dir.name,
                                "region_proxy": region_proxy,
                            }
                        )
                        seen_paths.add(selfie_str)

                video_path = next(
                    (
                        candidate
                        for candidate in [
                            csv_path.parent / "files" / video_rel,
                            dataset_dir / "files" / video_rel,
                            dataset_dir / video_rel,
                        ]
                        if candidate.exists()
                    ),
                    None,
                )
                if video_path is None:
                    continue
                for frame_path in _extract_video_frames(video_path.resolve(), frame_cache, frames_per_video=12):
                    frame_str = str(frame_path.resolve())
                    if frame_str in seen_paths:
                        continue
                    rows.append(
                        {
                            "image_path": frame_str,
                            "is_attack": 0,
                            "attack_type": "bona_fide",
                            "source_dataset": dataset_dir.name,
                            "region_proxy": region_proxy,
                        }
                    )
                    seen_paths.add(frame_str)

        for image_path in _valid_images(_list_images(dataset_dir)):
            label = _detect_binary_label(image_path)
            if label is None:
                continue
            image_str = str(image_path.resolve())
            if image_str in seen_paths:
                continue
            rows.append(
                {
                    "image_path": image_str,
                    "is_attack": int(label),
                    "attack_type": _detect_attack_type(image_path) if int(label) == 1 else "bona_fide",
                    "source_dataset": dataset_dir.name,
                    "region_proxy": "unknown",
                }
            )
            seen_paths.add(image_str)

        for video_path in _video_files(dataset_dir):
            label = _detect_binary_label(video_path)
            if label is None:
                continue
            for frame_path in _extract_video_frames(video_path.resolve(), frame_cache, frames_per_video=12):
                frame_str = str(frame_path.resolve())
                if frame_str in seen_paths:
                    continue
                rows.append(
                    {
                        "image_path": frame_str,
                        "is_attack": int(label),
                        "attack_type": _detect_attack_type(video_path) if int(label) == 1 else "bona_fide",
                        "source_dataset": dataset_dir.name,
                        "region_proxy": "unknown",
                    }
                )
                seen_paths.add(frame_str)
    return pd.DataFrame(rows)


def _append_subject_rows(
    rows: list[dict[str, str]],
    dataset_dir: Path,
    subject_root: Path,
    subject_key: str,
    region_proxy: str,
    min_images_per_subject: int,
) -> None:
    images = _valid_images(subject_root.rglob("*"))
    if len(images) < min_images_per_subject:
        return
    subject_id = f"{_source_id(dataset_dir)}_{_slug(subject_key)}"
    for image_path in sorted(images):
        rows.append(
            {
                "image_path": str(image_path),
                "subject_id": subject_id,
                "capture_type": _capture_type_for_path(image_path),
                "source_dataset": dataset_dir.name,
                "region_proxy": region_proxy,
            }
        )


def _build_celeba_identity_rows(dataset_dir: Path, min_images_per_subject: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    identity_files = sorted(dataset_dir.rglob("identity_CelebA.txt"))
    if not identity_files:
        return rows
    identity_file = identity_files[0]
    subject_to_images: dict[str, list[Path]] = {}
    for raw_line in identity_file.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) < 2:
            continue
        filename, subject_value = parts[0], parts[1]
        image_path = _existing_path(
            [
                identity_file.parent / filename,
                dataset_dir / filename,
                dataset_dir / "img_align_celeba" / filename,
                dataset_dir / "img_align_celeba" / "img_align_celeba" / filename,
            ]
        )
        if image_path is None or not _is_valid_image(image_path):
            continue
        subject_to_images.setdefault(subject_value, []).append(image_path)

    for subject_value, image_paths in subject_to_images.items():
        if len(image_paths) < min_images_per_subject:
            continue
        subject_id = f"{_source_id(dataset_dir)}_{_slug(subject_value)}"
        for image_path in sorted(image_paths):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "subject_id": subject_id,
                    "capture_type": "selfie",
                    "source_dataset": dataset_dir.name,
                    "region_proxy": "global",
                }
            )
    return rows


def build_identity_manifest(dataset_dirs: Iterable[str | Path], min_images_per_subject: int = 2) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        dataset_name = dataset_dir.name

        if "asian_kyc_photo_dataset" in dataset_name:
            files_root = dataset_dir / "files"
            for subject_root in sorted(path for path in files_root.iterdir() if path.is_dir()) if files_root.exists() else []:
                _append_subject_rows(
                    rows,
                    dataset_dir,
                    subject_root,
                    subject_root.name,
                    "southeast_asia",
                    min_images_per_subject,
                )
            continue

        if "selfie_and_official_id_photo_dataset_18k_images" in dataset_name:
            public_root = dataset_dir / "Selfie & id data - public sample"
            for subject_root in sorted(path for path in public_root.iterdir() if path.is_dir()) if public_root.exists() else []:
                _append_subject_rows(
                    rows,
                    dataset_dir,
                    subject_root,
                    f"public_{subject_root.name}",
                    "global",
                    min_images_per_subject,
                )

            diverse_root = dataset_dir / "AxonLabs_Diverse Selfie & ID Photo Dataset - samples"
            for region_root in sorted(path for path in diverse_root.iterdir() if path.is_dir()) if diverse_root.exists() else []:
                region_proxy = _region_proxy_from_group(region_root.name)
                for subject_root in sorted(path for path in region_root.iterdir() if path.is_dir()):
                    _append_subject_rows(
                        rows,
                        dataset_dir,
                        subject_root,
                        f"{region_root.name}_{subject_root.name}",
                        region_proxy,
                        min_images_per_subject,
                    )
            continue

        celeb_rows = _build_celeba_identity_rows(dataset_dir, min_images_per_subject)
        if celeb_rows:
            rows.extend(celeb_rows)
            continue

        for leaf_dir in [path for path in dataset_dir.rglob("*") if path.is_dir()]:
            images = _valid_images(leaf_dir.iterdir()) if leaf_dir.exists() else []
            if len(images) < min_images_per_subject:
                continue
            _append_subject_rows(
                rows,
                dataset_dir,
                leaf_dir,
                leaf_dir.relative_to(dataset_dir).as_posix(),
                "unknown",
                min_images_per_subject,
            )
    return pd.DataFrame(rows)


def generate_verification_pairs(frame: pd.DataFrame, positives_per_subject: int = 2, negatives_per_subject: int = 2, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    grouped = {subject_id: group.reset_index(drop=True) for subject_id, group in frame.groupby("subject_id")}
    subject_ids = sorted(grouped.keys())

    def _pair_metadata(left_row: pd.Series, right_row: pd.Series) -> dict[str, str]:
        left_region = str(left_row.get("region_proxy", "unknown"))
        right_region = str(right_row.get("region_proxy", "unknown"))
        pair_region = left_region if left_region == right_region else "mixed"
        return {
            "left_region_proxy": left_region,
            "right_region_proxy": right_region,
            "pair_region_proxy": pair_region,
            "left_capture_type": str(left_row.get("capture_type", "unknown")),
            "right_capture_type": str(right_row.get("capture_type", "unknown")),
            "left_source_dataset": str(left_row.get("source_dataset", "unknown")),
            "right_source_dataset": str(right_row.get("source_dataset", "unknown")),
        }

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
                        **_pair_metadata(group.iloc[left_idx], group.iloc[right_idx]),
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
                    **_pair_metadata(left_row, right_row),
                }
            )
    return pd.DataFrame(rows)


def build_age_gender_manifest(dataset_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        csv_candidates = sorted({path.resolve() for path in dataset_dir.rglob("fairface_label_*.csv")})
        if csv_candidates:
            for csv_path in csv_candidates:
                frame = pd.read_csv(csv_path)
                image_col = "file" if "file" in frame.columns else frame.columns[0]
                for _, row in frame.iterrows():
                    image_rel = Path(str(row[image_col]))
                    image_path = _existing_path(
                        [
                            csv_path.parent / image_rel,
                            dataset_dir / image_rel,
                            dataset_dir / csv_path.parent.name / image_rel,
                        ]
                    )
                    if image_path is None or not _is_valid_image(image_path):
                        continue
                    gender_raw = str(row.get("gender", "")).strip().lower()
                    gender_id = 1 if gender_raw.startswith("female") else 0
                    age_group = str(row.get("age", "unknown"))
                    race_label = str(row.get("race", "unknown")).strip()
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
                            "image_path": str(image_path),
                            "age": float(age_proxy),
                            "gender_id": int(gender_id),
                            "gender_label": "female" if gender_id == 1 else "male",
                            "age_group": age_group,
                            "race_label": race_label,
                            "skin_tone_proxy": "unknown",
                            "source_dataset": dataset_dir.name,
                            "region_proxy": _fairface_region_proxy(race_label),
                        }
                    )
            continue

        for image_path in _valid_images(_list_images(dataset_dir)):
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
        txt_candidates = [] if csv_candidates else list(dataset_dir.rglob("list_attr_celeba.txt"))
        if csv_candidates:
            for csv_path in csv_candidates:
                frame = pd.read_csv(csv_path)
                image_col = "image_id" if "image_id" in frame.columns else frame.columns[0]
                for _, row in frame.iterrows():
                    image_path = next((candidate for candidate in [dataset_dir / str(row[image_col]), dataset_dir / "img_align_celeba" / str(row[image_col]), dataset_dir / "img_align_celeba" / "img_align_celeba" / str(row[image_col])] if candidate.exists()), None)
                    if image_path is None or not _is_valid_image(image_path):
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
        if txt_candidates:
            txt_path = txt_candidates[0]
            frame = pd.read_csv(txt_path, delim_whitespace=True, skiprows=1)
            image_col = "image_id" if "image_id" in frame.columns else frame.columns[0]
            for _, row in frame.iterrows():
                image_path = next(
                    (
                        candidate
                        for candidate in [
                            dataset_dir / str(row[image_col]),
                            dataset_dir / "img_align_celeba" / str(row[image_col]),
                            dataset_dir / "img_align_celeba" / "img_align_celeba" / str(row[image_col]),
                        ]
                        if candidate.exists()
                    ),
                    None,
                )
                if image_path is None or not _is_valid_image(image_path):
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

        for image_path in _valid_images(_list_images(dataset_dir)):
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
        image_path = Path(str(row["image_path"])).expanduser().resolve()
        if not image_path.exists():
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "quality_score": _quality_proxy_score(str(image_path)),
                "source_dataset": row.get("source_dataset", "unknown"),
                "region_proxy": row.get("region_proxy", "unknown"),
                "age_group": row.get("age_group", "unknown"),
                "gender": row.get("gender", row.get("gender_label", row.get("gender_id", "unknown"))),
                "race_label": row.get("race_label", "unknown"),
                "skin_tone_proxy": row.get("skin_tone_proxy", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def build_celebamaskhq_parser_manifest(dataset_dirs: Iterable[str | Path], generated_mask_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(generated_mask_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_dir in [Path(item).expanduser().resolve() for item in dataset_dirs]:
        images_root = _existing_path(
            [
                dataset_dir / "CelebAMask-HQ" / "CelebAMask-HQ" / "CelebA-HQ-img",
                dataset_dir / "CelebA-HQ-img",
            ]
        )
        mask_root = _existing_path(
            [
                dataset_dir / "CelebAMask-HQ" / "CelebAMask-HQ" / "CelebAMask-HQ-mask-anno",
                dataset_dir / "CelebAMask-HQ-mask-anno",
            ]
        )
        if images_root is None or mask_root is None:
            continue

        images = [path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".png"}]
        if not images:
            continue

        mask_index: dict[str, list[Path]] = {}
        for mask_file in mask_root.rglob("*.png"):
            stem = mask_file.stem.split("_", 1)[0]
            mask_index.setdefault(stem, []).append(mask_file)

        for image_path in images:
            stem = image_path.stem.zfill(5)
            mask_files = mask_index.get(stem, [])
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

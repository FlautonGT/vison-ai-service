#!/usr/bin/env python3
"""Select, download, and prepare manifests for a training task on Vast.ai."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from training.vison_train.data.builders import (
    build_age_gender_manifest,
    build_attribute_manifest,
    build_celebamaskhq_parser_manifest,
    build_deepfake_manifest,
    build_identity_manifest,
    build_pad_manifest,
    build_quality_manifest,
    generate_verification_pairs,
    write_manifest,
)
from training.vison_train.data.inventory import select_datasets
from training.vison_train.data.kaggle import download_dataset
from training.vison_train.data.manifests import split_manifest


TASK_QUERY_MAP = {
    "deepfake": ["deepfake_detection"],
    "passive_pad": ["passive_pad"],
    "verification": ["verification"],
    "age_gender": ["age_estimation", "gender_classification"],
    "face_attributes": ["face_attributes"],
    "face_quality": ["verification", "age_estimation", "gender_classification", "face_attributes"],
    "face_parser": ["face_parsing"],
}


def _allowed_statuses(args: argparse.Namespace) -> list[str]:
    statuses = ["approved"]
    if args.allow_restricted:
        statuses.append("restricted")
    if args.allow_fallback:
        statuses.append("fallback_only")
    if args.allow_rejected:
        statuses.append("rejected")
    return statuses


def _select(task: str, args: argparse.Namespace) -> list[dict]:
    selected = []
    seen = set()
    for query_task in TASK_QUERY_MAP[task]:
        for item in select_datasets(
            task=query_task,
            preferred_region=args.preferred_region,
            require_commercial=not args.allow_noncommercial,
            require_modifiable=not args.allow_nonmodifiable,
            allowed_statuses=_allowed_statuses(args),
            path=args.inventory,
        ):
            dataset_id = item["dataset"]["id"]
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            selected.append(item)
    return selected[: args.max_datasets]


def _download(selected: list[dict], raw_root: Path, force: bool) -> list[Path]:
    dataset_dirs = []
    for item in selected:
        dataset = item["dataset"]
        kaggle_ref = dataset.get("download", {}).get("kaggle_ref")
        if not kaggle_ref:
            continue
        target_dir = raw_root / dataset["id"].replace("/", "__")
        download_dataset(kaggle_ref, target_dir, unzip=True, force=force)
        dataset_dirs.append(target_dir)
    return dataset_dirs


def _with_group_id(frame: pd.DataFrame, column: str = "image_path") -> pd.DataFrame:
    updated = frame.copy()
    updated["group_id"] = updated[column].map(lambda value: f"{Path(str(value)).parent.name}_{Path(str(value)).stem}")
    return updated


def _prepare_verification(dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    frame = build_identity_manifest(dataset_dirs)
    splits = split_manifest(frame, group_cols=["subject_id"], val_ratio=0.1, test_ratio=0.1, seed=42)
    train_path = write_manifest(splits["train"], manifest_dir / "train.csv")
    val_path = write_manifest(splits["val"], manifest_dir / "val.csv")
    test_path = write_manifest(splits["test"], manifest_dir / "test.csv")
    pairs_val = generate_verification_pairs(splits["val"])
    pairs_test = generate_verification_pairs(splits["test"])
    pair_val_path = write_manifest(pairs_val, manifest_dir / "pairs_val.csv")
    pair_test_path = write_manifest(pairs_test, manifest_dir / "pairs_test.csv")
    return {
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
        "test_manifest": str(test_path),
        "pair_val_manifest": str(pair_val_path),
        "pair_test_manifest": str(pair_test_path),
    }


def _prepare_binary(task: str, dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    frame = build_deepfake_manifest(dataset_dirs) if task == "deepfake" else build_pad_manifest(dataset_dirs)
    frame = _with_group_id(frame)
    splits = split_manifest(frame, group_cols=["group_id"], val_ratio=0.1, test_ratio=0.1, seed=42, stratify_col=frame.columns[1])
    return {
        "train_manifest": str(write_manifest(splits["train"], manifest_dir / "train.csv")),
        "val_manifest": str(write_manifest(splits["val"], manifest_dir / "val.csv")),
        "test_manifest": str(write_manifest(splits["test"], manifest_dir / "test.csv")),
    }


def _prepare_age_gender(dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    frame = build_age_gender_manifest(dataset_dirs)
    if "gender_id" in frame.columns:
        frame = frame.loc[frame["gender_id"] >= 0].reset_index(drop=True)
    frame = _with_group_id(frame)
    splits = split_manifest(frame, group_cols=["group_id"], val_ratio=0.1, test_ratio=0.1, seed=42, stratify_col="age_group")
    return {
        "train_manifest": str(write_manifest(splits["train"], manifest_dir / "train.csv")),
        "val_manifest": str(write_manifest(splits["val"], manifest_dir / "val.csv")),
        "test_manifest": str(write_manifest(splits["test"], manifest_dir / "test.csv")),
    }


def _prepare_attributes(dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    frame = build_attribute_manifest(dataset_dirs)
    frame = _with_group_id(frame)
    splits = split_manifest(frame, group_cols=["group_id"], val_ratio=0.1, test_ratio=0.1, seed=42)
    return {
        "train_manifest": str(write_manifest(splits["train"], manifest_dir / "train.csv")),
        "val_manifest": str(write_manifest(splits["val"], manifest_dir / "val.csv")),
        "test_manifest": str(write_manifest(splits["test"], manifest_dir / "test.csv")),
    }


def _prepare_quality(dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    source_frame = build_age_gender_manifest(dataset_dirs)
    if source_frame.empty:
        source_frame = build_attribute_manifest(dataset_dirs)
    frame = build_quality_manifest(source_frame)
    frame = _with_group_id(frame)
    splits = split_manifest(frame, group_cols=["group_id"], val_ratio=0.1, test_ratio=0.1, seed=42)
    return {
        "train_manifest": str(write_manifest(splits["train"], manifest_dir / "train.csv")),
        "val_manifest": str(write_manifest(splits["val"], manifest_dir / "val.csv")),
        "test_manifest": str(write_manifest(splits["test"], manifest_dir / "test.csv")),
    }


def _prepare_parser(dataset_dirs: list[Path], manifest_dir: Path) -> dict:
    frame = build_celebamaskhq_parser_manifest(dataset_dirs, manifest_dir / "generated_masks")
    frame = _with_group_id(frame)
    splits = split_manifest(frame, group_cols=["group_id"], val_ratio=0.1, test_ratio=0.1, seed=42)
    return {
        "train_manifest": str(write_manifest(splits["train"], manifest_dir / "train.csv")),
        "val_manifest": str(write_manifest(splits["val"], manifest_dir / "val.csv")),
        "test_manifest": str(write_manifest(splits["test"], manifest_dir / "test.csv")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare datasets and manifests for a Vison training task")
    parser.add_argument("--task", choices=sorted(TASK_QUERY_MAP.keys()), required=True)
    parser.add_argument("--preferred-region", default="indonesia")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--inventory", default="configs/datasets/dataset_inventory.json")
    parser.add_argument("--max-datasets", type=int, default=4)
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--allow-nonmodifiable", action="store_true")
    parser.add_argument("--allow-restricted", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--allow-rejected", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_root = (ROOT_DIR / args.data_root).resolve()
    manifest_dir = (ROOT_DIR / args.manifest_root / args.task).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    selected = _select(args.task, args)
    dataset_dirs = [raw_root / item["dataset"]["id"].replace("/", "__") for item in selected]
    if not args.skip_download:
        dataset_dirs = _download(selected, raw_root, force=args.force_download)

    if args.task in {"deepfake", "passive_pad"}:
        manifests = _prepare_binary(args.task, dataset_dirs, manifest_dir)
    elif args.task == "verification":
        manifests = _prepare_verification(dataset_dirs, manifest_dir)
    elif args.task == "age_gender":
        manifests = _prepare_age_gender(dataset_dirs, manifest_dir)
    elif args.task == "face_attributes":
        manifests = _prepare_attributes(dataset_dirs, manifest_dir)
    elif args.task == "face_quality":
        manifests = _prepare_quality(dataset_dirs, manifest_dir)
    else:
        manifests = _prepare_parser(dataset_dirs, manifest_dir)

    summary = {
        "task": args.task,
        "selected_datasets": selected,
        "dataset_dirs": [str(path) for path in dataset_dirs],
        "manifest_dir": str(manifest_dir),
        "manifests": manifests,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

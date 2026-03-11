#!/usr/bin/env python3
"""Collect checkpoints, exports, thresholds, and reports into a handoff directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.vison_train.config import load_json


CONFIGS = {
    "deepfake": "configs/training/deepfake_detection.json",
    "passive_pad": "configs/training/passive_pad.json",
    "face_quality": "configs/training/face_quality.json",
    "age_gender": "configs/training/age_gender.json",
    "face_attributes": "configs/training/face_attributes.json",
    "verification": "configs/training/verification.json",
    "face_parser": "configs/training/face_parser.json",
}


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect training artifacts for local handoff")
    parser.add_argument("--task", choices=sorted(CONFIGS.keys()), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_json(Path(CONFIGS[args.task]).resolve())
    checkpoint_dir = Path(config["optimization"]["checkpoint_dir"]).resolve()
    report_dir = Path(config["evaluation"]["report_dir"]).resolve()
    artifact_dir = checkpoint_dir.parent / "artifacts"
    output_dir = Path(args.output_dir).expanduser().resolve()

    _copy_tree(checkpoint_dir, output_dir / "checkpoints")
    _copy_tree(report_dir, output_dir / "reports")
    _copy_tree(artifact_dir, output_dir / "exports")

    print({"task": args.task, "output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

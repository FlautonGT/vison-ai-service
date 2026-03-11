#!/usr/bin/env python3
"""Convenience wrapper around the shared training CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.vison_train.cli import main as training_main


CONFIGS = {
    "deepfake": "configs/training/deepfake_detection.json",
    "passive_pad": "configs/training/passive_pad.json",
    "face_quality": "configs/training/face_quality.json",
    "age_gender": "configs/training/age_gender.json",
    "face_attributes": "configs/training/face_attributes.json",
    "verification": "configs/training/verification.json",
    "face_parser": "configs/training/face_parser.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a configured Vison training pipeline")
    parser.add_argument("--task", choices=sorted(CONFIGS.keys()), required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config_path = Path(CONFIGS[args.task]).resolve()
    argv = ["fit", "--config", str(config_path)]
    for item in args.override:
        argv.extend(["--override", item])
    return training_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

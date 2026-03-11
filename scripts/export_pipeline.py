#!/usr/bin/env python3
"""Export a trained task model to ONNX for local inference handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.vison_train.config import load_json
from training.vison_train.export import export_onnx


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
    parser = argparse.ArgumentParser(description="Export a trained Vison task model to ONNX")
    parser.add_argument("--task", choices=sorted(CONFIGS.keys()), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_json(Path(CONFIGS[args.task]).resolve())
    result = export_onnx(config, checkpoint_path=args.checkpoint, output_dir=args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

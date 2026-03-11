#!/usr/bin/env python3
"""Grouped manifest split wrapper for leakage-safe dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.vison_train.cli import main as training_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val/test manifests with group-disjoint splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--group-cols", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-col")
    parser.add_argument("--holdout-column")
    parser.add_argument("--holdout-values")
    args = parser.parse_args()
    argv = [
        "split-manifest",
        "--manifest",
        args.manifest,
        "--group-cols",
        args.group_cols,
        "--train-output",
        args.train_output,
        "--val-output",
        args.val_output,
        "--test-output",
        args.test_output,
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.stratify_col:
        argv.extend(["--stratify-col", args.stratify_col])
    if args.holdout_column:
        argv.extend(["--holdout-column", args.holdout_column])
    if args.holdout_values:
        argv.extend(["--holdout-values", args.holdout_values])
    return training_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

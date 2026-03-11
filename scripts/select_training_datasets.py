#!/usr/bin/env python3
"""Dataset selection wrapper for the inventory-first workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.vison_train.cli import main as training_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidate datasets from the inventory")
    parser.add_argument("--task", required=True)
    parser.add_argument("--preferred-region", default="indonesia")
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--allow-nonmodifiable", action="store_true")
    parser.add_argument("--allow-restricted", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--allow-rejected", action="store_true")
    args = parser.parse_args()
    argv = [
        "select-datasets",
        "--task",
        args.task,
        "--preferred-region",
        args.preferred_region,
    ]
    if args.allow_noncommercial:
        argv.append("--allow-noncommercial")
    if args.allow_nonmodifiable:
        argv.append("--allow-nonmodifiable")
    if args.allow_restricted:
        argv.append("--allow-restricted")
    if args.allow_fallback:
        argv.append("--allow-fallback")
    if args.allow_rejected:
        argv.append("--allow-rejected")
    return training_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

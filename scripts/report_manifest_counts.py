#!/usr/bin/env python3
"""Report row counts for prepared manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return max(sum(1 for _ in reader) - 1, 0)


def _count_unique(path: Path, column_name: str) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return len({str(row.get(column_name, "")).strip() for row in reader if str(row.get(column_name, "")).strip()})


def main() -> int:
    parser = argparse.ArgumentParser(description="Report manifest counts for one or more tasks")
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args()

    manifest_root = Path(args.manifest_root).expanduser().resolve()
    tasks = args.tasks or sorted(path.name for path in manifest_root.iterdir() if path.is_dir())

    payload: dict[str, dict[str, int]] = {}
    for task in tasks:
        task_root = manifest_root / task
        task_report: dict[str, int] = {}
        for split_name in ["train", "val", "test", "pairs_val", "pairs_test"]:
            path = task_root / f"{split_name}.csv"
            if path.exists():
                task_report[split_name] = _count_rows(path)
        if (task_root / "train.csv").exists():
            task_report["train_unique_subjects"] = _count_unique(task_root / "train.csv", "subject_id")
        payload[task] = task_report

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

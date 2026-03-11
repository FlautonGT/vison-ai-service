#!/usr/bin/env python3
"""Dataset selection wrapper for the inventory-first workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PREPARE_TASK_PATH = ROOT_DIR / "scripts" / "vastai_prepare_task.py"
SPEC = importlib.util.spec_from_file_location("vastai_prepare_task", PREPARE_TASK_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {PREPARE_TASK_PATH}")
vastai_prepare_task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vastai_prepare_task)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidate datasets from the inventory")
    parser.add_argument("--task", required=True)
    parser.add_argument("--preferred-region", default="indonesia")
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--allow-nonmodifiable", action="store_true")
    parser.add_argument("--allow-restricted", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--allow-rejected", action="store_true")
    parser.add_argument("--inventory", default="configs/datasets/dataset_inventory.json")
    parser.add_argument("--max-datasets", type=int, default=4)
    args = parser.parse_args()
    selected = vastai_prepare_task._select(args.task, args)
    payload = {
        "task": args.task,
        "preferred_region": args.preferred_region,
        "constraints": {
            "commercial_use_ok": not args.allow_noncommercial,
            "modifiable": not args.allow_nonmodifiable,
            "allowed_statuses": vastai_prepare_task._allowed_statuses(args),
        },
        "regional_assessment": vastai_prepare_task._regional_assessment(args.task, selected),
        "results": selected,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

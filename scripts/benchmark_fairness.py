#!/usr/bin/env python3
"""Demographic fairness snapshot for analyze endpoint on UTKFace labels."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests


RACE_MAP = {0: "WHITE", 1: "BLACK", 2: "ASIAN", 3: "INDIAN", 4: "OTHER"}


def _parse_utk_label(path: Path) -> dict | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = "MAN" if int(parts[1]) == 0 else "WOMAN"
        race = RACE_MAP.get(int(parts[2]), "OTHER")
        if age <= 30:
            age_group = "18-30"
        elif age <= 50:
            age_group = "31-50"
        elif age <= 70:
            age_group = "51-70"
        else:
            age_group = "71+"
        return {"age": age, "gender": gender, "race": race, "age_group": age_group}
    except Exception:
        return None


def _analyze(url: str, headers: dict, image_path: Path) -> dict | None:
    with image_path.open("rb") as fh:
        response = requests.post(
            f"{url}/api/face/analyze",
            headers=headers,
            files={"image": (image_path.name, fh, "image/jpeg")},
            timeout=120,
        )
    if response.status_code != 200:
        return None
    return response.json()


def _age_mid(age_range: dict) -> float:
    low = float(age_range.get("low", 0))
    high = float(age_range.get("high", low))
    return (low + high) * 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-secret", default=os.getenv("AI_SERVICE_SECRET", ""))
    parser.add_argument("--dataset-dir", default="./benchmark_data/UTKFace")
    parser.add_argument("--max-tests", type=int, default=1000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")

    images = sorted(list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png")))
    labeled = [(path, _parse_utk_label(path)) for path in images]
    labeled = [(path, label) for path, label in labeled if label is not None]
    random.seed(42)
    if len(labeled) > args.max_tests:
        labeled = random.sample(labeled, args.max_tests)

    headers = {"X-AI-Service-Key": args.api_secret} if args.api_secret else {}
    started = time.perf_counter()
    rows = []
    for idx, (path, label) in enumerate(labeled, 1):
        pred = _analyze(args.url, headers, path)
        if pred is None:
            continue
        pred_gender = pred.get("gender")
        pred_age = _age_mid(pred.get("ageRange", {}))
        rows.append(
            {
                "gt_gender": label["gender"],
                "gt_age": label["age"],
                "gt_race": label["race"],
                "gt_age_group": label["age_group"],
                "pred_gender": pred_gender,
                "pred_age": pred_age,
            }
        )
        if idx % 200 == 0:
            print(f"processed {idx}/{len(labeled)}")

    by_gender = defaultdict(list)
    by_race = defaultdict(list)
    by_age_group = defaultdict(list)
    for row in rows:
        by_gender[row["gt_gender"]].append(row)
        by_race[row["gt_race"]].append(row)
        by_age_group[row["gt_age_group"]].append(row)

    def summarize(items: list[dict]) -> dict:
        if not items:
            return {"count": 0, "gender_acc": None, "age_mae": None}
        gender_acc = sum(1 for r in items if r["gt_gender"] == r["pred_gender"]) / len(items) * 100.0
        age_mae = float(np.mean([abs(r["gt_age"] - r["pred_age"]) for r in items]))
        return {"count": len(items), "gender_acc": round(gender_acc, 4), "age_mae": round(age_mae, 4)}

    result = {
        "count": len(rows),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "overall": summarize(rows),
        "by_gender": {k: summarize(v) for k, v in sorted(by_gender.items())},
        "by_race": {k: summarize(v) for k, v in sorted(by_race.items())},
        "by_age_group": {k: summarize(v) for k, v in sorted(by_age_group.items())},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

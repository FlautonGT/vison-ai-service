#!/usr/bin/env python3
"""Compute FRVT-style FNMR@FMR metrics using compare endpoint scores."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import requests


def _collect_pairs(dataset_dir: Path, max_pairs: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    persons: dict[str, list[Path]] = {}
    for person in sorted(dataset_dir.iterdir()):
        if person.is_dir():
            imgs = sorted(list(person.glob("*.jpg")) + list(person.glob("*.png")))
            if imgs:
                persons[person.name] = imgs
    names = list(persons.keys())

    genuine: list[tuple[Path, Path]] = []
    for _, imgs in persons.items():
        if len(imgs) >= 2:
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    genuine.append((imgs[i], imgs[j]))
    if len(genuine) > max_pairs:
        random.seed(123)
        genuine = random.sample(genuine, max_pairs)

    impostor: list[tuple[Path, Path]] = []
    random.seed(123)
    for _ in range(max_pairs * 4):
        if len(names) < 2:
            break
        a, b = random.sample(names, 2)
        impostor.append((random.choice(persons[a]), random.choice(persons[b])))
    if len(impostor) > max_pairs:
        impostor = random.sample(impostor, max_pairs)
    return genuine, impostor


def _compare(url: str, headers: dict, left: Path, right: Path) -> float | None:
    with left.open("rb") as src, right.open("rb") as tgt:
        response = requests.post(
            f"{url}/api/face/compare",
            headers=headers,
            files={
                "sourceImage": (left.name, src, "image/jpeg"),
                "targetImage": (right.name, tgt, "image/jpeg"),
            },
            timeout=120,
        )
    if response.status_code != 200:
        return None
    payload = response.json()
    try:
        return float(payload.get("similarity", 0.0))
    except Exception:
        return None


def _frvt_metrics(genuine_scores: list[float], impostor_scores: list[float]) -> dict:
    g = np.asarray(genuine_scores, dtype=np.float32)
    i = np.asarray(impostor_scores, dtype=np.float32)
    thresholds = np.linspace(0.0, 100.0, 1001)

    results = {}
    for fmr_target in (0.0001, 0.001, 0.01):
        eligible = []
        for t in thresholds:
            fmr = float(np.mean(i >= t))
            fnmr = float(np.mean(g < t))
            if fmr <= fmr_target:
                eligible.append((t, fmr, fnmr))
        if not eligible:
            results[f"FNMR@FMR_{fmr_target*100:.02f}%"] = None
            continue
        best = min(eligible, key=lambda row: row[2])
        results[f"FNMR@FMR_{fmr_target*100:.02f}%"] = {
            "threshold": round(float(best[0]), 3),
            "fmr": round(float(best[1]) * 100.0, 6),
            "fnmr": round(float(best[2]) * 100.0, 6),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-secret", default=os.getenv("AI_SERVICE_SECRET", ""))
    parser.add_argument("--dataset-dir", default="./benchmark_data/lfw")
    parser.add_argument("--max-pairs", type=int, default=500)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")

    headers = {"X-AI-Service-Key": args.api_secret} if args.api_secret else {}
    genuine_pairs, impostor_pairs = _collect_pairs(dataset_dir, args.max_pairs)

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    started = time.perf_counter()

    for left, right in genuine_pairs:
        score = _compare(args.url, headers, left, right)
        if score is not None:
            genuine_scores.append(score)
    for left, right in impostor_pairs:
        score = _compare(args.url, headers, left, right)
        if score is not None:
            impostor_scores.append(score)

    metrics = _frvt_metrics(genuine_scores, impostor_scores)
    payload = {
        "dataset": str(dataset_dir),
        "genuine_scores": len(genuine_scores),
        "impostor_scores": len(impostor_scores),
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

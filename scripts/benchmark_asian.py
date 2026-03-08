#!/usr/bin/env python3
"""Benchmark compare endpoint on custom/asian-focused pair datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests


def _call_compare(base_url: str, headers: dict, left: Path, right: Path) -> tuple[dict, float]:
    started = time.perf_counter()
    with left.open("rb") as src, right.open("rb") as tgt:
        response = requests.post(
            f"{base_url}/api/face/compare",
            headers=headers,
            files={
                "sourceImage": (left.name, src, "image/jpeg"),
                "targetImage": (right.name, tgt, "image/jpeg"),
            },
            timeout=120,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code == 200:
        return response.json(), elapsed_ms
    return {"error": response.status_code, "detail": response.text[:200]}, elapsed_ms


def _build_pairs(dataset_dir: Path, max_pairs: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    persons: dict[str, list[Path]] = {}
    for person_dir in sorted(dataset_dir.iterdir()):
        if person_dir.is_dir():
            imgs = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
            if len(imgs) >= 1:
                persons[person_dir.name] = imgs

    matched: list[tuple[Path, Path]] = []
    for _, imgs in persons.items():
        if len(imgs) >= 2:
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    matched.append((imgs[i], imgs[j]))

    mismatched: list[tuple[Path, Path]] = []
    names = list(persons.keys())
    random.seed(42)
    for _ in range(max_pairs * 4):
        if len(names) < 2:
            break
        p1, p2 = random.sample(names, 2)
        mismatched.append((random.choice(persons[p1]), random.choice(persons[p2])))

    if len(matched) > max_pairs:
        matched = random.sample(matched, max_pairs)
    if len(mismatched) > max_pairs:
        mismatched = random.sample(mismatched, max_pairs)
    return matched, mismatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-secret", default=os.getenv("AI_SERVICE_SECRET", ""))
    parser.add_argument("--dataset-dir", default=os.getenv("ASIAN_DATASET_DIR", "./benchmark_data/asian_faces"))
    parser.add_argument("--max-pairs", type=int, default=200)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        fallback = Path("./benchmark_data/lfw")
        if fallback.exists():
            dataset_dir = fallback
        else:
            raise SystemExit(f"Dataset not found: {args.dataset_dir}")

    headers = {"X-AI-Service-Key": args.api_secret} if args.api_secret else {}
    matched, mismatched = _build_pairs(dataset_dir, args.max_pairs)
    print(f"dataset={dataset_dir} matched={len(matched)} mismatched={len(mismatched)}")

    tp = tn = fp = fn = 0
    latencies = []
    for left, right in matched:
        payload, latency = _call_compare(args.url, headers, left, right)
        latencies.append(latency)
        if payload.get("error"):
            continue
        if bool(payload.get("matched")):
            tp += 1
        else:
            fn += 1
    for left, right in mismatched:
        payload, latency = _call_compare(args.url, headers, left, right)
        latencies.append(latency)
        if payload.get("error"):
            continue
        if bool(payload.get("matched")):
            fp += 1
        else:
            tn += 1

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    result = {
        "dataset": str(dataset_dir),
        "pairs": {"matched": len(matched), "mismatched": len(mismatched)},
        "accuracy": round((tp + tn) / total * 100.0, 4),
        "f1": round(f1 * 100.0, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIST-style PAD benchmark (practical approximation) for Vison AI service.

Evaluates:
- /api/face/deepfake
- /api/face/verify-live

Metrics:
- APCER (Attack Presentation Classification Error Rate)
- BPCER (Bona Fide Presentation Classification Error Rate)
- ACER = (APCER + BPCER) / 2
- Accuracy / Precision / Recall / F1 (attack-positive)
- EER approximation + TPR@FAR for deepfake score

Usage:
  python scripts/benchmark_nist_style.py
  python scripts/benchmark_nist_style.py --real-count 200 --attack-count 200
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

# Reuse robust AI-face preparation from existing benchmark script.
from benchmark_real_vs_ai import collect_real_faces, download_ai_faces

BASE_URL = os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000")
API_SECRET = os.getenv("AI_SERVICE_SECRET", "test-123")
DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", "./benchmark_data"))
REPORT_DIR = Path(os.getenv("BENCHMARK_REPORT_DIR", "./benchmark_reports"))
HEADERS = {"X-AI-Service-Key": API_SECRET}


def log(message: str):
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def _healthcheck() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=8)
        return response.status_code == 200
    except Exception:
        return False


def _post_image(endpoint: str, image_path: Path, timeout: int = 120) -> tuple[dict, float]:
    url = f"{BASE_URL}{endpoint}"
    started = time.perf_counter()
    try:
        with image_path.open("rb") as handle:
            response = requests.post(
                url,
                headers=HEADERS,
                files={"image": (image_path.name, handle, "image/jpeg")},
                timeout=timeout,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code == 200:
            return response.json(), elapsed_ms
        return {"error": response.status_code, "detail": response.text[:250]}, elapsed_ms
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {"error": str(exc)}, elapsed_ms


def _prepare_datasets(real_count: int, attack_count: int) -> tuple[list[Path], list[Path]]:
    ai_dir = download_ai_faces(attack_count)
    attack_images = sorted(ai_dir.glob("*.jpg"))[:attack_count]
    if len(attack_images) < attack_count:
        raise RuntimeError(f"AI dataset insufficient: need {attack_count}, got {len(attack_images)}")

    real_images = collect_real_faces(real_count)
    if not real_images:
        raise RuntimeError("LFW dataset not found. Run benchmark_v2.py or benchmark_real_vs_ai.py first.")
    if len(real_images) < real_count:
        raise RuntimeError(f"Real dataset insufficient: need {real_count}, got {len(real_images)}")
    return list(real_images), attack_images


def _compute_confusion(
    records: list[dict],
    is_attack_positive: Callable[[dict], bool],
) -> dict:
    tp = fp = tn = fn = 0
    errors = 0
    latencies = []
    for row in records:
        if row.get("error"):
            errors += 1
            continue
        latencies.append(row["latency_ms"])
        pred_attack = bool(is_attack_positive(row["payload"]))
        is_attack = bool(row["is_attack"])
        if is_attack and pred_attack:
            tp += 1
        elif is_attack and not pred_attack:
            fn += 1
        elif (not is_attack) and pred_attack:
            fp += 1
        else:
            tn += 1

    attack_total = max(1, tp + fn)
    bona_fide_total = max(1, tn + fp)
    apcer = fn / attack_total
    bpcer = fp / bona_fide_total
    acer = (apcer + bpcer) / 2.0

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "errors": errors,
        "samples": int(tp + fp + tn + fn),
        "avg_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
        "APCER": round(apcer * 100.0, 4),
        "BPCER": round(bpcer * 100.0, 4),
        "ACER": round(acer * 100.0, 4),
        "accuracy": round(accuracy * 100.0, 4),
        "precision_attack": round(precision * 100.0, 4),
        "recall_attack": round(recall * 100.0, 4),
        "f1_attack": round(f1 * 100.0, 4),
    }


def _compute_apcer_by_attack_type(records: list[dict], is_attack_positive: Callable[[dict], bool]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        if not row.get("is_attack"):
            continue
        attack_label = str(row.get("attack_label") or "unknown").upper()
        grouped.setdefault(attack_label, []).append(row)

    result = {}
    for label, items in grouped.items():
        valid = [x for x in items if not x.get("error")]
        if not valid:
            result[label] = None
            continue
        misses = sum(1 for x in valid if not is_attack_positive(x["payload"]))
        apcer = misses / max(1, len(valid))
        result[label] = {
            "samples": len(valid),
            "APCER": round(apcer * 100.0, 4),
        }
    return result


def _deepfake_score(payload: dict) -> float:
    scores = payload.get("scores") or {}
    face_swap = float(scores.get("faceSwapScore", 0.0))
    ai_score = float(scores.get("aiGeneratedScore", 0.0))
    return max(face_swap, ai_score)


def _eer_from_scores(real_scores: list[float], attack_scores: list[float]) -> dict:
    if not real_scores or not attack_scores:
        return {
            "EER": None,
            "EER_threshold": None,
            "TPR_at_FAR_1pct": None,
            "TPR_at_FAR_0_1pct": None,
        }

    real = np.asarray(real_scores, dtype=np.float32)
    attack = np.asarray(attack_scores, dtype=np.float32)
    thresholds = np.linspace(0.0, 100.0, num=2001)

    fars = []  # false accept rate for attacks on bona fide (BPCER)
    frrs = []  # false reject rate for attacks (APCER)
    for threshold in thresholds:
        far = float(np.mean(real >= threshold))
        frr = float(np.mean(attack < threshold))
        fars.append(far)
        frrs.append(frr)

    fars_arr = np.asarray(fars, dtype=np.float32)
    frrs_arr = np.asarray(frrs, dtype=np.float32)
    idx = int(np.argmin(np.abs(fars_arr - frrs_arr)))
    eer = float((fars_arr[idx] + frrs_arr[idx]) * 0.5)
    eer_threshold = float(thresholds[idx])

    tpr_at_far_1 = None
    tpr_at_far_0_1 = None

    mask_1 = np.where(fars_arr <= 0.01)[0]
    if mask_1.size > 0:
        best = mask_1[np.argmax(1.0 - frrs_arr[mask_1])]
        tpr_at_far_1 = float((1.0 - frrs_arr[best]) * 100.0)

    mask_0_1 = np.where(fars_arr <= 0.001)[0]
    if mask_0_1.size > 0:
        best = mask_0_1[np.argmax(1.0 - frrs_arr[mask_0_1])]
        tpr_at_far_0_1 = float((1.0 - frrs_arr[best]) * 100.0)

    return {
        "EER": round(eer * 100.0, 4),
        "EER_threshold": round(eer_threshold, 4),
        "TPR_at_FAR_1pct": round(tpr_at_far_1, 4) if tpr_at_far_1 is not None else None,
        "TPR_at_FAR_0_1pct": round(tpr_at_far_0_1, 4) if tpr_at_far_0_1 is not None else None,
    }


def _bpcer_at_apcer(real_scores: list[float], attack_scores: list[float], targets=(0.05, 0.10)) -> dict:
    if not real_scores or not attack_scores:
        return {}
    real = np.asarray(real_scores, dtype=np.float32)
    attack = np.asarray(attack_scores, dtype=np.float32)
    thresholds = np.linspace(0.0, 100.0, num=2001)

    out = {}
    for target in targets:
        best = None
        for threshold in thresholds:
            apcer = float(np.mean(attack < threshold))
            bpcer = float(np.mean(real >= threshold))
            if apcer <= target:
                if best is None or bpcer < best[1]:
                    best = (threshold, bpcer, apcer)
        key = f"BPCER@APCER_{int(target*100)}pct"
        if best is None:
            out[key] = None
        else:
            out[key] = {
                "threshold": round(float(best[0]), 4),
                "bpcer": round(float(best[1]) * 100.0, 4),
                "apcer": round(float(best[2]) * 100.0, 4),
            }
    return out


def _run_endpoint_benchmark(
    endpoint: str,
    real_images: list[Path],
    attack_images: list[Path],
    is_attack_positive: Callable[[dict], bool],
) -> dict:
    records: list[dict] = []

    def _run_set(images: list[Path], is_attack: bool, label: str):
        for idx, image_path in enumerate(images, 1):
            payload, latency_ms = _post_image(endpoint, image_path)
            records.append(
                {
                    "is_attack": is_attack,
                    "attack_label": image_path.parent.name if is_attack else "REAL",
                    "payload": payload,
                    "latency_ms": latency_ms,
                    "error": "error" in payload,
                }
            )
            if idx % 50 == 0:
                log(f"  {label}: {idx}/{len(images)}")

    _run_set(real_images, is_attack=False, label="real")
    _run_set(attack_images, is_attack=True, label="attack")

    summary = _compute_confusion(records, is_attack_positive=is_attack_positive)
    return {"records": records, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-count", type=int, default=200)
    parser.add_argument("--attack-count", type=int, default=200)
    parser.add_argument(
        "--compute-score-curve",
        action="store_true",
        help="Compute score-based EER/TPR@FAR approximation for deepfake scores",
    )
    args = parser.parse_args()

    log("=" * 68)
    log("  NIST-Style PAD Benchmark (Practical Approximation)")
    log("=" * 68)
    if not _healthcheck():
        log(f"ERROR: Service not reachable at {BASE_URL}")
        sys.exit(1)
    log("Service OK")

    log("Preparing datasets...")
    real_images, attack_images = _prepare_datasets(args.real_count, args.attack_count)
    log(f"  Real images:   {len(real_images)}")
    log(f"  Attack images: {len(attack_images)}")

    log("\n[1/2] Deepfake endpoint benchmark")
    deepfake = _run_endpoint_benchmark(
        endpoint="/api/face/deepfake",
        real_images=real_images,
        attack_images=attack_images,
        is_attack_positive=lambda payload: bool(payload.get("isDeepfake")),
    )

    deepfake_curve = None
    deepfake_bpcer_apcer = None
    if args.compute_score_curve:
        deepfake_real_scores = [
            _deepfake_score(item["payload"])
            for item in deepfake["records"]
            if not item["error"] and not item["is_attack"]
        ]
        deepfake_attack_scores = [
            _deepfake_score(item["payload"])
            for item in deepfake["records"]
            if not item["error"] and item["is_attack"]
        ]
        deepfake_curve = _eer_from_scores(deepfake_real_scores, deepfake_attack_scores)
        deepfake_bpcer_apcer = _bpcer_at_apcer(deepfake_real_scores, deepfake_attack_scores)

    log("[2/2] Verify-live endpoint benchmark")
    verify_live = _run_endpoint_benchmark(
        endpoint="/api/face/verify-live",
        real_images=real_images,
        attack_images=attack_images,
        is_attack_positive=lambda payload: not bool(payload.get("isLive")),
    )

    report = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "base_url": BASE_URL,
        "dataset": {
            "real_count": len(real_images),
            "attack_count": len(attack_images),
            "source_real": str(DATA_DIR / "lfw*"),
            "source_attack": str(DATA_DIR / "ai_faces"),
        },
        "deepfake": deepfake["summary"],
        "verify_live": verify_live["summary"],
        "notes": [
            "This is a practical open-source approximation, not an official NIST submission score.",
            "APCER/BPCER/ACER follow PAD convention with attack-positive class.",
        ],
    }
    if deepfake_curve is not None:
        report["deepfake"]["score_curve"] = deepfake_curve
        report["deepfake"]["bpcer_at_apcer"] = deepfake_bpcer_apcer
        report["notes"].append("Score curve uses max(faceSwapScore, aiGeneratedScore) as approximation.")
    else:
        report["notes"].append("Score curve disabled (use --compute-score-curve to enable approximation).")

    report["deepfake"]["APCER_by_attack_type"] = _compute_apcer_by_attack_type(
        deepfake["records"],
        is_attack_positive=lambda payload: bool(payload.get("isDeepfake")),
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(report, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"nist_style_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Report saved: {out}")


if __name__ == "__main__":
    main()

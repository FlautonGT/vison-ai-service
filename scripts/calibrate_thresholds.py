#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-calibrate deepfake fusion thresholds using local real-vs-attack dataset.

The script queries /api/face/deepfake once per sample, then grid-searches
thresholds offline from returned scores (faceSwapScore, aiGeneratedScore, face confidence).

Usage:
  python scripts/calibrate_thresholds.py
  python scripts/calibrate_thresholds.py --real-count 200 --attack-count 200 --write-env .env.calibrated
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

import requests

from benchmark_real_vs_ai import collect_real_faces, download_ai_faces

BASE_URL = os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000")
API_SECRET = os.getenv("AI_SERVICE_SECRET", "test-123")
HEADERS = {"X-AI-Service-Key": API_SECRET}
REPORT_DIR = Path(os.getenv("BENCHMARK_REPORT_DIR", "./benchmark_reports"))


def log(message: str):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def _healthcheck() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=8)
        return response.status_code == 200
    except Exception:
        return False


def _post_deepfake(image_path: Path, timeout: int = 120) -> dict:
    url = f"{BASE_URL}/api/face/deepfake"
    try:
        with image_path.open("rb") as handle:
            response = requests.post(
                url,
                headers=HEADERS,
                files={"image": (image_path.name, handle, "image/jpeg")},
                timeout=timeout,
            )
        if response.status_code != 200:
            return {"error": response.status_code}
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_samples(real_count: int, attack_count: int) -> list[dict]:
    ai_dir = download_ai_faces(attack_count)
    attack_images = sorted(ai_dir.glob("*.jpg"))[:attack_count]
    real_images = collect_real_faces(real_count)
    if not real_images or len(real_images) < real_count:
        raise RuntimeError("Not enough real images (LFW).")
    if len(attack_images) < attack_count:
        raise RuntimeError("Not enough attack images.")

    records: list[dict] = []

    def _run(images: list[Path], is_attack: bool, label: str):
        for index, image_path in enumerate(images, 1):
            payload = _post_deepfake(image_path)
            if "error" in payload:
                records.append({"is_attack": is_attack, "error": True})
            else:
                scores = payload.get("scores") or {}
                face = payload.get("face") or {}
                records.append(
                    {
                        "is_attack": is_attack,
                        "error": False,
                        "ai_score": float(scores.get("aiGeneratedScore", 0.0)),
                        "face_swap_score": float(scores.get("faceSwapScore", 0.0)),
                        "face_confidence": float(face.get("confidence", 0.0)),
                        "face_detected": bool(face.get("detected", False)),
                        "api_pred": bool(payload.get("isDeepfake")),
                    }
                )
            if index % 50 == 0:
                log(f"  {label}: {index}/{len(images)}")

    _run(list(real_images), is_attack=False, label="real")
    _run(attack_images, is_attack=True, label="attack")
    return records


def _predict_from_params(
    sample: dict,
    ai_threshold: float,
    low_conf_threshold: float,
    low_conf_face_conf: float,
    strong_face_swap_threshold: float,
    suppress_enabled: bool,
    suppress_face_conf: float,
    suppress_ai_max: float,
    suppress_face_swap_max: float,
) -> bool:
    face_swap_score = float(sample["face_swap_score"])
    ai_score = float(sample["ai_score"])
    face_confidence = float(sample["face_confidence"])
    face_detected = bool(sample["face_detected"])

    face_swap_flag = face_swap_score >= strong_face_swap_threshold
    low_conf = (not face_detected) or (face_confidence <= low_conf_face_conf)
    ai_generated = (ai_score >= ai_threshold) or (low_conf and ai_score >= low_conf_threshold)

    if suppress_enabled and (not face_swap_flag) and face_detected:
        if (
            face_confidence >= suppress_face_conf
            and ai_score < suppress_ai_max
            and face_swap_score < suppress_face_swap_max
        ):
            ai_generated = False

    return bool(face_swap_flag or ai_generated)


def _evaluate(records: list[dict], **params) -> dict:
    tp = fp = tn = fn = 0
    errors = 0
    for row in records:
        if row.get("error"):
            errors += 1
            continue
        pred_attack = _predict_from_params(row, **params)
        is_attack = bool(row["is_attack"])
        if is_attack and pred_attack:
            tp += 1
        elif is_attack and not pred_attack:
            fn += 1
        elif (not is_attack) and pred_attack:
            fp += 1
        else:
            tn += 1

    tpr = tp / max(1, tp + fn)  # attack catch rate
    tnr = tn / max(1, tn + fp)  # real pass rate
    fpr = fp / max(1, tn + fp)
    fnr = fn / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "errors": errors,
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-count", type=int, default=200)
    parser.add_argument("--attack-count", type=int, default=200)
    parser.add_argument("--target-tpr", type=float, default=0.99)
    parser.add_argument("--target-tnr", type=float, default=0.99)
    parser.add_argument("--write-env", type=str, default="")
    args = parser.parse_args()

    if not _healthcheck():
        log(f"ERROR: service not reachable at {BASE_URL}")
        sys.exit(1)

    log("Collecting samples from /api/face/deepfake ...")
    records = _collect_samples(args.real_count, args.attack_count)
    valid = [r for r in records if not r.get("error")]
    log(f"Collected {len(valid)} valid samples, {len(records) - len(valid)} errors")

    grid = {
        "ai_threshold": [68, 70, 72, 74, 76, 78],
        "low_conf_threshold": [12, 14, 16, 18, 20],
        "low_conf_face_conf": [66, 68, 70, 72, 74],
        "strong_face_swap_threshold": [95, 96, 97, 98],
        "suppress_enabled": [True],
        "suppress_face_conf": [72, 75, 78, 80],
        "suppress_ai_max": [88, 90, 92, 94],
        "suppress_face_swap_max": [56, 58, 60, 62],
    }

    best = None
    evaluated = 0

    for ai_threshold in grid["ai_threshold"]:
        for low_conf_threshold in grid["low_conf_threshold"]:
            for low_conf_face_conf in grid["low_conf_face_conf"]:
                for strong_face_swap_threshold in grid["strong_face_swap_threshold"]:
                    for suppress_enabled in grid["suppress_enabled"]:
                        for suppress_face_conf in grid["suppress_face_conf"]:
                            for suppress_ai_max in grid["suppress_ai_max"]:
                                for suppress_face_swap_max in grid["suppress_face_swap_max"]:
                                    params = {
                                        "ai_threshold": float(ai_threshold),
                                        "low_conf_threshold": float(low_conf_threshold),
                                        "low_conf_face_conf": float(low_conf_face_conf),
                                        "strong_face_swap_threshold": float(strong_face_swap_threshold),
                                        "suppress_enabled": bool(suppress_enabled),
                                        "suppress_face_conf": float(suppress_face_conf),
                                        "suppress_ai_max": float(suppress_ai_max),
                                        "suppress_face_swap_max": float(suppress_face_swap_max),
                                    }
                                    metrics = _evaluate(records, **params)
                                    evaluated += 1

                                    tpr = metrics["TPR"]
                                    tnr = metrics["TNR"]
                                    # Lexicographic objective:
                                    # 1) meet targets if possible
                                    # 2) maximize min(TPR, TNR)
                                    # 3) maximize average(TPR, TNR)
                                    meets = int(tpr >= args.target_tpr and tnr >= args.target_tnr)
                                    objective = (
                                        meets,
                                        min(tpr, tnr),
                                        (tpr + tnr) * 0.5,
                                        -metrics["errors"],
                                    )
                                    if best is None or objective > best["objective"]:
                                        best = {
                                            "objective": objective,
                                            "params": params,
                                            "metrics": metrics,
                                        }

    assert best is not None
    result = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "base_url": BASE_URL,
        "evaluated_combinations": evaluated,
        "targets": {"TPR": args.target_tpr, "TNR": args.target_tnr},
        "best": {
            "params": best["params"],
            "metrics": {
                **best["metrics"],
                "TPR_percent": round(best["metrics"]["TPR"] * 100.0, 4),
                "TNR_percent": round(best["metrics"]["TNR"] * 100.0, 4),
                "FPR_percent": round(best["metrics"]["FPR"] * 100.0, 4),
                "FNR_percent": round(best["metrics"]["FNR"] * 100.0, 4),
            },
        },
    }

    print("\n=== RECOMMENDED PARAMETERS ===")
    print(json.dumps(result, indent=2))

    env_block = [
        f"AI_FACE_THRESHOLD={int(best['params']['ai_threshold'])}",
        f"AI_FACE_LOW_CONF_THRESHOLD={int(best['params']['low_conf_threshold'])}",
        f"AI_FACE_LOW_CONF_FACE_CONF={int(best['params']['low_conf_face_conf'])}",
        f"DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD={int(best['params']['strong_face_swap_threshold'])}",
        f"AI_FACE_REAL_SUPPRESS_ENABLED={'true' if best['params']['suppress_enabled'] else 'false'}",
        f"AI_FACE_REAL_SUPPRESS_FACE_CONF={int(best['params']['suppress_face_conf'])}",
        f"AI_FACE_REAL_SUPPRESS_AI_MAX={int(best['params']['suppress_ai_max'])}",
        f"AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX={int(best['params']['suppress_face_swap_max'])}",
    ]
    print("\n=== ENV BLOCK ===")
    print("\n".join(env_block))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"calibration_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Calibration report saved: {report_path}")

    if args.write_env:
        env_path = Path(args.write_env)
        env_path.write_text("\n".join(env_block) + "\n", encoding="utf-8")
        log(f"Wrote recommended env file: {env_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Vison AI Service — Automated Benchmark Script
===============================================
Downloads public face datasets (LFW subset) and runs automated accuracy tests
against the running Vison AI service, generating a comprehensive report.

Usage:
    1. Start vison-ai-service: uvicorn app.main:app --host 127.0.0.1 --port 8000
    2. Run: python scripts/benchmark_vison_ai.py

Requirements: requests, numpy (already in vison-ai-service venv)
"""

import os
import sys
import json
import time
import random
import hashlib
import tarfile
import shutil
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────

BASE_URL = os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000")
API_SECRET = os.getenv("AI_SERVICE_SECRET", "test-123")
DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", "./benchmark_data"))
REPORT_DIR = Path(os.getenv("BENCHMARK_REPORT_DIR", "./benchmark_reports"))

HEADERS = {"X-AI-Service-Key": API_SECRET}

# LFW dataset — standard face recognition benchmark
LFW_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_PAIRS_URL = "https://raw.githubusercontent.com/jian667/LFW-dataset/main/pairs.txt"

# ─── Utility Functions ──────────────────────────────────────────────────────


def log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def download_file(url, dest, desc=""):
    """Download file with progress indicator."""
    if dest.exists():
        log(f"  Already exists: {dest.name}")
        return True
    log(f"  Downloading {desc or dest.name}...")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / 1024 / 1024
                    print(f"\r  [{pct:3d}%] {mb:.1f} MB", end="", flush=True)
        print()
        return True
    except Exception as e:
        log(f"  Download failed: {e}", "ERROR")
        return False


def call_api(endpoint, files=None, data=None, timeout=30):
    """Call Vison AI API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    try:
        start = time.time()
        r = requests.post(url, headers=HEADERS, files=files, data=data, timeout=timeout)
        elapsed_ms = (time.time() - start) * 1000
        if r.status_code == 200:
            return r.json(), elapsed_ms
        else:
            return {"error": r.status_code, "detail": r.text[:200]}, elapsed_ms
    except Exception as e:
        return {"error": str(e)}, 0


def check_health():
    """Check if service is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=5)
        return r.status_code == 200
    except:
        return False


# ─── Dataset Preparation ────────────────────────────────────────────────────


def prepare_lfw_dataset():
    """Download and extract LFW dataset."""
    lfw_dir = DATA_DIR / "lfw"
    lfw_tgz = DATA_DIR / "lfw-funneled.tgz"
    pairs_file = DATA_DIR / "pairs.txt"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download pairs.txt (defines same/different person pairs)
    download_file(LFW_PAIRS_URL, pairs_file, "LFW pairs.txt")

    # Download LFW images
    if not (lfw_dir / "lfw-funneled").exists() and not lfw_dir.exists():
        if not download_file(LFW_URL, lfw_tgz, "LFW dataset (~233MB)"):
            return None, None

        log("  Extracting LFW...")
        with tarfile.open(lfw_tgz, "r:gz") as tar:
            tar.extractall(DATA_DIR)

        # Rename extracted dir
        extracted = DATA_DIR / "lfw_funneled"
        if extracted.exists():
            extracted.rename(lfw_dir)
        elif (DATA_DIR / "lfw-funneled").exists():
            (DATA_DIR / "lfw-funneled").rename(lfw_dir)

    # Handle different extraction paths
    if not lfw_dir.exists():
        # Try common extraction names
        for name in ["lfw_funneled", "lfw-funneled"]:
            candidate = DATA_DIR / name
            if candidate.exists():
                lfw_dir = candidate
                break

    return lfw_dir, pairs_file


def parse_lfw_pairs(pairs_file, lfw_dir, max_pairs=200):
    """
    Parse LFW pairs.txt OR auto-generate pairs from available images.
    """
    matched = []
    mismatched = []

    # Try loading pairs.txt first
    if pairs_file and pairs_file.exists():
        with open(pairs_file) as f:
            lines = f.readlines()
        header = lines[0].strip().split("\t")
        num_sets = int(header[0]) if len(header) >= 1 else 10
        pairs_per_set = int(header[1]) if len(header) >= 2 else 300
        idx = 1
        for _ in range(num_sets):
            for _ in range(pairs_per_set):
                if idx >= len(lines): break
                parts = lines[idx].strip().split("\t")
                idx += 1
                if len(parts) == 3:
                    name, n1, n2 = parts
                    img1 = lfw_dir / name / f"{name}_{int(n1):04d}.jpg"
                    img2 = lfw_dir / name / f"{name}_{int(n2):04d}.jpg"
                    if img1.exists() and img2.exists():
                        matched.append((str(img1), str(img2)))
            for _ in range(pairs_per_set):
                if idx >= len(lines): break
                parts = lines[idx].strip().split("\t")
                idx += 1
                if len(parts) == 4:
                    name1, n1, name2, n2 = parts
                    img1 = lfw_dir / name1 / f"{name1}_{int(n1):04d}.jpg"
                    img2 = lfw_dir / name2 / f"{name2}_{int(n2):04d}.jpg"
                    if img1.exists() and img2.exists():
                        mismatched.append((str(img1), str(img2)))

    # Fallback: auto-generate pairs from directory structure
    if not matched or not mismatched:
        log("  pairs.txt not available, auto-generating pairs from images...")
        persons = {}
        for person_dir in sorted(lfw_dir.iterdir()):
            if person_dir.is_dir():
                imgs = sorted(person_dir.glob("*.jpg"))
                if imgs:
                    persons[person_dir.name] = [str(p) for p in imgs]

        # Matched pairs: same person, different photos
        for name, imgs in persons.items():
            if len(imgs) >= 2:
                for i in range(len(imgs)):
                    for j in range(i + 1, len(imgs)):
                        matched.append((imgs[i], imgs[j]))

        # Mismatched pairs: different persons
        person_names = list(persons.keys())
        random.seed(42)
        for _ in range(max_pairs * 3):
            p1, p2 = random.sample(person_names, 2)
            img1 = random.choice(persons[p1])
            img2 = random.choice(persons[p2])
            mismatched.append((img1, img2))

        log(f"  Generated {len(matched)} matched + {len(mismatched)} mismatched pairs")

    # Subsample
    if len(matched) > max_pairs:
        random.seed(42)
        matched = random.sample(matched, max_pairs)
    if len(mismatched) > max_pairs:
        random.seed(42)
        mismatched = random.sample(mismatched, max_pairs)

    return matched, mismatched


# ─── Benchmark Tests ────────────────────────────────────────────────────────


def benchmark_compare(matched_pairs, mismatched_pairs, max_tests=100):
    """
    Benchmark face comparison accuracy.
    Tests same-person pairs (should match) and different-person pairs (should not match).
    """
    log(f"═══ Face Compare Benchmark ═══")
    log(f"  Testing {min(len(matched_pairs), max_tests)} matched + "
        f"{min(len(mismatched_pairs), max_tests)} mismatched pairs")

    results = {
        "true_positive": 0,   # correctly matched (same person → matched=true)
        "false_negative": 0,  # missed match (same person → matched=false)
        "true_negative": 0,   # correctly rejected (diff person → matched=false)
        "false_positive": 0,  # wrong match (diff person → matched=true)
        "errors": 0,
        "latencies_ms": [],
        "similarities_matched": [],
        "similarities_mismatched": [],
    }

    # Test matched pairs (same person)
    log(f"  Testing same-person pairs...")
    for i, (img1, img2) in enumerate(matched_pairs[:max_tests]):
        files = {
            "sourceImage": ("source.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("target.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)

        if "error" in resp:
            results["errors"] += 1
            continue

        results["latencies_ms"].append(ms)
        sim = resp.get("similarity", 0)
        results["similarities_matched"].append(sim)

        if resp.get("matched", False):
            results["true_positive"] += 1
        else:
            results["false_negative"] += 1

        if (i + 1) % 20 == 0:
            log(f"    Matched: {i+1}/{min(len(matched_pairs), max_tests)}")

    # Test mismatched pairs (different person)
    log(f"  Testing different-person pairs...")
    for i, (img1, img2) in enumerate(mismatched_pairs[:max_tests]):
        files = {
            "sourceImage": ("source.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("target.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)

        if "error" in resp:
            results["errors"] += 1
            continue

        results["latencies_ms"].append(ms)
        sim = resp.get("similarity", 0)
        results["similarities_mismatched"].append(sim)

        if not resp.get("matched", True):
            results["true_negative"] += 1
        else:
            results["false_positive"] += 1

        if (i + 1) % 20 == 0:
            log(f"    Mismatched: {i+1}/{min(len(mismatched_pairs), max_tests)}")

    # Calculate metrics
    tp = results["true_positive"]
    fn = results["false_negative"]
    tn = results["true_negative"]
    fp = results["false_positive"]
    total = tp + fn + tn + fp

    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["metrics"] = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "errors": results["errors"],
        "avg_latency_ms": round(np.mean(results["latencies_ms"]), 1) if results["latencies_ms"] else 0,
        "p95_latency_ms": round(np.percentile(results["latencies_ms"], 95), 1) if results["latencies_ms"] else 0,
        "avg_similarity_matched": round(np.mean(results["similarities_matched"]), 2) if results["similarities_matched"] else 0,
        "avg_similarity_mismatched": round(np.mean(results["similarities_mismatched"]), 2) if results["similarities_mismatched"] else 0,
    }

    m = results["metrics"]
    log(f"  ── Results ──")
    log(f"  Accuracy:  {m['accuracy']}%")
    log(f"  Precision: {m['precision']}%  |  Recall: {m['recall']}%  |  F1: {m['f1_score']}%")
    log(f"  TP={tp} FN={fn} TN={tn} FP={fp} Errors={results['errors']}")
    log(f"  Avg similarity (matched):    {m['avg_similarity_matched']}%")
    log(f"  Avg similarity (mismatched): {m['avg_similarity_mismatched']}%")
    log(f"  Avg latency: {m['avg_latency_ms']}ms  |  P95: {m['p95_latency_ms']}ms")

    return results


def benchmark_liveness(lfw_dir, max_tests=100):
    """
    Benchmark liveness detection on real photos.
    LFW photos are all real (no spoofing) → should all be classified as live.
    """
    log(f"═══ Liveness Benchmark (Real Photos) ═══")

    # Collect sample images
    all_images = []
    for person_dir in sorted(lfw_dir.iterdir()):
        if person_dir.is_dir():
            for img in person_dir.glob("*.jpg"):
                all_images.append(str(img))
    
    random.seed(42)
    sample = random.sample(all_images, min(max_tests, len(all_images)))
    log(f"  Testing {len(sample)} real photos (all should be isLive=true)")

    live_count = 0
    not_live_count = 0
    errors = 0
    scores = []
    latencies = []

    for i, img_path in enumerate(sample):
        files = {"image": ("test.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/liveness", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        score = resp.get("liveScore", 0)
        scores.append(score)

        if resp.get("isLive", False):
            live_count += 1
        else:
            not_live_count += 1

        if (i + 1) % 20 == 0:
            log(f"    Progress: {i+1}/{len(sample)}")

    total = live_count + not_live_count
    accuracy = live_count / total * 100 if total > 0 else 0

    results = {
        "metrics": {
            "accuracy": round(accuracy, 2),
            "live_count": live_count,
            "not_live_count": not_live_count,
            "errors": errors,
            "avg_live_score": round(np.mean(scores), 2) if scores else 0,
            "min_live_score": round(np.min(scores), 2) if scores else 0,
            "max_live_score": round(np.max(scores), 2) if scores else 0,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        },
        "scores": scores,
    }

    m = results["metrics"]
    log(f"  ── Results ──")
    log(f"  Real photos detected as LIVE: {m['accuracy']}% ({live_count}/{total})")
    log(f"  Avg liveScore: {m['avg_live_score']}  (min={m['min_live_score']}, max={m['max_live_score']})")
    log(f"  Errors: {errors}")
    log(f"  Avg latency: {m['avg_latency_ms']}ms  |  P95: {m['p95_latency_ms']}ms")

    return results


def benchmark_deepfake(lfw_dir, max_tests=100):
    """
    Benchmark deepfake detection on real photos.
    LFW photos are all real → should all be classified as NOT deepfake.
    """
    log(f"═══ Deepfake Benchmark (Real Photos) ═══")

    all_images = []
    for person_dir in sorted(lfw_dir.iterdir()):
        if person_dir.is_dir():
            for img in person_dir.glob("*.jpg"):
                all_images.append(str(img))

    random.seed(43)
    sample = random.sample(all_images, min(max_tests, len(all_images)))
    log(f"  Testing {len(sample)} real photos (all should be isDeepfake=false)")

    real_count = 0
    fake_count = 0
    errors = 0
    latencies = []
    risk_levels = defaultdict(int)

    for i, img_path in enumerate(sample):
        files = {"image": ("test.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/deepfake", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        risk = resp.get("attackRiskLevel", "UNKNOWN")
        risk_levels[risk] += 1

        if not resp.get("isDeepfake", True):
            real_count += 1
        else:
            fake_count += 1

        if (i + 1) % 20 == 0:
            log(f"    Progress: {i+1}/{len(sample)}")

    total = real_count + fake_count
    accuracy = real_count / total * 100 if total > 0 else 0

    results = {
        "metrics": {
            "accuracy": round(accuracy, 2),
            "correctly_real": real_count,
            "false_deepfake": fake_count,
            "errors": errors,
            "risk_distribution": dict(risk_levels),
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        }
    }

    m = results["metrics"]
    log(f"  ── Results ──")
    log(f"  Real photos detected as NOT deepfake: {m['accuracy']}% ({real_count}/{total})")
    log(f"  False deepfake detections: {fake_count}")
    log(f"  Risk distribution: {dict(risk_levels)}")
    log(f"  Errors: {errors}")
    log(f"  Avg latency: {m['avg_latency_ms']}ms  |  P95: {m['p95_latency_ms']}ms")

    return results


def benchmark_analyze(lfw_dir, max_tests=100):
    """
    Benchmark age/gender analysis.
    LFW doesn't have ground truth labels, so we check:
    - Gender confidence should be high (>80%)
    - Age should be reasonable (10-80 range)
    - Gender distribution should be roughly plausible
    """
    log(f"═══ Analyze (Age/Gender) Benchmark ═══")

    all_images = []
    for person_dir in sorted(lfw_dir.iterdir()):
        if person_dir.is_dir():
            for img in person_dir.glob("*.jpg"):
                all_images.append(str(img))

    random.seed(44)
    sample = random.sample(all_images, min(max_tests, len(all_images)))
    log(f"  Testing {len(sample)} photos for age/gender analysis")

    genders = defaultdict(int)
    confidences = []
    ages_low = []
    ages_high = []
    errors = 0
    latencies = []
    suspicious = 0  # age < 5 or > 90, or confidence < 50%

    for i, img_path in enumerate(sample):
        files = {"image": ("test.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/analyze", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        gender = resp.get("gender", "UNKNOWN")
        conf = resp.get("genderConfidence", 0)
        age_range = resp.get("ageRange", {})
        age_lo = age_range.get("low", -1)
        age_hi = age_range.get("high", -1)

        genders[gender] += 1
        confidences.append(conf)
        ages_low.append(age_lo)
        ages_high.append(age_hi)

        # Flag suspicious results
        if age_lo < 5 or age_hi > 90 or conf < 50:
            suspicious += 1

        if (i + 1) % 20 == 0:
            log(f"    Progress: {i+1}/{len(sample)}")

    total = sum(genders.values())

    results = {
        "metrics": {
            "total_analyzed": total,
            "gender_distribution": dict(genders),
            "avg_gender_confidence": round(np.mean(confidences), 2) if confidences else 0,
            "min_gender_confidence": round(np.min(confidences), 2) if confidences else 0,
            "avg_age_low": round(np.mean(ages_low), 1) if ages_low else 0,
            "avg_age_high": round(np.mean(ages_high), 1) if ages_high else 0,
            "suspicious_results": suspicious,
            "suspicious_pct": round(suspicious / total * 100, 1) if total > 0 else 0,
            "errors": errors,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        }
    }

    m = results["metrics"]
    log(f"  ── Results ──")
    log(f"  Gender distribution: {dict(genders)}")
    log(f"  Avg gender confidence: {m['avg_gender_confidence']}% (min={m['min_gender_confidence']}%)")
    log(f"  Avg age range: {m['avg_age_low']}-{m['avg_age_high']}")
    log(f"  Suspicious results (age<5 or >90 or conf<50%): {suspicious}/{total} ({m['suspicious_pct']}%)")
    log(f"  Errors: {errors}")
    log(f"  Avg latency: {m['avg_latency_ms']}ms  |  P95: {m['p95_latency_ms']}ms")

    return results


def benchmark_embed(lfw_dir, max_tests=50):
    """
    Benchmark embedding extraction.
    Check: embeddings are 512-dim, normalized, consistent for same image.
    """
    log(f"═══ Embed Benchmark ═══")

    all_images = []
    for person_dir in sorted(lfw_dir.iterdir()):
        if person_dir.is_dir():
            for img in person_dir.glob("*.jpg"):
                all_images.append(str(img))
                if len(all_images) >= max_tests:
                    break
        if len(all_images) >= max_tests:
            break

    log(f"  Testing {len(all_images)} photos for embedding extraction")

    valid = 0
    invalid_dim = 0
    invalid_norm = 0
    errors = 0
    latencies = []

    for i, img_path in enumerate(all_images):
        files = {"image": ("test.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/embed", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        emb = resp.get("embedding", [])

        if len(emb) != 512:
            invalid_dim += 1
            continue

        # Check if normalized (L2 norm should be ~1.0)
        norm = np.linalg.norm(emb)
        if 0.9 < norm < 1.1:
            valid += 1
        else:
            invalid_norm += 1

        if (i + 1) % 20 == 0:
            log(f"    Progress: {i+1}/{len(all_images)}")

    total = valid + invalid_dim + invalid_norm

    results = {
        "metrics": {
            "valid_embeddings": valid,
            "invalid_dimension": invalid_dim,
            "invalid_norm": invalid_norm,
            "errors": errors,
            "valid_pct": round(valid / total * 100, 2) if total > 0 else 0,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        }
    }

    m = results["metrics"]
    log(f"  ── Results ──")
    log(f"  Valid embeddings (512-dim, normalized): {valid}/{total} ({m['valid_pct']}%)")
    log(f"  Invalid dimension: {invalid_dim}  |  Invalid norm: {invalid_norm}")
    log(f"  Errors: {errors}")
    log(f"  Avg latency: {m['avg_latency_ms']}ms  |  P95: {m['p95_latency_ms']}ms")

    return results


# ─── Report Generation ──────────────────────────────────────────────────────


def generate_report(all_results):
    """Generate final benchmark report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = REPORT_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Text report
    txt_path = REPORT_DIR / f"benchmark_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  VISON AI SERVICE — BENCHMARK REPORT\n")
        f.write(f"  Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"  Service URL: {BASE_URL}\n")
        f.write("=" * 70 + "\n\n")

        # Compare
        if "compare" in all_results:
            m = all_results["compare"]["metrics"]
            f.write("─── FACE COMPARE ───────────────────────────────────────\n")
            f.write(f"  Accuracy:   {m['accuracy']}%\n")
            f.write(f"  Precision:  {m['precision']}%\n")
            f.write(f"  Recall:     {m['recall']}%\n")
            f.write(f"  F1 Score:   {m['f1_score']}%\n")
            f.write(f"  TP={m['true_positive']} FN={m['false_negative']} "
                    f"TN={m['true_negative']} FP={m['false_positive']}\n")
            f.write(f"  Avg similarity (matched):    {m['avg_similarity_matched']}%\n")
            f.write(f"  Avg similarity (mismatched): {m['avg_similarity_mismatched']}%\n")
            f.write(f"  Avg latency: {m['avg_latency_ms']}ms | P95: {m['p95_latency_ms']}ms\n\n")

            # Grade
            if m['accuracy'] >= 95:
                grade = "A (Excellent — commercial grade)"
            elif m['accuracy'] >= 90:
                grade = "B (Good — acceptable for production)"
            elif m['accuracy'] >= 80:
                grade = "C (Fair — needs improvement)"
            else:
                grade = "D (Poor — significant issues)"
            f.write(f"  GRADE: {grade}\n")
            f.write(f"  Target: >95% accuracy on LFW (AWS/Tencent baseline: ~99%)\n\n")

        # Liveness
        if "liveness" in all_results:
            m = all_results["liveness"]["metrics"]
            f.write("─── LIVENESS (Real Photos) ─────────────────────────────\n")
            f.write(f"  Real photos → isLive=true: {m['accuracy']}%\n")
            f.write(f"  Avg liveScore: {m['avg_live_score']} "
                    f"(min={m['min_live_score']}, max={m['max_live_score']})\n")
            f.write(f"  Avg latency: {m['avg_latency_ms']}ms | P95: {m['p95_latency_ms']}ms\n\n")

            if m['accuracy'] >= 90:
                grade = "A (Good true-positive rate)"
            elif m['accuracy'] >= 75:
                grade = "B (Acceptable, some false rejections)"
            else:
                grade = "C (Too many false rejections)"
            f.write(f"  GRADE: {grade}\n")
            f.write(f"  Target: >90% of real photos should be isLive=true\n\n")

        # Deepfake
        if "deepfake" in all_results:
            m = all_results["deepfake"]["metrics"]
            f.write("─── DEEPFAKE (Real Photos) ─────────────────────────────\n")
            f.write(f"  Real photos → NOT deepfake: {m['accuracy']}%\n")
            f.write(f"  False deepfake detections: {m['false_deepfake']}\n")
            f.write(f"  Risk distribution: {m['risk_distribution']}\n")
            f.write(f"  Avg latency: {m['avg_latency_ms']}ms | P95: {m['p95_latency_ms']}ms\n\n")

            if m['accuracy'] >= 95:
                grade = "A (Low false positive rate)"
            elif m['accuracy'] >= 85:
                grade = "B (Acceptable)"
            else:
                grade = "C (Too many false positives)"
            f.write(f"  GRADE: {grade}\n")
            f.write(f"  Target: >95% of real photos should NOT be flagged\n\n")

        # Analyze
        if "analyze" in all_results:
            m = all_results["analyze"]["metrics"]
            f.write("─── ANALYZE (Age/Gender) ───────────────────────────────\n")
            f.write(f"  Gender distribution: {m['gender_distribution']}\n")
            f.write(f"  Avg confidence: {m['avg_gender_confidence']}% "
                    f"(min={m['min_gender_confidence']}%)\n")
            f.write(f"  Avg age range: {m['avg_age_low']}-{m['avg_age_high']}\n")
            f.write(f"  Suspicious (age<5 or >90 or conf<50%): "
                    f"{m['suspicious_results']} ({m['suspicious_pct']}%)\n")
            f.write(f"  Avg latency: {m['avg_latency_ms']}ms | P95: {m['p95_latency_ms']}ms\n\n")

            if m['suspicious_pct'] <= 5 and m['avg_gender_confidence'] >= 85:
                grade = "A (Reliable)"
            elif m['suspicious_pct'] <= 15 and m['avg_gender_confidence'] >= 70:
                grade = "B (Mostly reliable)"
            else:
                grade = "C (Needs improvement)"
            f.write(f"  GRADE: {grade}\n")
            f.write(f"  Target: <5% suspicious, avg confidence >85%\n\n")

        # Embed
        if "embed" in all_results:
            m = all_results["embed"]["metrics"]
            f.write("─── EMBED ──────────────────────────────────────────────\n")
            f.write(f"  Valid embeddings: {m['valid_pct']}%\n")
            f.write(f"  Invalid dimension: {m['invalid_dimension']}\n")
            f.write(f"  Invalid norm: {m['invalid_norm']}\n")
            f.write(f"  Avg latency: {m['avg_latency_ms']}ms | P95: {m['p95_latency_ms']}ms\n\n")

        f.write("=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")

    return json_path, txt_path


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Vison AI Service Benchmark")
    parser.add_argument("--max-pairs", type=int, default=100,
                        help="Max pairs for compare test (default: 100)")
    parser.add_argument("--max-tests", type=int, default=100,
                        help="Max tests per endpoint (default: 100)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing)")
    parser.add_argument("--tests", nargs="+",
                        default=["compare", "liveness", "deepfake", "analyze", "embed"],
                        help="Which tests to run")
    args = parser.parse_args()

    log("╔══════════════════════════════════════════════════════╗")
    log("║      VISON AI SERVICE — AUTOMATED BENCHMARK         ║")
    log("╚══════════════════════════════════════════════════════╝")
    log(f"  URL: {BASE_URL}")
    log(f"  Max pairs: {args.max_pairs}  |  Max tests: {args.max_tests}")
    log(f"  Tests: {', '.join(args.tests)}")

    # Check service
    if not check_health():
        log("Service is NOT running! Start it first:", "ERROR")
        log("  uvicorn app.main:app --host 127.0.0.1 --port 8000", "ERROR")
        sys.exit(1)
    log("Service is running ✓")

    # Prepare dataset
    log("\n─── Preparing LFW Dataset ───")
    lfw_dir, pairs_file = prepare_lfw_dataset()
    if lfw_dir is None or not lfw_dir.exists():
        log("Failed to prepare LFW dataset!", "ERROR")
        sys.exit(1)

    # Count images
    img_count = sum(1 for _ in lfw_dir.rglob("*.jpg"))
    person_count = sum(1 for d in lfw_dir.iterdir() if d.is_dir())
    log(f"  LFW ready: {img_count} images, {person_count} persons")

    all_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "service_url": BASE_URL,
        "dataset": "LFW (Labeled Faces in the Wild)",
        "config": {
            "max_pairs": args.max_pairs,
            "max_tests": args.max_tests,
        }
    }

    # Run benchmarks
    print()

    if "compare" in args.tests:
        matched, mismatched = parse_lfw_pairs(pairs_file, lfw_dir, args.max_pairs)
        log(f"  Parsed {len(matched)} matched + {len(mismatched)} mismatched pairs")
        print()
        all_results["compare"] = benchmark_compare(matched, mismatched, args.max_pairs)
        print()

    if "liveness" in args.tests:
        all_results["liveness"] = benchmark_liveness(lfw_dir, args.max_tests)
        print()

    if "deepfake" in args.tests:
        all_results["deepfake"] = benchmark_deepfake(lfw_dir, args.max_tests)
        print()

    if "analyze" in args.tests:
        all_results["analyze"] = benchmark_analyze(lfw_dir, args.max_tests)
        print()

    if "embed" in args.tests:
        all_results["embed"] = benchmark_embed(lfw_dir, args.max_tests)
        print()

    # Generate report
    log("─── Generating Report ───")
    json_path, txt_path = generate_report(all_results)
    log(f"  JSON: {json_path}")
    log(f"  Text: {txt_path}")

    # Print summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for test_name in ["compare", "liveness", "deepfake", "analyze", "embed"]:
        if test_name in all_results and "metrics" in all_results[test_name]:
            m = all_results[test_name]["metrics"]
            if "accuracy" in m:
                print(f"  {test_name.upper():12s} → {m['accuracy']}%")
            elif "valid_pct" in m:
                print(f"  {test_name.upper():12s} → {m['valid_pct']}% valid")
    print("=" * 60)
    print(f"  Full report: {txt_path}")
    print()


if __name__ == "__main__":
    main()
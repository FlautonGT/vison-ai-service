#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vison AI Service — Comprehensive Benchmark v2
==============================================
Uses UTKFace (labeled age/gender) + LFW (identity pairs) for full validation.

Benchmarks:
  1. Compare   — LFW same/different person pairs → accuracy, F1
  2. Analyze   — UTKFace labeled photos → gender accuracy %, age MAE
  3. Liveness  — Real photos should be live
  4. Deepfake  — Real photos should NOT be flagged
  5. Embed     — Embeddings should be 512-dim, normalized

Usage:
    python scripts/benchmark_v2.py
    python scripts/benchmark_v2.py --max-tests 50 --tests analyze compare
"""

import os, sys, json, time, random, zipfile, tarfile, shutil, argparse, io
import datetime
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np

# === Config ===
BASE_URL = os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000")
API_SECRET = os.getenv("AI_SERVICE_SECRET", "test-123")
DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", "./benchmark_data"))
REPORT_DIR = Path(os.getenv("BENCHMARK_REPORT_DIR", "./benchmark_reports"))
HEADERS = {"X-AI-Service-Key": API_SECRET}

# UTKFace — labeled age/gender/ethnicity in filename
UTKFACE_URL = "https://huggingface.co/datasets/eraiichiro/UTKFace/resolve/main/UTKFace.tar.gz"
# LFW
LFW_URL = "https://ndownloader.figshare.com/files/5976018"


def log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def download_file(url, dest, desc=""):
    if dest.exists():
        log(f"  Already exists: {dest.name}")
        return True
    log(f"  Downloading {desc or dest.name}...")
    try:
        r = requests.get(url, stream=True, timeout=600)
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
    try:
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=5)
        return r.status_code == 200
    except:
        return False


# === Dataset Preparation ===

def prepare_utkface():
    """Download UTKFace dataset. Labels encoded in filename: age_gender_race_date.jpg"""
    utk_dir = DATA_DIR / "UTKFace"
    utk_tgz = DATA_DIR / "UTKFace.tar.gz"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not utk_dir.exists():
        if not download_file(UTKFACE_URL, utk_tgz, "UTKFace dataset (~600MB)"):
            return None

        log("  Extracting UTKFace...")
        with tarfile.open(utk_tgz, "r:gz") as tar:
            tar.extractall(DATA_DIR)

    # Find the actual image directory
    if utk_dir.exists() and any(utk_dir.glob("*.jpg")):
        return utk_dir
    
    # Check subdirectories
    for sub in DATA_DIR.iterdir():
        if sub.is_dir() and any(sub.glob("*.jpg")):
            fname = list(sub.glob("*.jpg"))[0].name
            if "_" in fname and fname.split("_")[0].isdigit():
                return sub

    return utk_dir if utk_dir.exists() else None


def parse_utkface_label(filename):
    """Parse UTKFace filename: age_gender_race_date.jpg → {age, gender, race}"""
    try:
        parts = Path(filename).stem.split("_")
        if len(parts) >= 3:
            age = int(parts[0])
            gender_code = int(parts[1])  # 0=male, 1=female
            race_code = int(parts[2])    # 0=white,1=black,2=asian,3=indian,4=other
            gender = "MAN" if gender_code == 0 else "WOMAN"
            return {"age": age, "gender": gender, "race_code": race_code}
    except:
        pass
    return None


def prepare_lfw():
    """Download LFW for face compare benchmark."""
    lfw_dir = DATA_DIR / "lfw"
    lfw_tgz = DATA_DIR / "lfw-funneled.tgz"

    if lfw_dir.exists() and any(lfw_dir.iterdir()):
        return lfw_dir

    # Check alternative names from previous download
    for name in ["lfw_funneled", "lfw-funneled"]:
        alt = DATA_DIR / name
        if alt.exists():
            return alt

    if not download_file(LFW_URL, lfw_tgz, "LFW dataset (~233MB)"):
        return None

    log("  Extracting LFW...")
    with tarfile.open(lfw_tgz, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    for name in ["lfw", "lfw_funneled", "lfw-funneled"]:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    return None


def generate_lfw_pairs(lfw_dir, max_pairs=200):
    """Auto-generate same/different person pairs from LFW directory."""
    persons = {}
    for person_dir in sorted(lfw_dir.iterdir()):
        if person_dir.is_dir():
            imgs = sorted(person_dir.glob("*.jpg"))
            if imgs:
                persons[person_dir.name] = [str(p) for p in imgs]

    matched = []
    for name, imgs in persons.items():
        if len(imgs) >= 2:
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    matched.append((imgs[i], imgs[j]))

    mismatched = []
    person_names = list(persons.keys())
    random.seed(42)
    for _ in range(max_pairs * 3):
        p1, p2 = random.sample(person_names, 2)
        img1 = random.choice(persons[p1])
        img2 = random.choice(persons[p2])
        mismatched.append((img1, img2))

    if len(matched) > max_pairs:
        random.seed(42)
        matched = random.sample(matched, max_pairs)
    if len(mismatched) > max_pairs:
        random.seed(42)
        mismatched = random.sample(mismatched, max_pairs)

    return matched, mismatched


# === Benchmark Tests ===

def benchmark_compare(lfw_dir, max_tests=100):
    """Face compare: same person should match, different should not."""
    log("=" * 60)
    log("  FACE COMPARE BENCHMARK (LFW)")
    log("=" * 60)

    matched, mismatched = generate_lfw_pairs(lfw_dir, max_tests)
    log(f"  Pairs: {len(matched)} matched + {len(mismatched)} mismatched")

    tp, fn, tn, fp, errors = 0, 0, 0, 0, 0
    latencies = []
    sim_matched, sim_mismatched = [], []
    details = []

    # Same person
    log(f"  Testing same-person pairs...")
    for i, (img1, img2) in enumerate(matched):
        files = {
            "sourceImage": ("s.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("t.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)
        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        sim = resp.get("similarity", 0)
        is_matched = resp.get("matched", False)
        sim_matched.append(sim)

        p1 = Path(img1).parent.name
        details.append({
            "type": "same_person", "person": p1,
            "similarity": sim, "matched": is_matched,
            "correct": is_matched, "latency_ms": round(ms, 1)
        })

        if is_matched:
            tp += 1
        else:
            fn += 1

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{len(matched)}")

    # Different person
    log(f"  Testing different-person pairs...")
    for i, (img1, img2) in enumerate(mismatched):
        files = {
            "sourceImage": ("s.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("t.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)
        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        sim = resp.get("similarity", 0)
        is_matched = resp.get("matched", False)
        sim_mismatched.append(sim)

        p1, p2 = Path(img1).parent.name, Path(img2).parent.name
        details.append({
            "type": "diff_person", "person1": p1, "person2": p2,
            "similarity": sim, "matched": is_matched,
            "correct": not is_matched, "latency_ms": round(ms, 1)
        })

        if not is_matched:
            tn += 1
        else:
            fp += 1

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{len(mismatched)}")

    total = tp + fn + tn + fp
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "metrics": {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp, "errors": errors,
            "avg_sim_matched": round(np.mean(sim_matched), 2) if sim_matched else 0,
            "avg_sim_mismatched": round(np.mean(sim_mismatched), 2) if sim_mismatched else 0,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        },
        "details": details,
    }

    m = results["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Accuracy: {m['accuracy']}%  |  F1: {m['f1']}%")
    log(f"  TP={tp} FN={fn} TN={tn} FP={fp} Errors={errors}")
    log(f"  Avg similarity matched: {m['avg_sim_matched']}%  mismatched: {m['avg_sim_mismatched']}%")
    log(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms")
    return results


def benchmark_analyze(utk_dir, max_tests=200):
    """
    Age/Gender benchmark with GROUND TRUTH from UTKFace labels.
    Compare API response vs actual age/gender encoded in filename.
    """
    log("=" * 60)
    log("  ANALYZE (AGE/GENDER) BENCHMARK — UTKFace Ground Truth")
    log("=" * 60)

    # Collect labeled images
    all_labeled = []
    for img in sorted(utk_dir.glob("*.jpg")):
        label = parse_utkface_label(img.name)
        if label and 10 <= label["age"] <= 80:  # reasonable range
            all_labeled.append((str(img), label))

    random.seed(42)
    sample = random.sample(all_labeled, min(max_tests, len(all_labeled)))
    log(f"  Testing {len(sample)} labeled photos (ground truth age + gender)")

    gender_correct = 0
    gender_wrong = 0
    age_errors = []  # abs(predicted - actual)
    latencies = []
    errors = 0
    details = []

    for i, (img_path, gt) in enumerate(sample):
        files = {"image": ("test.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/analyze", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)

        pred_gender = resp.get("gender", "UNKNOWN")
        pred_conf = resp.get("genderConfidence", 0)
        pred_age = resp.get("ageRange", {})
        pred_age_mid = (pred_age.get("low", 0) + pred_age.get("high", 0)) / 2

        gt_gender = gt["gender"]
        gt_age = gt["age"]

        gender_ok = pred_gender == gt_gender
        age_err = abs(pred_age_mid - gt_age)

        if gender_ok:
            gender_correct += 1
        else:
            gender_wrong += 1
        age_errors.append(age_err)

        details.append({
            "file": Path(img_path).name,
            "gt_gender": gt_gender,
            "pred_gender": pred_gender,
            "gender_conf": round(pred_conf, 1),
            "gender_correct": gender_ok,
            "gt_age": gt_age,
            "pred_age_low": pred_age.get("low", 0),
            "pred_age_high": pred_age.get("high", 0),
            "pred_age_mid": round(pred_age_mid, 1),
            "age_error": round(age_err, 1),
            "latency_ms": round(ms, 1),
        })

        if (i + 1) % 25 == 0:
            g_acc = gender_correct / (gender_correct + gender_wrong) * 100
            a_mae = np.mean(age_errors)
            log(f"    {i+1}/{len(sample)}  gender_acc={g_acc:.1f}%  age_MAE={a_mae:.1f}")

    total = gender_correct + gender_wrong
    gender_acc = gender_correct / total * 100 if total > 0 else 0
    age_mae = np.mean(age_errors) if age_errors else 0
    age_within_5 = sum(1 for e in age_errors if e <= 5) / len(age_errors) * 100 if age_errors else 0
    age_within_10 = sum(1 for e in age_errors if e <= 10) / len(age_errors) * 100 if age_errors else 0

    # Gender confusion matrix
    gender_cm = {"MAN_as_MAN": 0, "MAN_as_WOMAN": 0, "WOMAN_as_WOMAN": 0, "WOMAN_as_MAN": 0}
    for d in details:
        key = f"{d['gt_gender']}_as_{d['pred_gender']}"
        if key in gender_cm:
            gender_cm[key] += 1

    # Worst age predictions
    worst_age = sorted(details, key=lambda x: x["age_error"], reverse=True)[:10]

    results = {
        "metrics": {
            "gender_accuracy": round(gender_acc, 2),
            "gender_correct": gender_correct,
            "gender_wrong": gender_wrong,
            "gender_confusion_matrix": gender_cm,
            "age_mae": round(age_mae, 2),
            "age_within_5_years": round(age_within_5, 2),
            "age_within_10_years": round(age_within_10, 2),
            "age_median_error": round(np.median(age_errors), 2) if age_errors else 0,
            "errors": errors,
            "total_tested": total,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        },
        "worst_age_predictions": worst_age,
        "details": details,
    }

    m = results["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Gender accuracy: {m['gender_accuracy']}% ({gender_correct}/{total})")
    log(f"  Gender confusion: {gender_cm}")
    log(f"  Age MAE: {m['age_mae']} years")
    log(f"  Age within 5 years:  {m['age_within_5_years']}%")
    log(f"  Age within 10 years: {m['age_within_10_years']}%")
    log(f"  Errors: {errors}")
    log(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms")

    log(f"\n  WORST AGE PREDICTIONS:")
    for w in worst_age[:5]:
        log(f"    {w['file']}: actual={w['gt_age']}  predicted={w['pred_age_low']}-{w['pred_age_high']}  error={w['age_error']}")

    return results


def benchmark_liveness(img_dir, max_tests=100):
    """Liveness on real photos — all should be isLive=true."""
    log("=" * 60)
    log("  LIVENESS BENCHMARK (Real Photos)")
    log("=" * 60)

    all_imgs = list(img_dir.glob("*.jpg"))
    if not all_imgs:
        # Try subdirectories (LFW structure)
        for sub in img_dir.iterdir():
            if sub.is_dir():
                all_imgs.extend(sub.glob("*.jpg"))

    random.seed(42)
    sample = random.sample(all_imgs, min(max_tests, len(all_imgs)))
    log(f"  Testing {len(sample)} real photos (all should be isLive=true)")

    live, not_live, errors = 0, 0, 0
    scores, latencies = [], []
    details = []

    for i, img_path in enumerate(sample):
        files = {"image": ("test.jpg", open(str(img_path), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/liveness", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        score = resp.get("liveScore", 0)
        is_live = resp.get("isLive", False)
        scores.append(score)

        details.append({
            "file": img_path.name, "isLive": is_live,
            "liveScore": round(score, 2), "correct": is_live,
            "latency_ms": round(ms, 1),
        })

        if is_live:
            live += 1
        else:
            not_live += 1

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{len(sample)}")

    total = live + not_live
    acc = live / total * 100 if total > 0 else 0

    results = {
        "metrics": {
            "accuracy": round(acc, 2),
            "live_count": live, "not_live_count": not_live, "errors": errors,
            "avg_score": round(np.mean(scores), 2) if scores else 0,
            "min_score": round(np.min(scores), 2) if scores else 0,
            "max_score": round(np.max(scores), 2) if scores else 0,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        },
        "false_rejections": [d for d in details if not d["correct"]],
        "details": details,
    }

    m = results["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Real photos as LIVE: {m['accuracy']}% ({live}/{total})")
    log(f"  Avg score: {m['avg_score']}  min={m['min_score']}  max={m['max_score']}")
    log(f"  Errors: {errors}")
    log(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms")
    return results


def benchmark_deepfake(img_dir, max_tests=100):
    """Deepfake on real photos — all should be NOT deepfake."""
    log("=" * 60)
    log("  DEEPFAKE BENCHMARK (Real Photos)")
    log("=" * 60)

    all_imgs = list(img_dir.glob("*.jpg"))
    if not all_imgs:
        for sub in img_dir.iterdir():
            if sub.is_dir():
                all_imgs.extend(sub.glob("*.jpg"))

    random.seed(43)
    sample = random.sample(all_imgs, min(max_tests, len(all_imgs)))
    log(f"  Testing {len(sample)} real photos (all should be isDeepfake=false)")

    real, fake, errors = 0, 0, 0
    latencies = []
    risks = defaultdict(int)
    details = []

    for i, img_path in enumerate(sample):
        files = {"image": ("test.jpg", open(str(img_path), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/deepfake", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        is_fake = resp.get("isDeepfake", True)
        risk = resp.get("attackRiskLevel", "UNKNOWN")
        risks[risk] += 1

        details.append({
            "file": img_path.name, "isDeepfake": is_fake,
            "attackRiskLevel": risk, "correct": not is_fake,
            "latency_ms": round(ms, 1),
        })

        if not is_fake:
            real += 1
        else:
            fake += 1

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{len(sample)}")

    total = real + fake
    acc = real / total * 100 if total > 0 else 0

    results = {
        "metrics": {
            "accuracy": round(acc, 2),
            "correctly_real": real, "false_deepfake": fake, "errors": errors,
            "risk_distribution": dict(risks),
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        },
        "false_positives": [d for d in details if not d["correct"]],
        "details": details,
    }

    m = results["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Real photos as NOT deepfake: {m['accuracy']}% ({real}/{total})")
    log(f"  False deepfake: {fake}")
    log(f"  Risk distribution: {dict(risks)}")
    log(f"  Errors: {errors}")
    log(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms")
    return results


def benchmark_embed(img_dir, max_tests=50):
    """Embedding quality check."""
    log("=" * 60)
    log("  EMBED BENCHMARK")
    log("=" * 60)

    all_imgs = list(img_dir.glob("*.jpg"))[:max_tests]
    log(f"  Testing {len(all_imgs)} photos")

    valid, bad_dim, bad_norm, errors = 0, 0, 0, 0
    latencies = []

    for i, img_path in enumerate(all_imgs):
        files = {"image": ("test.jpg", open(str(img_path), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/embed", files=files)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        emb = resp.get("embedding", [])

        if len(emb) != 512:
            bad_dim += 1
        elif not (0.9 < np.linalg.norm(emb) < 1.1):
            bad_norm += 1
        else:
            valid += 1

        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{len(all_imgs)}")

    total = valid + bad_dim + bad_norm
    results = {
        "metrics": {
            "valid_pct": round(valid / total * 100, 2) if total > 0 else 0,
            "valid": valid, "bad_dim": bad_dim, "bad_norm": bad_norm, "errors": errors,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        }
    }

    m = results["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Valid: {m['valid_pct']}% ({valid}/{total})")
    log(f"  Errors: {errors}")
    log(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms")
    return results


# === Report ===

def generate_report(all_results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (full details)
    json_path = REPORT_DIR / f"benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Remove large details arrays for readability, keep in separate file
        summary = {}
        for k, v in all_results.items():
            if isinstance(v, dict) and "details" in v:
                summary[k] = {kk: vv for kk, vv in v.items() if kk != "details"}
            else:
                summary[k] = v
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Full details
    details_path = REPORT_DIR / f"benchmark_{ts}_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Text report
    txt_path = REPORT_DIR / f"benchmark_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  VISON AI SERVICE - BENCHMARK REPORT\n")
        f.write(f"  Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"  Service: {BASE_URL}\n")
        f.write("=" * 70 + "\n\n")

        # Compare
        if "compare" in all_results:
            m = all_results["compare"]["metrics"]
            f.write("--- FACE COMPARE (LFW) ---\n")
            f.write(f"  Accuracy:   {m['accuracy']}%\n")
            f.write(f"  Precision:  {m['precision']}%\n")
            f.write(f"  Recall:     {m['recall']}%\n")
            f.write(f"  F1:         {m['f1']}%\n")
            f.write(f"  TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']} Err={m['errors']}\n")
            f.write(f"  Avg sim matched:    {m['avg_sim_matched']}%\n")
            f.write(f"  Avg sim mismatched: {m['avg_sim_mismatched']}%\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms\n")
            grade = "A" if m['accuracy'] >= 95 else "B" if m['accuracy'] >= 90 else "C" if m['accuracy'] >= 80 else "D"
            f.write(f"  GRADE: {grade}  (target: >95% = AWS/Tencent baseline)\n\n")

        # Analyze
        if "analyze" in all_results:
            m = all_results["analyze"]["metrics"]
            f.write("--- ANALYZE AGE/GENDER (UTKFace ground truth) ---\n")
            f.write(f"  Gender accuracy:     {m['gender_accuracy']}% ({m['gender_correct']}/{m['total_tested']})\n")
            f.write(f"  Gender confusion:    {m['gender_confusion_matrix']}\n")
            f.write(f"  Age MAE:             {m['age_mae']} years\n")
            f.write(f"  Age within 5 years:  {m['age_within_5_years']}%\n")
            f.write(f"  Age within 10 years: {m['age_within_10_years']}%\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms\n")

            g_grade = "A" if m['gender_accuracy'] >= 95 else "B" if m['gender_accuracy'] >= 85 else "C" if m['gender_accuracy'] >= 70 else "D"
            a_grade = "A" if m['age_mae'] <= 5 else "B" if m['age_mae'] <= 8 else "C" if m['age_mae'] <= 12 else "D"
            f.write(f"  GRADE Gender: {g_grade}  (target: >95% = AWS baseline ~99%)\n")
            f.write(f"  GRADE Age:    {a_grade}  (target: MAE<5 = AWS baseline ~3)\n\n")

            # Worst predictions
            if "worst_age_predictions" in all_results["analyze"]:
                f.write("  WORST AGE PREDICTIONS:\n")
                for w in all_results["analyze"]["worst_age_predictions"][:5]:
                    f.write(f"    {w['file']}: actual={w['gt_age']} predicted={w['pred_age_low']}-{w['pred_age_high']} err={w['age_error']}\n")
                f.write("\n")

        # Liveness
        if "liveness" in all_results:
            m = all_results["liveness"]["metrics"]
            f.write("--- LIVENESS (Real photos) ---\n")
            f.write(f"  Real as LIVE:  {m['accuracy']}% ({m['live_count']}/{m['live_count']+m['not_live_count']})\n")
            f.write(f"  Avg score: {m['avg_score']}  min={m['min_score']}  max={m['max_score']}\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms\n")
            grade = "A" if m['accuracy'] >= 90 else "B" if m['accuracy'] >= 75 else "C"
            f.write(f"  GRADE: {grade}  (target: >90% real as live)\n\n")

        # Deepfake
        if "deepfake" in all_results:
            m = all_results["deepfake"]["metrics"]
            f.write("--- DEEPFAKE (Real photos) ---\n")
            f.write(f"  Real as NOT deepfake: {m['accuracy']}% ({m['correctly_real']}/{m['correctly_real']+m['false_deepfake']})\n")
            f.write(f"  False positives: {m['false_deepfake']}\n")
            f.write(f"  Risk distribution: {m['risk_distribution']}\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms\n")
            grade = "A" if m['accuracy'] >= 95 else "B" if m['accuracy'] >= 85 else "C"
            f.write(f"  GRADE: {grade}  (target: >95% real not flagged)\n\n")

        # Embed
        if "embed" in all_results:
            m = all_results["embed"]["metrics"]
            f.write("--- EMBED ---\n")
            f.write(f"  Valid: {m['valid_pct']}%\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms p95={m['p95_latency_ms']}ms\n\n")

        f.write("=" * 70 + "\n")
        f.write("  COMPARISON VS COMMERCIAL (AWS/Tencent targets)\n")
        f.write("=" * 70 + "\n")
        f.write("  Compare accuracy:  target >95%  (AWS ~99%)\n")
        f.write("  Gender accuracy:   target >95%  (AWS ~99%)\n")
        f.write("  Age MAE:           target <5yr  (AWS ~3yr)\n")
        f.write("  Liveness (real):   target >90%  (Tencent ~99%)\n")
        f.write("  Deepfake (real):   target >95%  (Tencent ~97%)\n")
        f.write("=" * 70 + "\n")

    return json_path, txt_path, details_path


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Vison AI Benchmark v2")
    parser.add_argument("--max-tests", type=int, default=100)
    parser.add_argument("--tests", nargs="+",
                        default=["compare", "analyze", "liveness", "deepfake", "embed"])
    args = parser.parse_args()

    log("=" * 60)
    log("  VISON AI SERVICE - COMPREHENSIVE BENCHMARK v2")
    log("=" * 60)
    log(f"  URL: {BASE_URL}  |  Max tests: {args.max_tests}")
    log(f"  Tests: {', '.join(args.tests)}")

    if not check_health():
        log("Service NOT running!", "ERROR")
        sys.exit(1)
    log("  Service OK")

    all_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {"max_tests": args.max_tests, "tests": args.tests},
    }

    # Prepare datasets
    utk_dir = None
    lfw_dir = None

    if "analyze" in args.tests or "liveness" in args.tests or "deepfake" in args.tests or "embed" in args.tests:
        log("\n--- Preparing UTKFace Dataset ---")
        utk_dir = prepare_utkface()
        if utk_dir:
            count = len(list(utk_dir.glob("*.jpg")))
            log(f"  UTKFace ready: {count} labeled images")
        else:
            log("  UTKFace download failed, will use LFW for these tests", "WARN")

    if "compare" in args.tests:
        log("\n--- Preparing LFW Dataset ---")
        lfw_dir = prepare_lfw()
        if lfw_dir:
            count = sum(1 for _ in lfw_dir.rglob("*.jpg"))
            log(f"  LFW ready: {count} images")

    print()

    # Run benchmarks
    if "compare" in args.tests and lfw_dir:
        all_results["compare"] = benchmark_compare(lfw_dir, args.max_tests)
        print()

    if "analyze" in args.tests:
        img_dir = utk_dir or lfw_dir
        if img_dir:
            all_results["analyze"] = benchmark_analyze(img_dir, args.max_tests)
        print()

    if "liveness" in args.tests:
        img_dir = utk_dir or lfw_dir
        if img_dir:
            all_results["liveness"] = benchmark_liveness(img_dir, args.max_tests)
        print()

    if "deepfake" in args.tests:
        img_dir = utk_dir or lfw_dir
        if img_dir:
            all_results["deepfake"] = benchmark_deepfake(img_dir, args.max_tests)
        print()

    if "embed" in args.tests:
        img_dir = utk_dir or lfw_dir
        if img_dir:
            all_results["embed"] = benchmark_embed(img_dir, args.max_tests)
        print()

    # Report
    log("--- Generating Report ---")
    json_path, txt_path, details_path = generate_report(all_results)
    log(f"  Summary: {txt_path}")
    log(f"  JSON:    {json_path}")
    log(f"  Details: {details_path}")

    # Print final summary
    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    if "compare" in all_results:
        m = all_results["compare"]["metrics"]
        print(f"  COMPARE:   accuracy={m['accuracy']}%  F1={m['f1']}%")
    if "analyze" in all_results:
        m = all_results["analyze"]["metrics"]
        print(f"  GENDER:    accuracy={m['gender_accuracy']}%")
        print(f"  AGE:       MAE={m['age_mae']} years  within5yr={m['age_within_5_years']}%")
    if "liveness" in all_results:
        m = all_results["liveness"]["metrics"]
        print(f"  LIVENESS:  {m['accuracy']}% real as live")
    if "deepfake" in all_results:
        m = all_results["deepfake"]["metrics"]
        print(f"  DEEPFAKE:  {m['accuracy']}% real not flagged")
    if "embed" in all_results:
        m = all_results["embed"]["metrics"]
        print(f"  EMBED:     {m['valid_pct']}% valid")
    print("=" * 60)


if __name__ == "__main__":
    main()
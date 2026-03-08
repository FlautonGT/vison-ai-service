#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vison AI — Comprehensive Industry Benchmark (1000+ faces)
==========================================================
Uses the SAME datasets that AWS, Tencent, InsightFace, AdaFace benchmark on.

Datasets:
  - LFW (13K faces)     → Compare same/diff person, Embed
  - UTKFace (23K faces)  → Analyze age/gender with ground truth
  - AI Generated (100)   → Deepfake + Liveness vs AI faces
  - Real photos          → Deepfake + Liveness vs real faces

Tests (1000+ total face evaluations):
  1. COMPARE    — 250 matched + 250 mismatched pairs = 500 comparisons
  2. ANALYZE    — 500 photos with ground truth age + gender
  3. LIVENESS   — 200 real + 100 AI = 300 tests
  4. DEEPFAKE   — 200 real + 100 AI = 300 tests
  5. EMBED      — 200 embedding extractions + consistency check

Usage:
    python scripts/benchmark_1k.py
    python scripts/benchmark_1k.py --scale large    (2500+ evaluations)
    python scripts/benchmark_1k.py --scale small    (200 evaluations, quick)
"""

import os, sys, json, time, random, tarfile, shutil, argparse, datetime
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

# Scale presets
SCALES = {
    "small":  {"compare": 50,  "analyze": 100, "liveness": 50,  "deepfake": 50,  "embed": 50,  "ai_faces": 50},
    "medium": {"compare": 250, "analyze": 500, "liveness": 200, "deepfake": 200, "embed": 200, "ai_faces": 100},
    "large":  {"compare": 500, "analyze": 1000,"liveness": 500, "deepfake": 500, "embed": 500, "ai_faces": 200},
}

LFW_URL = "https://ndownloader.figshare.com/files/5976018"
UTKFACE_URLS = [
    "https://huggingface.co/datasets/wildchlamydia/utkface/resolve/main/data/UTKFace.tar.gz",
    "https://huggingface.co/datasets/raulc0399/utkface/resolve/main/data/train-00000-of-00003.parquet",
]


def log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def call_api(endpoint, files=None, timeout=30):
    url = f"{BASE_URL}{endpoint}"
    try:
        start = time.time()
        r = requests.post(url, headers=HEADERS, files=files, timeout=timeout)
        ms = (time.time() - start) * 1000
        if r.status_code == 200:
            return r.json(), ms
        return {"error": r.status_code, "detail": r.text[:200]}, ms
    except Exception as e:
        return {"error": str(e)}, 0


def download_file(url, dest, desc=""):
    if dest.exists():
        log(f"  Cached: {dest.name}")
        return True
    log(f"  Downloading {desc or dest.name}...")
    try:
        r = requests.get(url, stream=True, timeout=600,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dl = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                dl += len(chunk)
                if total:
                    print(f"\r  [{dl*100//total:3d}%] {dl/1024/1024:.1f}MB", end="", flush=True)
        print()
        return True
    except Exception as e:
        log(f"  Failed: {e}", "ERROR")
        if dest.exists():
            dest.unlink()
        return False


# === Dataset Prep ===

def prep_lfw():
    for name in ["lfw", "lfw_funneled", "lfw-funneled"]:
        d = DATA_DIR / name
        if d.exists() and any(d.iterdir()):
            return d
    tgz = DATA_DIR / "lfw-funneled.tgz"
    if not download_file(LFW_URL, tgz, "LFW (~233MB)"):
        return None
    log("  Extracting LFW...")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    for name in ["lfw", "lfw_funneled", "lfw-funneled"]:
        d = DATA_DIR / name
        if d.exists():
            return d
    return None


def prep_utkface():
    utk = DATA_DIR / "UTKFace"
    if utk.exists() and len(list(utk.glob("*.*jpg*"))) > 100:
        return utk
    # Try download
    tgz = DATA_DIR / "UTKFace.tar.gz"
    for url in UTKFACE_URLS:
        if download_file(url, tgz, "UTKFace"):
            try:
                log("  Extracting UTKFace...")
                with tarfile.open(tgz, "r:gz") as tar:
                    tar.extractall(DATA_DIR)
                if utk.exists():
                    return utk
            except:
                pass
    # Check if already there from manual download
    if utk.exists():
        return utk
    return None


def prep_ai_faces(count=100):
    ai_dir = DATA_DIR / "ai_faces"
    ai_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(ai_dir.glob("*.jpg")))
    if existing >= count:
        return ai_dir
    log(f"  Downloading {count - existing} AI faces...")
    for i in range(existing, count):
        try:
            r = requests.get("https://thispersondoesnotexist.com",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200 and len(r.content) > 5000:
                (ai_dir / f"ai_{i:04d}.jpg").write_bytes(r.content)
                if (i + 1) % 25 == 0:
                    log(f"    {i+1}/{count}")
                time.sleep(1)
        except:
            pass
    return ai_dir


def gen_lfw_pairs(lfw_dir, max_pairs):
    persons = {}
    for d in sorted(lfw_dir.iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg"))
            if imgs:
                persons[d.name] = [str(p) for p in imgs]

    matched = []
    for name, imgs in persons.items():
        if len(imgs) >= 2:
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    matched.append((imgs[i], imgs[j], name))

    mismatched = []
    names = list(persons.keys())
    random.seed(42)
    for _ in range(max_pairs * 3):
        p1, p2 = random.sample(names, 2)
        mismatched.append((
            random.choice(persons[p1]),
            random.choice(persons[p2]),
            f"{p1} vs {p2}"
        ))

    random.seed(42)
    if len(matched) > max_pairs:
        matched = random.sample(matched, max_pairs)
    if len(mismatched) > max_pairs:
        mismatched = random.sample(mismatched, max_pairs)

    return matched, mismatched


def parse_utk_label(filename):
    try:
        parts = Path(filename).stem.replace(".jpg.chip", "").split("_")
        if len(parts) >= 3:
            age = int(parts[0])
            gender = "MAN" if int(parts[1]) == 0 else "WOMAN"
            race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
            race = race_map.get(int(parts[2]), "Unknown")
            if 1 <= age <= 100:
                return {"age": age, "gender": gender, "race": race}
    except:
        pass
    return None


# === Benchmark Functions ===

def bench_compare(lfw_dir, max_pairs):
    log("=" * 65)
    log("  1. FACE COMPARE — LFW Dataset (Industry Standard)")
    log("     Used by: InsightFace, AdaFace, ArcFace, CosFace")
    log("=" * 65)

    matched, mismatched = gen_lfw_pairs(lfw_dir, max_pairs)
    total_tests = len(matched) + len(mismatched)
    log(f"  {len(matched)} same-person + {len(mismatched)} diff-person = {total_tests} pairs")

    tp, fn, tn, fp, errors = 0, 0, 0, 0, 0
    lats, sim_m, sim_mm = [], [], []
    details = []

    for i, (img1, img2, label) in enumerate(matched):
        files = {
            "sourceImage": ("s.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("t.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)
        if "error" in resp:
            errors += 1
            continue
        lats.append(ms)
        sim = resp.get("similarity", 0)
        sim_m.append(sim)
        ok = resp.get("matched", False)
        details.append({"type": "same", "label": label, "sim": sim, "matched": ok, "correct": ok})
        if ok: tp += 1
        else: fn += 1
        if (i + 1) % 50 == 0:
            log(f"    Same-person: {i+1}/{len(matched)}")

    for i, (img1, img2, label) in enumerate(mismatched):
        files = {
            "sourceImage": ("s.jpg", open(img1, "rb"), "image/jpeg"),
            "targetImage": ("t.jpg", open(img2, "rb"), "image/jpeg"),
        }
        resp, ms = call_api("/api/face/compare", files=files)
        if "error" in resp:
            errors += 1
            continue
        lats.append(ms)
        sim = resp.get("similarity", 0)
        sim_mm.append(sim)
        ok = not resp.get("matched", True)
        details.append({"type": "diff", "label": label, "sim": sim, "matched": not ok, "correct": ok})
        if ok: tn += 1
        else: fp += 1
        if (i + 1) % 50 == 0:
            log(f"    Diff-person: {i+1}/{len(mismatched)}")

    total = tp + fn + tn + fp
    acc = (tp + tn) / total * 100 if total else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # FAR/FRR at various thresholds
    all_scores_match = sim_m
    all_scores_nonmatch = sim_mm
    
    far_frr = []
    for thr in [40, 50, 60, 70, 80, 90]:
        far = sum(1 for s in all_scores_nonmatch if s >= thr) / len(all_scores_nonmatch) * 100 if all_scores_nonmatch else 0
        frr = sum(1 for s in all_scores_match if s < thr) / len(all_scores_match) * 100 if all_scores_match else 0
        far_frr.append({"threshold": thr, "FAR": round(far, 2), "FRR": round(frr, 2)})

    result = {
        "metrics": {
            "accuracy": round(acc, 2), "precision": round(prec, 2),
            "recall": round(rec, 2), "f1": round(f1, 2),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp, "errors": errors,
            "avg_sim_matched": round(np.mean(sim_m), 2) if sim_m else 0,
            "avg_sim_mismatched": round(np.mean(sim_mm), 2) if sim_mm else 0,
            "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
            "p95_latency_ms": round(np.percentile(lats, 95), 1) if lats else 0,
        },
        "far_frr_curve": far_frr,
        "details": details,
    }

    m = result["metrics"]
    log(f"\n  RESULTS: Accuracy={m['accuracy']}%  F1={m['f1']}%")
    log(f"  TP={tp} FN={fn} TN={tn} FP={fp} Err={errors}")
    log(f"  Avg sim matched={m['avg_sim_matched']}%  mismatched={m['avg_sim_mismatched']}%")
    log(f"  Latency: avg={m['avg_latency_ms']}ms  p95={m['p95_latency_ms']}ms")
    log(f"  FAR/FRR curve:")
    for x in far_frr:
        log(f"    threshold={x['threshold']}%  FAR={x['FAR']}%  FRR={x['FRR']}%")
    return result


def bench_analyze(utk_dir, max_tests):
    log("=" * 65)
    log("  2. ANALYZE (Age/Gender) — UTKFace Ground Truth Labels")
    log("     Used by: FairFace, DeepFace, InsightFace benchmarks")
    log("=" * 65)

    labeled = []
    for img in sorted(utk_dir.glob("*.*jpg*")):
        lbl = parse_utk_label(img.name)
        if lbl and 10 <= lbl["age"] <= 80:
            labeled.append((str(img), lbl))

    random.seed(42)
    sample = random.sample(labeled, min(max_tests, len(labeled)))
    log(f"  Testing {len(sample)} labeled photos")

    g_ok, g_fail = 0, 0
    age_errors = []
    lats = []
    errors = 0
    details = []
    gender_cm = {"MAN_as_MAN": 0, "MAN_as_WOMAN": 0, "WOMAN_as_WOMAN": 0, "WOMAN_as_MAN": 0}
    age_by_decade = defaultdict(list)  # track age error per decade

    for i, (img_path, gt) in enumerate(sample):
        files = {"image": ("f.jpg", open(img_path, "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/analyze", files=files)
        if "error" in resp:
            errors += 1
            continue
        lats.append(ms)

        pg = resp.get("gender", "UNKNOWN")
        pc = resp.get("genderConfidence", 0)
        pa = resp.get("ageRange", {})
        pa_mid = (pa.get("low", 0) + pa.get("high", 0)) / 2
        ae = abs(pa_mid - gt["age"])

        gok = (pg == gt["gender"])
        if gok: g_ok += 1
        else: g_fail += 1

        key = f"{gt['gender']}_as_{pg}"
        if key in gender_cm:
            gender_cm[key] += 1

        age_errors.append(ae)
        decade = (gt["age"] // 10) * 10
        age_by_decade[decade].append(ae)

        details.append({
            "file": Path(img_path).name,
            "gt_gender": gt["gender"], "pred_gender": pg, "gender_conf": round(pc, 1),
            "gender_ok": gok,
            "gt_age": gt["age"], "pred_age": f"{pa.get('low',0)}-{pa.get('high',0)}",
            "age_error": round(ae, 1),
            "race": gt.get("race", ""),
        })

        if (i + 1) % 100 == 0:
            gacc = g_ok / (g_ok + g_fail) * 100 if (g_ok + g_fail) else 0
            amae = np.mean(age_errors) if age_errors else 0
            log(f"    {i+1}/{len(sample)}  gender={gacc:.1f}%  age_MAE={amae:.1f}")

    total = g_ok + g_fail
    gacc = g_ok / total * 100 if total else 0
    amae = np.mean(age_errors) if age_errors else 0
    a5 = sum(1 for e in age_errors if e <= 5) / len(age_errors) * 100 if age_errors else 0
    a10 = sum(1 for e in age_errors if e <= 10) / len(age_errors) * 100 if age_errors else 0

    # Age accuracy by decade
    age_decade_mae = {}
    for dec, errs in sorted(age_by_decade.items()):
        age_decade_mae[f"{dec}-{dec+9}"] = round(np.mean(errs), 1)

    worst = sorted(details, key=lambda x: x.get("age_error", 0), reverse=True)[:10]

    result = {
        "metrics": {
            "gender_accuracy": round(gacc, 2),
            "gender_correct": g_ok, "gender_wrong": g_fail,
            "gender_confusion_matrix": gender_cm,
            "age_mae": round(amae, 2),
            "age_within_5yr": round(a5, 2),
            "age_within_10yr": round(a10, 2),
            "age_mae_by_decade": age_decade_mae,
            "errors": errors, "total": total,
            "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
        },
        "worst_age": worst,
        "details": details,
    }

    m = result["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Gender accuracy: {m['gender_accuracy']}% ({g_ok}/{total})")
    log(f"  Gender confusion: {gender_cm}")
    log(f"  Age MAE: {m['age_mae']}yr  within5={m['age_within_5yr']}%  within10={m['age_within_10yr']}%")
    log(f"  Age MAE by decade: {age_decade_mae}")
    log(f"  Errors: {errors}")
    for w in worst[:3]:
        log(f"  Worst: {w['file']} actual={w['gt_age']} pred={w['pred_age']} err={w['age_error']}")
    return result


def bench_liveness_deepfake(real_dir, ai_dir, max_real, max_ai):
    """Test liveness AND deepfake on real + AI faces."""
    log("=" * 65)
    log("  3+4. LIVENESS & DEEPFAKE — Real vs AI-Generated")
    log("       AI source: thispersondoesnotexist.com (StyleGAN)")
    log("=" * 65)

    # Collect images
    real_imgs = list(real_dir.rglob("*.jpg"))
    random.seed(42)
    real_sample = random.sample(real_imgs, min(max_real, len(real_imgs)))

    ai_imgs = sorted(ai_dir.glob("*.jpg"))[:max_ai]
    log(f"  Real: {len(real_sample)}  |  AI: {len(ai_imgs)}")

    results = {}

    # --- Liveness Real ---
    log(f"\n  [Liveness x Real] Testing {len(real_sample)} real faces...")
    live_ok, live_fail, errs = 0, 0, 0
    scores, lats = [], []
    for i, img in enumerate(real_sample):
        files = {"image": ("f.jpg", open(str(img), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/liveness", files=files)
        if "error" in resp:
            errs += 1; continue
        lats.append(ms)
        s = resp.get("liveScore", 0)
        scores.append(s)
        if resp.get("isLive", False): live_ok += 1
        else: live_fail += 1
        if (i+1) % 50 == 0: log(f"    {i+1}/{len(real_sample)}")

    t = live_ok + live_fail
    results["liveness_real"] = {
        "accuracy": round(live_ok/t*100, 2) if t else 0,
        "correct": live_ok, "wrong": live_fail, "errors": errs,
        "avg_score": round(np.mean(scores), 2) if scores else 0,
        "min_score": round(np.min(scores), 2) if scores else 0,
        "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
    }
    log(f"  >> Real as LIVE: {results['liveness_real']['accuracy']}%")

    # --- Liveness AI ---
    log(f"\n  [Liveness x AI] Testing {len(ai_imgs)} AI faces...")
    live_ok, live_fail, errs = 0, 0, 0
    scores, lats = [], []
    for i, img in enumerate(ai_imgs):
        files = {"image": ("f.jpg", open(str(img), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/liveness", files=files)
        if "error" in resp:
            errs += 1; continue
        lats.append(ms)
        s = resp.get("liveScore", 0)
        scores.append(s)
        if not resp.get("isLive", True): live_ok += 1  # correctly rejected
        else: live_fail += 1
        if (i+1) % 50 == 0: log(f"    {i+1}/{len(ai_imgs)}")

    t = live_ok + live_fail
    results["liveness_ai"] = {
        "ai_caught": round(live_ok/t*100, 2) if t else 0,
        "ai_passed": live_fail, "ai_blocked": live_ok, "errors": errs,
        "avg_score": round(np.mean(scores), 2) if scores else 0,
        "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
    }
    log(f"  >> AI caught by liveness: {results['liveness_ai']['ai_caught']}%")

    # --- Deepfake Real ---
    log(f"\n  [Deepfake x Real] Testing {len(real_sample)} real faces...")
    df_ok, df_fail, errs = 0, 0, 0
    lats = []
    risks = defaultdict(int)
    for i, img in enumerate(real_sample):
        files = {"image": ("f.jpg", open(str(img), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/deepfake", files=files)
        if "error" in resp:
            errs += 1; continue
        lats.append(ms)
        risks[resp.get("attackRiskLevel", "?")] += 1
        if not resp.get("isDeepfake", True): df_ok += 1
        else: df_fail += 1
        if (i+1) % 50 == 0: log(f"    {i+1}/{len(real_sample)}")

    t = df_ok + df_fail
    results["deepfake_real"] = {
        "accuracy": round(df_ok/t*100, 2) if t else 0,
        "correct": df_ok, "false_positive": df_fail, "errors": errs,
        "risk_dist": dict(risks),
        "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
    }
    log(f"  >> Real NOT flagged: {results['deepfake_real']['accuracy']}%")

    # --- Deepfake AI ---
    log(f"\n  [Deepfake x AI] Testing {len(ai_imgs)} AI faces...")
    df_ok, df_fail, errs = 0, 0, 0
    lats = []
    risks = defaultdict(int)
    wrong_list = []
    for i, img in enumerate(ai_imgs):
        files = {"image": ("f.jpg", open(str(img), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/deepfake", files=files)
        if "error" in resp:
            errs += 1; continue
        lats.append(ms)
        risk = resp.get("attackRiskLevel", "?")
        risks[risk] += 1
        if resp.get("isDeepfake", False): df_ok += 1
        else:
            df_fail += 1
            wrong_list.append({"file": img.name, "risk": risk})
        if (i+1) % 50 == 0: log(f"    {i+1}/{len(ai_imgs)}")

    t = df_ok + df_fail
    results["deepfake_ai"] = {
        "ai_caught": round(df_ok/t*100, 2) if t else 0,
        "ai_detected": df_ok, "ai_missed": df_fail, "errors": errs,
        "risk_dist": dict(risks),
        "missed_samples": wrong_list[:10],
        "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
    }
    log(f"  >> AI caught by deepfake: {results['deepfake_ai']['ai_caught']}%")

    # Combined security
    ai_pass_live = 100 - results["liveness_ai"]["ai_caught"]
    ai_pass_df = 100 - results["deepfake_ai"]["ai_caught"]
    combined_bypass = (ai_pass_live / 100) * (ai_pass_df / 100) * 100
    results["combined"] = {
        "ai_pass_liveness_pct": round(ai_pass_live, 2),
        "ai_pass_deepfake_pct": round(ai_pass_df, 2),
        "combined_bypass_pct": round(combined_bypass, 2),
        "effective_block_rate": round(100 - combined_bypass, 2),
    }

    return results


def bench_embed(lfw_dir, max_tests):
    log("=" * 65)
    log("  5. EMBED — Embedding Quality + Consistency")
    log("=" * 65)

    # Get persons with 2+ photos for consistency check
    persons = {}
    for d in sorted(lfw_dir.iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg"))
            if len(imgs) >= 2:
                persons[d.name] = [str(p) for p in imgs]

    # Test 1: Extract embeddings
    all_imgs = list(lfw_dir.rglob("*.jpg"))[:max_tests]
    log(f"  Extracting {len(all_imgs)} embeddings...")

    valid, bad_dim, bad_norm, errs = 0, 0, 0, 0
    lats = []
    for i, img in enumerate(all_imgs):
        files = {"image": ("f.jpg", open(str(img), "rb"), "image/jpeg")}
        resp, ms = call_api("/api/face/embed", files=files)
        if "error" in resp:
            errs += 1; continue
        lats.append(ms)
        emb = resp.get("embedding", [])
        if len(emb) != 512: bad_dim += 1
        elif not (0.9 < np.linalg.norm(emb) < 1.1): bad_norm += 1
        else: valid += 1
        if (i+1) % 50 == 0: log(f"    {i+1}/{len(all_imgs)}")

    t = valid + bad_dim + bad_norm

    # Test 2: Consistency — same person should have similar embeddings
    log(f"  Consistency check: same person = similar embeddings...")
    consistency_tests = 0
    consistency_pass = 0
    person_list = list(persons.items())[:20]  # check 20 persons

    for name, imgs in person_list:
        embs = []
        for img in imgs[:3]:  # max 3 per person
            files = {"image": ("f.jpg", open(img, "rb"), "image/jpeg")}
            resp, ms = call_api("/api/face/embed", files=files)
            if "error" not in resp:
                emb = resp.get("embedding", [])
                if len(emb) == 512:
                    embs.append(np.array(emb))

        if len(embs) >= 2:
            for j in range(len(embs)):
                for k in range(j+1, len(embs)):
                    sim = np.dot(embs[j], embs[k])
                    consistency_tests += 1
                    if sim > 0.5:  # same person should have >0.5 cosine sim
                        consistency_pass += 1

    cons_rate = consistency_pass / consistency_tests * 100 if consistency_tests else 0

    result = {
        "metrics": {
            "valid_pct": round(valid/t*100, 2) if t else 0,
            "valid": valid, "bad_dim": bad_dim, "bad_norm": bad_norm, "errors": errs,
            "consistency_rate": round(cons_rate, 2),
            "consistency_tests": consistency_tests,
            "avg_latency_ms": round(np.mean(lats), 1) if lats else 0,
            "p95_latency_ms": round(np.percentile(lats, 95), 1) if lats else 0,
        }
    }

    m = result["metrics"]
    log(f"\n  RESULTS:")
    log(f"  Valid embeddings: {m['valid_pct']}% ({valid}/{t})")
    log(f"  Consistency (same person sim>0.5): {m['consistency_rate']}% ({consistency_pass}/{consistency_tests})")
    log(f"  Latency: avg={m['avg_latency_ms']}ms  p95={m['p95_latency_ms']}ms")
    return result


# === Report ===

def gen_report(results, scale_name, total_evals):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON summary (no details)
    summary = {}
    for k, v in results.items():
        if isinstance(v, dict):
            summary[k] = {kk: vv for kk, vv in v.items()
                          if kk not in ["details", "wrong_samples", "worst_age"]}
        else:
            summary[k] = v

    json_path = REPORT_DIR / f"benchmark_1k_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Full details
    det_path = REPORT_DIR / f"benchmark_1k_{ts}_details.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Text report
    txt_path = REPORT_DIR / f"benchmark_1k_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  VISON AI SERVICE - COMPREHENSIVE INDUSTRY BENCHMARK\n")
        f.write(f"  Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"  Scale: {scale_name}  |  Total evaluations: {total_evals}\n")
        f.write(f"  Datasets: LFW, UTKFace, thispersondoesnotexist.com\n")
        f.write("=" * 70 + "\n\n")

        # Compare
        if "compare" in results:
            m = results["compare"]["metrics"]
            f.write("--- 1. FACE COMPARE (LFW) ---\n")
            f.write(f"  Accuracy: {m['accuracy']}%  F1: {m['f1']}%\n")
            f.write(f"  TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']}\n")
            f.write(f"  Avg sim matched={m['avg_sim_matched']}%  mismatched={m['avg_sim_mismatched']}%\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms\n")
            if "far_frr_curve" in results["compare"]:
                f.write("  FAR/FRR:\n")
                for x in results["compare"]["far_frr_curve"]:
                    f.write(f"    thr={x['threshold']}%  FAR={x['FAR']}%  FRR={x['FRR']}%\n")
            g = "A" if m['accuracy'] >= 99 else "B" if m['accuracy'] >= 95 else "C"
            f.write(f"  GRADE: {g}  (InsightFace R100: 99.83% LFW)\n\n")

        # Analyze
        if "analyze" in results:
            m = results["analyze"]["metrics"]
            f.write("--- 2. ANALYZE AGE/GENDER (UTKFace) ---\n")
            f.write(f"  Gender: {m['gender_accuracy']}%\n")
            f.write(f"  Confusion: {m['gender_confusion_matrix']}\n")
            f.write(f"  Age MAE: {m['age_mae']}yr  within5={m['age_within_5yr']}%  within10={m['age_within_10yr']}%\n")
            f.write(f"  Age by decade: {m['age_mae_by_decade']}\n")
            gg = "A" if m['gender_accuracy'] >= 95 else "B" if m['gender_accuracy'] >= 85 else "C"
            ag = "A" if m['age_mae'] <= 5 else "B" if m['age_mae'] <= 8 else "C"
            f.write(f"  GRADE Gender: {gg}  Age: {ag}\n\n")

        # Liveness + Deepfake
        if "liveness_deepfake" in results:
            ld = results["liveness_deepfake"]
            f.write("--- 3. LIVENESS ---\n")
            f.write(f"  Real as LIVE:   {ld['liveness_real']['accuracy']}%\n")
            f.write(f"  AI caught:      {ld['liveness_ai']['ai_caught']}%\n\n")
            f.write("--- 4. DEEPFAKE ---\n")
            f.write(f"  Real NOT flagged: {ld['deepfake_real']['accuracy']}%\n")
            f.write(f"  AI caught:        {ld['deepfake_ai']['ai_caught']}%\n")
            f.write(f"  Real risk dist:   {ld['deepfake_real']['risk_dist']}\n")
            f.write(f"  AI risk dist:     {ld['deepfake_ai']['risk_dist']}\n\n")
            f.write("--- COMBINED SECURITY ---\n")
            c = ld["combined"]
            f.write(f"  AI bypass liveness: {c['ai_pass_liveness_pct']}%\n")
            f.write(f"  AI bypass deepfake: {c['ai_pass_deepfake_pct']}%\n")
            f.write(f"  AI bypass BOTH:     {c['combined_bypass_pct']}%\n")
            f.write(f"  BLOCK RATE:         {c['effective_block_rate']}%\n\n")

        # Embed
        if "embed" in results:
            m = results["embed"]["metrics"]
            f.write("--- 5. EMBED ---\n")
            f.write(f"  Valid: {m['valid_pct']}%  Consistency: {m['consistency_rate']}%\n")
            f.write(f"  Latency: avg={m['avg_latency_ms']}ms\n\n")

        # Industry comparison
        f.write("=" * 70 + "\n")
        f.write("  COMPARISON VS INDUSTRY BENCHMARKS\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'Metric':<30s} {'Vison AI':>10s} {'ArcFace':>10s} {'AWS':>10s}\n")
        f.write("  " + "-" * 60 + "\n")
        
        if "compare" in results:
            v = results["compare"]["metrics"]["accuracy"]
            f.write(f"  {'LFW Compare Accuracy':<30s} {v:>9.2f}% {'99.83':>9s}% {'~99':>9s}%\n")
        if "analyze" in results:
            v = results["analyze"]["metrics"]["gender_accuracy"]
            f.write(f"  {'Gender Accuracy':<30s} {v:>9.2f}% {'~97':>9s}% {'~99':>9s}%\n")
            v = results["analyze"]["metrics"]["age_mae"]
            f.write(f"  {'Age MAE (years)':<30s} {v:>9.1f}  {'~3':>9s}  {'~3':>9s}\n")
        if "liveness_deepfake" in results:
            ld = results["liveness_deepfake"]
            v = ld["liveness_real"]["accuracy"]
            f.write(f"  {'Liveness Real Accept':<30s} {v:>9.2f}% {'N/A':>9s}  {'~99':>9s}%\n")
            v = ld["deepfake_ai"]["ai_caught"]
            f.write(f"  {'Deepfake AI Catch':<30s} {v:>9.2f}% {'N/A':>9s}  {'~95':>9s}%\n")
            v = ld["combined"]["effective_block_rate"]
            f.write(f"  {'Combined Block Rate':<30s} {v:>9.2f}%\n")

        f.write("=" * 70 + "\n")

    return json_path, txt_path, det_path


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Vison AI 1K Benchmark")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    args = parser.parse_args()

    cfg = SCALES[args.scale]

    log("=" * 65)
    log("  VISON AI — COMPREHENSIVE INDUSTRY BENCHMARK")
    log(f"  Scale: {args.scale}")
    log(f"  Compare: {cfg['compare']*2} pairs  Analyze: {cfg['analyze']}")
    log(f"  Liveness: {cfg['liveness']}+{cfg['ai_faces']}  Deepfake: {cfg['deepfake']}+{cfg['ai_faces']}")
    log(f"  Embed: {cfg['embed']}")
    total = cfg['compare']*2 + cfg['analyze'] + (cfg['liveness']+cfg['ai_faces'])*2 + cfg['embed']
    log(f"  Total evaluations: ~{total}")
    log("=" * 65)

    try:
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=5)
        assert r.status_code == 200
    except:
        log("Service NOT running!", "ERROR")
        sys.exit(1)
    log("Service OK\n")

    # Prep datasets
    log("--- Preparing Datasets ---")
    lfw_dir = prep_lfw()
    utk_dir = prep_utkface()
    ai_dir = prep_ai_faces(cfg["ai_faces"])

    if lfw_dir:
        log(f"  LFW: {sum(1 for _ in lfw_dir.rglob('*.jpg'))} images")
    if utk_dir:
        log(f"  UTKFace: {len(list(utk_dir.glob('*.*jpg*')))} images")
    log(f"  AI faces: {len(list(ai_dir.glob('*.jpg')))} images")
    print()

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scale": args.scale,
        "config": cfg,
    }

    # Run benchmarks
    if lfw_dir:
        results["compare"] = bench_compare(lfw_dir, cfg["compare"])
        print()

    if utk_dir:
        results["analyze"] = bench_analyze(utk_dir, cfg["analyze"])
        print()

    real_dir = lfw_dir or utk_dir
    if real_dir and ai_dir:
        results["liveness_deepfake"] = bench_liveness_deepfake(
            real_dir, ai_dir, cfg["liveness"], cfg["ai_faces"]
        )
        print()

    if lfw_dir:
        results["embed"] = bench_embed(lfw_dir, cfg["embed"])
        print()

    # Report
    log("--- Generating Report ---")
    json_p, txt_p, det_p = gen_report(results, args.scale, total)
    log(f"  Summary: {txt_p}")
    log(f"  JSON: {json_p}")

    # Final
    print()
    print("=" * 65)
    print("  FINAL SCORECARD")
    print("=" * 65)
    if "compare" in results:
        m = results["compare"]["metrics"]
        print(f"  COMPARE:      {m['accuracy']}% accuracy  F1={m['f1']}%")
    if "analyze" in results:
        m = results["analyze"]["metrics"]
        print(f"  GENDER:       {m['gender_accuracy']}% accuracy")
        print(f"  AGE:          MAE={m['age_mae']}yr  within5yr={m['age_within_5yr']}%")
    if "liveness_deepfake" in results:
        ld = results["liveness_deepfake"]
        print(f"  LIVENESS:     real={ld['liveness_real']['accuracy']}%  ai_caught={ld['liveness_ai']['ai_caught']}%")
        print(f"  DEEPFAKE:     real={ld['deepfake_real']['accuracy']}%  ai_caught={ld['deepfake_ai']['ai_caught']}%")
        print(f"  BLOCK RATE:   {ld['combined']['effective_block_rate']}%")
    if "embed" in results:
        m = results["embed"]["metrics"]
        print(f"  EMBED:        {m['valid_pct']}% valid  consistency={m['consistency_rate']}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
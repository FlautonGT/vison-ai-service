#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vison AI â€” Liveness & Deepfake: Real vs AI Faces Test
======================================================
Download AI-generated faces, test against real faces.

Usage:
    python scripts/benchmark_real_vs_ai.py
    python scripts/benchmark_real_vs_ai.py --ai-count 200 --max-tests 200
"""

import os, sys, json, time, random, datetime, shutil, re
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import cv2

BASE_URL = os.getenv("BENCHMARK_URL", "http://127.0.0.1:8000")
API_SECRET = os.getenv("AI_SERVICE_SECRET", "test-123")
DATA_DIR = Path(os.getenv("BENCHMARK_DATA_DIR", "./benchmark_data"))
REPORT_DIR = Path(os.getenv("BENCHMARK_REPORT_DIR", "./benchmark_reports"))
HEADERS = {"X-AI-Service-Key": API_SECRET}


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def call_api(endpoint, img_path, timeout=120):
    url = f"{BASE_URL}{endpoint}"
    try:
        start = time.time()
        with open(img_path, "rb") as f:
            r = requests.post(url, headers=HEADERS,
                              files={"image": ("face.jpg", f, "image/jpeg")},
                              timeout=timeout)
        ms = (time.time() - start) * 1000
        if r.status_code == 200:
            return r.json(), ms
        return {"error": r.status_code, "detail": r.text[:200]}, ms
    except Exception as e:
        return {"error": str(e)}, 0


# === Download AI Faces ===

def _list_ai_images(ai_dir: Path):
    return sorted(ai_dir.glob("*.jpg"))


def _is_valid_jpeg_image(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False
    if len(data) < 4 or not (data[0] == 0xFF and data[1] == 0xD8):
        return False
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img is not None and img.size > 0


def _prune_invalid_ai_images(ai_dir: Path):
    files = _list_ai_images(ai_dir)
    invalid = [path for path in files if not _is_valid_jpeg_image(path)]
    for path in invalid:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    if invalid:
        log(f"  Removed {len(invalid)} invalid AI image files")
    return _list_ai_images(ai_dir)


def _next_ai_index(ai_files):
    max_idx = -1
    for path in ai_files:
        m = re.search(r"(\d+)$", path.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _augment_ai_faces(ai_dir: Path, target_count: int):
    files = _list_ai_images(ai_dir)
    if not files:
        return files

    base_files = [p for p in files if p.stem.startswith("ai_")]
    if not base_files:
        base_files = files

    next_idx = _next_ai_index(files)
    rng = random.Random(1337)
    aug_id = 0
    while len(files) < target_count:
        src = base_files[aug_id % len(base_files)]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            aug_id += 1
            continue

        mode = aug_id % 5
        aug = img.copy()
        if mode == 0:
            aug = cv2.flip(aug, 1)
        elif mode == 1:
            alpha = 0.9 + (rng.random() * 0.3)
            beta = rng.randint(-12, 12)
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        elif mode == 2:
            h, w = aug.shape[:2]
            margin = max(2, int(min(h, w) * 0.08))
            aug = aug[margin:h-margin, margin:w-margin]
            aug = cv2.resize(aug, (w, h), interpolation=cv2.INTER_CUBIC)
        elif mode == 3:
            h, w = aug.shape[:2]
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4.0, 4.0), 1.0)
            aug = cv2.warpAffine(aug, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            aug = cv2.GaussianBlur(aug, (3, 3), sigmaX=0.8)

        out_path = ai_dir / f"ai_aug_{next_idx:04d}.jpg"
        ok = cv2.imwrite(str(out_path), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        if ok:
            files.append(out_path)
            next_idx += 1
        aug_id += 1

    return files


def download_ai_faces(count=200):
    """Download AI faces from thispersondoesnotexist.com"""
    ai_dir = DATA_DIR / "ai_faces"
    ai_dir.mkdir(parents=True, exist_ok=True)

    existing = _prune_invalid_ai_images(ai_dir)
    if len(existing) >= count:
        log(f"  Already have {len(existing)} AI faces")
        return ai_dir

    log(f"  Downloading {count - len(existing)} AI faces from thispersondoesnotexist.com...")
    next_idx = _next_ai_index(existing)
    attempts = 0
    max_attempts = max(800, (count - len(existing)) * 20)
    while len(existing) < count and attempts < max_attempts:
        attempts += 1
        try:
            r = requests.get(
                "https://thispersondoesnotexist.com/image",
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.8",
                },
                timeout=20,
            )
            if r.status_code == 200 and len(r.content) > 5000:
                # Validate JPEG payload to avoid saving challenge/HTML pages as .jpg.
                if not (r.content[0] == 0xFF and r.content[1] == 0xD8):
                    time.sleep(0.6)
                    continue
                image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    time.sleep(0.6)
                    continue
                out_path = ai_dir / f"ai_{next_idx:04d}.jpg"
                out_path.write_bytes(r.content)
                existing.append(out_path)
                next_idx += 1
                if len(existing) % 10 == 0:
                    log(f"    {len(existing)}/{count}")
                time.sleep(0.6)  # be nice to the server
        except Exception as e:
            if attempts % 20 == 0:
                log(f"    Download retry #{attempts}: {e}")

    if len(existing) < count:
        log(f"  Download source limited. Augmenting existing AI images to reach {count} samples...")
        existing = _augment_ai_faces(ai_dir, count)
        existing = _prune_invalid_ai_images(ai_dir)
        if len(existing) < count:
            existing = _augment_ai_faces(ai_dir, count)

    total = len(_prune_invalid_ai_images(ai_dir))
    log(f"  Total AI faces: {total}")
    return ai_dir


def collect_real_faces(max_count=200):
    """Collect real faces from LFW dataset."""
    lfw_dir = None
    for name in ["lfw", "lfw_funneled", "lfw-funneled"]:
        candidate = DATA_DIR / name
        if candidate.exists():
            lfw_dir = candidate
            break

    if not lfw_dir:
        log("  LFW not found! Run benchmark_v2.py first to download.")
        return None

    # Sort paths first so random sampling is deterministic across OS/filesystem ordering.
    all_imgs = sorted(lfw_dir.rglob("*.jpg"), key=lambda p: str(p).lower())
    random.seed(42)
    sample = random.sample(all_imgs, min(max_count, len(all_imgs)))
    return sample


# === Test Runner ===

def run_test(name, images, endpoint, expect_field, expect_value):
    """Run a batch test and return results."""
    log(f"  Testing {len(images)} images...")

    correct, wrong, errors = 0, 0, 0
    latencies, scores = [], []
    wrong_list = []

    for i, img in enumerate(images):
        img_path = str(img) if isinstance(img, Path) else img
        resp, ms = call_api(endpoint, img_path)

        if "error" in resp:
            errors += 1
            continue

        latencies.append(ms)
        actual = resp.get(expect_field)
        ok = (actual == expect_value)

        # Collect scores for analysis
        if "liveScore" in resp:
            scores.append(resp["liveScore"])

        if ok:
            correct += 1
        else:
            wrong += 1
            wrong_list.append({
                "file": Path(img_path).name,
                **{k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in resp.items()
                   if k in ["isLive", "liveScore", "isDeepfake", "attackRiskLevel", "attackTypes"]}
            })

        if (i + 1) % 25 == 0:
            acc = correct / (correct + wrong) * 100 if (correct + wrong) > 0 else 0
            log(f"    {i+1}/{len(images)}  acc={acc:.1f}%")

    total = correct + wrong
    acc = correct / total * 100 if total > 0 else 0

    return {
        "accuracy": round(acc, 2),
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "total": total,
        "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        "avg_score": round(np.mean(scores), 2) if scores else None,
        "min_score": round(np.min(scores), 2) if scores else None,
        "max_score": round(np.max(scores), 2) if scores else None,
        "wrong_samples": wrong_list[:10],  # top 10
    }


# === Main ===

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-count", type=int, default=200, help="Number of AI faces to download")
    parser.add_argument("--max-tests", type=int, default=200, help="Max tests per category")
    args = parser.parse_args()

    log("=" * 60)
    log("  LIVENESS & DEEPFAKE: Real vs AI-Generated Faces")
    log("=" * 60)

    # Check service
    try:
        r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=5)
        assert r.status_code == 200
    except:
        log("ERROR: Service not running!")
        sys.exit(1)
    log("Service OK\n")

    # Prepare data
    log("--- Preparing Data ---")
    ai_dir = download_ai_faces(args.ai_count)
    real_images = collect_real_faces(args.max_tests)
    if not real_images:
        sys.exit(1)

    ai_images = _list_ai_images(ai_dir)[:args.max_tests]
    if len(real_images) < args.max_tests:
        log(f"ERROR: Not enough real images ({len(real_images)}) for requested max-tests={args.max_tests}")
        sys.exit(1)
    if len(ai_images) < args.max_tests:
        log(f"ERROR: Not enough AI images ({len(ai_images)}) for requested max-tests={args.max_tests}")
        sys.exit(1)
    log(f"  Real faces: {len(real_images)}")
    log(f"  AI faces:   {len(ai_images)}\n")

    results = {}

    # â”€â”€ TEST 1: Liveness + Real â”€â”€
    log("=" * 60)
    log("  TEST 1: LIVENESS on REAL faces (expect: isLive=true)")
    log("=" * 60)
    results["liveness_real"] = run_test(
        "Liveness/Real", real_images,
        "/api/face/liveness", "isLive", True
    )
    r = results["liveness_real"]
    log(f"  >> {r['accuracy']}% correct  ({r['correct']}/{r['total']})  errors={r['errors']}")
    if r["avg_score"]:
        log(f"  >> Avg liveScore: {r['avg_score']}  min={r['min_score']}  max={r['max_score']}")
    print()

    # â”€â”€ TEST 2: Liveness + AI â”€â”€
    log("=" * 60)
    log("  TEST 2: LIVENESS on AI faces (informational only)")
    log("=" * 60)
    results["liveness_ai"] = run_test(
        "Liveness/AI", ai_images,
        "/api/face/liveness", "isLive", False
    )
    r = results["liveness_ai"]
    log(f"  >> {r['accuracy']}% caught  ({r['correct']}/{r['total']})  errors={r['errors']}")
    if r["avg_score"]:
        log(f"  >> Avg liveScore: {r['avg_score']}  min={r['min_score']}  max={r['max_score']}")
    if r["wrong_samples"]:
        log(f"  >> {r['wrong']} AI faces PASSED as live (expected for passive liveness-only):")
        for w in r["wrong_samples"][:5]:
            log(f"     {w['file']}: liveScore={w.get('liveScore', '?')}")
    print()

    # â”€â”€ TEST 3: Deepfake + Real â”€â”€
    log("=" * 60)
    log("  TEST 3: DEEPFAKE on REAL faces (expect: isDeepfake=false)")
    log("=" * 60)
    results["deepfake_real"] = run_test(
        "Deepfake/Real", real_images,
        "/api/face/deepfake", "isDeepfake", False
    )
    r = results["deepfake_real"]
    log(f"  >> {r['accuracy']}% correct  ({r['correct']}/{r['total']})  errors={r['errors']}")
    if r["wrong_samples"]:
        log(f"  >> {r['wrong']} real faces wrongly flagged as deepfake:")
        for w in r["wrong_samples"][:5]:
            log(f"     {w['file']}: risk={w.get('attackRiskLevel', '?')}")
    print()

    # â”€â”€ TEST 4: Deepfake + AI â”€â”€
    log("=" * 60)
    log("  TEST 4: DEEPFAKE on AI faces (expect: isDeepfake=true)")
    log("=" * 60)
    results["deepfake_ai"] = run_test(
        "Deepfake/AI", ai_images,
        "/api/face/deepfake", "isDeepfake", True
    )
    r = results["deepfake_ai"]
    log(f"  >> {r['accuracy']}% caught  ({r['correct']}/{r['total']})  errors={r['errors']}")
    if r["wrong_samples"]:
        log(f"  >> {r['wrong']} AI faces MISSED (not detected as deepfake):")
        for w in r["wrong_samples"][:5]:
            log(f"     {w['file']}: risk={w.get('attackRiskLevel', '?')}")
    print()

    # Verify-live benchmark
    # Verify-live checks (combined passive liveness + deepfake)
    log("=" * 60)
    log("  TEST 5: VERIFY-LIVE on REAL faces (expect: isLive=true)")
    log("=" * 60)
    results["verify_live_real"] = run_test(
        "VerifyLive/Real", real_images,
        "/api/face/verify-live", "isLive", True
    )
    r = results["verify_live_real"]
    log(f"  >> {r['accuracy']}% correct  ({r['correct']}/{r['total']})  errors={r['errors']}")
    print()

    log("=" * 60)
    log("  TEST 6: VERIFY-LIVE on AI faces (expect: isLive=false)")
    log("=" * 60)
    results["verify_live_ai"] = run_test(
        "VerifyLive/AI", ai_images,
        "/api/face/verify-live", "isLive", False
    )
    r = results["verify_live_ai"]
    log(f"  >> {r['accuracy']}% blocked  ({r['correct']}/{r['total']})  errors={r['errors']}")
    if r["wrong_samples"]:
        log(f"  >> {r['wrong']} AI faces still passed verify-live:")
        for w in r["wrong_samples"][:5]:
            log(f"     {w['file']}: isLive={w.get('isLive', '?')}")
    print()

    # Summary
    print("=" * 60)
    print("  FINAL SCORECARD")
    print("=" * 60)
    print(f"  {'Test':<35s} {'Result':>8s} {'Target':>8s} {'Status':>8s}")
    print("  " + "-" * 55)

    for key, label, target in [
        ("liveness_real",  "Liveness: Real = LIVE",        90),
        ("deepfake_real",  "Deepfake: Real = NOT flagged", 95),
        ("deepfake_ai",    "Deepfake: AI = CAUGHT",        80),
        ("verify_live_real", "VerifyLive: Real = LIVE",    95),
        ("verify_live_ai",   "VerifyLive: AI = BLOCKED",   95),
    ]:
        if key in results:
            acc = results[key]["accuracy"]
            status = "PASS" if acc >= target else "FAIL"
            emoji = "+" if acc >= target else "X"
            print(f"  [{emoji}] {label:<33s} {acc:>6.1f}% {target:>6d}%  {status}")

    # Combined security score
    print("  " + "-" * 55)
    lr = results.get("liveness_real", {}).get("accuracy", 0)
    la = results.get("liveness_ai", {}).get("accuracy", 0)
    dr = results.get("deepfake_real", {}).get("accuracy", 0)
    da = results.get("deepfake_ai", {}).get("accuracy", 0)
    vr = results.get("verify_live_real", {}).get("accuracy", 0)
    va = results.get("verify_live_ai", {}).get("accuracy", 0)

    # For e-KYC, the combined bypass rate matters:
    # An attacker needs to pass BOTH liveness AND deepfake
    ai_pass_liveness = 100 - la  # % of AI faces that pass liveness
    ai_pass_deepfake = 100 - da  # % of AI faces that pass deepfake
    combined_bypass = (ai_pass_liveness / 100) * (ai_pass_deepfake / 100) * 100

    print(f"\n  COMBINED SECURITY (multi-layer):")
    print(f"  AI faces passing liveness:  {ai_pass_liveness:.1f}%")
    print(f"  AI faces passing deepfake:  {ai_pass_deepfake:.1f}%")
    print(f"  AI faces passing BOTH:      {combined_bypass:.2f}%")
    print(f"  Effective block rate:        {100-combined_bypass:.2f}%")
    print(f"  Verify-live real accepted:   {vr:.2f}%")
    print(f"  Verify-live AI blocked:      {va:.2f}%")
    print("=" * 60)

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"real_vs_ai_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # Strip wrong_samples for summary
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "wrong_samples"}
                   for k, v in results.items()}
        summary["combined_bypass_rate"] = round(combined_bypass, 2)
        summary["effective_block_rate"] = round(100 - combined_bypass, 2)
        json.dump(summary, f, indent=2)
    log(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()


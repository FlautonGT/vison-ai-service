"""
Dataset Preparation untuk Vison AI Deepfake Training
=====================================================
Mengumpulkan dan rename foto dari berbagai sumber ke format:
  benchmark_data/id/real/  → foto wajah Asia real
  benchmark_data/id/ai/    → foto AI generated/fake

Format file: 00001_AGE_GENDER.jpg
  AGE    = umur (0 jika tidak diketahui)
  GENDER = 1 (pria), 2 (wanita), 0 (tidak diketahui)

Cara pakai:
  python prepare_dataset.py --step all
  python prepare_dataset.py --step real   # hanya kumpul foto real
  python prepare_dataset.py --step fake   # hanya kumpul foto fake
  python prepare_dataset.py --step rename-real  # rename foto yang sudah ada di real/
"""

import os
import re
import csv
import shutil
import random
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
REAL_DIR      = BASE_DIR / "benchmark_data" / "id" / "real"
AI_DIR        = BASE_DIR / "benchmark_data" / "id" / "ai"
DL_DIR        = BASE_DIR / "benchmark_data" / "_downloads"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UTKFACE_PAT   = re.compile(r"^(\d+)_(\d)_(\d)_")

# ─────────────────────────────────────────────
def get_next_index(directory: Path) -> int:
    existing = []
    for f in directory.glob("*"):
        if f.suffix.lower() in SUPPORTED_EXT or f.name.lower().endswith(".jpg.chip.jpg"):
            try:
                existing.append(int(f.stem.split("_")[0]))
            except (ValueError, IndexError):
                pass
    return max(existing, default=0) + 1

def is_image(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".jpg.chip.jpg") or p.suffix.lower() in SUPPORTED_EXT

def safe_copy(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"\n  [WARN] copy gagal {src.name}: {e}")
        return False

def print_progress(current, total, prefix=""):
    pct = current / total * 100 if total > 0 else 0
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"\r{prefix} [{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)

def copy_batch(photos, dest_dir, label="Copy", start_idx=None):
    """Copy list of (path, age, gender) ke dest_dir dengan rename."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if start_idx is None:
        start_idx = get_next_index(dest_dir)
    copied = 0
    for i, (photo, age, gender) in enumerate(photos):
        ext = ".jpg" if photo.name.lower().endswith(".chip.jpg") else photo.suffix.lower()
        if ext not in SUPPORTED_EXT:
            ext = ".jpg"
        new_name = f"{start_idx + i:05d}_{age}_{gender}{ext}"
        if safe_copy(photo, dest_dir / new_name):
            copied += 1
        print_progress(i + 1, len(photos), label)
    print(f"\n  ✅ {copied} foto disalin ke {dest_dir.name}/")
    return copied

# ─────────────────────────────────────────────
# RENAME foto real yang sudah ada
# ─────────────────────────────────────────────
def rename_existing_real():
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    renamed_pat = re.compile(r"^\d{5}_\d+_\d+\.", re.IGNORECASE)
    to_rename, already_ok = [], []
    for f in sorted(REAL_DIR.glob("*")):
        if not is_image(f):
            continue
        if renamed_pat.match(f.name):
            already_ok.append(f)
        else:
            to_rename.append(f)

    print(f"\n📁 {REAL_DIR}")
    print(f"   Sudah benar : {len(already_ok)}")
    print(f"   Perlu rename: {len(to_rename)}")
    if not to_rename:
        print("   ✅ Semua sudah format benar")
        return

    start_idx = get_next_index(REAL_DIR)
    for i, photo in enumerate(to_rename):
        m = UTKFACE_PAT.match(photo.name)
        if m:
            age    = min(max(int(m.group(1)), 0), 100)
            gender = 1 if m.group(2) == "0" else 2
        else:
            age, gender = 0, 0
        ext = ".jpg" if photo.name.lower().endswith(".chip.jpg") else photo.suffix.lower()
        new_path = REAL_DIR / f"{start_idx + i:05d}_{age}_{gender}{ext}"
        os.rename(photo, new_path)
        print_progress(i + 1, len(to_rename), "Rename")
    print(f"\n  ✅ {len(to_rename)} foto di-rename")

# ─────────────────────────────────────────────
# SOURCE 1: UTKFace (sudah ada + utk2)
# ─────────────────────────────────────────────
def collect_utkface(target=5000):
    print("\n[UTKFace] Scanning...")
    photos = []
    for base in [DL_DIR, DL_DIR / "utk2"]:
        for subdir in [base / "UTKFace", base / "crop_part1",
                       base / "utkface-aligned-labeled" / "images", base]:
            if not subdir.is_dir():
                continue
            for f in subdir.iterdir():
                if f.is_file() and is_image(f):
                    m = UTKFACE_PAT.match(f.name)
                    if m and m.group(3) == "2":  # Asian only
                        photos.append((f, min(max(int(m.group(1)),0),100),
                                       1 if m.group(2)=="0" else 2))

    # Deduplicate
    seen, unique = set(), []
    for p, a, g in photos:
        if p.name not in seen:
            seen.add(p.name)
            unique.append((p, a, g))

    print(f"  Ditemukan {len(unique)} foto Asian dari UTKFace")
    random.shuffle(unique)
    return unique[:target]

# ─────────────────────────────────────────────
# SOURCE 2: Diverse Asian Facial Ages
# ─────────────────────────────────────────────
AGE_FOLDER_MAP = {
    "10 tuổi 15": (12, 0), "15 tuổi 20": (17, 0), "20 tuổi 25": (22, 0),
    "25 tuổi 30": (27, 0), "30 đến 40 tuổi": (35, 0),
    "40 đến 50 tuổi": (45, 0), "50 đến 60 tuổi": (55, 0),
}

def collect_diverse_asian(target=5000):
    base = DL_DIR / "asian"
    # Coba Data_all dulu, fallback ke Data
    data_dir = None
    for candidate in [base / "Data_all", base / "Data"]:
        if candidate.exists() and candidate.is_dir():
            data_dir = candidate
            break
    if data_dir is None:
        print("  [SKIP] Diverse Asian tidak ditemukan")
        return []

    print("\n[Diverse Asian] Scanning...")
    photos = []
    # Pakai os.scandir untuk handle path encoding Vietnamese di Windows
    import os
    try:
        with os.scandir(str(data_dir)) as it:
            subdirs = [(e.name, Path(e.path)) for e in it if e.is_dir()]
    except Exception as e:
        print(f"  [WARN] scandir gagal: {e}")
        subdirs = []

    for folder_name, folder_path in subdirs:
        nums = re.findall(r"\d+", folder_name)
        age = min(max(int(nums[0]), 0), 100) if nums else 0
        gender = 0
        try:
            with os.scandir(str(folder_path)) as fit:
                for entry in fit:
                    if entry.is_file():
                        f = Path(entry.path)
                        if is_image(f):
                            photos.append((f, age, gender))
        except Exception:
            pass

    print(f"  Ditemukan {len(photos)} foto dari {data_dir.name}/")
    random.shuffle(photos)
    return photos[:target]

# ─────────────────────────────────────────────
# SOURCE 3: Asian KYC Dataset
# ─────────────────────────────────────────────
def collect_kyc(target=3000):
    base = DL_DIR / "kyc" / "files"
    if not base.exists():
        base = DL_DIR / "kyc"
    if not base.exists():
        print("  [SKIP] KYC dataset tidak ditemukan")
        return []

    print("\n[Asian KYC] Scanning...")
    photos = []
    for f in base.rglob("*"):
        if f.is_file() and is_image(f):
            # KYC: ambil Selfie saja (bukan ID card)
            if "selfie" in f.name.lower() or "Selfie" in f.name:
                photos.append((f, 0, 0))
            elif not any(x in f.name.lower() for x in ["id_", "id-", "card"]):
                photos.append((f, 0, 0))

    print(f"  Ditemukan {len(photos)} foto selfie")
    random.shuffle(photos)
    return photos[:target]

# ─────────────────────────────────────────────
# SOURCE 4: 140k Fake Faces
# ─────────────────────────────────────────────
def collect_fake_faces(target=10000):
    base = DL_DIR / "fakefaces" / "real_vs_fake" / "real-vs-fake"
    fake_dirs = [
        base / "train" / "fake",
        base / "valid" / "fake",
        base / "test"  / "fake",
    ]

    print("\n[140k Fake Faces] Scanning...")
    photos = []
    for d in fake_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and is_image(f):
                    photos.append((f, 0, 0))

    print(f"  Ditemukan {len(photos)} foto fake")
    random.shuffle(photos)
    return photos[:target]

# ─────────────────────────────────────────────
# MAIN COLLECTORS
# ─────────────────────────────────────────────
def collect_all_real(target=10000):
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    current = sum(1 for f in REAL_DIR.glob("*") if is_image(f))
    needed  = max(0, target - current)
    print(f"\n📸 Real photos: {current} ada, butuh {needed} lagi (target {target})")
    if needed <= 0:
        print("  ✅ Sudah cukup!")
        return

    # Kumpul dari semua source
    utk     = collect_utkface(target=min(needed, 5000))
    diverse = collect_diverse_asian(target=min(needed, 4000))
    kyc     = collect_kyc(target=min(needed, 2000))

    all_real = utk + diverse + kyc
    random.shuffle(all_real)
    all_real = all_real[:needed]

    print(f"\n  Total akan disalin: {len(all_real)} foto real")
    copy_batch(all_real, REAL_DIR, label="Real ")

def collect_all_fake(target=10000):
    AI_DIR.mkdir(parents=True, exist_ok=True)
    current = sum(1 for f in AI_DIR.glob("*") if is_image(f))
    needed  = max(0, target - current)
    print(f"\n🤖 AI photos: {current} ada, butuh {needed} lagi (target {target})")
    if needed <= 0:
        print("  ✅ Sudah cukup!")
        return

    fakes = collect_fake_faces(target=needed)
    if not fakes:
        print("  ❌ Tidak ada foto fake ditemukan")
        print(f"  Pastikan folder: {DL_DIR / 'fakefaces'} ada")
        return

    print(f"\n  Total akan disalin: {len(fakes)} foto fake")
    copy_batch(fakes, AI_DIR, label="Fake ")

# ─────────────────────────────────────────────
def print_summary():
    real = sum(1 for f in REAL_DIR.glob("*") if is_image(f)) if REAL_DIR.exists() else 0
    fake = sum(1 for f in AI_DIR.glob("*")   if is_image(f)) if AI_DIR.exists()   else 0
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    print(f"  Real : {real:,}  →  {REAL_DIR.name}/")
    print(f"  Fake : {fake:,}  →  {AI_DIR.name}/")
    print(f"  Total: {real+fake:,}")
    print("="*50)
    if real >= 1000 and fake >= 1000:
        print("  ✅ Dataset siap untuk training!")
        print("  Jalankan:")
        print("  python scripts/fine_tune_indonesian.py \\")
        print("    --data-dir benchmark_data/id \\")
        print("    --output-dir /opt/models/")
    else:
        if real < 1000:  print(f"  ⚠️  Real kurang ({real}/1000 min)")
        if fake < 1000:  print(f"  ⚠️  Fake kurang ({fake}/1000 min)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all","real","fake","rename-real"],
                        default="all")
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    print("="*50)
    print("  Vison AI - Dataset Preparation")
    print("="*50)

    if args.step in ("rename-real", "all"):
        print("\n[STEP 1] Rename existing real photos...")
        rename_existing_real()

    if args.step in ("real", "all"):
        print("\n[STEP 2] Collect real photos...")
        collect_all_real(target=args.count)

    if args.step in ("fake", "all"):
        print("\n[STEP 3] Collect fake photos...")
        collect_all_fake(target=args.count)

    print_summary()

if __name__ == "__main__":
    main()
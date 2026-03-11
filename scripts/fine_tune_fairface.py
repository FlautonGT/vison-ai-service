#!/usr/bin/env python3
"""
Fine-tune FairFace + Face Detection untuk Vison AI
===================================================
Models yang dihasilkan:
  1. fairface_indonesia.onnx   - FairFace fine-tuned untuk wajah Asia/Indonesia
  2. (SCRFD tidak di-fine-tune lewat script ini - butuh mmdet framework)

Dataset:
  - benchmark_data/_downloads/fairface/   - FairFace labels + images
  - benchmark_data/_downloads/UTKFace/    - UTKFace with race labels
  - benchmark_data/id/real/               - Real Indonesia faces

FairFace output:
  - race: 7 classes (White, Black, Latino, East Asian, Southeast Asian, Indian, Middle Eastern)
  - gender: Male/Female
  - age: 9 groups (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)

Usage:
    python scripts/fine_tune_fairface.py --epochs 20 --batch-size 64
"""

import argparse
import csv
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DL_DIR     = BASE_DIR / "benchmark_data" / "_downloads"
OUTPUT_DIR = BASE_DIR / "benchmark_data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RACE_CLASSES   = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
GENDER_CLASSES = ["Male", "Female"]
AGE_CLASSES    = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

NUM_RACE   = len(RACE_CLASSES)    # 7
NUM_GENDER = len(GENDER_CLASSES)  # 2
NUM_AGE    = len(AGE_CLASSES)     # 9


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class FairFaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: list of (img_path, race_idx, gender_idx, age_idx)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, race, gender, age = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(race,   dtype=torch.long),
            torch.tensor(gender, dtype=torch.long),
            torch.tensor(age,    dtype=torch.long),
        )


def load_fairface_samples(max_samples=80000):
    samples = []
    fairface_dir = DL_DIR / "fairface"

    # Try FairFace CSV labels
    for csv_name in ["fairface_label_train.csv", "train_labels.csv", "labels.csv"]:
        csv_path = fairface_dir / csv_name
        if not csv_path.exists():
            # Search recursively
            found = list(fairface_dir.rglob(csv_name))
            if found:
                csv_path = found[0]
            else:
                continue

        logger.info("  Loading FairFace CSV: %s", csv_path)
        img_root = csv_path.parent

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # FairFace CSV columns: file, age, gender, race, service_test
                img_path = img_root / row.get("file", "")
                if not img_path.exists():
                    img_path = img_root / "train" / row.get("file", "").split("/")[-1]
                if not img_path.exists():
                    continue

                race_str   = row.get("race", "").strip()
                gender_str = row.get("gender", "").strip()
                age_str    = row.get("age", "").strip()

                # Map to index
                race_idx = next((i for i, r in enumerate(RACE_CLASSES) if r.lower() in race_str.lower()), -1)
                if race_idx == -1:
                    continue
                gender_idx = 0 if "male" in gender_str.lower() and "female" not in gender_str.lower() else 1
                age_idx    = next((i for i, a in enumerate(AGE_CLASSES) if a == age_str), -1)
                if age_idx == -1:
                    # Try to find closest
                    age_idx = 3  # default 20-29

                samples.append((img_path, race_idx, gender_idx, age_idx))

        if samples:
            break

    # Supplement with UTKFace (has race label: 0=white, 1=black, 2=asian, 3=indian, 4=others)
    import re
    utk_pat = re.compile(r"^(\d+)_(\d)_(\d)_")
    utk_dirs = [
        DL_DIR / "UTKFace",
        DL_DIR / "crop_part1",
        DL_DIR / "utk2" / "utkface-aligned-labeled" / "images",
    ]
    utk_race_map = {0: 0, 1: 1, 2: 3, 3: 5, 4: 0}  # UTK→FairFace race map

    utk_count = 0
    for utk_dir in utk_dirs:
        if not utk_dir.exists():
            continue
        for img_path in utk_dir.rglob("*.jpg"):
            m = utk_pat.match(img_path.name)
            if not m:
                continue
            age_val    = int(m.group(1))
            gender_val = int(m.group(2))  # 0=male, 1=female
            race_val   = int(m.group(3))
            if race_val > 4:
                continue

            race_idx   = utk_race_map.get(race_val, 0)
            gender_idx = gender_val
            # Age to bucket
            age_idx = min(int(age_val / 10), 8) if age_val < 70 else 8
            if age_val <= 2: age_idx = 0
            elif age_val <= 9: age_idx = 1
            elif age_val <= 19: age_idx = 2
            elif age_val <= 29: age_idx = 3
            elif age_val <= 39: age_idx = 4
            elif age_val <= 49: age_idx = 5
            elif age_val <= 59: age_idx = 6
            elif age_val <= 69: age_idx = 7
            else: age_idx = 8

            samples.append((img_path, race_idx, gender_idx, age_idx))
            utk_count += 1

    logger.info("  FairFace CSV samples: %d | UTKFace samples: %d", len(samples) - utk_count, utk_count)

    if len(samples) < 500:
        logger.error("Tidak cukup data FairFace. Pastikan dataset sudah didownload.")
        return None, None

    random.shuffle(samples)
    samples = samples[:max_samples]
    split   = int(len(samples) * 0.85)
    return samples[:split], samples[split:]


TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Model - EfficientNet multitask
# ─────────────────────────────────────────────
class FairFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        feat_dim = 1280

        self.race_head   = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, NUM_RACE))
        self.gender_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, NUM_GENDER))
        self.age_head    = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, NUM_AGE))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.race_head(x), self.gender_head(x), self.age_head(x)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_fairface(model, train_loader, val_loader, device, epochs):
    ce = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    save_path = OUTPUT_DIR / "fairface_indonesia.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, race, gender, age in train_loader:
            imgs   = imgs.to(device)
            race   = race.to(device)
            gender = gender.to(device)
            age    = age.to(device)

            optimizer.zero_grad()
            r_out, g_out, a_out = model(imgs)
            loss = ce(r_out, race) + ce(g_out, gender) + ce(a_out, age)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        r_correct = g_correct = a_correct = total = 0
        with torch.no_grad():
            for imgs, race, gender, age in val_loader:
                imgs = imgs.to(device)
                r_out, g_out, a_out = model(imgs)
                r_correct += (r_out.argmax(1).cpu() == race).sum().item()
                g_correct += (g_out.argmax(1).cpu() == gender).sum().item()
                a_correct += (a_out.argmax(1).cpu() == age).sum().item()
                total += len(race)

        r_acc = r_correct / total * 100
        g_acc = g_correct / total * 100
        a_acc = a_correct / total * 100
        mean_acc = (r_acc + g_acc + a_acc) / 3

        scheduler.step()
        logger.info(
            "Epoch %2d/%d — loss=%.4f | Race=%.1f%% | Gender=%.1f%% | Age=%.1f%% | Mean=%.1f%%",
            epoch, epochs, total_loss / len(train_loader), r_acc, g_acc, a_acc, mean_acc
        )

        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model.state_dict(), save_path)
            logger.info("  -> Saved best (Mean=%.1f%%)", best_acc)

    return best_acc


# ─────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────
def export_onnx(model, device):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    dummy    = torch.randn(1, 3, 224, 224)
    onnx_path = OUTPUT_DIR / "fairface_indonesia.onnx"

    torch.onnx.export(
        model.cpu(), dummy, str(onnx_path),
        input_names=["input"],
        output_names=["race_logits", "gender_logits", "age_logits"],
        dynamic_axes={
            "input":        {0: "batch_size"},
            "race_logits":  {0: "batch_size"},
            "gender_logits":{0: "batch_size"},
            "age_logits":   {0: "batch_size"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_path))
    logger.info("✅ ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)

    int8_path = OUTPUT_DIR / "fairface_indonesia_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("✅ INT8 quantized: %s (%.1f MB)", int8_path.name, int8_path.stat().st_size / 1e6)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=80000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("\n%s\nLOADING FAIRFACE DATASET\n%s", "=" * 60, "=" * 60)
    train_samples, val_samples = load_fairface_samples(max_samples=args.max_samples)
    if train_samples is None:
        return

    logger.info("Train: %d | Val: %d", len(train_samples), len(val_samples))

    train_ds     = FairFaceDataset(train_samples, TRAIN_TF)
    val_ds       = FairFaceDataset(val_samples,   VAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("\n%s\nTRAINING: FAIRFACE\n%s", "=" * 60, "=" * 60)
    model = FairFaceModel().to(device)
    train_fairface(model, train_loader, val_loader, device, args.epochs)

    logger.info("\nExporting ONNX...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "fairface_indonesia.pth", map_location=device))
    export_onnx(model, device)

    logger.info("\n%s", "=" * 60)
    logger.info("NEXT STEP: Copy ke VPS")
    logger.info("=" * 60)
    logger.info("  scp benchmark_data/models/fairface_indonesia*.onnx user@vps:/opt/models/")
    logger.info("  Update .env: AGE_GENDER_VIT_MODEL bisa tetap, FairFace diupdate otomatis")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
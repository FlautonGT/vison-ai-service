#!/usr/bin/env python3
"""
Fine-tune Age/Gender + Face Attributes untuk Vison AI
======================================================
Dataset: CelebA (200k foto, 40 attributes)
         UTKFace Asian (label umur/gender)

Models yang dihasilkan:
  1. age_gender_indonesia.onnx  - estimasi umur + gender untuk wajah Asia
  2. face_attributes.onnx       - deteksi kacamata, topi, masker, dll

Usage:
    python scripts/fine_tune_age_gender.py --task age-gender --epochs 20
    python scripts/fine_tune_age_gender.py --task attributes --epochs 15
    python scripts/fine_tune_age_gender.py --task all --epochs 20
"""

import argparse
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DL_DIR     = BASE_DIR / "benchmark_data" / "_downloads"
OUTPUT_DIR = BASE_DIR / "benchmark_data" / "models"

_celeba_base = DL_DIR / "celeba" / "img_align_celeba"
CELEBA_IMG_DIR  = (_celeba_base / "img_align_celeba") if (_celeba_base / "img_align_celeba").exists() else _celeba_base
CELEBA_ATTR_CSV = DL_DIR / "celeba" / "list_attr_celeba.csv"
UTK_DIRS        = [
    DL_DIR / "UTKFace",
    DL_DIR / "crop_part1",
    DL_DIR / "utk2" / "utkface-aligned-labeled" / "images",
]

UTKFACE_PAT = re.compile(r"^(\d+)_(\d)_(\d)_")

# Attributes yang relevan untuk Vison AI
ATTRIBUTES = [
    # Occlusion - wajah tertutup
    "Eyeglasses",           # kacamata biasa
    "Wearing_Hat",          # topi
    # Masker tidak ada di CelebA, didetect via obstructions dataset nanti

    # Jenggot/kumis - affect face match confidence
    "No_Beard",             # tidak ada jenggot (inverse = ada jenggot)
    "Goatee",               # janggut
    "Mustache",             # kumis
    "Sideburns",            # cambang

    # Ekspresi - untuk active liveness
    "Mouth_Slightly_Open",  # buka mulut
    "Smiling",              # senyum

    # Gender & age clues
    "Male",                 # gender
    "Young",                # muda
    "Bald",                 # botak

    # Makeup/appearance
    "Heavy_Makeup",         # makeup tebal
    "Wearing_Lipstick",     # lipstik
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Dataset: Age/Gender (UTKFace Asian)
# ─────────────────────────────────────────────
class AgeGenderDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: list of (path, age, gender) gender: 0=male, 1=female
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age, gender = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        age_norm = torch.tensor(age / 100.0, dtype=torch.float32)
        gender_t = torch.tensor(gender, dtype=torch.long)
        return img, age_norm, gender_t


def load_utk_samples(target=20000):
    """Load UTKFace Asian samples dengan label umur/gender."""
    samples = []
    seen = set()
    for base in UTK_DIRS:
        if not base.exists():
            continue
        for f in base.iterdir():
            if not f.is_file():
                continue
            name = f.name
            if name in seen:
                continue
            if not (name.lower().endswith((".jpg", ".png", ".jpg.chip.jpg"))):
                continue
            m = UTKFACE_PAT.match(name)
            if not m:
                continue
            # Semua race (bukan hanya Asian) untuk age/gender yang lebih general
            age    = min(max(int(m.group(1)), 0), 100)
            gender = 0 if m.group(2) == "0" else 1  # 0=male, 1=female
            samples.append((f, age, gender))
            seen.add(name)

    logger.info(f"  UTKFace samples: {len(samples)}")
    random.shuffle(samples)
    return samples[:target]


# ─────────────────────────────────────────────
# Dataset: Face Attributes (CelebA)
# ─────────────────────────────────────────────
class FaceAttributeDataset(Dataset):
    def __init__(self, samples, attr_labels, transform=None):
        # samples: list of image paths
        # attr_labels: numpy array [N, num_attrs] binary 0/1
        self.samples    = samples
        self.attr_labels = attr_labels
        self.transform  = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(self.attr_labels[idx], dtype=torch.float32)
        return img, labels


def load_celeba_samples(max_samples=50000):
    """Load CelebA samples dengan attribute labels."""
    if not CELEBA_ATTR_CSV.exists():
        logger.error(f"CelebA attribute CSV tidak ditemukan: {CELEBA_ATTR_CSV}")
        return [], np.array([])
    if not CELEBA_IMG_DIR.exists():
        logger.error(f"CelebA image dir tidak ditemukan: {CELEBA_IMG_DIR}")
        return [], np.array([])

    logger.info("  Loading CelebA attributes CSV...")
    df = pd.read_csv(CELEBA_ATTR_CSV)

    # Cek kolom yang tersedia
    available_attrs = [a for a in ATTRIBUTES if a in df.columns]
    if not available_attrs:
        logger.error(f"Tidak ada attribute yang cocok di CSV. Kolom: {list(df.columns[:10])}")
        return [], np.array([])

    logger.info(f"  Attributes yang dipakai: {available_attrs}")

    # Convert -1/1 ke 0/1
    attr_matrix = df[available_attrs].values
    attr_matrix = (attr_matrix + 1) // 2  # -1 → 0, 1 → 1

    # Get image paths
    if "image_id" in df.columns:
        img_names = df["image_id"].values
    else:
        img_names = df.iloc[:, 0].values

    samples, labels = [], []
    for i, name in enumerate(img_names[:max_samples]):
        img_path = CELEBA_IMG_DIR / str(name)
        if img_path.exists():
            samples.append(img_path)
            labels.append(attr_matrix[i])

    logger.info(f"  CelebA samples loaded: {len(samples)}")
    return samples, np.array(labels), available_attrs


# ─────────────────────────────────────────────
# Model: Age/Gender (EfficientNet-B0 multitask)
# ─────────────────────────────────────────────
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.3)
        self.age_head  = nn.Linear(in_features, 1)    # regression
        self.gender_head = nn.Linear(in_features, 2)  # classification

    def forward(self, x):
        feat   = self.dropout(self.backbone(x))
        age    = torch.sigmoid(self.age_head(feat))    # 0-1 (×100 = umur)
        gender = self.gender_head(feat)                # logits [male, female]
        return age, gender


# ─────────────────────────────────────────────
# Model: Face Attributes (EfficientNet-B0 multi-label)
# ─────────────────────────────────────────────
class FaceAttributeModel(nn.Module):
    def __init__(self, num_attrs: int):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features, num_attrs)

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        return self.classifier(feat)  # raw logits, apply sigmoid for inference


# ─────────────────────────────────────────────
# Training: Age/Gender
# ─────────────────────────────────────────────
def train_age_gender(epochs=20, batch_size=64, lr=1e-3, val_split=0.15):
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Age/Gender Model")
    logger.info("="*60)

    samples = load_utk_samples(target=30000)
    if len(samples) < 100:
        logger.error("Tidak cukup data UTKFace. Pastikan dataset sudah didownload.")
        return

    # Split train/val
    random.shuffle(samples)
    n_val  = int(len(samples) * val_split)
    val_s, train_s = samples[:n_val], samples[n_val:]
    logger.info(f"  Train: {len(train_s)} | Val: {len(val_s)}")

    train_ds = AgeGenderDataset(train_s, TRAIN_TRANSFORM)
    val_ds   = AgeGenderDataset(val_s,   VAL_TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = AgeGenderModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    age_criterion    = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()

    best_mae, best_gender_acc = 999, 0
    patience, no_improve = 7, 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for imgs, ages, genders in train_dl:
            imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            optimizer.zero_grad()
            age_pred, gender_pred = model(imgs)
            loss = age_criterion(age_pred.squeeze(), ages) + 0.5 * gender_criterion(gender_pred, genders)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Val
        model.eval()
        all_ages_pred, all_ages_true = [], []
        all_gender_pred, all_gender_true = [], []
        with torch.no_grad():
            for imgs, ages, genders in val_dl:
                imgs = imgs.to(DEVICE)
                age_pred, gender_pred = model(imgs)
                all_ages_pred.extend((age_pred.squeeze().cpu().numpy() * 100).tolist())
                all_ages_true.extend((ages.numpy() * 100).tolist())
                all_gender_pred.extend(gender_pred.argmax(1).cpu().numpy().tolist())
                all_gender_true.extend(genders.numpy().tolist())

        mae         = mean_absolute_error(all_ages_true, all_ages_pred)
        gender_acc  = accuracy_score(all_gender_true, all_gender_pred) * 100
        train_loss /= len(train_dl)

        logger.info(
            f"Epoch {epoch:2d}/{epochs} — loss={train_loss:.4f} | "
            f"Age MAE={mae:.2f}y | Gender Acc={gender_acc:.1f}% | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best
        improved = False
        if mae < best_mae - 0.1:
            best_mae = mae
            improved = True
        if gender_acc > best_gender_acc + 0.1:
            best_gender_acc = gender_acc
            improved = True

        if improved:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "age_gender_indonesia_best.pth")
            logger.info(f"  -> Saved best (MAE={best_mae:.2f}y, Gender={best_gender_acc:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Export ONNX
    export_age_gender_onnx(model)


def export_age_gender_onnx(model):
    """Export age/gender model ke ONNX dengan output terpisah."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    onnx_path = OUTPUT_DIR / "age_gender_indonesia.onnx"

    # Wrapper untuk export kedua output
    class AgeGenderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            age, gender = self.m(x)
            # age: [B,1] → umur 0-100
            # gender: [B,2] → softmax probabilities
            age_years = age * 100
            gender_prob = torch.softmax(gender, dim=1)
            return age_years, gender_prob

    wrapper = AgeGenderWrapper(model).to(DEVICE)
    wrapper.eval()

    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        export_params=True, opset_version=14,
        input_names=["input"],
        output_names=["age", "gender_prob"],
        dynamic_axes={"input": {0: "batch"}, "age": {0: "batch"}, "gender_prob": {0: "batch"}},
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    logger.info(f"\n✅ Age/Gender ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    logger.info("   Output[0] = age (0-100 tahun)")
    logger.info("   Output[1] = gender_prob [male, female]")


# ─────────────────────────────────────────────
# Training: Face Attributes
# ─────────────────────────────────────────────
def train_attributes(epochs=15, batch_size=64, lr=1e-3, val_split=0.15, max_samples=50000):
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Face Attributes Model")
    logger.info("="*60)

    result = load_celeba_samples(max_samples=max_samples)
    if len(result) == 3:
        samples, labels, available_attrs = result
    else:
        logger.error("Gagal load CelebA")
        return

    if len(samples) < 100:
        logger.error("Tidak cukup data CelebA.")
        return

    num_attrs = len(available_attrs)
    logger.info(f"  Num attributes: {num_attrs} → {available_attrs}")

    # Split
    indices = list(range(len(samples)))
    random.shuffle(indices)
    n_val = int(len(indices) * val_split)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_s = [samples[i] for i in train_idx]
    val_s   = [samples[i] for i in val_idx]
    train_l = labels[train_idx]
    val_l   = labels[val_idx]

    logger.info(f"  Train: {len(train_s)} | Val: {len(val_s)}")

    train_ds = FaceAttributeDataset(train_s, train_l, TRAIN_TRANSFORM)
    val_ds   = FaceAttributeDataset(val_s,   val_l,   VAL_TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = FaceAttributeModel(num_attrs).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_acc, no_improve, patience = 0, 0, 5

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs = imgs.to(DEVICE)
                preds = torch.sigmoid(model(imgs))
                all_preds.append((preds.cpu().numpy() > 0.5).astype(int))
                all_labels.append(lbls.numpy().astype(int))

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Per-attribute accuracy
        attr_accs = [(all_preds[:, i] == all_labels[:, i]).mean() * 100
                     for i in range(num_attrs)]
        mean_acc = np.mean(attr_accs)
        train_loss /= len(train_dl)

        attr_str = " | ".join(f"{available_attrs[i][:8]}={attr_accs[i]:.0f}%"
                              for i in range(min(5, num_attrs)))
        logger.info(
            f"Epoch {epoch:2d}/{epochs} — loss={train_loss:.4f} | "
            f"MeanAcc={mean_acc:.1f}% | {attr_str}"
        )

        if mean_acc > best_acc + 0.1:
            best_acc = mean_acc
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "attributes": available_attrs,
                "num_attrs": num_attrs,
            }, OUTPUT_DIR / "face_attributes_best.pth")
            logger.info(f"  -> Saved best (MeanAcc={best_acc:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Export ONNX
    export_attributes_onnx(model, available_attrs)


def export_attributes_onnx(model, available_attrs):
    """Export face attributes model ke ONNX."""
    model.eval()

    class AttrWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.sigmoid(self.m(x))  # probabilities 0-1

    wrapper = AttrWrapper(model).to(DEVICE)
    wrapper.eval()

    dummy     = torch.randn(1, 3, 224, 224).to(DEVICE)
    onnx_path = OUTPUT_DIR / "face_attributes.onnx"

    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        export_params=True, opset_version=14,
        input_names=["input"],
        output_names=["attributes"],
        dynamic_axes={"input": {0: "batch"}, "attributes": {0: "batch"}},
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    logger.info(f"\n✅ Face Attributes ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    logger.info(f"   Output shape: [batch, {len(available_attrs)}]")
    logger.info(f"   Attribute order: {available_attrs}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Age/Gender + Face Attributes")
    parser.add_argument("--task",       choices=["age-gender", "attributes", "all"], default="all")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--max-celeba", type=int,   default=50000,
                        help="Max CelebA samples untuk attributes training")
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.task in ("age-gender", "all"):
        train_age_gender(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
        )

    if args.task in ("attributes", "all"):
        train_attributes(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            max_samples=args.max_celeba,
        )

    logger.info("\n" + "="*60)
    logger.info("NEXT STEP: Copy model ke VPS")
    logger.info("="*60)
    logger.info("  scp benchmark_data/models/age_gender_indonesia.onnx user@vps:/opt/models/")
    logger.info("  scp benchmark_data/models/face_attributes.onnx user@vps:/opt/models/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Fine-tune Deepfake Detection Models for Vison AI
=================================================
Output models:
  1. deepfake_efficientnet_indonesia.onnx  - EfficientNet-B0 fine-tuned on Asian faces
  2. deepfake_vit_indonesia.onnx           - ViT fine-tuned on Asian faces  
  3. deepfake_npr_indonesia.onnx           - NPR ResNet18 fine-tuned on Asian faces

Purpose:
  Reduce false positives on real Asian/Indonesian faces.
  Base models (efficientnet_b0, community_forensics_vit) are biased toward
  Western faces and incorrectly flag Indonesian faces as AI-generated.

Dataset:
  - benchmark_data/id/real/              (10k real faces, majority Asian/Indonesian)
  - benchmark_data/id/ai/                (10k AI-generated faces, StyleGAN)
  - benchmark_data/_downloads/fakefaces/ (additional StyleGAN fake faces)

Target metrics:
  - Accuracy > 99%
  - FPR (real flagged as fake) < 0.5%  ← fintech standard
  - FNR (fake missed) < 1%

Usage:
    python scripts/fine_tune_deepfake.py --task all --epochs 20 --batch-size 32
    python scripts/fine_tune_deepfake.py --task efficientnet --epochs 20
    python scripts/fine_tune_deepfake.py --task vit --epochs 15
    python scripts/fine_tune_deepfake.py --task npr --epochs 20
"""

import argparse
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
REAL_DIR   = BASE_DIR / "benchmark_data" / "id" / "real"
FAKE_DIR   = BASE_DIR / "benchmark_data" / "id" / "ai"
OUTPUT_DIR = BASE_DIR / "benchmark_data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label) where 0=real, 1=fake
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def load_samples(max_per_class=12000):
    real_imgs = list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png"))
    fake_imgs = list(FAKE_DIR.glob("*.jpg")) + list(FAKE_DIR.glob("*.png"))

    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)
    real_imgs = real_imgs[:max_per_class]
    fake_imgs = fake_imgs[:max_per_class]

    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    random.shuffle(samples)

    split = int(len(samples) * 0.85)
    return samples[:split], samples[split:]


TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_binary(model, train_loader, val_loader, device, epochs, save_path, model_name):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_fpr = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs).squeeze(1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = torch.sigmoid(model(imgs).squeeze(1))
                preds = (out > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc  = (all_preds == all_labels).mean() * 100
        # FPR = real wrongly flagged as fake
        real_mask = all_labels == 0
        fpr = (all_preds[real_mask] == 1).mean() * 100 if real_mask.sum() > 0 else 0.0
        # FNR = fake missed
        fake_mask = all_labels == 1
        fnr = (all_preds[fake_mask] == 0).mean() * 100 if fake_mask.sum() > 0 else 0.0

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        logger.info(
            "Epoch %2d/%d — loss=%.4f | Acc=%.1f%% | FPR=%.2f%% | FNR=%.2f%% | lr=%.2e",
            epoch, epochs, avg_loss, acc, fpr, fnr, scheduler.get_last_lr()[0]
        )

        if acc > best_acc or (acc == best_acc and fpr < best_fpr):
            best_acc = acc
            best_fpr = fpr
            torch.save(model.state_dict(), save_path.with_suffix(".pth"))
            logger.info("  -> Saved best (Acc=%.1f%%, FPR=%.2f%%)", best_acc, best_fpr)

    logger.info("Training %s done. Best Acc=%.1f%% FPR=%.2f%%", model_name, best_acc, best_fpr)
    return best_acc, best_fpr


# ─────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────
def export_onnx(model, save_path, model_name):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = OUTPUT_DIR / f"{save_path.stem}.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_path))
    size_mb = onnx_path.stat().st_size / 1e6
    logger.info("✅ ONNX exported: %s (%.1f MB)", onnx_path.name, size_mb)

    # INT8
    int8_path = OUTPUT_DIR / f"{save_path.stem}_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    size_int8 = int8_path.stat().st_size / 1e6
    logger.info("✅ INT8 quantized: %s (%.1f MB)", int8_path.name, size_int8)
    return onnx_path, int8_path


# ─────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────
def build_efficientnet():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, 1),
    )
    return m


def build_vit():
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    m.heads = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.heads.head.in_features, 1),
    )
    return m


def build_npr():
    """NPR = ResNet18 — lightweight, good for frequency artifacts."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, 1),
    )
    return m


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["efficientnet", "vit", "npr", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=12000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Verify data
    real_count = len(list(REAL_DIR.glob("*.jpg"))) + len(list(REAL_DIR.glob("*.png")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    logger.info("Real faces: %d | Fake faces: %d", real_count, fake_count)
    if real_count < 100 or fake_count < 100:
        logger.error("Tidak cukup data. Real: %d, Fake: %d", real_count, fake_count)
        return

    train_samples, val_samples = load_samples(max_per_class=args.max_samples)
    logger.info("Train: %d | Val: %d", len(train_samples), len(val_samples))

    train_ds = DeepfakeDataset(train_samples, TRAIN_TF)
    val_ds   = DeepfakeDataset(val_samples, VAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    tasks = ["efficientnet", "vit", "npr"] if args.task == "all" else [args.task]

    for task in tasks:
        logger.info("\n%s", "=" * 60)
        logger.info("TRAINING: %s", task.upper())
        logger.info("=" * 60)

        if task == "efficientnet":
            model = build_efficientnet().to(device)
            save_path = OUTPUT_DIR / "deepfake_efficientnet_indonesia"
        elif task == "vit":
            model = build_vit().to(device)
            save_path = OUTPUT_DIR / "deepfake_vit_indonesia"
        elif task == "npr":
            model = build_npr().to(device)
            save_path = OUTPUT_DIR / "deepfake_npr_indonesia"

        train_binary(model, train_loader, val_loader, device, args.epochs, save_path, task)

        # Load best weights and export
        model.load_state_dict(torch.load(save_path.with_suffix(".pth"), map_location=device))
        export_onnx(model, save_path, task)

    logger.info("\n%s", "=" * 60)
    logger.info("NEXT STEP: Copy ke VPS")
    logger.info("=" * 60)
    logger.info("  scp benchmark_data/models/deepfake_*_indonesia*.onnx user@vps:/opt/models/")
    logger.info("  Lalu update .env DEEPFAKE_MODELS dan AI_FACE_DETECTOR_MODEL")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
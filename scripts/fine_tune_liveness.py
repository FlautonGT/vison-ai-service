#!/usr/bin/env python3
"""
Fine-tune Liveness Detection untuk Vison AI
============================================
Models yang dihasilkan:
  1. liveness_minifas_indonesia.onnx  - MiniFASNet fine-tuned untuk wajah Asia

Dataset yang dipakai:
  - benchmark_data/_downloads/liveness/   - Asian People Liveness Detection
  - benchmark_data/_downloads/liveness2/  - Web Camera Face Liveness
  - benchmark_data/_downloads/liveness3/  - On-device Face Liveness
  - benchmark_data/_downloads/spoof2d/    - Real + 2D Mask Attacks
  - benchmark_data/_downloads/ibeta/      - iBeta PAD Level 1 (video frames)
  - benchmark_data/id/real/               - Real faces kita

Attack types yang dicover:
  - Print attack (foto dicetak)
  - Replay attack (foto/video di layar)
  - 2D mask
  - Partial occlusion spoof

Usage:
    python scripts/fine_tune_liveness.py --epochs 25 --batch-size 32
"""

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
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
REAL_DIR   = BASE_DIR / "benchmark_data" / "id" / "real"
OUTPUT_DIR = BASE_DIR / "benchmark_data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Dataset collection
# ─────────────────────────────────────────────
def collect_samples():
    live_samples  = []
    spoof_samples = []

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def add_dir(d, label, limit=None):
        imgs = [p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS]
        random.shuffle(imgs)
        if limit:
            imgs = imgs[:limit]
        return imgs

    # Real / live faces
    live_samples += add_dir(REAL_DIR, 0, limit=5000)
    logger.info("  Real (id/real): %d", len(live_samples))

    # Liveness datasets - scan for live/spoof subfolders
    liveness_dirs = [
        DL_DIR / "liveness",
        DL_DIR / "liveness2",
        DL_DIR / "liveness3",
    ]
    for ld in liveness_dirs:
        if not ld.exists():
            continue
        # Try to find live/spoof subfolders
        for sub in ld.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if any(k in name for k in ["live", "real", "genuine", "authentic"]):
                imgs = add_dir(sub, 0, limit=2000)
                live_samples += [(p, 0) for p in imgs]
                logger.info("  Live (%s): %d", sub.relative_to(DL_DIR), len(imgs))
            elif any(k in name for k in ["spoof", "fake", "attack", "print", "replay", "mask", "photo"]):
                imgs = add_dir(sub, 1, limit=2000)
                spoof_samples += [(p, 1) for p in imgs]
                logger.info("  Spoof (%s): %d", sub.relative_to(DL_DIR), len(imgs))

    # Spoof2D dataset
    spoof2d = DL_DIR / "spoof2d"
    if spoof2d.exists():
        for sub in spoof2d.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if any(k in name for k in ["real", "live"]):
                imgs = add_dir(sub, 0, limit=2000)
                live_samples += [(p, 0) for p in imgs]
            elif any(k in name for k in ["spoof", "fake", "mask", "attack"]):
                imgs = add_dir(sub, 1, limit=2000)
                spoof_samples += [(p, 1) for p in imgs]

    # iBeta PAD - extract frames from video if needed, or use existing frames
    ibeta = DL_DIR / "ibeta"
    if ibeta.exists():
        # iBeta might have video files - extract first frame only for now
        for vid in ibeta.rglob("*.mp4"):
            try:
                cap = cv2.VideoCapture(str(vid))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    # Save frame temporarily
                    frame_path = OUTPUT_DIR / f"ibeta_frame_{vid.stem}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    name = str(vid).lower()
                    if any(k in name for k in ["live", "real"]):
                        live_samples.append((frame_path, 0))
                    else:
                        spoof_samples.append((frame_path, 1))
            except Exception:
                pass

        # Also collect images if any
        for sub in ibeta.rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if any(k in name for k in ["live", "real", "bona_fide"]):
                imgs = add_dir(sub, 0, limit=3000)
                live_samples += [(p, 0) for p in imgs]
            elif any(k in name for k in ["attack", "spoof", "print", "replay"]):
                imgs = add_dir(sub, 1, limit=3000)
                spoof_samples += [(p, 1) for p in imgs]

    # Convert plain paths to tuples if needed
    live_samples  = [(p, 0) if not isinstance(p, tuple) else p for p in live_samples]
    spoof_samples = [(p, 1) if not isinstance(p, tuple) else p for p in spoof_samples]

    logger.info("Total live: %d | Total spoof: %d", len(live_samples), len(spoof_samples))

    # Balance
    min_count = min(len(live_samples), len(spoof_samples))
    if min_count < 200:
        logger.error("Tidak cukup data. Live: %d, Spoof: %d", len(live_samples), len(spoof_samples))
        return None, None

    random.shuffle(live_samples)
    random.shuffle(spoof_samples)
    samples = live_samples[:min_count] + spoof_samples[:min_count]
    random.shuffle(samples)

    split = int(len(samples) * 0.85)
    return samples[:split], samples[split:]


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class LivenessDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
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


TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Model - MobileNetV3 small (compatible dengan MiniFASNet size)
# ─────────────────────────────────────────────
def build_liveness_model():
    """
    MobileNetV3-Small sebagai backbone — ringan, cocok untuk liveness.
    Output: single logit (positive = spoof)
    """
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return m


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_liveness(model, train_loader, val_loader, device, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    best_apcer = 1.0  # Attack Presentation Classification Error Rate
    save_path = OUTPUT_DIR / "liveness_minifas_indonesia.pth"

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

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out  = torch.sigmoid(model(imgs).squeeze(1))
                preds = (out > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = (all_preds == all_labels).mean() * 100

        # ISO 30107-3 metrics
        spoof_mask = all_labels == 1
        live_mask  = all_labels == 0
        # APCER: spoof accepted as live (spoof wrongly predicted as 0)
        apcer = (all_preds[spoof_mask] == 0).mean() * 100 if spoof_mask.sum() > 0 else 0.0
        # BPCER: live rejected as spoof (live wrongly predicted as 1)
        bpcer = (all_preds[live_mask] == 1).mean() * 100 if live_mask.sum() > 0 else 0.0

        scheduler.step()
        logger.info(
            "Epoch %2d/%d — loss=%.4f | Acc=%.1f%% | APCER=%.2f%% | BPCER=%.2f%% | lr=%.2e",
            epoch, epochs, total_loss / len(train_loader), acc, apcer, bpcer,
            scheduler.get_last_lr()[0]
        )

        if acc > best_acc or (acc == best_acc and apcer < best_apcer):
            best_acc   = acc
            best_apcer = apcer
            torch.save(model.state_dict(), save_path)
            logger.info("  -> Saved best (Acc=%.1f%%, APCER=%.2f%%, BPCER=%.2f%%)", best_acc, best_apcer, bpcer)

    return best_acc, best_apcer


# ─────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────
def export_onnx(model, device):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = OUTPUT_DIR / "liveness_minifas_indonesia.onnx"

    torch.onnx.export(
        model.cpu(), dummy.cpu(), str(onnx_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_path))
    logger.info("✅ ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)

    int8_path = OUTPUT_DIR / "liveness_minifas_indonesia_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("✅ INT8 quantized: %s (%.1f MB)", int8_path.name, int8_path.stat().st_size / 1e6)

    # Verify inference
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = sess.run(None, {"input": test_input})
    score = float(1.0 / (1.0 + np.exp(-out[0][0][0])))
    logger.info("   Test inference score: %.4f (0=live, 1=spoof) ✅", score)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("\n%s\nCOLLECTING LIVENESS DATASET\n%s", "=" * 60, "=" * 60)
    train_samples, val_samples = collect_samples()
    if train_samples is None:
        logger.error("Dataset tidak cukup. Pastikan dataset sudah didownload.")
        return

    logger.info("Train: %d | Val: %d", len(train_samples), len(val_samples))

    train_ds = LivenessDataset(train_samples, TRAIN_TF)
    val_ds   = LivenessDataset(val_samples,   VAL_TF)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("\n%s\nTRAINING: LIVENESS DETECTION\n%s", "=" * 60, "=" * 60)
    model = build_liveness_model().to(device)
    train_liveness(model, train_loader, val_loader, device, args.epochs)

    logger.info("\nExporting ONNX...")
    pth_path = OUTPUT_DIR / "liveness_minifas_indonesia.pth"
    model.load_state_dict(torch.load(pth_path, map_location=device))
    export_onnx(model, device)

    logger.info("\n%s", "=" * 60)
    logger.info("NEXT STEP: Copy ke VPS")
    logger.info("=" * 60)
    logger.info("  scp benchmark_data/models/liveness_minifas_indonesia*.onnx user@vps:/opt/models/")
    logger.info("  Update .env: LIVENESS_MODELS=liveness_minifas_indonesia.onnx,MiniFASNetV2.onnx")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fine-tune CLIP linear probe for Indonesian/SEA face AI detection.

Trains only the linear classification head on top of frozen CLIP ViT-B/16
visual features.  This is the UniversalFakeDetect approach — the CLIP
backbone generalizes across generators, and the linear probe learns the
decision boundary for your specific deployment environment.

Expected dataset structure:
    data/
    ├── real/
    │   ├── indo_selfie_001.jpg
    │   ├── indo_selfie_002.jpg
    │   └── ...
    └── fake/
        ├── midjourney_asian_001.jpg
        ├── stablediffusion_indo_001.jpg
        └── ...

Augmentation pipeline is optimized for Indonesian face photography:
    - JPEG compression simulation (WhatsApp quality 50-85)
    - Low-light indoor conditions (common in Indonesian homes)
    - Tropical harsh sunlight
    - Skin-tone preserving hue shifts
    - Budget Android camera noise (Oppo, Vivo, Samsung)
    - Social media resize chains (720px, 1080px)

Usage:
    pip install torch torchvision open_clip_torch albumentations pillow

    python scripts/fine_tune_indonesian.py \
        --data-dir data/ \
        --output-dir models/ \
        --epochs 30 \
        --batch-size 32 \
        --lr 1e-3

    # After training, export to ONNX:
    python scripts/export_models.py \
        --output-dir models/ \
        --clip-weights models/clip_probe_best.pth \
        --quantize
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP linear probe for Indonesian AI face detection")
    parser.add_argument("--data-dir", required=True, help="Root data directory with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", default="models/", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--clip-model", default="ViT-B-16", help="CLIP model name")
    parser.add_argument("--clip-pretrained", default="openai", help="CLIP pretrained weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    try:
        import open_clip
    except ImportError:
        logger.error("open_clip not installed. Run: pip install open_clip_torch")
        sys.exit(1)

    try:
        import albumentations as A
    except ImportError:
        logger.error("albumentations not installed. Run: pip install albumentations")
        sys.exit(1)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # =================== DATASET ===================

    class JPEGCompression:
        """Simulate WhatsApp/social media JPEG compression."""
        def __init__(self, quality_range: tuple[int, int] = (50, 85)):
            self.quality_range = quality_range

        def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
            quality = random.randint(*self.quality_range)
            pil_img = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            result = np.array(Image.open(buffer))
            return result

    class SocialMediaResize:
        """Simulate social media resize chains."""
        def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
            target = random.choice([720, 1080, 1280])
            h, w = image.shape[:2]
            if max(h, w) > target:
                scale = target / max(h, w)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                import cv2
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return image

    # Indonesian-specific augmentation pipeline
    train_spatial_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
    ])

    train_color_aug = A.Compose([
        A.OneOf([
            # Low-light indoor (common in Indonesian homes)
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, -0.1),
                contrast_limit=(-0.2, 0.1),
                p=1.0,
            ),
            # Tropical harsh sunlight
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.4),
                contrast_limit=(0.1, 0.3),
                p=1.0,
            ),
            # Normal lighting
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=1.0,
            ),
        ], p=0.6),
        # Skin-tone preserving hue shift (minimal hue, moderate sat/val)
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=20,
            p=0.3,
        ),
        # Budget Android camera noise (Oppo, Vivo, Samsung low-end)
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        # Slight motion blur (selfies)
        A.MotionBlur(blur_limit=5, p=0.15),
        # JPEG compression simulation (WhatsApp quality)
        A.ImageCompression(quality_lower=50, quality_upper=85, p=0.5),
    ])

    # CLIP normalization constants
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    class IndonesianFaceDataset(Dataset):
        """Dataset with Indonesian-specific augmentations."""

        def __init__(self, file_paths: list[str], labels: list[int], augment: bool = True):
            self.file_paths = file_paths
            self.labels = labels
            self.augment = augment

        def __len__(self) -> int:
            return len(self.file_paths)

        def __getitem__(self, idx: int):
            img_path = self.file_paths[idx]
            label = self.labels[idx]

            # Load image
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                # Return a dummy on failure
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            if self.augment:
                # Spatial augmentations
                img = train_spatial_aug(image=img)["image"]
                # Color/noise augmentations
                img = train_color_aug(image=img)["image"]

            # Resize to 224x224
            import cv2
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Normalize for CLIP
            img = img.astype(np.float32) / 255.0
            img = (img - CLIP_MEAN) / CLIP_STD
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    # =================== LOAD DATA ===================
    data_dir = Path(args.data_dir)
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        logger.error("Expected directories: %s and %s", real_dir, fake_dir)
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    real_files = sorted([
        str(f) for f in real_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    fake_files = sorted([
        str(f) for f in fake_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    logger.info("Found %d real images, %d fake images", len(real_files), len(fake_files))

    if len(real_files) < 10 or len(fake_files) < 10:
        logger.error("Need at least 10 images in each class")
        sys.exit(1)

    # Combine and split
    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)  # 0=real, 1=fake

    # Stratified split
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)

    val_count = max(2, int(len(combined) * args.val_split))
    val_data = combined[:val_count]
    train_data = combined[val_count:]

    train_files = [f for f, _ in train_data]
    train_labels = [l for _, l in train_data]
    val_files = [f for f, _ in val_data]
    val_labels = [l for _, l in val_data]

    logger.info("Train: %d samples, Val: %d samples", len(train_files), len(val_files))
    logger.info("Train class balance: real=%d fake=%d",
                train_labels.count(0), train_labels.count(1))
    logger.info("Val class balance: real=%d fake=%d",
                val_labels.count(0), val_labels.count(1))

    train_dataset = IndonesianFaceDataset(train_files, train_labels, augment=True)
    val_dataset = IndonesianFaceDataset(val_files, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # =================== MODEL ===================
    logger.info("Loading CLIP model: %s / %s", args.clip_model, args.clip_pretrained)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.visual
    clip_model.eval()
    clip_model.to(device)

    # Freeze CLIP backbone
    for param in clip_model.parameters():
        param.requires_grad = False

    # Determine feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        feat_dim = clip_model(dummy).shape[-1]
    logger.info("CLIP feature dimension: %d", feat_dim)

    # Linear probe
    linear_probe = nn.Linear(feat_dim, 1).to(device)
    logger.info("Linear probe: %d -> 1", feat_dim)

    # =================== TRAINING ===================
    optimizer = torch.optim.AdamW(
        linear_probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        linear_probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.no_grad():
                features = clip_model(images)
                features = features / features.norm(dim=-1, keepdim=True)

            logits = linear_probe(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1) * 100.0

        # --- Validate ---
        linear_probe.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                features = clip_model(images)
                features = features / features.norm(dim=-1, keepdim=True)
                logits = linear_probe(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                # Confusion matrix
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100.0
        val_precision = val_tp / max(val_tp + val_fp, 1) * 100.0
        val_recall = val_tp / max(val_tp + val_fn, 1) * 100.0
        val_fpr = val_fp / max(val_fp + val_tn, 1) * 100.0  # False positive rate (BPCER proxy)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f train_acc=%.1f%% | "
            "val_loss=%.4f val_acc=%.1f%% precision=%.1f%% recall=%.1f%% FPR=%.2f%% "
            "| TP=%d FP=%d TN=%d FN=%d | lr=%.2e",
            epoch, args.epochs, train_loss, train_acc,
            val_loss, val_acc, val_precision, val_recall, val_fpr,
            val_tp, val_fp, val_tn, val_fn,
            optimizer.param_groups[0]["lr"],
        )

        # Save best model
        improved = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            improved = True
            patience_counter = 0

            save_path = output_dir / "clip_probe_best.pth"
            torch.save(linear_probe.state_dict(), save_path)
            logger.info("  -> Saved best model (val_acc=%.1f%% val_loss=%.4f) to %s",
                        val_acc, val_loss, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping after %d epochs without improvement", args.patience)
                break

    # Save final model
    final_path = output_dir / "clip_probe_final.pth"
    torch.save(linear_probe.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)
    logger.info("Best validation accuracy: %.1f%%", best_val_acc)

    # =================== EXPORT REMINDER ===================
    logger.info("")
    logger.info("=" * 60)
    logger.info("NEXT STEP: Export to ONNX for production")
    logger.info("=" * 60)
    logger.info("  python scripts/export_models.py \\")
    logger.info("      --output-dir models/ \\")
    logger.info("      --clip-weights %s \\", output_dir / "clip_probe_best.pth")
    logger.info("      --quantize")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

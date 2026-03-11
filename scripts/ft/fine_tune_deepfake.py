#!/usr/bin/env python3
"""
Fine-tune Deepfake Detection — Vison AI
========================================
Models output:
  deepfake_efficientnet_indonesia_int8.onnx  — EfficientNet-B4
  deepfake_vit_indonesia_int8.onnx           — ViT-B/16
  deepfake_npr_indonesia_int8.onnx           — ResNet50 (NPR)

Target metrics (fintech Indonesia):
  Accuracy > 99.5% | FPR < 0.3% | FNR < 0.5%

Usage:
  python scripts/fine_tune_deepfake.py --task all --base-dir /root/vison-training/benchmark_data
  python scripts/fine_tune_deepfake.py --task efficientnet --epochs 25 --batch-size 64 --mixed-precision
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 128)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def load_samples(real_dir, fake_dir, dl_dir, max_per_class=15000):
    real = [p for p in real_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    fake = [p for p in fake_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    # Extra fake sources
    for extra in ["fakefaces", "fakefaces2", "deepfakefaces"]:
        d = dl_dir / extra
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in IMG_EXTS and "fake" in str(p).lower():
                    fake.append(p)

    random.shuffle(real); random.shuffle(fake)
    real = real[:max_per_class]; fake = fake[:max_per_class]
    logger.info("  Real: %d | Fake: %d", len(real), len(fake))

    samples = [(p, 0) for p in real] + [(p, 1) for p in fake]
    random.shuffle(samples)
    split = int(len(samples) * 0.85)
    return samples[:split], samples[split:]


def get_tf(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


def train_model(model, train_loader, val_loader, device, epochs, save_path, name, use_amp=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1)
    scaler   = GradScaler() if use_amp else None
    best_acc = 0.0; best_fpr = 1.0; no_improve = 0; patience = 6

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    loss = criterion(model(imgs).squeeze(1), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = criterion(model(imgs).squeeze(1), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        probs, lbls = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                p = torch.sigmoid(model(imgs.to(device)).squeeze(1))
                probs.extend(p.cpu().numpy()); lbls.extend(labels.numpy())

        probs = np.array(probs); lbls = np.array(lbls)

        # Find optimal threshold — penalize FPR heavily
        best_t, best_combo = 0.5, -999
        for t in np.arange(0.3, 0.8, 0.005):
            preds = (probs >= t).astype(int)
            acc   = (preds == lbls).mean() * 100
            rm    = lbls == 0
            fpr   = (preds[rm] == 1).mean() * 100 if rm.sum() > 0 else 100
            fnr   = (preds[lbls==1] == 0).mean() * 100 if (lbls==1).sum() > 0 else 100
            combo = acc - fpr * 3 - fnr
            if combo > best_combo: best_combo, best_t = combo, t

        preds = (probs >= best_t).astype(int)
        acc   = (preds == lbls).mean() * 100
        rm    = lbls == 0; fm = lbls == 1
        fpr   = (preds[rm] == 1).mean() * 100 if rm.sum() > 0 else 0.0
        fnr   = (preds[fm] == 0).mean() * 100 if fm.sum() > 0 else 0.0

        logger.info("Epoch %2d/%d loss=%.4f | Acc=%.2f%% FPR=%.2f%% FNR=%.2f%% thresh=%.3f",
                    epoch, epochs, total_loss/len(train_loader), acc, fpr, fnr, best_t)

        if acc > best_acc or (abs(acc-best_acc) < 0.1 and fpr < best_fpr):
            best_acc = acc; best_fpr = fpr; no_improve = 0
            torch.save({"model_state": model.state_dict(), "threshold": best_t,
                        "acc": acc, "fpr": fpr, "fnr": fnr}, save_path)
            logger.info("  → Saved | Acc=%.2f%% FPR=%.2f%% FNR=%.2f%%", acc, fpr, fnr)
        else:
            no_improve += 1
            if no_improve >= patience and epoch > 10:
                logger.info("  Early stopping epoch %d", epoch); break

    logger.info("Done %s | Best Acc=%.2f%% FPR=%.2f%%", name, best_acc, best_fpr)


def export_onnx(model, stem, output_dir, threshold=0.5):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort

    model.eval().cpu()
    onnx_path = output_dir / f"{stem}.onnx"
    torch.onnx.export(model, torch.randn(1,3,224,224), str(onnx_path),
                      input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input":{0:"batch"},"logit":{0:"batch"}},
                      opset_version=17)
    onnx.checker.check_model(str(onnx_path))
    logger.info("✅ ONNX: %s (%.1fMB)", onnx_path.name, onnx_path.stat().st_size/1e6)

    int8 = output_dir / f"{stem}_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8), weight_type=QuantType.QInt8)
    logger.info("✅ INT8: %s (%.1fMB)", int8.name, int8.stat().st_size/1e6)
    (output_dir / f"{stem}_threshold.txt").write_text(str(round(threshold,4)))

    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"input": np.random.randn(1,3,224,224).astype(np.float32)})[0]
    logger.info("   Test logit=%.4f prob=%.4f ✅", out[0][0], 1/(1+np.exp(-out[0][0])))


def build_efficientnet():
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f,256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256,1))
    return m

def build_vit():
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    in_f = m.heads.head.in_features
    m.heads = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,1))
    return m

def build_npr():
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,1))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["efficientnet","vit","npr","all"], default="all")
    parser.add_argument("--base-dir", default="benchmark_data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--max-samples", type=int, default=15000)
    args = parser.parse_args()

    BASE       = Path(args.base_dir)
    REAL_DIR   = BASE / "id" / "real"
    FAKE_DIR   = BASE / "id" / "ai"
    DL_DIR     = BASE / "_downloads"
    OUTPUT_DIR = BASE / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.mixed_precision and device.type == "cuda"
    logger.info("Device: %s | AMP: %s | GPU: %s", device, use_amp,
                torch.cuda.get_device_name(0) if device.type=="cuda" else "N/A")

    real_c = sum(1 for p in REAL_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    fake_c = sum(1 for p in FAKE_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    logger.info("Dataset — Real: %d | Fake: %d", real_c, fake_c)
    if real_c < 500 or fake_c < 500:
        logger.error("Insufficient data. Run prepare_dataset.py first.")
        return

    train_s, val_s = load_samples(REAL_DIR, FAKE_DIR, DL_DIR, args.max_samples)
    logger.info("Train: %d | Val: %d", len(train_s), len(val_s))

    labels  = [s[1] for s in train_s]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1.0/counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(DeepfakeDataset(train_s, get_tf(True)),
                              batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(DeepfakeDataset(val_s, get_tf(False)),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    builders = {
        "efficientnet": (build_efficientnet, "deepfake_efficientnet_indonesia"),
        "vit":          (build_vit,          "deepfake_vit_indonesia"),
        "npr":          (build_npr,          "deepfake_npr_indonesia"),
    }
    tasks = list(builders.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        logger.info("\n%s\nTRAINING: %s\n%s", "="*60, task.upper(), "="*60)
        build_fn, stem = builders[task]
        model     = build_fn().to(device)
        save_path = OUTPUT_DIR / f"{stem}.pth"
        train_model(model, train_loader, val_loader, device,
                    args.epochs, save_path, task, use_amp)
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        export_onnx(model, stem, OUTPUT_DIR, ckpt.get("threshold", 0.5))

    logger.info("\n✅ DEEPFAKE DONE")


if __name__ == "__main__":
    main()
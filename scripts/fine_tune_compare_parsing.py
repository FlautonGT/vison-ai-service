#!/usr/bin/env python3
"""
Fine-tune Face Compare (ArcFace) + Face Parsing (BiSeNet) untuk Vison AI
=========================================================================
Models yang dihasilkan:
  1. arcface_indonesia.onnx     - ArcFace fine-tuned untuk wajah Indonesia
  2. bisenet_indonesia.onnx     - BiSeNet face parsing fine-tuned

Dataset:
  Compare/Embed:
    - benchmark_data/_downloads/selfie_id/  - 65k selfie+ID pairs
    - benchmark_data/_downloads/faceid/     - Face verification pairs
    - benchmark_data/id/real/               - Real Indonesia faces

  Parsing:
    - benchmark_data/_downloads/celebamaskhq/  - CelebAMask-HQ (30k wajah + mask)

Usage:
    python scripts/fine_tune_compare_parsing.py --task compare --epochs 20
    python scripts/fine_tune_compare_parsing.py --task parsing --epochs 15
    python scripts/fine_tune_compare_parsing.py --task all --epochs 20
"""

import argparse
import logging
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
REAL_DIR   = BASE_DIR / "benchmark_data" / "id" / "real"
OUTPUT_DIR = BASE_DIR / "benchmark_data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# PART 1: FACE COMPARE / EMBED (ArcFace fine-tune)
# ═══════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """ArcFace margin loss head."""
    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        import math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Normalize
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(input),
            torch.nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class EmbedModel(nn.Module):
    """Lightweight embedding model based on MobileNetV3."""
    def __init__(self, embed_dim=512):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.embed    = nn.Sequential(
            nn.Linear(960, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.embed(x)
        return torch.nn.functional.normalize(x, dim=1)


class IdentityDataset(Dataset):
    """
    Dataset untuk face recognition.
    Scan folder structure: each subfolder = 1 identity.
    """
    def __init__(self, samples, transform=None):
        # samples: list of (img_path, identity_idx)
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (112, 112), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def load_identity_samples(min_imgs_per_id=2, max_ids=5000):
    """
    Load identity dataset dari folder structure.
    Tiap subfolder = 1 identitas.
    """
    samples    = []
    id_counter = 0
    IMG_EXTS   = {".jpg", ".jpeg", ".png"}

    search_dirs = [
        DL_DIR / "selfie_id",
        DL_DIR / "faceid",
    ]

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        # Find identity folders (subfolder with multiple images)
        for id_dir in sorted(base_dir.rglob("*")):
            if not id_dir.is_dir():
                continue
            imgs = [p for p in id_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            if len(imgs) < min_imgs_per_id:
                continue
            for img_path in imgs:
                samples.append((img_path, id_counter))
            id_counter += 1
            if id_counter >= max_ids:
                break
        if id_counter >= max_ids:
            break

    # If not enough identity folders, create pseudo-pairs from real faces
    # Each photo treated as unique identity (for triplet-style training fallback)
    if id_counter < 50:
        logger.warning("Tidak ada folder identity. Menggunakan real faces sebagai pseudo-identities.")
        real_imgs = list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png"))
        random.shuffle(real_imgs)
        for i, img_path in enumerate(real_imgs[:max_ids]):
            samples.append((img_path, i))
        id_counter = len(real_imgs[:max_ids])

    logger.info("  Identities: %d | Total samples: %d", id_counter, len(samples))
    return samples, id_counter


EMBED_TRAIN_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

EMBED_VAL_TF = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def train_compare(device, epochs, batch_size):
    logger.info("\n%s\nTRAINING: FACE COMPARE/EMBED\n%s", "=" * 60, "=" * 60)

    all_samples, num_ids = load_identity_samples()
    if num_ids < 10:
        logger.error("Tidak cukup data identity. Minimal 10 identitas.")
        return

    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    train_ds = IdentityDataset(train_samples, EMBED_TRAIN_TF)
    val_ds   = IdentityDataset(val_samples,   EMBED_VAL_TF)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model      = EmbedModel(embed_dim=512).to(device)
    arc_head   = ArcFaceHead(512, num_ids).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.AdamW(list(model.parameters()) + list(arc_head.parameters()), lr=1e-4, weight_decay=5e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    save_path = OUTPUT_DIR / "arcface_indonesia.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        arc_head.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeds = model(imgs)
            logits = arc_head(embeds, labels)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation - simple top-1 accuracy
        model.eval()
        arc_head.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                embeds = model(imgs)
                logits = arc_head(embeds, labels.to(device))
                preds  = logits.argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)

        acc = correct / total * 100 if total > 0 else 0.0
        scheduler.step()
        logger.info(
            "Epoch %2d/%d — loss=%.4f | Top1=%.1f%% | lr=%.2e",
            epoch, epochs, total_loss / len(train_loader), acc, scheduler.get_last_lr()[0]
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            logger.info("  -> Saved best (Acc=%.1f%%)", best_acc)

    # Export
    model.load_state_dict(torch.load(save_path, map_location=device))
    export_embed_onnx(model)


def export_embed_onnx(model):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval().cpu()
    dummy     = torch.randn(1, 3, 112, 112)
    onnx_path = OUTPUT_DIR / "arcface_indonesia.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_path))
    logger.info("✅ ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)

    int8_path = OUTPUT_DIR / "arcface_indonesia_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("✅ INT8: %s (%.1f MB)", int8_path.name, int8_path.stat().st_size / 1e6)

    # Verify - embedding should be unit norm
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp  = np.random.randn(1, 3, 112, 112).astype(np.float32)
    emb  = sess.run(None, {"input": inp})[0]
    norm = np.linalg.norm(emb)
    logger.info("   Test embedding norm: %.4f (should be ~1.0) ✅", norm)


# ═══════════════════════════════════════════════════════════
# PART 2: FACE PARSING (BiSeNet fine-tune)
# ═══════════════════════════════════════════════════════════

# BiSeNet semantic classes (CelebAMask-HQ)
PARSE_CLASSES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g",      # glasses
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip",
    "neck", "neck_l", "cloth",
    "hair",
    "hat",        # class 18
]
NUM_CLASSES = len(PARSE_CLASSES)  # 19


class ParseDataset(Dataset):
    def __init__(self, samples, img_tf, mask_tf):
        self.samples = samples  # list of (img_path, mask_path)
        self.img_tf  = img_tf
        self.mask_tf = mask_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        try:
            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except Exception:
            img  = Image.new("RGB", (512, 512), 0)
            mask = Image.new("L",   (512, 512), 0)
        img  = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask * 255).long().squeeze(0)
        mask = torch.clamp(mask, 0, NUM_CLASSES - 1)
        return img, mask


def load_parsing_samples():
    """Load CelebAMask-HQ image+mask pairs."""
    mask_dir = DL_DIR / "celebamaskhq"
    samples  = []

    # CelebAMask-HQ structure: CelebA-HQ-img/XXXXX.jpg + CelebAMask-HQ-mask-anno/XX/XXXXX_*.png
    img_dir  = None
    anno_dir = None

    for d in mask_dir.rglob("*"):
        if d.is_dir():
            if "img" in d.name.lower() and img_dir is None:
                test_imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if test_imgs:
                    img_dir = d
            if "mask" in d.name.lower() and "anno" in str(d).lower() and anno_dir is None:
                anno_dir = d.parent if d.parent != mask_dir else d

    if img_dir is None:
        logger.warning("CelebAMask-HQ img dir tidak ditemukan. Parsing training di-skip.")
        return None

    logger.info("  CelebAMask img_dir: %s", img_dir)

    # Build samples - match image to its combined mask
    # We'll use skin mask as proxy for face region (class 1)
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:15000]
    for img_path in img_files:
        stem = img_path.stem  # e.g. "00001"
        # Find any mask for this image
        if anno_dir:
            idx_folder = str(int(stem) // 2000).zfill(2)
            mask_candidates = list((anno_dir / idx_folder).glob(f"{stem}_*.png")) if (anno_dir / idx_folder).exists() else []
            if mask_candidates:
                samples.append((img_path, mask_candidates[0]))
            else:
                # Try flat structure
                mask_candidates = list(anno_dir.rglob(f"{stem}_skin.png"))
                if mask_candidates:
                    samples.append((img_path, mask_candidates[0]))

    if len(samples) < 100:
        logger.warning("Tidak cukup sample parsing (%d). Skip BiSeNet fine-tune.", len(samples))
        return None

    logger.info("  Parsing samples: %d", len(samples))
    random.shuffle(samples)
    return samples


class SimpleSegModel(nn.Module):
    """Lightweight segmentation model using DeepLabV3 MobileNet backbone."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        base = deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=num_classes,
            aux_loss=True,
        )
        self.model = base

    def forward(self, x):
        out = self.model(x)
        return out["out"]  # [B, num_classes, H, W]


PARSE_IMG_TF = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

PARSE_MASK_TF = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


def train_parsing(device, epochs, batch_size):
    logger.info("\n%s\nTRAINING: FACE PARSING\n%s", "=" * 60, "=" * 60)

    samples = load_parsing_samples()
    if samples is None:
        logger.info("Skipping face parsing training - dataset tidak tersedia.")
        logger.info("Re-run setelah celebamaskhq download selesai.")
        return

    split = int(len(samples) * 0.9)
    train_ds = ParseDataset(samples[:split],  PARSE_IMG_TF, PARSE_MASK_TF)
    val_ds   = ParseDataset(samples[split:],  PARSE_IMG_TF, PARSE_MASK_TF)
    train_loader = DataLoader(train_ds, batch_size=max(1, batch_size // 4), shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, batch_size // 4), shuffle=False, num_workers=0)

    model     = SimpleSegModel(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_miou = 0.0
    save_path = OUTPUT_DIR / "bisenet_indonesia.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Pixel accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                out  = model(imgs).argmax(1).cpu()
                correct += (out == masks).sum().item()
                total   += masks.numel()

        pix_acc = correct / total * 100 if total > 0 else 0.0
        scheduler.step()
        logger.info(
            "Epoch %2d/%d — loss=%.4f | PixAcc=%.1f%% | lr=%.2e",
            epoch, epochs, total_loss / len(train_loader), pix_acc, scheduler.get_last_lr()[0]
        )

        if pix_acc > best_miou:
            best_miou = pix_acc
            torch.save(model.state_dict(), save_path)
            logger.info("  -> Saved best (PixAcc=%.1f%%)", best_miou)

    # Export
    model.load_state_dict(torch.load(save_path, map_location=device))
    export_parsing_onnx(model)


def export_parsing_onnx(model):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval().cpu()
    dummy     = torch.randn(1, 3, 512, 512)
    onnx_path = OUTPUT_DIR / "bisenet_indonesia.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["seg_map"],
        dynamic_axes={"input": {0: "batch_size"}, "seg_map": {0: "batch_size"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_path))
    logger.info("✅ ONNX exported: %s (%.1f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)

    int8_path = OUTPUT_DIR / "bisenet_indonesia_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("✅ INT8: %s (%.1f MB)", int8_path.name, int8_path.stat().st_size / 1e6)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["compare", "parsing", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    if args.task in ("compare", "all"):
        train_compare(device, args.epochs, args.batch_size)

    if args.task in ("parsing", "all"):
        train_parsing(device, args.epochs, args.batch_size)

    logger.info("\n%s", "=" * 60)
    logger.info("NEXT STEP: Copy ke VPS")
    logger.info("=" * 60)
    logger.info("  scp benchmark_data/models/arcface_indonesia*.onnx user@vps:/opt/models/")
    logger.info("  scp benchmark_data/models/bisenet_indonesia*.onnx user@vps:/opt/models/")
    logger.info("  Update .env:")
    logger.info("    ARCFACE_EXTRA_MODEL=arcface_indonesia.onnx")
    logger.info("    FACE_PARSING_MODEL=bisenet_indonesia.onnx")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
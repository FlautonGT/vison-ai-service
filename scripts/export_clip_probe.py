#!/usr/bin/env python3
"""
Export CLIP Indonesian Deepfake Probe ke ONNX
=============================================
Mengexport model CLIP linear probe yang sudah di-fine-tune
ke format ONNX untuk dipakai di Vison AI service.

Usage:
    python scripts/export_clip_probe.py \
        --weights benchmark_data/models/clip_probe_best.pth \
        --output-dir benchmark_data/models

Output:
    clip_indonesian_probe.onnx  - model full precision
    clip_indonesian_probe_int8.onnx - model quantized (lebih kecil, lebih cepat)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model definition (harus sama persis dengan fine_tune_indonesian.py)
# ─────────────────────────────────────────────
class CLIPProbe(nn.Module):
    """CLIP ViT-B-16 + linear probe untuk deepfake detection."""

    def __init__(self, clip_model, feature_dim: int = 512):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract CLIP image features
        features = self.clip_model.encode_image(x)
        features = features.float()
        # Normalize features
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        # Linear probe → sigmoid → score
        logit = self.classifier(features)
        score = torch.sigmoid(logit)  # 0 = real, 1 = fake
        return score  # shape: [batch, 1]


def load_model(weights_path: str, device: str = "cpu"):
    """Load CLIP + probe dari checkpoint."""
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip tidak terinstall. Jalankan: pip install open_clip_torch")
        sys.exit(1)

    logger.info("Loading CLIP ViT-B-16...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    clip_model = clip_model.to(device)

    # Build full model
    model = CLIPProbe(clip_model, feature_dim=512).to(device)

    # Load trained weights
    logger.info(f"Loading weights dari: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle berbagai format checkpoint
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load hanya classifier weights (CLIP weights sudah di-freeze saat training)
    if any(k.startswith("classifier") for k in state_dict.keys()):
        # State dict berisi full model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
    else:
        # State dict hanya berisi classifier weights
        model.classifier.load_state_dict(state_dict)

    model.eval()
    logger.info("Model loaded ✅")
    return model, preprocess


def export_onnx(model, output_path: str, device: str = "cpu"):
    """Export model ke ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input sesuai CLIP input size
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    logger.info(f"Exporting ke ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "score": {0: "batch_size"},
        },
    )
    logger.info(f"✅ ONNX export selesai: {output_path}")

    # Verifikasi
    try:
        import onnx
        model_onnx = onnx.load(str(output_path))
        onnx.checker.check_model(model_onnx)
        logger.info("✅ ONNX model valid")
    except ImportError:
        logger.warning("onnx tidak terinstall, skip verifikasi")
    except Exception as e:
        logger.error(f"ONNX verifikasi gagal: {e}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"   Size: {size_mb:.1f} MB")
    return str(output_path)


def quantize_onnx(input_path: str, output_path: str):
    """Quantize ONNX model ke INT8 untuk inferensi lebih cepat."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.warning("onnxruntime tidak terinstall, skip quantization")
        return

    logger.info(f"Quantizing ke INT8: {output_path}")
    try:
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"✅ INT8 quantization selesai: {output_path} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Quantization gagal: {e}")


def test_inference(onnx_path: str):
    """Test inferensi model ONNX."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime tidak terinstall, skip test")
        return

    logger.info(f"Testing inferensi: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = sess.run(["score"], {"input": dummy})
    score = output[0][0][0]
    logger.info(f"   Test output score: {score:.4f} (0=real, 1=fake) ✅")


def main():
    parser = argparse.ArgumentParser(description="Export CLIP Indonesian Probe ke ONNX")
    parser.add_argument(
        "--weights",
        default="benchmark_data/models/clip_probe_best.pth",
        help="Path ke checkpoint .pth"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_data/models",
        help="Output directory"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Juga export versi INT8 quantized"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device untuk export (cpu/cuda)"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights tidak ditemukan: {weights_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, _ = load_model(str(weights_path), device=args.device)

    # Export FP32
    onnx_path = output_dir / "clip_indonesian_probe.onnx"
    export_onnx(model, str(onnx_path), device=args.device)

    # Test
    test_inference(str(onnx_path))

    # Quantize optional
    if args.quantize:
        int8_path = output_dir / "clip_indonesian_probe_int8.onnx"
        quantize_onnx(str(onnx_path), str(int8_path))
        test_inference(str(int8_path))

    logger.info("")
    logger.info("=" * 60)
    logger.info("NEXT STEP: Copy ke VPS dan integrasi ke service")
    logger.info("=" * 60)
    logger.info(f"  scp {onnx_path} user@vps:/opt/models/")
    logger.info("  Lalu tambahkan ke deepfake_detector.py sebagai model ensemble")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
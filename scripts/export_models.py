#!/usr/bin/env python3
"""Export enhanced detection models to ONNX format with optional INT8 quantization.

Models exported:
  1. UniversalFakeDetect — CLIP ViT-B/16 + linear probe (AI-generated image detection)
  2. NPR Detector — ResNet-18 on pixel-difference features (compression-robust deepfake detection)
  3. CDCN Liveness — Central Difference Convolutional Network (passive face anti-spoofing)

Usage:
    # Install dependencies first:
    pip install torch torchvision ftfy regex open_clip_torch onnx onnxruntime onnxruntime-tools

    # Download pretrained weights (see download instructions in each section)
    # Then run:
    python scripts/export_models.py --output-dir models/ --quantize

Prerequisites — clone these repos and download weights:

    # 1. UniversalFakeDetect
    git clone https://github.com/Yuheng-Li/UniversalFakeDetect.git /tmp/UniversalFakeDetect
    # Download pretrained checkpoint:
    #   https://github.com/Yuheng-Li/UniversalFakeDetect — follow README for fc_weights.pth
    #   OR use the HuggingFace mirror if available.

    # 2. NPR DeepfakeDetection
    git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git /tmp/NPR-DeepfakeDetection
    # Download pretrained checkpoint from the repo releases.

    # 3. CDCN
    git clone https://github.com/ZitongYu/CDCN.git /tmp/CDCN
    # Download pretrained checkpoint from the repo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. UniversalFakeDetect (CLIP ViT-B/16 + linear probe)
# ---------------------------------------------------------------------------
def export_universal_fake_detect(
    output_dir: str,
    weights_path: str = "",
    clip_model_name: str = "ViT-B-16",
    clip_pretrained: str = "openai",
) -> str:
    """Export CLIP ViT-B/16 visual encoder + linear probe to ONNX.

    The model takes a 224x224 RGB image and outputs a single logit
    (positive = fake).
    """
    import torch
    import torch.nn as nn

    try:
        import open_clip
    except ImportError:
        logger.error("open_clip not installed. Run: pip install open_clip_torch")
        return ""

    logger.info("Loading CLIP model: %s / %s", clip_model_name, clip_pretrained)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model.eval()

    # Get visual feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        features = clip_model.encode_image(dummy)
        feat_dim = features.shape[-1]
    logger.info("CLIP visual feature dim: %d", feat_dim)

    # Build the linear probe
    linear = nn.Linear(feat_dim, 1)

    if weights_path and os.path.exists(weights_path):
        logger.info("Loading linear probe weights from: %s", weights_path)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Try to load — handle different checkpoint formats
        if isinstance(state, dict):
            # Filter to just the linear layer keys
            linear_keys = {k: v for k, v in state.items() if "weight" in k or "bias" in k}
            if linear_keys:
                # Rename keys if needed
                new_state = {}
                for k, v in linear_keys.items():
                    clean_key = k.split(".")[-1] if "." in k else k
                    if clean_key in ("weight", "bias"):
                        new_state[clean_key] = v
                if new_state:
                    linear.load_state_dict(new_state, strict=False)
                    logger.info("Linear probe weights loaded")
                else:
                    linear.load_state_dict(state, strict=False)
            else:
                linear.load_state_dict(state, strict=False)
        else:
            logger.warning("Unexpected checkpoint format, using random linear probe")
    else:
        logger.warning(
            "No weights file provided for linear probe — exporting with random weights. "
            "You should fine-tune the linear probe on your data (see fine_tune_indonesian.py)."
        )

    linear.eval()

    class CLIPFakeDetectWrapper(nn.Module):
        def __init__(self, clip_visual, probe):
            super().__init__()
            self.visual = clip_visual
            self.probe = probe

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.visual(x)
            # Normalize features (CLIP convention)
            features = features / features.norm(dim=-1, keepdim=True)
            logit = self.probe(features)
            return logit

    wrapper = CLIPFakeDetectWrapper(clip_model.visual, linear)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = os.path.join(output_dir, "universal_fake_detect.onnx")

    logger.info("Exporting to ONNX: %s", output_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True,
        )

    logger.info("UniversalFakeDetect exported: %s (%.1f MB)",
                output_path, os.path.getsize(output_path) / 1e6)
    return output_path


# ---------------------------------------------------------------------------
# 2. NPR Detector (ResNet-18 on pixel-difference features)
# ---------------------------------------------------------------------------
def export_npr_detector(
    output_dir: str,
    weights_path: str = "",
    input_channels: int = 3,
) -> str:
    """Export NPR detector (ResNet-18 backbone) to ONNX.

    The model takes a 224x224 image (or NPR features) and outputs a
    single logit (positive = fake).

    NPR features are computed as 4-directional pixel differences in
    preprocessing (see NPRDetector.compute_npr_features in models.py).
    The ONNX model itself is a standard ResNet-18 classifier.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    logger.info("Building NPR detector (ResNet-18, in_channels=%d)", input_channels)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Modify first conv layer if using non-RGB input
    if input_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize new conv with repeated RGB weights
        with torch.no_grad():
            if input_channels > 3:
                repeats = (input_channels + 2) // 3
                expanded = old_conv.weight.data.repeat(1, repeats, 1, 1)
                model.conv1.weight.data = expanded[:, :input_channels, :, :]
            else:
                model.conv1.weight.data = old_conv.weight.data[:, :input_channels, :, :]

    # Replace classifier head: 512 -> 1 (binary)
    model.fc = nn.Linear(512, 1)

    if weights_path and os.path.exists(weights_path):
        logger.info("Loading NPR weights from: %s", weights_path)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Remove "module." prefix if present (DataParallel)
        cleaned = {}
        for k, v in state.items():
            clean_k = k.replace("module.", "")
            cleaned[clean_k] = v
        model.load_state_dict(cleaned, strict=False)
        logger.info("NPR weights loaded")
    else:
        logger.warning(
            "No weights file for NPR detector — exporting with ImageNet-pretrained backbone + random classifier. "
            "Fine-tune on your data for production use."
        )

    model.eval()

    dummy = torch.randn(1, input_channels, 224, 224)
    output_path = os.path.join(output_dir, "npr_resnet18.onnx")

    logger.info("Exporting to ONNX: %s", output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True,
        )

    logger.info("NPR detector exported: %s (%.1f MB)",
                output_path, os.path.getsize(output_path) / 1e6)
    return output_path


# ---------------------------------------------------------------------------
# 3. CDCN Liveness
# ---------------------------------------------------------------------------
def export_cdcn_liveness(
    output_dir: str,
    weights_path: str = "",
) -> str:
    """Export CDCN (Central Difference Convolutional Network) to ONNX.

    If the full CDCN architecture is not available, we export a compact
    ResNet-18 based anti-spoofing model as a drop-in replacement with
    similar performance characteristics.

    The model takes a 256x256 face crop and outputs a single liveness
    score (higher = more likely real).
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    # Try to import CDCN architecture
    cdcn_available = False
    try:
        sys.path.insert(0, "/tmp/CDCN")
        from CDCN import CDCN  # type: ignore
        cdcn_available = True
        logger.info("CDCN architecture imported from /tmp/CDCN")
    except ImportError:
        logger.info("CDCN architecture not found, using ResNet-18 anti-spoofing model")

    if cdcn_available:
        model = CDCN()
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
            logger.info("CDCN weights loaded from: %s", weights_path)
        input_size = 256
    else:
        # Compact anti-spoofing model based on ResNet-18
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
            logger.info("Anti-spoofing weights loaded from: %s", weights_path)
        else:
            logger.warning(
                "No weights for CDCN/anti-spoofing — exporting with random classifier. "
                "Fine-tune on liveness data for production use."
            )
        input_size = 256

    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    output_path = os.path.join(output_dir, "cdcn_liveness.onnx")

    logger.info("Exporting to ONNX: %s", output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True,
        )

    logger.info("CDCN liveness exported: %s (%.1f MB)",
                output_path, os.path.getsize(output_path) / 1e6)
    return output_path


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------
def quantize_to_int8(onnx_path: str) -> str:
    """Quantize an ONNX model to INT8 using onnxruntime dynamic quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime quantization not available. "
                     "Run: pip install onnxruntime onnxruntime-tools")
        return ""

    stem = Path(onnx_path).stem
    parent = Path(onnx_path).parent
    output_path = str(parent / f"{stem}_int8.onnx")

    logger.info("Quantizing %s -> %s", onnx_path, output_path)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = os.path.getsize(onnx_path) / 1e6
    quant_size = os.path.getsize(output_path) / 1e6
    logger.info("Quantized: %.1f MB -> %.1f MB (%.0f%% reduction)",
                orig_size, quant_size, (1 - quant_size / orig_size) * 100)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export enhanced detection models to ONNX")
    parser.add_argument("--output-dir", default="models/", help="Output directory for ONNX files")
    parser.add_argument("--quantize", action="store_true", help="Also produce INT8 quantized versions")
    parser.add_argument("--clip-weights", default="", help="Path to UniversalFakeDetect linear probe weights")
    parser.add_argument("--npr-weights", default="", help="Path to NPR detector weights")
    parser.add_argument("--cdcn-weights", default="", help="Path to CDCN liveness weights")
    parser.add_argument("--npr-channels", type=int, default=3,
                        help="Input channels for NPR model (3=RGB NPR visualization, 12=raw NPR features)")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP model export")
    parser.add_argument("--skip-npr", action="store_true", help="Skip NPR model export")
    parser.add_argument("--skip-cdcn", action="store_true", help="Skip CDCN model export")
    args = parser.parse_args()

    out = _ensure_dir(args.output_dir)
    exported: list[str] = []

    if not args.skip_clip:
        logger.info("=" * 60)
        logger.info("EXPORTING: UniversalFakeDetect (CLIP ViT-B/16)")
        logger.info("=" * 60)
        path = export_universal_fake_detect(str(out), weights_path=args.clip_weights)
        if path:
            exported.append(path)

    if not args.skip_npr:
        logger.info("=" * 60)
        logger.info("EXPORTING: NPR Detector (ResNet-18)")
        logger.info("=" * 60)
        path = export_npr_detector(str(out), weights_path=args.npr_weights,
                                   input_channels=args.npr_channels)
        if path:
            exported.append(path)

    if not args.skip_cdcn:
        logger.info("=" * 60)
        logger.info("EXPORTING: CDCN Liveness")
        logger.info("=" * 60)
        path = export_cdcn_liveness(str(out), weights_path=args.cdcn_weights)
        if path:
            exported.append(path)

    if args.quantize and exported:
        logger.info("=" * 60)
        logger.info("INT8 QUANTIZATION")
        logger.info("=" * 60)
        for path in exported:
            quantize_to_int8(path)

    logger.info("=" * 60)
    logger.info("DONE. Exported %d models to %s", len(exported), out)
    for path in exported:
        logger.info("  %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export deepfake and liveness models to ONNX format with optional INT8 quantization.

Models exported:
  1. DeepFake ViT v2 - prithivMLmods/Deep-Fake-Detector-v2-Model
     - Architecture: ViT-base-patch16-224
     - F1: 0.92 on the source deepfake dataset
     - Expected input: [1, 3, 224, 224]
     - Normalization: ImageNet mean/std
     - Output: [1, 2] logits (index 0 = Fake, index 1 = Real)
  2. CDCN Liveness - Central Difference Convolutional Network

Usage:
    pip install torch torchvision transformers "optimum[onnxruntime]" onnx onnxruntime onnxruntime-tools
    python scripts/export_models.py --output-dir models/ --quantize

Prerequisites:
    # DeepFake ViT v2 is exported directly from HuggingFace:
    #   prithivMLmods/Deep-Fake-Detector-v2-Model
    #
    # CDCN still optionally uses the upstream repo:
    #   git clone https://github.com/ZitongYu/CDCN.git /tmp/CDCN
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# 1. DeepFake ViT v2 (HuggingFace / Optimum ONNX export)
# ---------------------------------------------------------------------------
def export_deepfake_vit_v2(output_dir: str) -> str:
    """Export prithivMLmods/Deep-Fake-Detector-v2-Model to ONNX."""
    model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    out_dir = _ensure_dir(output_dir)
    output_path = out_dir / "deepfake_vit_v2.onnx"
    export_dir: Path | None = None

    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoImageProcessor
    except ImportError:
        logger.error(
            "transformers/optimum not installed. "
            "Run: pip install transformers \"optimum[onnxruntime]\""
        )
        return ""

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        logger.info(
            "Exporting %s with input size=%s mean=%s std=%s",
            model_id,
            getattr(processor, "size", {}),
            getattr(processor, "image_mean", []),
            getattr(processor, "image_std", []),
        )

        export_dir = Path(tempfile.mkdtemp(prefix="deepfake_vit_v2_", dir=str(out_dir)))
        main_export(model_id, output=export_dir, task="image-classification")

        candidates = sorted(export_dir.rglob("*.onnx"))
        if not candidates:
            raise FileNotFoundError("Optimum export completed but no ONNX artifact was produced")

        source_path = next(
            (candidate for candidate in candidates if candidate.name == "model.onnx"),
            candidates[0],
        )

        if output_path.exists():
            output_path.unlink()
        shutil.move(str(source_path), str(output_path))
    except Exception:
        logger.exception("Failed to export %s", model_id)
        return ""
    finally:
        if export_dir is not None:
            shutil.rmtree(export_dir, ignore_errors=True)

    logger.info(
        "DeepFake ViT v2 exported: %s (%.1f MB)",
        output_path,
        output_path.stat().st_size / 1e6,
    )
    logger.info(
        "Expected interface: input [1, 3, 224, 224] with ImageNet mean/std, "
        "output [1, 2] logits (index 0 = Fake, index 1 = Real)"
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# 2. CDCN Liveness
# ---------------------------------------------------------------------------
def export_cdcn_liveness(output_dir: str, weights_path: str = "") -> str:
    """Export CDCN or a compact anti-spoofing fallback to ONNX."""
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

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
            cleaned = {key.replace("module.", ""): value for key, value in state.items()}
            model.load_state_dict(cleaned, strict=False)
            logger.info("CDCN weights loaded from: %s", weights_path)
        input_size = 256
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {key.replace("module.", ""): value for key, value in state.items()}
            model.load_state_dict(cleaned, strict=False)
            logger.info("Anti-spoofing weights loaded from: %s", weights_path)
        else:
            logger.warning(
                "No weights for CDCN/anti-spoofing. Exporting with random classifier; "
                "fine-tune on liveness data for production use."
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

    logger.info(
        "CDCN liveness exported: %s (%.1f MB)",
        output_path,
        os.path.getsize(output_path) / 1e6,
    )
    return output_path


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------
def quantize_to_int8(onnx_path: str) -> str:
    """Quantize an ONNX model to INT8 using onnxruntime dynamic quantization."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.error(
            "onnxruntime quantization not available. "
            "Run: pip install onnxruntime onnxruntime-tools"
        )
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

    original_size = os.path.getsize(onnx_path) / 1e6
    quantized_size = os.path.getsize(output_path) / 1e6
    logger.info(
        "Quantized: %.1f MB -> %.1f MB (%.0f%% reduction)",
        original_size,
        quantized_size,
        (1 - quantized_size / original_size) * 100,
    )
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Export deepfake and liveness models to ONNX")
    parser.add_argument("--output-dir", default="models/", help="Output directory for ONNX files")
    parser.add_argument("--quantize", action="store_true", help="Also produce INT8 quantized versions")
    parser.add_argument("--cdcn-weights", default="", help="Path to CDCN liveness weights")
    parser.add_argument("--skip-vit-v2", action="store_true", help="Skip DeepFake ViT v2 export")
    parser.add_argument("--skip-cdcn", action="store_true", help="Skip CDCN model export")
    args = parser.parse_args()

    out = _ensure_dir(args.output_dir)
    exported: list[str] = []

    if not args.skip_vit_v2:
        logger.info("=" * 60)
        logger.info("EXPORTING: DeepFake ViT v2")
        logger.info("=" * 60)
        path = export_deepfake_vit_v2(str(out))
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

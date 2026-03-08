#!/usr/bin/env python3
"""Generate INT8 variants for ONNX models using dynamic quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def _iter_models(model_dir: Path, include: list[str] | None) -> list[Path]:
    if include:
        return [model_dir / name for name in include]
    return sorted(path for path in model_dir.glob("*.onnx") if not path.name.endswith("_int8.onnx"))


def _quantize(src: Path, dst: Path):
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "./models"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model filenames")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    models = _iter_models(model_dir, args.models)
    if not models:
        print("No models found to quantize.")
        return

    print(f"Model dir: {model_dir}")
    for src in models:
        if not src.exists():
            print(f"[skip] missing: {src.name}")
            continue
        dst = src.with_name(f"{src.stem}_int8.onnx")
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst.name} already exists")
            continue
        try:
            _quantize(src, dst)
            src_mb = src.stat().st_size / (1024 * 1024)
            dst_mb = dst.stat().st_size / (1024 * 1024)
            saving = (1.0 - (dst_mb / max(src_mb, 1e-6))) * 100.0
            print(f"[ok] {src.name} -> {dst.name} ({src_mb:.1f}MB -> {dst_mb:.1f}MB, save {saving:.1f}%)")
        except Exception as exc:
            print(f"[fail] {src.name}: {exc}")


if __name__ == "__main__":
    main()

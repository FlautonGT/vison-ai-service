"""Download ONNX models required by the pure inference service."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models")

# (filename, url, description)
MODELS = [
    (
        "scrfd_10g_bnkps.onnx",
        # Official InsightFace buffalo_l detection model (det_10g) mapped to Tier 3 filename.
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "SCRFD 10G detector (Tier 3 default)",
    ),
    (
        "scrfd_2.5g_bnkps.onnx",
        "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx",
        "SCRFD 2.5G fallback detector",
    ),
    (
        "w600k_r50.onnx",
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        "ArcFace ResNet50 (Tier 3 default, 512-dim)",
    ),
    (
        "w600k_mbf.onnx",
        "https://huggingface.co/onnx-community/arcface-onnx/resolve/main/arcface.onnx",
        "ArcFace MobileFaceNet fallback",
    ),
    (
        "MiniFASNetV2.onnx",
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
        "Liveness ensemble primary model",
    ),
    (
        "MiniFASNetV1SE.onnx",
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
        "Liveness ensemble secondary model",
    ),
    (
        "deepfake_efficientnet_b0.onnx",
        "https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX/resolve/main/onnx/model.onnx",
        "Deepfake detector model #1",
    ),
    (
        "community_forensics_vit.onnx",
        "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT-ONNX/resolve/main/onnx/model.onnx",
        "CommunityForensics ViT deepfake - ensemble member 2, different training data from model #1",
    ),
    (
        "ai_vs_deepfake_vs_real.onnx",
        "https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX/resolve/main/onnx/model.onnx",
        "Optional AI detector #2 (3-class: artificial/deepfake/real) for stronger fusion",
    ),
    (
        "bisenet_face_parsing.onnx",
        "https://huggingface.co/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser.onnx",
        "BiSeNet face parsing model",
    ),
    (
        "genderage.onnx",
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
        "InsightFace age/gender model",
    ),
    (
        "age_gender_vit.onnx",
        "https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx",
        "ViT age/gender model (secondary ensemble for analyze endpoint)",
    ),
    (
        "fairface.onnx",
        "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx",
        "FairFace age/gender/race model (balanced demographics, good for Asian faces)",
    ),
]

# (filename, archive_url, member_path, description)
ARCHIVE_MODELS = [
    (
        "glintr100.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        "antelopev2/glintr100.onnx",
        "ArcFace R100 (antelopev2) - stronger recognition model option",
    ),
]


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vison-ai-service/2.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        status = getattr(response, "status", 200)
        if status not in (200, 206):
            raise RuntimeError(f"Unexpected HTTP status {status}")
        return response.read()


def _download(url: str, target_path: str):
    payload = _download_bytes(url)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(payload)
    os.replace(tmp_name, target_path)


def download_model(model_dir: str, filename: str, url: str, description: str):
    target_path = os.path.join(model_dir, filename)
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  [skip] {filename} ({size_mb:.1f} MB) - already exists")
        return

    print(f"  [download] {filename}")
    print(f"    url: {url}")
    print(f"    desc: {description}")
    try:
        _download(url, target_path)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  [ok] {filename} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"  [fail] {filename}: {exc}")


def download_model_from_archive(
    model_dir: str,
    filename: str,
    archive_url: str,
    member_path: str,
    description: str,
):
    target_path = os.path.join(model_dir, filename)
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  [skip] {filename} ({size_mb:.1f} MB) - already exists")
        return

    print(f"  [download] {filename} (from archive)")
    print(f"    archive: {archive_url}")
    print(f"    member: {member_path}")
    print(f"    desc: {description}")

    try:
        archive_bytes = _download_bytes(archive_url)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            with zf.open(member_path) as src, tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_name = tmp.name
                shutil.copyfileobj(src, tmp)
        os.replace(tmp_name, target_path)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  [ok] {filename} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"  [fail] {filename}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")

    for filename, url, description in MODELS:
        download_model(model_dir, filename, url, description)

    for filename, archive_url, member_path, description in ARCHIVE_MODELS:
        download_model_from_archive(model_dir, filename, archive_url, member_path, description)

    print("Done.")


if __name__ == "__main__":
    main()

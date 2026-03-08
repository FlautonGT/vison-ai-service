"""Model loading, benchmark, ensemble, and degradation checks."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if "MODEL_DIR" not in os.environ:
    local_models = ROOT_DIR / "models"
    if local_models.exists():
        os.environ["MODEL_DIR"] = str(local_models)

from app.core.config import settings
from app.core.models import FaceDetectionResult, LivenessChecker, ModelRegistry
from app.services.face_processing import FaceProcessor


def _dummy_image(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    cv2.ellipse(image, (cx, cy), (90, 120), 0, 0, 360, (180, 170, 160), -1)
    cv2.circle(image, (cx - 30, cy - 20), 8, (20, 20, 20), -1)
    cv2.circle(image, (cx + 30, cy - 20), 8, (20, 20, 20), -1)
    cv2.ellipse(image, (cx, cy + 40), (28, 10), 0, 0, 180, (40, 40, 40), 2)
    return image


def _fallback_face(image: np.ndarray) -> FaceDetectionResult:
    h, w = image.shape[:2]
    return FaceDetectionResult(
        bbox=np.array([w * 0.3, h * 0.2, w * 0.7, h * 0.85], dtype=np.float32),
        score=0.99,
        landmarks=None,
        image_width=w,
        image_height=h,
    )


def _timeit(iterations: int, fn):
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000.0)
    return {
        "avgMs": round(mean(times), 2),
        "minMs": round(min(times), 2),
        "maxMs": round(max(times), 2),
        "iterations": iterations,
    }


def _file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


async def _graceful_degradation_check() -> dict:
    model_dir = Path(settings.MODEL_DIR)
    liveness_files = [name.strip() for name in settings.LIVENESS_MODELS.split(",") if name.strip()]
    if len(liveness_files) < 2:
        return {"checked": False, "reason": "Need at least 2 liveness models configured"}

    candidate = model_dir / liveness_files[1]
    if not candidate.exists():
        return {"checked": False, "reason": f"Model file missing: {candidate.name}"}

    backup = candidate.with_suffix(candidate.suffix + ".bak")
    if backup.exists():
        return {"checked": False, "reason": f"Backup already exists: {backup.name}"}

    candidate.rename(backup)
    try:
        registry = ModelRegistry()
        await registry.load_all()
        status = registry.get_status()
        registry.unload_all()
    finally:
        backup.rename(candidate)

    ok = bool(status.get("ready")) and int(status.get("livenessModels", 0)) >= 1
    return {
        "checked": True,
        "passed": ok,
        "status": status,
    }


async def main():
    report = {"timings": {}, "models": {}, "checks": {}, "notes": []}
    registry = ModelRegistry()
    await registry.load_all()

    image = _dummy_image()
    faces = registry.face_detector.detect_all(image) if registry.face_detector else []
    face = faces[0] if faces else _fallback_face(image)
    processor = FaceProcessor(image, face)
    face_crop = processor.for_arcface()

    # Basic model information.
    model_dir = Path(settings.MODEL_DIR)
    report["models"]["providers"] = settings.ONNX_PROVIDERS
    report["models"]["registryStatus"] = registry.get_status()
    report["models"]["processMemoryMb"] = round(registry.get_memory_usage_mb(), 2)
    report["models"]["modelFileSizesMb"] = {
        "scrfd": _file_size_mb(model_dir / settings.SCRFD_MODEL),
        "arcface": _file_size_mb(model_dir / settings.ARCFACE_MODEL),
        "arcface_extra": _file_size_mb(model_dir / settings.ARCFACE_EXTRA_MODEL),
        "liveness_primary": _file_size_mb(model_dir / settings.liveness_model_list[0]) if settings.liveness_model_list else 0,
        "liveness_secondary": _file_size_mb(model_dir / settings.liveness_model_list[1]) if len(settings.liveness_model_list) > 1 else 0,
        "deepfake_1": _file_size_mb(model_dir / settings.deepfake_model_list[0]) if settings.deepfake_model_list else 0,
        "deepfake_2": _file_size_mb(model_dir / settings.deepfake_model_list[1]) if len(settings.deepfake_model_list) > 1 else 0,
        "face_parsing": _file_size_mb(model_dir / settings.FACE_PARSING_MODEL),
        "age_gender": _file_size_mb(model_dir / settings.AGE_GENDER_MODEL),
        "age_gender_vit": _file_size_mb(model_dir / settings.AGE_GENDER_VIT_MODEL),
    }

    iterations = 10

    # Benchmark detector.
    report["timings"]["faceDetector"] = _timeit(iterations, lambda: registry.face_detector.detect_all(image))

    # Benchmark recognizer.
    report["timings"]["faceRecognizer"] = _timeit(iterations, lambda: registry.face_recognizer.get_embedding(face_crop))

    # Liveness ensemble benchmark.
    liveness_crop = processor.for_liveness()
    report["timings"]["livenessEnsemble"] = _timeit(iterations, lambda: registry.liveness_checker.predict(liveness_crop))
    live_score, is_live = registry.liveness_checker.predict(liveness_crop)
    report["checks"]["livenessOutput"] = {"score": live_score, "isLive": is_live}

    # Single-model vs ensemble consistency check.
    if registry.liveness_checker and registry.liveness_checker.model_count >= 2:
        single_scores = []
        for model_info in registry.liveness_checker.models:
            single_scores.append(registry.liveness_checker._infer_single(model_info, liveness_crop))  # noqa: SLF001
        ensemble_prob = live_score / 100.0
        report["checks"]["livenessEnsembleConsistency"] = {
            "singleScores": [round(s, 4) for s in single_scores],
            "ensembleScore": round(ensemble_prob, 4),
            "insideRange": min(single_scores) <= ensemble_prob <= max(single_scores),
        }
        report["notes"].append(
            "Accuracy uplift vs single model needs labeled anti-spoof dataset; this script verifies ensemble execution and consistency."
        )
    else:
        report["checks"]["livenessEnsembleConsistency"] = {
            "singleScores": [],
            "ensembleScore": round(live_score / 100.0, 4),
            "insideRange": True,
        }

    # Deepfake ensemble benchmark.
    deepfake_crop = processor.for_deepfake()
    report["timings"]["deepfakeEnsemble"] = _timeit(iterations, lambda: registry.deepfake_detector.predict(deepfake_crop))
    report["checks"]["deepfakeOutput"] = registry.deepfake_detector.predict(deepfake_crop)

    # Face parsing benchmark.
    attrs_crop = processor.for_attributes()
    report["timings"]["faceParsing"] = _timeit(iterations, lambda: registry.face_parser.predict_attributes(attrs_crop))
    report["checks"]["faceParsingOutput"] = registry.face_parser.predict_attributes(attrs_crop)

    # Age/gender benchmark.
    age_crop = processor.for_age_gender()
    age_crop_vit = processor.for_deepfake()
    report["timings"]["ageGender"] = _timeit(
        iterations,
        lambda: registry.age_gender.predict(age_crop, face_crop=age_crop_vit),
    )
    report["checks"]["ageGenderOutput"] = registry.age_gender.predict(age_crop, face_crop=age_crop_vit)

    # Graceful degradation test: temporarily hide 1 liveness model.
    report["checks"]["gracefulDegradation"] = await _graceful_degradation_check()

    registry.unload_all()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

"""FastAPI entrypoint for stateless pure inference service."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import face_router
from app.api.middleware import AIServiceAuth, RequestHardeningMiddleware
from app.core.config import settings
from app.core.models import ModelRegistry
from app.core.service_catalog import load_service_catalog

# --- Logging configuration ---
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


async def _warmup_models(registry: ModelRegistry):
    """Run one dummy inference per model so first request is faster."""
    start = time.time()
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
    dummy_face_hires = np.zeros((224, 224, 3), dtype=np.uint8)

    try:
        if registry.face_detector and registry.face_detector.session is not None:
            registry.face_detector.detect_all(dummy_image)
    except Exception:
        logger.exception("SCRFD warmup failed")

    try:
        if registry.face_recognizer and registry.face_recognizer.session is not None:
            registry.face_recognizer.get_embedding(dummy_face)
    except Exception:
        logger.exception("ArcFace warmup failed")

    try:
        if registry.face_recognizer_extra and registry.face_recognizer_extra.session is not None:
            registry.face_recognizer_extra.get_embedding(dummy_face)
    except Exception:
        logger.exception("ArcFace extra warmup failed")

    try:
        if registry.adaface_recognizer and registry.adaface_recognizer.is_loaded:
            registry.adaface_recognizer.get_embedding(dummy_face)
    except Exception:
        logger.exception("AdaFace warmup failed")

    try:
        if registry.liveness_checker:
            registry.liveness_checker.predict(dummy_face)
    except Exception:
        logger.exception("Liveness warmup failed")

    try:
        if registry.deepfake_detector:
            registry.deepfake_detector.predict(dummy_face)
    except Exception:
        logger.exception("Deepfake warmup failed")

    try:
        if registry.ai_face_detector and registry.ai_face_detector.session is not None:
            registry.ai_face_detector.predict(dummy_image)
    except Exception:
        logger.exception("AI face detector warmup failed")

    try:
        if registry.ai_face_detector_extra and registry.ai_face_detector_extra.session is not None:
            registry.ai_face_detector_extra.predict(dummy_image)
    except Exception:
        logger.exception("AI face detector extra warmup failed")

    try:
        if registry.npr_detector and registry.npr_detector.is_loaded:
            registry.npr_detector.predict(dummy_face)
    except Exception:
        logger.exception("NPR detector warmup failed")

    try:
        if registry.clip_fake_detector and registry.clip_fake_detector.is_loaded:
            registry.clip_fake_detector.predict(dummy_face)
    except Exception:
        logger.exception("CLIP fake detector warmup failed")

    try:
        if registry.deepfake_vit_v2 and registry.deepfake_vit_v2.is_loaded:
            registry.deepfake_vit_v2.predict(dummy_face_hires)
    except Exception:
        logger.exception("Deepfake ViT v2 warmup failed")

    try:
        if registry.cdcn_liveness and registry.cdcn_liveness.is_loaded:
            registry.cdcn_liveness.predict(dummy_face)
    except Exception:
        logger.exception("CDCN liveness warmup failed")

    try:
        if registry.face_parser and registry.face_parser.session is not None:
            registry.face_parser.predict_attributes(dummy_face)
    except Exception:
        logger.exception("Face parsing warmup failed")

    try:
        if registry.age_gender and registry.age_gender.is_loaded:
            registry.age_gender.predict(
                dummy_face,
                face_crop=dummy_face,
                face_crop_hires=dummy_face_hires,
            )
    except Exception:
        logger.exception("Age/gender warmup failed")

    try:
        mivolo = getattr(getattr(registry, "age_gender", None), "mivolo_estimator", None)
        if mivolo is not None and mivolo.is_loaded:
            mivolo.predict(dummy_face_hires)
    except Exception:
        logger.exception("MiVOLO warmup failed")

    logger.info("Model warmup finished in %.2fs", time.time() - start)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup and release on shutdown."""
    if not settings.DEBUG and not settings.AI_SERVICE_SECRET:
        raise RuntimeError(
            "AI_SERVICE_SECRET is required in production (non-debug) mode. "
            "Set FACE_AI_DEBUG=true for local development without a secret."
        )
    logger.info("Loading AI models...")
    start = time.time()

    registry = ModelRegistry()
    await registry.load_all()
    app.state.service_catalog = load_service_catalog(settings.MODEL_REGISTRY_CONFIG or None)
    app.state.models = registry
    app.state.started_at = time.time()
    app.state.total_requests = 0
    app.state.total_latency_ms = 0.0
    app.state.metrics_lock = threading.Lock()
    await _warmup_models(registry)

    logger.info("All models loaded in %.2fs", time.time() - start)
    logger.info("RAM usage: %.0f MB", registry.get_memory_usage_mb())

    try:
        yield
    finally:
        logger.info("Shutting down AI service...")
        registry.unload_all()


app = FastAPI(
    title="Vison Face AI Service",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    lifespan=lifespan,
)
app.add_middleware(AIServiceAuth)
app.add_middleware(RequestHardeningMiddleware)


@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal Server Error",
            "error": {"code": "INTERNAL_ERROR", "detail": "An unexpected error occurred"},
        },
    )
app.include_router(face_router.router, prefix="/api/face")


@app.get("/health")
async def health():
    has_models = hasattr(app.state, "models")
    registry = app.state.models if has_models else None
    status = registry.get_status() if registry else {}
    uptime = int(time.time() - getattr(app.state, "started_at", time.time()))
    total_requests = int(getattr(app.state, "total_requests", 0))
    total_latency_ms = float(getattr(app.state, "total_latency_ms", 0.0))
    avg_latency_ms = round(total_latency_ms / total_requests, 2) if total_requests > 0 else 0.0

    scrfd_model_name = settings.SCRFD_MODEL
    if registry and registry.face_detector is not None:
        scrfd_model_name = registry.face_detector.model_name

    arcface_model_name = settings.ARCFACE_MODEL
    if registry and registry.face_recognizer is not None:
        arcface_model_name = registry.face_recognizer.model_name
    arcface_extra_model_name = settings.ARCFACE_EXTRA_MODEL
    if registry and getattr(registry, "face_recognizer_extra", None) is not None:
        arcface_extra_model_name = registry.face_recognizer_extra.model_name
    adaface_model_name = settings.ADAFACE_MODEL
    adaface_loaded = False
    if registry and getattr(registry, "adaface_recognizer", None) is not None and registry.adaface_recognizer.is_loaded:
        adaface_model_name = registry.adaface_recognizer.model_name
        adaface_loaded = True

    liveness_models = settings.liveness_model_list
    if registry and registry.liveness_checker is not None:
        liveness_models = registry.liveness_checker.model_names

    deepfake_models = settings.deepfake_model_list
    if registry and registry.deepfake_detector is not None:
        deepfake_models = registry.deepfake_detector.model_names

    ai_face_model = settings.AI_FACE_DETECTOR_MODEL
    if registry and registry.ai_face_detector is not None:
        ai_face_model = registry.ai_face_detector.model_name
    ai_face_model_extra = settings.AI_FACE_EXTRA_DETECTOR_MODEL
    if registry and registry.ai_face_detector_extra is not None:
        ai_face_model_extra = registry.ai_face_detector_extra.model_name

    face_parsing_model = settings.FACE_PARSING_MODEL
    if registry and registry.face_parser is not None:
        face_parsing_model = registry.face_parser.model_name

    age_gender_model = settings.AGE_GENDER_MODEL
    if registry and registry.age_gender is not None:
        age_gender_model = registry.age_gender.model_name
    age_gender_vit_model = settings.AGE_GENDER_VIT_MODEL
    if (
        registry
        and registry.age_gender is not None
        and getattr(registry.age_gender, "vit_estimator", None) is not None
        and registry.age_gender.vit_estimator.is_loaded
    ):
        age_gender_vit_model = registry.age_gender.vit_estimator.model_name

    fairface_model = settings.AGE_GENDER_FAIRFACE_MODEL
    fairface_loaded = False
    if (
        registry
        and registry.age_gender is not None
        and getattr(registry.age_gender, "fairface_estimator", None) is not None
        and registry.age_gender.fairface_estimator.is_loaded
    ):
        fairface_model = registry.age_gender.fairface_estimator.model_path
        fairface_loaded = True
    mivolo_model = settings.MIVOLO_MODEL
    mivolo_loaded = False
    if (
        registry
        and registry.age_gender is not None
        and getattr(registry.age_gender, "mivolo_estimator", None) is not None
        and registry.age_gender.mivolo_estimator.is_loaded
    ):
        mivolo_model = registry.age_gender.mivolo_estimator.model_name
        mivolo_loaded = True

    npr_model = settings.NPR_MODEL
    npr_loaded = bool(registry and registry.npr_detector and registry.npr_detector.is_loaded) if registry else False
    clip_fake_model = settings.CLIP_FAKE_MODEL
    clip_fake_loaded = bool(registry and registry.clip_fake_detector and registry.clip_fake_detector.is_loaded) if registry else False
    deepfake_vit_v2_model = settings.DEEPFAKE_VIT_V2_MODEL
    deepfake_vit_v2_loaded = bool(registry and registry.deepfake_vit_v2 and registry.deepfake_vit_v2.is_loaded) if registry else False
    if deepfake_vit_v2_loaded:
        deepfake_vit_v2_model = registry.deepfake_vit_v2.model_name
    cdcn_model = settings.CDCN_MODEL
    cdcn_loaded = bool(registry and registry.cdcn_liveness and registry.cdcn_liveness.is_loaded) if registry else False

    model_details = {
        "scrfd": {
            "loaded": bool(status.get("faceDetector", False)),
            "model": scrfd_model_name,
        },
        "arcface": {
            "loaded": bool(status.get("faceRecognizer", False)),
            "model": arcface_model_name,
        },
        "arcface_extra": {
            "loaded": bool(status.get("faceRecognizerExtra", False)),
            "model": arcface_extra_model_name,
        },
        "adaface": {
            "loaded": adaface_loaded,
            "model": adaface_model_name,
        },
        "liveness": {
            "loaded": int(status.get("livenessModels", 0)) > 0,
            "models": liveness_models,
        },
        "deepfake": {
            "loaded": int(status.get("deepfakeModels", 0)) > 0,
            "models": deepfake_models,
        },
        "ai_face_detector": {
            "loaded": bool(status.get("aiFaceDetector", False)),
            "model": ai_face_model,
        },
        "ai_face_detector_extra": {
            "loaded": bool(status.get("aiFaceDetectorExtra", False)),
            "model": ai_face_model_extra,
        },
        "npr_detector": {
            "loaded": npr_loaded,
            "model": npr_model,
        },
        "clip_fake_detector": {
            "loaded": clip_fake_loaded,
            "model": clip_fake_model,
        },
        "deepfake_vit_v2": {
            "loaded": deepfake_vit_v2_loaded,
            "model": deepfake_vit_v2_model,
        },
        "cdcn_liveness": {
            "loaded": cdcn_loaded,
            "model": cdcn_model,
        },
        "face_parsing": {
            "loaded": bool(status.get("faceParser", False)),
            "model": face_parsing_model,
        },
        "genderage": {
            "loaded": bool(status.get("ageGender", False)),
            "model": age_gender_model,
        },
        "genderage_vit": {
            "loaded": bool(
                registry
                and registry.age_gender is not None
                and getattr(registry.age_gender, "vit_estimator", None) is not None
                and registry.age_gender.vit_estimator.is_loaded
            ),
            "model": age_gender_vit_model,
        },
        "fairface": {
            "loaded": fairface_loaded,
            "model": os.path.basename(fairface_model) if fairface_loaded else fairface_model,
        },
        "mivolo": {
            "loaded": mivolo_loaded,
            "model": mivolo_model,
        },
    }

    models_ready = bool(has_models and registry.is_ready())
    payload = {
        "status": "ok" if models_ready else "degraded",
        "models_loaded": models_ready,
        "models": model_details,
        "memoryMb": round(float(registry.get_memory_usage_mb()), 2) if has_models else 0.0,
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "avg_latency_ms": avg_latency_ms,
        "service_catalog_version": getattr(getattr(app.state, "service_catalog", None), "version", "unknown"),
        "capabilities": getattr(getattr(app.state, "service_catalog", None), "endpoint_names", lambda: [])(),
    }
    return JSONResponse(content=payload, status_code=200 if models_ready else 503)

"""Face inference API endpoints (stateless, no database)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.models import FaceDetectionResult
from app.services.face_processing import FaceProcessor
from app.services.image_utils import ImageValidationError, read_image

logger = logging.getLogger(__name__)
router = APIRouter()
DETECTION_MAX_DIM_RETRY = 640
DETECTION_RETRY_THRESHOLDS = (0.35, 0.25, 0.2)
DETECTION_TILE_GRIDS = ((2, 2), (3, 3), (4, 4))
DETECTION_TILE_OVERLAP = 0.4
DETECTION_TILE_MIN_DIM = 700
DETECTION_TIME_BUDGET_SEC = 0.8  # max total time for all detection retries + tiling
GENERAL_EXECUTOR = ThreadPoolExecutor(max_workers=6)
VERIFY_LIVE_EXECUTOR = ThreadPoolExecutor(max_workers=4)
COMPARE_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _error_response(status_code: int, code: str, message: str, detail: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "error": {"code": code, "detail": detail or message},
        },
    )


def _handle_image_error(error: ImageValidationError) -> JSONResponse:
    return _error_response(error.status_code, error.code, error.message, error.detail)


def _handle_face_detection_error(error_code: str) -> JSONResponse:
    detail_map = {
        "FACE_NOT_DETECTED": "No complete face detected. Please provide a full-face image",
        "MULTIPLE_FACES": "Multiple faces detected. Proceeding with the largest face",
        "FACE_TOO_SMALL": "Detected face is too small for reliable inference",
        "FACE_ATTRIBUTE_NOT_ALLOWED": "Face attributes are not allowed (mask, hat, or glasses detected)",
        "FACE_QUALITY_TOO_LOW": "Face quality is too low. Improve sharpness and brightness, then retry",
    }
    detail = detail_map.get(error_code, "Internal Server Error")

    if error_code in ("FACE_NOT_DETECTED", "FACE_TOO_SMALL", "FACE_ATTRIBUTE_NOT_ALLOWED", "FACE_QUALITY_TOO_LOW"):
        return _error_response(422, error_code, "Invalid Field Format", detail)
    if error_code == "MULTIPLE_FACES":
        return _error_response(400, error_code, "Invalid Field Format", detail)
    return _error_response(500, "MODEL_ERROR", "Internal Server Error")


def _parse_bool_form(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in ("true", "1", "yes")


def _round2(value: float) -> float:
    return round(float(value), 2)


def _non_negative_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(score):
        return 0.0
    return max(0.0, score)


async def _run_in_executor(executor, func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))


async def _await_submitted(*futures):
    return await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))


def _set_request_observability(request: Request, model_timings: Optional[dict] = None, result_summary: Optional[dict] = None):
    if model_timings is not None:
        request.state.model_timings = model_timings
    if result_summary is not None:
        request.state.result_summary = result_summary


def _build_face_payload(face, models) -> dict:
    del models
    if face is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "boundingBox": None,
            "landmarks": None,
        }

    # SCRFD raw scores are typically 0.5-0.99 range.  Scale to AWS-comparable
    # confidence where clearly-detected faces report 99-100%.
    raw_pct = _non_negative_score(face.score) * 100.0
    if raw_pct >= 80.0:
        # SCRFD 80-100 → report 97-100 (clearly visible face)
        confidence = 97.0 + ((raw_pct - 80.0) / 20.0) * 3.0
    elif raw_pct >= 60.0:
        # SCRFD 60-80 → report 85-97
        confidence = 85.0 + ((raw_pct - 60.0) / 20.0) * 12.0
    elif raw_pct >= 50.0:
        confidence = 70.0 + ((raw_pct - 50.0) / 10.0) * 15.0
    else:
        confidence = raw_pct * 1.4
    return {
        "detected": face.face_detected,
        "confidence": _round2(_non_negative_score(confidence)),
        "boundingBox": face.bounding_box_relative,
        "landmarks": face.landmarks_relative,
    }


def _build_validation_payload(
    face,
    image: np.ndarray,
    models,
    validate_attributes: bool,
    validate_quality: bool,
    processor: Optional[FaceProcessor] = None,
) -> dict:
    payload = {"attributes": None, "quality": None}
    if face is None:
        return payload

    proc = processor or FaceProcessor(image, face)

    if validate_attributes and models.face_parser:
        payload["attributes"] = models.face_parser.predict_attributes(proc.for_attributes())

    if validate_quality:
        from app.services.quality import validate_quality as check_quality

        payload["quality"] = check_quality(
            image,
            proc.for_quality(),
            landmarks=face.landmarks if face is not None else None,
            face_bbox=face.bbox if face is not None else None,
        )

    return payload


def _check_validation_errors(
    face,
    image: np.ndarray,
    models,
    validate_attributes: bool,
    validate_quality: bool,
    processor: Optional[FaceProcessor] = None,
) -> Optional[JSONResponse]:
    proc = processor or FaceProcessor(image, face)

    if validate_attributes and models.face_parser:
        attrs = models.face_parser.predict_attributes(proc.for_attributes())
        if attrs.get("hasMask") or attrs.get("hasHat") or attrs.get("hasGlasses"):
            return _handle_face_detection_error("FACE_ATTRIBUTE_NOT_ALLOWED")

    if validate_quality:
        from app.services.quality import check_quality_passed, validate_quality as check_quality

        quality = check_quality(
            image,
            proc.for_quality(),
            landmarks=face.landmarks if face is not None else None,
            face_bbox=face.bbox if face is not None else None,
        )
        if not check_quality_passed(quality):
            return _handle_face_detection_error("FACE_QUALITY_TOO_LOW")

    return None


def _face_area_ratio(face: FaceDetectionResult, image: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32).tolist()
    face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = float(max(1, image.shape[0] * image.shape[1]))
    return float(face_area / image_area)


def _largest_face(faces: list[FaceDetectionResult]) -> FaceDetectionResult:
    return max(
        faces,
        key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
    )


def _resize_for_detection(image: np.ndarray, max_dim: int = DETECTION_MAX_DIM_RETRY) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _maybe_equalize_for_detection(image: np.ndarray) -> np.ndarray:
    if not settings.SCRFD_PRE_EQUALIZE or image is None or image.size == 0:
        return image

    threshold = float(settings.SCRFD_PRE_EQUALIZE_BRIGHTNESS_THRESHOLD)
    try:
        if image.ndim == 2:
            brightness = float(image.mean())
            if brightness >= threshold:
                return image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            logger.info(
                "Applied grayscale SCRFD pre-equalization: brightness=%.1f threshold=%.1f",
                brightness,
                threshold,
            )
            return clahe.apply(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = float(hsv[:, :, 2].mean())
        if brightness >= threshold:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        logger.info(
            "Applied SCRFD pre-equalization: brightness=%.1f threshold=%.1f",
            brightness,
            threshold,
        )
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        logger.exception("SCRFD pre-equalization failed")
        return image


def _resolve_compare_threshold(
    base_threshold: float,
    source_quality: Optional[dict],
    target_quality: Optional[dict],
) -> float:
    threshold = float(base_threshold)
    strategy = settings.ARCFACE_ADAPTIVE_STRATEGY.strip().lower()
    if strategy == "fixed" or source_quality is None or target_quality is None:
        return threshold

    avg_quality = (
        _non_negative_score(source_quality.get("score", 0.0))
        + _non_negative_score(target_quality.get("score", 0.0))
    ) * 0.5
    low = float(settings.ARCFACE_ADAPTIVE_LOW)
    high = float(settings.ARCFACE_ADAPTIVE_HIGH)

    if strategy == "permissive":
        if avg_quality < 50.0:
            return min(threshold, low)
        return threshold

    if strategy not in ("", "conservative"):
        logger.warning("Unknown ARCFACE_ADAPTIVE_STRATEGY=%s, using conservative", strategy)

    if avg_quality > 80.0:
        return max(threshold, high)
    if avg_quality < 50.0:
        return max(threshold, low)
    return threshold


def _map_faces_to_original(
    faces: list[FaceDetectionResult],
    scale: float,
    image_width: int,
    image_height: int,
) -> list[FaceDetectionResult]:
    if scale <= 0:
        return faces
    if abs(scale - 1.0) < 1e-6:
        return faces

    inv_scale = 1.0 / scale
    remapped: list[FaceDetectionResult] = []
    for face in faces:
        bbox = np.asarray(face.bbox, dtype=np.float32) * inv_scale
        bbox[0] = np.clip(bbox[0], 0, image_width - 1)
        bbox[1] = np.clip(bbox[1], 0, image_height - 1)
        bbox[2] = np.clip(bbox[2], 0, image_width - 1)
        bbox[3] = np.clip(bbox[3], 0, image_height - 1)

        landmarks = None
        if face.landmarks is not None:
            landmarks = np.asarray(face.landmarks, dtype=np.float32) * inv_scale
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, image_width - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, image_height - 1)

        remapped.append(
            FaceDetectionResult(
                bbox=bbox.astype(np.float32),
                score=float(face.score),
                landmarks=landmarks.astype(np.float32) if landmarks is not None else None,
                image_width=image_width,
                image_height=image_height,
            )
        )
    return remapped


def _offset_face_to_original(
    face: FaceDetectionResult,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> FaceDetectionResult:
    bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    bbox[0] += float(offset_x)
    bbox[1] += float(offset_y)
    bbox[2] += float(offset_x)
    bbox[3] += float(offset_y)
    bbox[0] = np.clip(bbox[0], 0, image_width - 1)
    bbox[1] = np.clip(bbox[1], 0, image_height - 1)
    bbox[2] = np.clip(bbox[2], 0, image_width - 1)
    bbox[3] = np.clip(bbox[3], 0, image_height - 1)

    landmarks = None
    if face.landmarks is not None:
        landmarks = np.asarray(face.landmarks, dtype=np.float32).copy()
        landmarks[:, 0] += float(offset_x)
        landmarks[:, 1] += float(offset_y)
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, image_width - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, image_height - 1)

    return FaceDetectionResult(
        bbox=bbox.astype(np.float32),
        score=float(face.score),
        landmarks=landmarks.astype(np.float32) if landmarks is not None else None,
        image_width=image_width,
        image_height=image_height,
    )


def _merge_faces_with_nms(models, faces: list[FaceDetectionResult]) -> list[FaceDetectionResult]:
    if not faces:
        return []

    det = np.array(
        [
            [
                float(face.bbox[0]),
                float(face.bbox[1]),
                float(face.bbox[2]),
                float(face.bbox[3]),
                float(face.score),
            ]
            for face in faces
        ],
        dtype=np.float32,
    )
    keep = models.face_detector._nms(det, models.face_detector.nms_threshold)  # pylint: disable=protected-access
    return [faces[i] for i in keep]


def _detect_faces_with_tiling(image: np.ndarray, models, deadline: float = 0.0) -> list[FaceDetectionResult]:
    image_h, image_w = image.shape[:2]
    if max(image_h, image_w) < DETECTION_TILE_MIN_DIM:
        return []

    aggregated: list[FaceDetectionResult] = []
    for rows, cols in DETECTION_TILE_GRIDS:
        if deadline > 0.0 and time.perf_counter() >= deadline:
            break
        # If a coarser grid already found faces, skip finer grids.
        if aggregated:
            break
        cell_w = float(image_w) / float(cols)
        cell_h = float(image_h) / float(rows)
        tile_w = max(1, int(round(cell_w * (1.0 + DETECTION_TILE_OVERLAP))))
        tile_h = max(1, int(round(cell_h * (1.0 + DETECTION_TILE_OVERLAP))))

        for row in range(rows):
            for col in range(cols):
                if deadline > 0.0 and time.perf_counter() >= deadline:
                    break
                cx = int(round((col + 0.5) * cell_w))
                cy = int(round((row + 0.5) * cell_h))
                x1 = max(0, cx - tile_w // 2)
                y1 = max(0, cy - tile_h // 2)
                x2 = min(image_w, x1 + tile_w)
                y2 = min(image_h, y1 + tile_h)
                x1 = max(0, x2 - tile_w)
                y1 = max(0, y2 - tile_h)

                tile = image[y1:y2, x1:x2]
                if tile is None or tile.size == 0:
                    continue

                # Use lowest threshold directly for tiles (already last-resort fallback).
                tile_faces = models.face_detector.detect_all(
                    tile, score_threshold=DETECTION_RETRY_THRESHOLDS[-1],
                )
                if not tile_faces:
                    continue

                for face in tile_faces:
                    aggregated.append(_offset_face_to_original(face, x1, y1, image_w, image_h))

    if not aggregated:
        return []

    merged = _merge_faces_with_nms(models, aggregated)
    logger.info(
        "Face detection recovered with tile fallback: image=%sx%s raw_faces=%d merged_faces=%d",
        image_w,
        image_h,
        len(aggregated),
        len(merged),
    )
    return merged


def _detect_faces_with_retries(image: np.ndarray, models, enable_retries: bool = True) -> list[FaceDetectionResult]:
    start = time.perf_counter()
    deadline = start + DETECTION_TIME_BUDGET_SEC
    detect_image = _maybe_equalize_for_detection(image)

    faces = models.face_detector.detect_all(detect_image)
    if faces:
        return faces

    if not enable_retries:
        return []

    height, width = detect_image.shape[:2]
    resized, scale = _resize_for_detection(detect_image, DETECTION_MAX_DIM_RETRY)
    if scale < 1.0 and time.perf_counter() < deadline:
        faces = models.face_detector.detect_all(resized)
        if faces:
            logger.info(
                "Face detection recovered after resize retry: orig=%sx%s resized=%sx%s",
                width,
                height,
                resized.shape[1],
                resized.shape[0],
            )
            return _map_faces_to_original(faces, scale, width, height)

    for threshold in DETECTION_RETRY_THRESHOLDS:
        if time.perf_counter() >= deadline:
            logger.info("Face detection retry budget exhausted after %.0fms", (time.perf_counter() - start) * 1000)
            return []
        faces = models.face_detector.detect_all(detect_image, score_threshold=threshold)
        if faces:
            logger.info("Face detection recovered with lower threshold=%.2f", threshold)
            return faces
        if scale < 1.0 and time.perf_counter() < deadline:
            resized_faces = models.face_detector.detect_all(resized, score_threshold=threshold)
            if resized_faces:
                logger.info(
                    "Face detection recovered with resize+threshold retry: threshold=%.2f",
                    threshold,
                )
                return _map_faces_to_original(resized_faces, scale, width, height)

    if time.perf_counter() < deadline:
        faces = _detect_faces_with_tiling(detect_image, models, deadline=deadline)
        if faces:
            return faces

    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 200:
        logger.info("Face detection retries exhausted in %.0fms (budget=%.0fms)", elapsed_ms, DETECTION_TIME_BUDGET_SEC * 1000)
    return []


def _run_deepfake_fusion(models, image: np.ndarray, processor: Optional[FaceProcessor], face: Optional[FaceDetectionResult]):
    """3-stage cascade AI/deepfake detection with compression awareness.

    Stage 1: Fast screening (NPR + EfficientNet). If both < fast_exit → early return real.
    Stage 2: CLIP-based UniversalFakeDetect (if available).
    Stage 3: Adaptive weighted fusion with compression awareness, small-face boost,
             hard-block, and consensus rules.
    """
    from app.services.face_processing import estimate_jpeg_quality_from_array, normalize_lighting

    timings: dict[str, float] = {}

    # Pre-compute crops once (cached in processor).
    deepfake_face_crop = processor.for_deepfake() if processor is not None else None
    # Use artifact-preserving bbox crop for AI detection (NOT ArcFace-aligned).
    ai_face_crop = processor.for_ai_detection() if processor is not None else None

    small_face_for_ai_crop = False
    face_area = 0.0
    if face is not None:
        face_area = _face_area_ratio(face, image)
        small_face_for_ai_crop = face_area < float(settings.FACE_MIN_AREA_RATIO)

    do_crop_check = (
        (settings.AI_FACE_ALWAYS_CROP_CHECK or small_face_for_ai_crop)
        and ai_face_crop is not None
        and ai_face_crop.size > 0
    )

    # Estimate JPEG compression quality from pixel statistics.
    jpeg_quality = estimate_jpeg_quality_from_array(image)
    is_low_quality_jpeg = jpeg_quality < int(settings.JPEG_QUALITY_LOW_THRESHOLD)

    # For low-light images, prepare a lighting-normalized crop for AI detection.
    ai_face_crop_normalized = None
    if ai_face_crop is not None and ai_face_crop.size > 0:
        ai_face_crop_normalized = normalize_lighting(ai_face_crop)

    face_confidence = _non_negative_score(face.score) * 100.0 if face is not None else 0.0

    # =================== PARALLEL BRANCH DEFINITIONS ===================

    # --- Branch 1: Face-swap detector (uses ArcFace-aligned crop — correct for face swaps) ---
    def _branch_faceswap():
        if deepfake_face_crop is None:
            return {"isDeepfake": False, "attackRiskLevel": "LOW_RISK", "attackTypes": [], "score": 0.0}, 0.0
        try:
            start = time.perf_counter()
            result = models.deepfake_detector.predict(deepfake_face_crop)
            result = dict(result)
            result["score"] = _non_negative_score(result.get("score", 0.0))
            return result, (time.perf_counter() - start) * 1000.0
        except Exception:
            logger.exception("Face-swap detector branch failed")
            return {"isDeepfake": False, "attackRiskLevel": "LOW_RISK", "attackTypes": [], "score": 0.0}, 0.0

    # --- Branch 2: NPR detector (fast, compression-robust) ---
    def _branch_npr():
        npr = getattr(models, "npr_detector", None)
        if not npr or not npr.is_loaded:
            return {"isFake": False, "fakeScore": 0.0}, 0.0
        crop = ai_face_crop_normalized if ai_face_crop_normalized is not None else ai_face_crop
        if crop is None or crop.size == 0:
            return {"isFake": False, "fakeScore": 0.0}, 0.0
        try:
            start = time.perf_counter()
            result = npr.predict(crop)
            return result, (time.perf_counter() - start) * 1000.0
        except Exception:
            logger.exception("NPR branch failed")
            return {"isFake": False, "fakeScore": 0.0}, 0.0

    # --- Branch 3: AI primary detector (EfficientNet, full + crop) ---
    def _branch_ai_primary():
        ai_detector = getattr(models, "ai_face_detector", None)
        if not ai_detector or ai_detector.session is None:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0
        try:
            start = time.perf_counter()
            result = ai_detector.predict(image)
            full_ms = (time.perf_counter() - start) * 1000.0
            full_score = _non_negative_score(result.get("aiScore", 0.0))
            crop_ms = 0.0
            crop_score = 0.0
            if do_crop_check and ai_face_crop is not None:
                start = time.perf_counter()
                crop_result = ai_detector.predict(ai_face_crop)
                crop_ms = (time.perf_counter() - start) * 1000.0
                crop_score = _non_negative_score(crop_result.get("aiScore", 0.0))
                if crop_score > full_score:
                    result = crop_result
            return result, full_ms, crop_ms, full_score, crop_score
        except Exception:
            logger.exception("AI primary detector branch failed")
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0

    # --- Branch 4: AI extra detector (full + crop) ---
    def _branch_ai_extra():
        ai_detector_extra = getattr(models, "ai_face_detector_extra", None)
        if not ai_detector_extra or ai_detector_extra.session is None:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0
        try:
            start = time.perf_counter()
            result = ai_detector_extra.predict(image)
            full_ms = (time.perf_counter() - start) * 1000.0
            full_score = _non_negative_score(result.get("aiScore", 0.0))
            crop_ms = 0.0
            crop_score = 0.0
            if do_crop_check and ai_face_crop is not None:
                start = time.perf_counter()
                crop_result = ai_detector_extra.predict(ai_face_crop)
                crop_ms = (time.perf_counter() - start) * 1000.0
                crop_score = _non_negative_score(crop_result.get("aiScore", 0.0))
                if crop_score > full_score:
                    result = crop_result
            return result, full_ms, crop_ms, full_score, crop_score
        except Exception:
            logger.exception("AI extra detector branch failed")
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0

    # --- Branch 5: CLIP-based UniversalFakeDetect ---
    def _branch_clip():
        clip_det = getattr(models, "clip_fake_detector", None)
        if not clip_det or not clip_det.is_loaded:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0
        try:
            # Run on full image
            start = time.perf_counter()
            full_result = clip_det.predict(image)
            full_ms = (time.perf_counter() - start) * 1000.0
            full_score = _non_negative_score(full_result.get("aiScore", 0.0))
            # Run on face crop
            crop_ms = 0.0
            crop_score = 0.0
            best_result = full_result
            if ai_face_crop is not None and ai_face_crop.size > 0:
                start = time.perf_counter()
                crop_result = clip_det.predict(ai_face_crop)
                crop_ms = (time.perf_counter() - start) * 1000.0
                crop_score = _non_negative_score(crop_result.get("aiScore", 0.0))
                if crop_score > full_score:
                    best_result = crop_result
            return best_result, full_ms, crop_ms, full_score, crop_score
        except Exception:
            logger.exception("CLIP branch failed")
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0

    def _run_deepfake_vit_v2_branch():
        vit_v2 = getattr(models, "deepfake_vit_v2", None)
        if not vit_v2 or not vit_v2.is_loaded:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0
        branch_input = ai_face_crop if ai_face_crop is not None and ai_face_crop.size > 0 else image
        if branch_input is None or branch_input.size == 0:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0
        try:
            start = time.perf_counter()
            result = vit_v2.predict(branch_input)
            result = dict(result)
            result["aiScore"] = _non_negative_score(result.get("aiScore", 0.0))
            return result, (time.perf_counter() - start) * 1000.0
        except Exception:
            logger.exception("Deepfake ViT v2 branch failed")
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0

    # =================== RUN ALL BRANCHES IN PARALLEL ===================
    fs_future = GENERAL_EXECUTOR.submit(_branch_faceswap)
    npr_future = GENERAL_EXECUTOR.submit(_branch_npr)
    ai_p_future = GENERAL_EXECUTOR.submit(_branch_ai_primary)
    ai_e_future = GENERAL_EXECUTOR.submit(_branch_ai_extra)
    clip_future = GENERAL_EXECUTOR.submit(_branch_clip)
    vit_v2_future = GENERAL_EXECUTOR.submit(_run_deepfake_vit_v2_branch)

    face_swap_result, fs_ms = fs_future.result()
    timings["deepfake_faceswap_ms"] = _round2(fs_ms)

    npr_result, npr_ms = npr_future.result()
    timings["deepfake_npr_ms"] = _round2(npr_ms)

    ai_primary_result, ai_p_full_ms, ai_p_crop_ms, ai_primary_full_score, ai_primary_crop_score = ai_p_future.result()
    timings["deepfake_ai_model_full_ms"] = _round2(ai_p_full_ms)
    timings["deepfake_ai_model_crop_ms"] = _round2(ai_p_crop_ms)
    timings["deepfake_ai_model_ms"] = _round2(ai_p_full_ms + ai_p_crop_ms)

    ai_extra_result, ai_e_full_ms, ai_e_crop_ms, ai_extra_full_score, ai_extra_crop_score = ai_e_future.result()
    timings["deepfake_ai_model_extra_full_ms"] = _round2(ai_e_full_ms)
    timings["deepfake_ai_model_extra_crop_ms"] = _round2(ai_e_crop_ms)
    timings["deepfake_ai_model_extra_ms"] = _round2(ai_e_full_ms + ai_e_crop_ms)

    clip_result, clip_full_ms, clip_crop_ms, clip_full_score, clip_crop_score = clip_future.result()
    timings["deepfake_clip_full_ms"] = _round2(clip_full_ms)
    timings["deepfake_clip_crop_ms"] = _round2(clip_crop_ms)
    timings["deepfake_clip_ms"] = _round2(clip_full_ms + clip_crop_ms)

    vit_v2_result, vit_v2_ms = vit_v2_future.result()
    timings["deepfake_vit_v2_ms"] = _round2(vit_v2_ms)
    timings["jpeg_quality_estimate"] = float(jpeg_quality)

    # =================== EXTRACT SCORES ===================
    face_swap_score = _non_negative_score(face_swap_result.get("score", 0.0))
    npr_score = _non_negative_score(npr_result.get("fakeScore", 0.0))
    ai_primary_score = _non_negative_score(ai_primary_result.get("aiScore", 0.0))
    ai_extra_score = _non_negative_score(ai_extra_result.get("aiScore", 0.0))
    clip_score = _non_negative_score(clip_result.get("aiScore", 0.0))
    vit_v2_score = _non_negative_score(vit_v2_result.get("aiScore", 0.0))

    # =================== STAGE 1: FAST SCREENING ===================
    # If both NPR and EfficientNet are very low confidence → early exit as real.
    # This saves ~200ms by skipping heavy fusion logic.
    fast_exit_threshold = float(settings.AI_FAST_EXIT_THRESHOLD) * 100.0  # convert to percent
    npr_available = bool(getattr(models, "npr_detector", None) and models.npr_detector.is_loaded)
    fast_exit = False
    if npr_available and npr_score < fast_exit_threshold and ai_primary_score < fast_exit_threshold:
        # Double-check: face swap must also be low
        if face_swap_score < float(settings.DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD):
            fast_exit = True

    # Don't fast-exit if face confidence is low (suspicious)
    if fast_exit and face_confidence < float(settings.AI_FACE_LOW_CONF_FACE_CONF):
        fast_exit = False

    if fast_exit:
        fusion_result = {
            "isDeepfake": False,
            "attackRiskLevel": "LOW_RISK",
            "attackTypes": [],
            "scores": {
                "faceSwapScore": _round2(_non_negative_score(face_swap_score)),
                "aiGeneratedScore": _round2(_non_negative_score(max(npr_score, ai_primary_score))),
            },
        }
        if settings.DEBUG:
            logger.info(
                "Deepfake fusion FAST EXIT: npr=%.2f ai_primary=%.2f fs=%.2f jpeg_q=%d face_conf=%.2f",
                npr_score, ai_primary_score, face_swap_score, jpeg_quality, face_confidence,
            )
        return fusion_result, timings

    # =================== STAGE 2 & 3: ADAPTIVE WEIGHTED FUSION ===================
    # Collect all AI evidence scores.
    ai_evidence_scores: list[float] = [
        float(ai_primary_full_score),
        float(ai_primary_crop_score),
        float(ai_extra_full_score),
        float(ai_extra_crop_score),
        float(ai_primary_score),
        float(ai_extra_score),
    ]
    # Add NPR and CLIP scores to evidence pool.
    if npr_available and npr_score > 0.0:
        ai_evidence_scores.append(npr_score)
    clip_available = bool(getattr(models, "clip_fake_detector", None) and models.clip_fake_detector.is_loaded)
    if clip_available:
        if clip_full_score > 0.0:
            ai_evidence_scores.append(clip_full_score)
        if clip_crop_score > 0.0:
            ai_evidence_scores.append(clip_crop_score)
        if clip_score > 0.0:
            ai_evidence_scores.append(clip_score)
    vit_v2_available = bool(getattr(models, "deepfake_vit_v2", None) and models.deepfake_vit_v2.is_loaded)
    if vit_v2_available and vit_v2_score > 0.0:
        ai_evidence_scores.append(vit_v2_score)

    # --- Adaptive weighting based on compression quality ---
    freq_penalty = float(settings.COMPRESSION_FREQ_WEIGHT_PENALTY)
    small_face_boost = float(settings.SMALL_FACE_BOOST_MULTIPLIER)

    # Base weights for each detector family (on 0-100 scale scores).
    w_primary = max(0.0, float(settings.AI_FACE_PRIMARY_WEIGHT))    # EfficientNet
    w_extra = max(0.0, float(settings.AI_FACE_EXTRA_WEIGHT))        # ai_vs_deepfake_vs_real
    w_npr = 0.25 if npr_available else 0.0
    w_clip = 0.35 if clip_available else 0.0
    w_vit_v2 = max(0.0, float(settings.DEEPFAKE_VIT_V2_WEIGHT)) if vit_v2_available else 0.0

    if is_low_quality_jpeg:
        # Reduce frequency-domain model weights (EfficientNet, ViT) — they
        # confuse JPEG blocking artifacts with GAN artifacts.
        w_primary *= freq_penalty
        w_extra *= freq_penalty
        w_vit_v2 *= freq_penalty
        # Boost compression-robust models.
        w_npr *= 1.3
        w_clip *= 1.2

    if small_face_for_ai_crop:
        # Small faces benefit from crop-based scores.
        w_npr *= small_face_boost
        w_clip *= small_face_boost
        w_vit_v2 *= small_face_boost

    # Compute weighted AI score.
    weighted_scores: list[tuple[float, float]] = []  # (weight, score)
    if timings["deepfake_ai_model_ms"] > 0.0:
        weighted_scores.append((w_primary, ai_primary_score))
    if timings["deepfake_ai_model_extra_ms"] > 0.0:
        weighted_scores.append((w_extra, ai_extra_score))
    if npr_available and npr_ms > 0.0:
        weighted_scores.append((w_npr, npr_score))
    if clip_available and timings["deepfake_clip_ms"] > 0.0:
        weighted_scores.append((w_clip, clip_score))
    if vit_v2_available and timings["deepfake_vit_v2_ms"] > 0.0:
        weighted_scores.append((w_vit_v2, vit_v2_score))

    if weighted_scores:
        weight_sum = sum(w for w, _ in weighted_scores)
        if weight_sum <= 0.0:
            ai_score = float(sum(s for _, s in weighted_scores) / len(weighted_scores))
        else:
            ai_score = float(sum(w * s for w, s in weighted_scores) / weight_sum)
    else:
        ai_score = 0.0

    # For small/far faces, prefer strongest AI evidence.
    if small_face_for_ai_crop:
        ai_score = max(ai_score, ai_primary_score, ai_extra_score, clip_score, npr_score, vit_v2_score)
    if settings.AI_FACE_ALWAYS_CROP_CHECK:
        all_crop_scores = [ai_primary_full_score, ai_primary_crop_score,
                           ai_extra_full_score, ai_extra_crop_score]
        if clip_available:
            all_crop_scores.extend([clip_full_score, clip_crop_score])
        if vit_v2_available:
            all_crop_scores.append(vit_v2_score)
        ai_score = max(ai_score, *all_crop_scores)

    # Linear calibration for environment-specific tuning (default is identity).
    ai_score = float(
        np.clip(
            ai_score * float(settings.AI_FACE_CALIBRATION_ALPHA) + float(settings.AI_FACE_CALIBRATION_BETA),
            0.0,
            100.0,
        )
    )

    # =================== DECISION LOGIC ===================
    ai_threshold = float(settings.AI_FACE_THRESHOLD)
    ai_low_conf_threshold = float(settings.AI_FACE_LOW_CONF_THRESHOLD)
    low_conf_face_limit = float(settings.AI_FACE_LOW_CONF_FACE_CONF)

    low_conf_or_missing_face = (face is None) or (face_confidence <= low_conf_face_limit)
    ai_generated_primary = ai_score >= ai_threshold
    ai_generated_low_conf = low_conf_or_missing_face and ai_score >= ai_low_conf_threshold

    # Hard block: any single model > threshold → immediate flag.
    hard_block_single = float(settings.AI_HARD_BLOCK_SINGLE)
    evidence_nonzero = [value for value in ai_evidence_scores if value > 0.0]
    ai_generated_hard_block = bool(evidence_nonzero and max(evidence_nonzero) >= hard_block_single)

    # Consensus: 3+ models > threshold → flag.
    consensus_threshold = float(settings.AI_CONSENSUS_THRESHOLD)
    consensus_min = max(1, int(settings.AI_CONSENSUS_MIN_MODELS))
    consensus_count = sum(1 for value in evidence_nonzero if value >= consensus_threshold)
    ai_generated_consensus_multi = consensus_count >= consensus_min

    # Legacy vote/trigger thresholds (kept for backward compat).
    hard_block_threshold = float(settings.AI_FACE_HARD_BLOCK_THRESHOLD)
    vote_threshold = float(settings.AI_FACE_VOTE_THRESHOLD)
    vote_min_count = max(1, int(settings.AI_FACE_VOTE_MIN_COUNT))
    any_trigger_threshold = float(settings.AI_FACE_ANY_TRIGGER_THRESHOLD)
    vote_count = sum(1 for value in evidence_nonzero if value >= vote_threshold)
    ai_generated_legacy_hard = bool(evidence_nonzero and max(evidence_nonzero) >= hard_block_threshold)
    ai_generated_vote = vote_count >= vote_min_count
    ai_generated_any_trigger = bool(evidence_nonzero and max(evidence_nonzero) >= any_trigger_threshold)

    consensus_ai_threshold = float(settings.AI_FACE_CONSENSUS_AI_THRESHOLD)
    consensus_face_swap_threshold = float(settings.AI_FACE_CONSENSUS_FACE_SWAP_THRESHOLD)
    ai_generated_consensus_legacy = False
    if consensus_ai_threshold <= 100.0:
        ai_generated_consensus_legacy = (ai_score >= consensus_ai_threshold) and (
            face_swap_score >= consensus_face_swap_threshold
        )

    ai_generated = (
        ai_generated_primary
        or ai_generated_low_conf
        or ai_generated_consensus_legacy
        or ai_generated_hard_block
        or ai_generated_consensus_multi
        or ai_generated_legacy_hard
        or ai_generated_vote
        or (ai_generated_any_trigger and face_swap_score >= consensus_face_swap_threshold)
    )

    # --- Real suppress: high-confidence real face with moderate AI scores ---
    face_swap_threshold = float(settings.DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD)
    face_swap_flag = face_swap_score >= face_swap_threshold
    likely_real_suppressed = False
    if settings.AI_FACE_REAL_SUPPRESS_ENABLED and not face_swap_flag and face is not None:
        skip_real_suppress = (
            (small_face_for_ai_crop and ai_score >= ai_threshold)
            or ai_generated_hard_block
            or ai_generated_vote
            or ai_generated_consensus_multi
        )
        if not skip_real_suppress:
            if (
                face_confidence >= float(settings.AI_FACE_REAL_SUPPRESS_FACE_CONF)
                and ai_score < float(settings.AI_FACE_REAL_SUPPRESS_AI_MAX)
                and face_swap_score < float(settings.AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX)
            ):
                ai_generated = False
                likely_real_suppressed = True

    # --- Compression-aware false-positive suppression ---
    # For heavily compressed images (WhatsApp, low-quality JPEG) with a high-confidence
    # real face, require stronger evidence before flagging as AI.
    if is_low_quality_jpeg and ai_generated and not face_swap_flag and not ai_generated_hard_block:
        if face_confidence >= 72.0:
            # Require at least 2 models > 70% for compressed images
            strong_evidence_count = sum(1 for value in evidence_nonzero if value >= 70.0)
            if strong_evidence_count < 2:
                ai_generated = False
                likely_real_suppressed = True
                if settings.DEBUG:
                    logger.info(
                        "Compression-aware suppress: jpeg_q=%d face_conf=%.2f strong_evidence=%d ai=%.2f",
                        jpeg_quality, face_confidence, strong_evidence_count, ai_score,
                    )

    is_deepfake = bool(face_swap_flag or ai_generated)

    attack_types: list[str] = []
    if face_swap_flag:
        attack_types.extend(face_swap_result.get("attackTypes", []) or ["SYNTHETIC_IMAGE"])
    if ai_generated:
        attack_types.append("AI_GENERATED")
    # Keep order stable and unique
    attack_types = list(dict.fromkeys(attack_types))

    max_score = max(face_swap_score, ai_score)
    if max_score >= 80.0:
        risk = "HIGH_RISK"
    elif max_score >= 50.0:
        risk = "MEDIUM_RISK"
    else:
        risk = "LOW_RISK"

    fusion_result = {
        "isDeepfake": is_deepfake,
        "attackRiskLevel": risk,
        "attackTypes": attack_types,
        "scores": {
            "faceSwapScore": _round2(_non_negative_score(face_swap_score)),
            "aiGeneratedScore": _round2(_non_negative_score(ai_score)),
        },
    }
    if settings.DEBUG:
        logger.info(
            "Deepfake fusion: face_conf=%.2f small_face=%s jpeg_q=%d ai=%.2f "
            "ai_primary=%.2f (full=%.2f crop=%.2f) ai_extra=%.2f (full=%.2f crop=%.2f) "
            "npr=%.2f clip=%.2f (full=%.2f crop=%.2f) vit_v2=%.2f fs=%.2f "
            "hard_single=%s consensus_multi=%s(%d/%d) legacy_hard=%s vote=%s(%d/%d) "
            "real_suppressed=%s deepfake=%s",
            face_confidence, small_face_for_ai_crop, jpeg_quality, ai_score,
            ai_primary_score, ai_primary_full_score, ai_primary_crop_score,
            ai_extra_score, ai_extra_full_score, ai_extra_crop_score,
            npr_score, clip_score, clip_full_score, clip_crop_score, vit_v2_score, face_swap_score,
            ai_generated_hard_block, ai_generated_consensus_multi, consensus_count, consensus_min,
            ai_generated_legacy_hard, ai_generated_vote, vote_count, vote_min_count,
            likely_real_suppressed, is_deepfake,
        )
    return fusion_result, timings


def _detect_and_validate(
    image: np.ndarray,
    models,
    validate_attributes: bool,
    validate_quality: bool,
    allow_no_face: bool = False,
):
    faces = _detect_faces_with_retries(image, models, enable_retries=True)
    face = None

    if not faces:
        if allow_no_face:
            return None, None, None
        return None, None, _handle_face_detection_error("FACE_NOT_DETECTED")
    else:
        if len(faces) > 1:
            logger.warning("Multiple faces detected (%d), using largest face", len(faces))
        face = _largest_face(faces)

    if face is not None:
        area_ratio = _face_area_ratio(face, image)
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32).tolist()
        face_w = max(0.0, float(x2 - x1))
        face_h = max(0.0, float(y2 - y1))
        min_side = min(face_w, face_h)

        # Hard guard only for extremely tiny detections.
        too_small_hard = (
            area_ratio < float(settings.FACE_MIN_AREA_RATIO_HARD)
            or min_side < float(settings.FACE_MIN_PIXELS_HARD)
        )
        if too_small_hard:
            return face, None, _handle_face_detection_error("FACE_TOO_SMALL")

        # Soft guard: allow small faces and rely on alignment upsample/autocrop.
        if area_ratio < float(settings.FACE_MIN_AREA_RATIO):
            if bool(settings.ALLOW_SMALL_FACE_AUTOCROP):
                setattr(face, "_small_face_autocrop", True)
                if settings.DEBUG:
                    logger.info(
                        "Small face accepted via autocrop: area_ratio=%.4f min_side=%.1f bbox=%s",
                        area_ratio,
                        min_side,
                        np.asarray(face.bbox, dtype=np.float32).tolist(),
                    )
            else:
                return face, None, _handle_face_detection_error("FACE_TOO_SMALL")

    processor = FaceProcessor(image, face) if face is not None else None
    err = _check_validation_errors(
        face,
        image,
        models,
        validate_attributes,
        validate_quality,
        processor=processor,
    ) if processor is not None else None
    if err:
        return face, processor, err
    return face, processor, None


def _get_face_embedding(models, face_aligned_112: np.ndarray) -> tuple[np.ndarray, dict]:
    timings: dict[str, float] = {}
    embeddings: list[np.ndarray] = []
    weights: list[float] = []

    t0 = time.perf_counter()
    primary_emb = models.face_recognizer.get_embedding(face_aligned_112)
    timings["arcface_primary_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    embeddings.append(primary_emb)
    weights.append(max(0.0, float(settings.ARCFACE_PRIMARY_WEIGHT)))

    extra = getattr(models, "face_recognizer_extra", None)
    if extra is not None and extra.session is not None:
        try:
            t0 = time.perf_counter()
            extra_emb = extra.get_embedding(face_aligned_112)
            timings["arcface_extra_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
            embeddings.append(extra_emb)
            weights.append(max(0.0, float(settings.ARCFACE_EXTRA_WEIGHT)))
        except Exception:
            logger.exception("ArcFace extra model embedding failed, fallback to primary")

    adaface = getattr(models, "adaface_recognizer", None)
    if adaface is not None and adaface.is_loaded:
        try:
            t0 = time.perf_counter()
            adaface_emb = adaface.get_embedding(face_aligned_112)
            timings["adaface_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
            embeddings.append(adaface_emb)
            weights.append(max(0.0, float(getattr(adaface, "weight", settings.ADAFACE_WEIGHT))))
        except Exception:
            logger.exception("AdaFace embedding failed, fallback to ArcFace-only fusion")

    if len(embeddings) == 1:
        return embeddings[0], timings

    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    else:
        weights = [weight / weight_sum for weight in weights]

    fused = np.zeros_like(embeddings[0], dtype=np.float32)
    for emb, weight in zip(embeddings, weights):
        fused += emb.astype(np.float32) * float(weight)
    norm = float(np.linalg.norm(fused))
    if norm > 0.0:
        fused = fused / norm
    return fused.astype(np.float32), timings


def _endpoint_guard(handler):
    @wraps(handler)
    async def wrapper(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        except ImageValidationError as error:
            return _handle_image_error(error)
        except Exception:
            logger.exception("Unhandled exception in endpoint %s", handler.__name__)
            return _error_response(500, "MODEL_ERROR", "Internal Server Error")

    return wrapper


@router.post("/compare")
@_endpoint_guard
async def compare_faces(
    request: Request,
    sourceImage: UploadFile = File(...),
    targetImage: UploadFile = File(...),
    similarityThreshold: Optional[str] = Form(None),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    threshold = float(settings.COMPARE_THRESHOLD_DEFAULT)
    if similarityThreshold:
        try:
            threshold = float(similarityThreshold.strip())
            if threshold < 0 or threshold > 100:
                return _error_response(
                    400,
                    "INVALID_FIELD_FORMAT",
                    "Invalid Field Format",
                    "similarityThreshold must be between 0 and 100",
                )
        except ValueError:
            return _error_response(
                400,
                "INVALID_FIELD_FORMAT",
                "Invalid Field Format",
                "similarityThreshold must be a number",
            )

    source_img, target_img = await asyncio.gather(
        read_image(sourceImage, "sourceImage"),
        read_image(targetImage, "targetImage"),
    )

    timings: dict[str, float] = {}
    source_quality: Optional[dict] = None
    target_quality: Optional[dict] = None

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    src_detect_fut = loop.run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate, source_img, models, validate_attrs, validate_qual,
    )
    tgt_detect_fut = loop.run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate, target_img, models, validate_attrs, validate_qual,
    )
    (source_face, source_proc, src_err), (target_face, target_proc, tgt_err) = await asyncio.gather(
        src_detect_fut, tgt_detect_fut,
    )
    timings["detect_parallel_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if src_err:
        return src_err
    if tgt_err:
        return tgt_err

    adaptive_strategy = settings.ARCFACE_ADAPTIVE_STRATEGY.strip().lower()
    if settings.ARCFACE_ADAPTIVE_COMPARE and adaptive_strategy != "fixed":
        from app.services.quality import validate_quality as check_quality

        t0 = time.perf_counter()
        src_qual_fut = loop.run_in_executor(GENERAL_EXECUTOR, lambda: check_quality(
            source_img, source_proc.for_quality(),
            landmarks=source_face.landmarks if source_face is not None else None,
            face_bbox=source_face.bbox if source_face is not None else None,
        ))
        tgt_qual_fut = loop.run_in_executor(GENERAL_EXECUTOR, lambda: check_quality(
            target_img, target_proc.for_quality(),
            landmarks=target_face.landmarks if target_face is not None else None,
            face_bbox=target_face.bbox if target_face is not None else None,
        ))
        source_quality, target_quality = await asyncio.gather(src_qual_fut, tgt_qual_fut)
        timings["adaptive_quality_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
        threshold = _resolve_compare_threshold(threshold, source_quality, target_quality)

    def _embed_from_processor(proc: FaceProcessor):
        return _get_face_embedding(models, proc.for_arcface())

    t0 = time.perf_counter()
    if settings.COMPARE_PARALLEL_EMBEDDING:
        src_future = COMPARE_EMBED_EXECUTOR.submit(_embed_from_processor, source_proc)
        tgt_future = COMPARE_EMBED_EXECUTOR.submit(_embed_from_processor, target_proc)
        (source_emb, source_emb_timings), (target_emb, target_emb_timings) = await _await_submitted(
            src_future,
            tgt_future,
        )
    else:
        source_emb, source_emb_timings = await _run_in_executor(
            COMPARE_EMBED_EXECUTOR,
            _embed_from_processor,
            source_proc,
        )
        target_emb, target_emb_timings = await _run_in_executor(
            COMPARE_EMBED_EXECUTOR,
            _embed_from_processor,
            target_proc,
        )
    cosine_sim = models.face_recognizer.cosine_similarity(source_emb, target_emb)
    similarity_percent = models.face_recognizer.similarity_to_percent(cosine_sim)
    timings["arcface_similarity_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    for key, value in source_emb_timings.items():
        timings[f"source_{key}"] = value
    for key, value in target_emb_timings.items():
        timings[f"target_{key}"] = value

    response_payload = {
        "matched": similarity_percent >= threshold,
        "similarity": _round2(similarity_percent),
        "threshold": _round2(threshold),
        "sourceFace": _build_face_payload(source_face, models),
        "targetFace": _build_face_payload(target_face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            target_face,
            target_img,
            models,
            validate_attrs,
            validate_qual,
            processor=target_proc,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={"matched": response_payload["matched"], "similarity": response_payload["similarity"]},
    )
    return response_payload


@router.post("/liveness")
@_endpoint_guard
async def liveness_check(
    request: Request,
    image: UploadFile = File(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    img = await read_image(image)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    def _predict_liveness_sync():
        return models.liveness_checker.predict(processor.for_liveness())

    live_score, is_live = await _run_in_executor(GENERAL_EXECUTOR, _predict_liveness_sync)
    live_score = _non_negative_score(live_score)
    timings["liveness_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    response_payload = {
        "isLive": is_live,
        "liveScore": _round2(live_score),
        "face": _build_face_payload(face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            face,
            img,
            models,
            validate_attrs,
            validate_qual,
            processor=processor,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={"isLive": response_payload["isLive"], "liveScore": response_payload["liveScore"]},
    )
    return response_payload


@router.post("/deepfake")
@_endpoint_guard
async def deepfake_check(
    request: Request,
    image: UploadFile = File(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    img = await read_image(image)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    result, fusion_timings = await _run_in_executor(
        VERIFY_LIVE_EXECUTOR,
        _run_deepfake_fusion,
        models,
        img,
        processor,
        face,
    )
    timings["deepfake_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(fusion_timings)
    result_scores = result.get("scores", {"faceSwapScore": 0.0, "aiGeneratedScore": 0.0})
    response_payload = {
        "isDeepfake": result["isDeepfake"],
        "attackRiskLevel": result["attackRiskLevel"],
        "attackTypes": result["attackTypes"],
        "scores": {
            "faceSwapScore": _round2(_non_negative_score(result_scores.get("faceSwapScore", 0.0))),
            "aiGeneratedScore": _round2(_non_negative_score(result_scores.get("aiGeneratedScore", 0.0))),
        },
        "face": _build_face_payload(face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            face,
            img,
            models,
            validate_attrs,
            validate_qual,
            processor=processor,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={
            "isDeepfake": response_payload["isDeepfake"],
            "attackRiskLevel": response_payload["attackRiskLevel"],
        },
    )
    return response_payload


@router.post("/analyze")
@_endpoint_guard
async def analyze_face(
    request: Request,
    image: UploadFile = File(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    img = await read_image(image)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    def _predict_age_gender_sync():
        age_face_crop = processor.for_age_gender()
        age_face_crop_hires = processor.for_age_gender_hires()
        age_input = img if getattr(models.age_gender, "backend", "") == "insightface" else age_face_crop
        return models.age_gender.predict(
            age_input,
            face_crop=age_face_crop,
            face_crop_hires=age_face_crop_hires,
        )

    t0 = time.perf_counter()
    result = await _run_in_executor(GENERAL_EXECUTOR, _predict_age_gender_sync)
    timings["age_gender_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    response_payload = {
        "gender": result["gender"],
        "genderConfidence": _round2(result["genderConfidence"]),
        "ageRange": result["ageRange"],
        "face": _build_face_payload(face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            face,
            img,
            models,
            validate_attrs,
            validate_qual,
            processor=processor,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={
            "gender": response_payload["gender"],
            "ageRange": response_payload["ageRange"],
        },
    )
    return response_payload


@router.post("/verify-live")
@_endpoint_guard
async def verify_live(
    request: Request,
    image: UploadFile = File(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    del validateAttributes, validateQuality
    models = request.app.state.models

    img = await read_image(image)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attributes=False,
        validate_quality=False,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    from app.services.quality import validate_quality as check_quality

    # Pre-compute crops before submitting to threads (avoids contention on FaceProcessor cache).
    quality_target = processor.for_quality()
    live_crop = processor.for_liveness()
    processor.for_deepfake()  # warm cache for deepfake fusion threads
    processor.for_ai_detection()  # warm cache for AI detection crop

    face_landmarks = face.landmarks
    face_bbox = face.bbox

    def _run_quality():
        start = time.perf_counter()
        result = check_quality(img, quality_target, landmarks=face_landmarks, face_bbox=face_bbox)
        return result, (time.perf_counter() - start) * 1000.0

    def _run_liveness():
        """Run MiniFASNet ensemble + optional CDCN, return fused score."""
        start = time.perf_counter()
        minifas_score, minifas_live = models.liveness_checker.predict(live_crop)
        minifas_score = _non_negative_score(minifas_score)

        cdcn = getattr(models, "cdcn_liveness", None)
        cdcn_score = 0.0
        cdcn_live = False
        if cdcn is not None and cdcn.is_loaded:
            try:
                cdcn_score, cdcn_live = cdcn.predict(live_crop)
                cdcn_score = _non_negative_score(cdcn_score)
            except Exception:
                logger.exception("CDCN liveness failed, using MiniFASNet only")

        # Weighted fusion of MiniFASNet and CDCN.
        if cdcn is not None and cdcn.is_loaded and cdcn_score > 0.0:
            w_minifas = max(0.0, float(settings.LIVENESS_MINIFAS_WEIGHT))
            w_cdcn = max(0.0, float(settings.CDCN_WEIGHT))
            w_total = w_minifas + w_cdcn
            if w_total > 0.0:
                fused_score = (minifas_score * w_minifas + cdcn_score * w_cdcn) / w_total
            else:
                fused_score = (minifas_score + cdcn_score) * 0.5
            fused_live = fused_score > (float(settings.LIVENESS_THRESHOLD) * 100.0)
        else:
            fused_score = minifas_score
            fused_live = minifas_live

        fused_score = _non_negative_score(fused_score)
        elapsed = (time.perf_counter() - start) * 1000.0
        return (fused_score, fused_live), elapsed

    def _run_deepfake():
        start = time.perf_counter()
        value, detail_timings = _run_deepfake_fusion(models, img, processor, face)
        return value, detail_timings, (time.perf_counter() - start) * 1000.0

    try:
        parallel_start = time.perf_counter()
        quality_future = VERIFY_LIVE_EXECUTOR.submit(_run_quality)
        liveness_future = VERIFY_LIVE_EXECUTOR.submit(_run_liveness)
        deepfake_future = VERIFY_LIVE_EXECUTOR.submit(_run_deepfake)
        (
            (quality, quality_ms),
            ((live_score, liveness_is_live), liveness_ms),
            (deepfake_result, deepfake_detail_timings, deepfake_ms),
        ) = await _await_submitted(
            quality_future,
            liveness_future,
            deepfake_future,
        )
        timings["parallel_ms"] = _round2((time.perf_counter() - parallel_start) * 1000.0)
        timings["quality_ms"] = _round2(quality_ms)
        timings["liveness_ms"] = _round2(liveness_ms)
        timings["deepfake_ms"] = _round2(deepfake_ms)
        timings.update(deepfake_detail_timings)
    except Exception:
        logger.exception("verify-live model execution failed")
        return _error_response(
            500,
            "MODEL_ERROR",
            "Internal Server Error",
            "Failed to run liveness/deepfake models",
        )

    quality_gate_passed = True
    if settings.VERIFY_LIVE_QUALITY_GATE:
        quality_gate_passed = _non_negative_score(quality.get("score", 0.0)) >= float(
            settings.VERIFY_LIVE_QUALITY_MIN_SCORE
        )

    live_score = _non_negative_score(live_score)

    final_is_live = bool(
        liveness_is_live
        and not deepfake_result.get("isDeepfake", False)
        and quality_gate_passed
    )
    response_payload = {
        "isLive": final_is_live,
        "liveScore": _round2(live_score),
        "isDeepfake": bool(deepfake_result.get("isDeepfake", False)),
        "attackRiskLevel": deepfake_result.get("attackRiskLevel", "LOW_RISK"),
        "attackTypes": deepfake_result.get("attackTypes", []),
        "face": _build_face_payload(face, models),
        "validation": {
            "quality": quality,
            "qualityGatePassed": quality_gate_passed,
        },
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={
            "isLive": response_payload["isLive"],
            "isDeepfake": response_payload["isDeepfake"],
            "attackRiskLevel": response_payload["attackRiskLevel"],
            "qualityGatePassed": quality_gate_passed,
        },
    )
    return response_payload


@router.post("/embed")
@_endpoint_guard
async def extract_embedding(
    request: Request,
    image: UploadFile = File(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    img = await read_image(image)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    def _extract_embedding_sync():
        return _get_face_embedding(models, processor.for_arcface())

    embedding, emb_timings = await _run_in_executor(
        COMPARE_EMBED_EXECUTOR,
        _extract_embedding_sync,
    )
    timings["arcface_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(emb_timings)
    response_payload = {
        "embedding": embedding.astype(np.float32).tolist(),
        "face": _build_face_payload(face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            face,
            img,
            models,
            validate_attrs,
            validate_qual,
            processor=processor,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={"embedding_dim": int(embedding.shape[0])},
    )
    return response_payload


@router.post("/similarity")
@_endpoint_guard
async def compute_similarity(
    request: Request,
    image: UploadFile = File(...),
    embeddingStored: str = Form(...),
    validateAttributes: Optional[str] = Form(None),
    validateQuality: Optional[str] = Form(None),
):
    models = request.app.state.models
    validate_attrs = _parse_bool_form(validateAttributes)
    validate_qual = _parse_bool_form(validateQuality)

    try:
        stored_list = json.loads(embeddingStored)
        if not isinstance(stored_list, list) or len(stored_list) != 512:
            return _error_response(
                400,
                "INVALID_FIELD_FORMAT",
                "Invalid Field Format",
                "embeddingStored must be a JSON array of 512 floats",
            )
        stored_emb = np.asarray(stored_list, dtype=np.float32)
        # Validate and normalize stored embedding to prevent corrupt data
        if not np.all(np.isfinite(stored_emb)):
            return _error_response(
                400,
                "INVALID_FIELD_FORMAT",
                "Invalid Field Format",
                "embeddingStored contains non-finite values (NaN/Inf)",
            )
        emb_norm = float(np.linalg.norm(stored_emb))
        if emb_norm < 1e-6:
            return _error_response(
                400,
                "INVALID_FIELD_FORMAT",
                "Invalid Field Format",
                "embeddingStored is a zero vector",
            )
        # Re-normalize to unit vector (embeddings must be L2-normalized for cosine similarity)
        stored_emb = stored_emb / emb_norm
    except (json.JSONDecodeError, ValueError, TypeError):
        return _error_response(
            400,
            "INVALID_FIELD_FORMAT",
            "Invalid Field Format",
            "embeddingStored must be valid JSON array",
        )

    img = await read_image(image)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    face, processor, err = await _run_in_executor(
        GENERAL_EXECUTOR,
        _detect_and_validate,
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    def _compute_similarity_sync():
        query_emb_local, emb_timings_local = _get_face_embedding(models, processor.for_arcface())
        cosine_sim_local = models.face_recognizer.cosine_similarity(query_emb_local, stored_emb)
        similarity_percent_local = models.face_recognizer.similarity_to_percent(cosine_sim_local)
        return query_emb_local, emb_timings_local, similarity_percent_local

    query_emb, emb_timings, similarity_percent = await _run_in_executor(
        COMPARE_EMBED_EXECUTOR,
        _compute_similarity_sync,
    )
    timings["arcface_similarity_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(emb_timings)

    response_payload = {
        "similarity": _round2(similarity_percent),
        "queryEmbedding": query_emb.astype(np.float32).tolist(),
        "face": _build_face_payload(face, models),
        "validation": await _run_in_executor(
            GENERAL_EXECUTOR,
            _build_validation_payload,
            face,
            img,
            models,
            validate_attrs,
            validate_qual,
            processor=processor,
        ),
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={"similarity": response_payload["similarity"]},
    )
    return response_payload

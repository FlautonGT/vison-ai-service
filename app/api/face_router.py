"""Face inference API endpoints (stateless, no database)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
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
    raw_pct = face.score * 100.0
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
        "confidence": _round2(confidence),
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

    faces = models.face_detector.detect_all(image)
    if faces:
        return faces

    if not enable_retries:
        return []

    height, width = image.shape[:2]
    resized, scale = _resize_for_detection(image, DETECTION_MAX_DIM_RETRY)
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
        faces = models.face_detector.detect_all(image, score_threshold=threshold)
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
        faces = _detect_faces_with_tiling(image, models, deadline=deadline)
        if faces:
            return faces

    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 200:
        logger.info("Face detection retries exhausted in %.0fms (budget=%.0fms)", elapsed_ms, DETECTION_TIME_BUDGET_SEC * 1000)
    return []


def _run_deepfake_fusion(models, image: np.ndarray, processor: Optional[FaceProcessor], face: Optional[FaceDetectionResult]):
    timings: dict[str, float] = {}

    # Pre-compute crop once (cached in processor).
    deepfake_face_crop = processor.for_deepfake() if processor is not None else None

    ai_face_crop = None
    small_face_for_ai_crop = False
    if processor is not None and face is not None:
        ai_face_crop = deepfake_face_crop
        small_face_for_ai_crop = _face_area_ratio(face, image) < float(settings.FACE_MIN_AREA_RATIO)

    do_crop_check = (settings.AI_FACE_ALWAYS_CROP_CHECK or small_face_for_ai_crop) and ai_face_crop is not None and ai_face_crop.size > 0

    # --- Branch 1: Face-swap detector ---
    def _branch_faceswap():
        if deepfake_face_crop is None:
            return {"isDeepfake": False, "attackRiskLevel": "LOW_RISK", "attackTypes": [], "score": 0.0}, 0.0
        start = time.perf_counter()
        result = models.deepfake_detector.predict(deepfake_face_crop)
        return result, (time.perf_counter() - start) * 1000.0

    # --- Branch 2: AI primary detector (full + crop) ---
    def _branch_ai_primary():
        ai_detector = getattr(models, "ai_face_detector", None)
        if not ai_detector or ai_detector.session is None:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0
        start = time.perf_counter()
        result = ai_detector.predict(image)
        full_ms = (time.perf_counter() - start) * 1000.0
        full_score = float(result.get("aiScore", 0.0))
        crop_ms = 0.0
        crop_score = 0.0
        if do_crop_check:
            start = time.perf_counter()
            crop_result = ai_detector.predict(ai_face_crop)
            crop_ms = (time.perf_counter() - start) * 1000.0
            crop_score = float(crop_result.get("aiScore", 0.0))
            if crop_score > full_score:
                result = crop_result
        return result, full_ms, crop_ms, full_score, crop_score

    # --- Branch 3: AI extra detector (full + crop) ---
    def _branch_ai_extra():
        ai_detector_extra = getattr(models, "ai_face_detector_extra", None)
        if not ai_detector_extra or ai_detector_extra.session is None:
            return {"isAIGenerated": False, "aiScore": 0.0}, 0.0, 0.0, 0.0, 0.0
        start = time.perf_counter()
        result = ai_detector_extra.predict(image)
        full_ms = (time.perf_counter() - start) * 1000.0
        full_score = float(result.get("aiScore", 0.0))
        crop_ms = 0.0
        crop_score = 0.0
        if do_crop_check:
            start = time.perf_counter()
            crop_result = ai_detector_extra.predict(ai_face_crop)
            crop_ms = (time.perf_counter() - start) * 1000.0
            crop_score = float(crop_result.get("aiScore", 0.0))
            if crop_score > full_score:
                result = crop_result
        return result, full_ms, crop_ms, full_score, crop_score

    # Run all three branches in parallel.
    fs_future = GENERAL_EXECUTOR.submit(_branch_faceswap)
    ai_p_future = GENERAL_EXECUTOR.submit(_branch_ai_primary)
    ai_e_future = GENERAL_EXECUTOR.submit(_branch_ai_extra)

    face_swap_result, fs_ms = fs_future.result()
    timings["deepfake_faceswap_ms"] = _round2(fs_ms)

    ai_primary_result, ai_p_full_ms, ai_p_crop_ms, ai_primary_full_score, ai_primary_crop_score = ai_p_future.result()
    timings["deepfake_ai_model_full_ms"] = _round2(ai_p_full_ms)
    timings["deepfake_ai_model_crop_ms"] = _round2(ai_p_crop_ms)
    timings["deepfake_ai_model_ms"] = _round2(ai_p_full_ms + ai_p_crop_ms)

    ai_extra_result, ai_e_full_ms, ai_e_crop_ms, ai_extra_full_score, ai_extra_crop_score = ai_e_future.result()
    timings["deepfake_ai_model_extra_full_ms"] = _round2(ai_e_full_ms)
    timings["deepfake_ai_model_extra_crop_ms"] = _round2(ai_e_crop_ms)
    timings["deepfake_ai_model_extra_ms"] = _round2(ai_e_full_ms + ai_e_crop_ms)

    face_swap_score = float(face_swap_result.get("score", 0.0))
    ai_primary_score = float(ai_primary_result.get("aiScore", 0.0))
    ai_extra_score = float(ai_extra_result.get("aiScore", 0.0))
    ai_evidence_scores = [
        float(ai_primary_full_score),
        float(ai_primary_crop_score),
        float(ai_extra_full_score),
        float(ai_extra_crop_score),
        float(ai_primary_score),
        float(ai_extra_score),
    ]

    ai_weights = []
    ai_weighted_scores = []
    if timings["deepfake_ai_model_ms"] > 0.0:
        ai_weights.append(max(0.0, float(settings.AI_FACE_PRIMARY_WEIGHT)))
        ai_weighted_scores.append(ai_primary_score)
    if timings["deepfake_ai_model_extra_ms"] > 0.0:
        ai_weights.append(max(0.0, float(settings.AI_FACE_EXTRA_WEIGHT)))
        ai_weighted_scores.append(ai_extra_score)

    if ai_weighted_scores:
        weight_sum = sum(ai_weights)
        if weight_sum <= 0.0:
            ai_score = float(sum(ai_weighted_scores) / len(ai_weighted_scores))
        else:
            ai_score = float(sum(w * s for w, s in zip(ai_weights, ai_weighted_scores)) / weight_sum)
    else:
        ai_score = 0.0

    # For small/far faces, prefer strongest AI evidence from full-frame or face-crop branch.
    if small_face_for_ai_crop:
        ai_score = max(ai_score, ai_primary_score, ai_extra_score)
    if settings.AI_FACE_ALWAYS_CROP_CHECK:
        ai_score = max(ai_score, ai_primary_full_score, ai_primary_crop_score, ai_extra_full_score, ai_extra_crop_score)

    # Linear calibration for environment-specific tuning (default is identity).
    ai_score = float(
        np.clip(
            ai_score * float(settings.AI_FACE_CALIBRATION_ALPHA) + float(settings.AI_FACE_CALIBRATION_BETA),
            0.0,
            100.0,
        )
    )

    ai_threshold = float(settings.AI_FACE_THRESHOLD)
    ai_low_conf_threshold = float(settings.AI_FACE_LOW_CONF_THRESHOLD)
    low_conf_face_limit = float(settings.AI_FACE_LOW_CONF_FACE_CONF)

    face_confidence = float(face.score * 100.0) if face is not None else 0.0
    low_conf_or_missing_face = (face is None) or (face_confidence <= low_conf_face_limit)
    ai_generated_primary = ai_score >= ai_threshold
    ai_generated_low_conf = low_conf_or_missing_face and ai_score >= ai_low_conf_threshold
    hard_block_threshold = float(settings.AI_FACE_HARD_BLOCK_THRESHOLD)
    vote_threshold = float(settings.AI_FACE_VOTE_THRESHOLD)
    vote_min_count = max(1, int(settings.AI_FACE_VOTE_MIN_COUNT))
    any_trigger_threshold = float(settings.AI_FACE_ANY_TRIGGER_THRESHOLD)
    evidence_nonzero = [value for value in ai_evidence_scores if value > 0.0]
    vote_count = sum(1 for value in evidence_nonzero if value >= vote_threshold)
    ai_generated_hard_block = bool(evidence_nonzero and max(evidence_nonzero) >= hard_block_threshold)
    ai_generated_vote = vote_count >= vote_min_count
    ai_generated_any_trigger = bool(evidence_nonzero and max(evidence_nonzero) >= any_trigger_threshold)
    consensus_ai_threshold = float(settings.AI_FACE_CONSENSUS_AI_THRESHOLD)
    consensus_face_swap_threshold = float(settings.AI_FACE_CONSENSUS_FACE_SWAP_THRESHOLD)
    ai_generated_consensus = False
    if consensus_ai_threshold <= 100.0:
        ai_generated_consensus = (ai_score >= consensus_ai_threshold) and (
            face_swap_score >= consensus_face_swap_threshold
        )
    ai_generated = (
        ai_generated_primary
        or ai_generated_low_conf
        or ai_generated_consensus
        or ai_generated_hard_block
        or ai_generated_vote
        or (ai_generated_any_trigger and face_swap_score >= consensus_face_swap_threshold)
    )

    face_swap_threshold = float(settings.DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD)
    face_swap_flag = face_swap_score >= face_swap_threshold
    likely_real_suppressed = False
    if settings.AI_FACE_REAL_SUPPRESS_ENABLED and not face_swap_flag and face is not None:
        skip_real_suppress = (
            (small_face_for_ai_crop and ai_score >= ai_threshold)
            or ai_generated_hard_block
            or ai_generated_vote
        )
        if not skip_real_suppress:
            if (
                face_confidence >= float(settings.AI_FACE_REAL_SUPPRESS_FACE_CONF)
                and ai_score < float(settings.AI_FACE_REAL_SUPPRESS_AI_MAX)
                and face_swap_score < float(settings.AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX)
            ):
                ai_generated = False
                likely_real_suppressed = True

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
            "faceSwapScore": _round2(face_swap_score),
            "aiGeneratedScore": _round2(ai_score),
        },
    }
    if settings.DEBUG:
        logger.info(
            "Deepfake fusion: face_conf=%.2f small_face=%s ai=%.2f ai_primary=%.2f (full=%.2f crop=%.2f) ai_extra=%.2f (full=%.2f crop=%.2f) fs=%.2f high_th=%.2f low_th=%.2f hard_th=%.2f vote_th=%.2f vote_count=%d vote_min=%d any_th=%.2f low_conf_limit=%.2f consensus_ai_th=%.2f consensus_fs_th=%.2f low_conf=%s consensus=%s hard=%s vote=%s real_suppressed=%s deepfake=%s",
            face_confidence,
            small_face_for_ai_crop,
            ai_score,
            ai_primary_score,
            ai_primary_full_score,
            ai_primary_crop_score,
            ai_extra_score,
            ai_extra_full_score,
            ai_extra_crop_score,
            face_swap_score,
            ai_threshold,
            ai_low_conf_threshold,
            hard_block_threshold,
            vote_threshold,
            vote_count,
            vote_min_count,
            any_trigger_threshold,
            low_conf_face_limit,
            consensus_ai_threshold,
            consensus_face_swap_threshold,
            low_conf_or_missing_face,
            ai_generated_consensus,
            ai_generated_hard_block,
            ai_generated_vote,
            likely_real_suppressed,
            is_deepfake,
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

    if settings.ARCFACE_ADAPTIVE_COMPARE:
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

        avg_quality = float(source_quality.get("score", 0.0) + target_quality.get("score", 0.0)) * 0.5
        if avg_quality < 50.0:
            threshold = min(threshold, float(settings.ARCFACE_ADAPTIVE_LOW))
        elif avg_quality > 80.0:
            threshold = max(threshold, float(settings.ARCFACE_ADAPTIVE_HIGH))

    source_arc = source_proc.for_arcface()
    target_arc = target_proc.for_arcface()
    t0 = time.perf_counter()
    if settings.COMPARE_PARALLEL_EMBEDDING:
        src_future = COMPARE_EMBED_EXECUTOR.submit(_get_face_embedding, models, source_arc)
        tgt_future = COMPARE_EMBED_EXECUTOR.submit(_get_face_embedding, models, target_arc)
        source_emb, source_emb_timings = src_future.result()
        target_emb, target_emb_timings = tgt_future.result()
    else:
        source_emb, source_emb_timings = _get_face_embedding(models, source_arc)
        target_emb, target_emb_timings = _get_face_embedding(models, target_arc)
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
        "validation": _build_validation_payload(
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
    face, processor, err = _detect_and_validate(
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    live_score, is_live = models.liveness_checker.predict(processor.for_liveness())
    timings["liveness_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    response_payload = {
        "isLive": is_live,
        "liveScore": _round2(live_score),
        "face": _build_face_payload(face, models),
        "validation": _build_validation_payload(
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
    face, processor, err = _detect_and_validate(
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    result, fusion_timings = _run_deepfake_fusion(models, img, processor, face)
    timings["deepfake_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(fusion_timings)
    response_payload = {
        "isDeepfake": result["isDeepfake"],
        "attackRiskLevel": result["attackRiskLevel"],
        "attackTypes": result["attackTypes"],
        "scores": result.get("scores", {"faceSwapScore": 0.0, "aiGeneratedScore": 0.0}),
        "face": _build_face_payload(face, models),
        "validation": _build_validation_payload(
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
    face, processor, err = _detect_and_validate(
        img,
        models,
        validate_attrs,
        validate_qual,
    )
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    age_input = img if getattr(models.age_gender, "backend", "") == "insightface" else processor.for_age_gender()
    age_face_crop = processor.for_deepfake()
    t0 = time.perf_counter()
    result = models.age_gender.predict(age_input, face_crop=age_face_crop)
    timings["age_gender_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    response_payload = {
        "gender": result["gender"],
        "genderConfidence": _round2(result["genderConfidence"]),
        "ageRange": result["ageRange"],
        "face": _build_face_payload(face, models),
        "validation": _build_validation_payload(
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
    face, processor, err = _detect_and_validate(
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

    face_landmarks = face.landmarks
    face_bbox = face.bbox

    def _run_quality():
        start = time.perf_counter()
        result = check_quality(img, quality_target, landmarks=face_landmarks, face_bbox=face_bbox)
        return result, (time.perf_counter() - start) * 1000.0

    def _run_liveness():
        start = time.perf_counter()
        value = models.liveness_checker.predict(live_crop)
        return value, (time.perf_counter() - start) * 1000.0

    def _run_deepfake():
        start = time.perf_counter()
        value, detail_timings = _run_deepfake_fusion(models, img, processor, face)
        return value, detail_timings, (time.perf_counter() - start) * 1000.0

    try:
        parallel_start = time.perf_counter()
        quality_future = VERIFY_LIVE_EXECUTOR.submit(_run_quality)
        liveness_future = VERIFY_LIVE_EXECUTOR.submit(_run_liveness)
        deepfake_future = VERIFY_LIVE_EXECUTOR.submit(_run_deepfake)
        quality, quality_ms = quality_future.result()
        (live_score, liveness_is_live), liveness_ms = liveness_future.result()
        deepfake_result, deepfake_detail_timings, deepfake_ms = deepfake_future.result()
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

    final_is_live = bool(liveness_is_live and not deepfake_result.get("isDeepfake", False))
    response_payload = {
        "isLive": final_is_live,
        "liveScore": _round2(live_score),
        "isDeepfake": bool(deepfake_result.get("isDeepfake", False)),
        "attackRiskLevel": deepfake_result.get("attackRiskLevel", "LOW_RISK"),
        "attackTypes": deepfake_result.get("attackTypes", []),
        "face": _build_face_payload(face, models),
        "validation": {"quality": quality},
    }
    _set_request_observability(
        request,
        model_timings=timings,
        result_summary={
            "isLive": response_payload["isLive"],
            "isDeepfake": response_payload["isDeepfake"],
            "attackRiskLevel": response_payload["attackRiskLevel"],
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
    face, processor, err = _detect_and_validate(img, models, validate_attrs, validate_qual)
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    embedding, emb_timings = _get_face_embedding(models, processor.for_arcface())
    timings["arcface_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(emb_timings)
    response_payload = {
        "embedding": embedding.astype(np.float32).tolist(),
        "face": _build_face_payload(face, models),
        "validation": _build_validation_payload(
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
    face, processor, err = _detect_and_validate(img, models, validate_attrs, validate_qual)
    timings["detect_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    if err:
        return err

    t0 = time.perf_counter()
    query_emb, emb_timings = _get_face_embedding(models, processor.for_arcface())
    cosine_sim = models.face_recognizer.cosine_similarity(query_emb, stored_emb)
    similarity_percent = models.face_recognizer.similarity_to_percent(cosine_sim)
    timings["arcface_similarity_ms"] = _round2((time.perf_counter() - t0) * 1000.0)
    timings.update(emb_timings)

    response_payload = {
        "similarity": _round2(similarity_percent),
        "queryEmbedding": query_emb.astype(np.float32).tolist(),
        "face": _build_face_payload(face, models),
        "validation": _build_validation_payload(
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

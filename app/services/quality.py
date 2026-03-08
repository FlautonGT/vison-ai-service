"""Face quality validation with ISO-style metrics."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import settings


def validate_quality(
    image: np.ndarray,
    face_crop: np.ndarray,
    landmarks: np.ndarray | None = None,
    face_bbox: np.ndarray | None = None,
) -> dict:
    """
    Compute quality metrics with backward-compatible keys.

    Required stable keys:
    - score
    - sharpness
    - brightness
    """
    sharpness = _compute_sharpness(face_crop)
    brightness = _compute_brightness(face_crop)
    pose = estimate_pose_from_landmarks(landmarks)
    inter_eye = compute_inter_eye_distance(image, landmarks, face_bbox)
    illum = compute_illumination_uniformity(face_crop)
    contrast = compute_contrast(face_crop)

    # Component scores on 0-100 range.
    sharpness_score = float(np.clip(sharpness, 0.0, 100.0))
    brightness_score = float(np.clip(100.0 - abs(brightness - 55.0) * 2.0, 0.0, 100.0))
    frontal_score = (
        100.0
        if pose["isFrontal"]
        else float(np.clip(100.0 - (abs(pose["yaw"]) + abs(pose["pitch"])) * 2.0, 0.0, 100.0))
    )
    illum_score = float(
        np.clip(
            100.0 - (illum["asymmetry"] / max(settings.QUALITY_MAX_ILLUM_ASYMMETRY, 1e-6)) * 100.0,
            0.0,
            100.0,
        )
    )
    inter_eye_score = float(
        np.clip(
            (inter_eye["pixels"] / max(settings.QUALITY_MIN_INTER_EYE_PX, 1e-6)) * 100.0,
            0.0,
            100.0,
        )
    )
    contrast_score = float(np.clip((contrast / max(settings.QUALITY_MIN_CONTRAST, 1e-6)) * 100.0, 0.0, 100.0))

    # Weighted composite score (ISO-style approximation).
    score = (
        sharpness_score * 0.25
        + brightness_score * 0.15
        + frontal_score * 0.20
        + illum_score * 0.15
        + inter_eye_score * 0.15
        + contrast_score * 0.10
    )

    return {
        "score": round(float(np.clip(score, 0.0, 100.0)), 2),
        "sharpness": round(sharpness_score, 2),
        "brightness": round(float(np.clip(brightness, 0.0, 100.0)), 2),
        "pose": {
            "yaw": round(float(pose["yaw"]), 2),
            "pitch": round(float(pose["pitch"]), 2),
            "isFrontal": bool(pose["isFrontal"]),
        },
        "interEyeDistance": {
            "pixels": round(float(inter_eye["pixels"]), 2),
            "ratio": round(float(inter_eye["ratio"]), 4),
            "adequate": bool(inter_eye["adequate"]),
        },
        "illumination": {
            "leftBrightness": round(float(illum["leftBrightness"]), 2),
            "rightBrightness": round(float(illum["rightBrightness"]), 2),
            "asymmetry": round(float(illum["asymmetry"]), 2),
            "isUniform": bool(illum["isUniform"]),
        },
        "contrast": round(float(contrast), 2),
    }


def check_quality_passed(quality: dict) -> bool:
    """Threshold checks for quality gate."""
    return (
        quality["score"] >= settings.QUALITY_MIN_SCORE
        and quality["sharpness"] >= settings.QUALITY_MIN_SHARPNESS
        and quality["brightness"] >= settings.QUALITY_MIN_BRIGHTNESS
        and quality["brightness"] <= settings.QUALITY_MAX_BRIGHTNESS
    )


def _to_gray(face_crop: np.ndarray) -> np.ndarray:
    if face_crop is None or face_crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if len(face_crop.shape) == 2:
        return face_crop
    return cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)


def _compute_sharpness(face_crop: np.ndarray) -> float:
    """Laplacian variance with logarithmic normalization for high-variance regions."""
    gray = _to_gray(face_crop)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(max(0.0, laplacian.var()))
    return float(min(100.0, 50.0 * np.log1p(variance / 50.0)))


def _compute_brightness(face_crop: np.ndarray) -> float:
    if face_crop is None or face_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return float(v_channel.mean() / 255.0 * 100.0)


def estimate_pose_from_landmarks(landmarks: np.ndarray | None) -> dict:
    """Estimate rough yaw/pitch in degrees from 5-point landmarks."""
    if landmarks is None:
        return {"yaw": 0.0, "pitch": 0.0, "isFrontal": True}
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return {"yaw": 0.0, "pitch": 0.0, "isFrontal": True}

    left_eye = pts[0]
    right_eye = pts[1]
    nose = pts[2]

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-6:
        return {"yaw": 0.0, "pitch": 0.0, "isFrontal": True}

    left_nose = float(np.linalg.norm(nose - left_eye))
    right_nose = float(np.linalg.norm(nose - right_eye))
    yaw = float(np.clip(((left_nose - right_nose) / eye_dist) * 55.0, -45.0, 45.0))

    eye_mid = (left_eye + right_eye) * 0.5
    pitch_ratio = float((nose[1] - eye_mid[1]) / eye_dist)
    pitch = float(np.clip((pitch_ratio - 0.55) * 80.0, -45.0, 45.0))

    is_frontal = (
        abs(yaw) <= float(settings.QUALITY_POSE_MAX_ABS_DEG)
        and abs(pitch) <= float(settings.QUALITY_POSE_MAX_ABS_DEG)
    )
    return {"yaw": yaw, "pitch": pitch, "isFrontal": is_frontal}


def compute_inter_eye_distance(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    face_bbox: np.ndarray | None,
) -> dict:
    h, w = image.shape[:2]
    pixels = 0.0
    if landmarks is not None:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 2:
            pixels = float(np.linalg.norm(pts[1] - pts[0]))
    if pixels <= 0.0 and face_bbox is not None:
        x1, y1, x2, y2 = np.asarray(face_bbox, dtype=np.float32).tolist()
        pixels = float(max(0.0, x2 - x1) * 0.35)
    ratio = float(pixels / max(float(max(h, w)), 1.0))
    adequate = pixels >= float(settings.QUALITY_MIN_INTER_EYE_PX)
    return {"pixels": pixels, "ratio": ratio, "adequate": adequate}


def compute_illumination_uniformity(face_crop: np.ndarray) -> dict:
    gray = _to_gray(face_crop)
    if gray.size == 0:
        return {
            "leftBrightness": 0.0,
            "rightBrightness": 0.0,
            "asymmetry": 100.0,
            "isUniform": False,
        }
    h, w = gray.shape[:2]
    mid = max(1, w // 2)
    left = gray[:, :mid]
    right = gray[:, mid:]
    left_b = float(left.mean() / 255.0 * 100.0) if left.size else 0.0
    right_b = float(right.mean() / 255.0 * 100.0) if right.size else 0.0
    denom = max((left_b + right_b) * 0.5, 1e-6)
    asym = float(abs(left_b - right_b) / denom * 100.0)
    is_uniform = asym <= float(settings.QUALITY_MAX_ILLUM_ASYMMETRY)
    return {
        "leftBrightness": left_b,
        "rightBrightness": right_b,
        "asymmetry": asym,
        "isUniform": is_uniform,
    }


def compute_contrast(face_crop: np.ndarray) -> float:
    gray = _to_gray(face_crop)
    return float(gray.std())


def compute_face_size_ratio(bbox: np.ndarray, image_shape: tuple) -> float:
    """Ratio of face area to image area."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    image_area = w * h
    return float(face_area / max(image_area, 1))

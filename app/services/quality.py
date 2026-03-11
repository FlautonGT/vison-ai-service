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
    attributes: dict | None = None,
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
    face_symmetry = compute_face_symmetry(face_crop, landmarks)
    occlusion_score = compute_occlusion_score(attributes)
    background_clutter = compute_background_clutter(image, face_bbox)

    # Component scores on 0-100 range.
    sharpness_score = float(np.clip(sharpness, 0.0, 100.0))
    brightness_target = float(settings.QUALITY_BRIGHTNESS_TARGET)
    brightness_score = float(np.clip(100.0 - abs(brightness - brightness_target) * 2.0, 0.0, 100.0))
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
    background_clean_score = float(np.clip(100.0 - background_clutter, 0.0, 100.0))

    # Weighted composite score (ISO-style approximation).
    score = (
        sharpness_score * 0.22
        + brightness_score * 0.13
        + frontal_score * 0.15
        + illum_score * 0.12
        + inter_eye_score * 0.10
        + contrast_score * 0.08
        + float(face_symmetry["score"]) * 0.10
        + occlusion_score * 0.07
        + background_clean_score * 0.03
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
        "faceSymmetry": {
            "score": round(float(face_symmetry["score"]), 2),
            "brightnessAsymmetry": round(float(face_symmetry["brightnessAsymmetry"]), 2),
            "eyeSymmetryError": round(float(face_symmetry["eyeSymmetryError"]), 2),
        },
        "occlusionScore": round(float(occlusion_score), 2),
        "backgroundClutter": round(float(background_clutter), 2),
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


def compute_face_symmetry(face_crop: np.ndarray, landmarks: np.ndarray | None) -> dict:
    illum = compute_illumination_uniformity(face_crop)
    brightness_asymmetry = float(np.clip(illum["asymmetry"], 0.0, 100.0))

    eye_symmetry_error = 0.0
    if landmarks is not None:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 3:
            left_eye = pts[0]
            right_eye = pts[1]
            nose = pts[2]
            eye_dist = float(np.linalg.norm(right_eye - left_eye))
            if eye_dist > 1e-6:
                eye_level = abs(float(left_eye[1] - right_eye[1])) / eye_dist * 100.0
                eye_mid_x = float((left_eye[0] + right_eye[0]) * 0.5)
                nose_center = abs(float(nose[0] - eye_mid_x)) / eye_dist * 100.0
                eye_symmetry_error = float(np.clip((eye_level * 0.6) + (nose_center * 0.4), 0.0, 100.0))

    score = float(np.clip(100.0 - (brightness_asymmetry * 0.55 + eye_symmetry_error * 0.45), 0.0, 100.0))
    return {
        "score": score,
        "brightnessAsymmetry": brightness_asymmetry,
        "eyeSymmetryError": eye_symmetry_error,
    }


def compute_occlusion_score(attributes: dict | None) -> float:
    if not attributes:
        return 100.0

    mask_cov = float(attributes.get("maskCoverage", 0.55 if attributes.get("hasMask") else 0.0))
    hat_cov = float(attributes.get("hatCoverage", 0.18 if attributes.get("hasHat") else 0.0))
    glasses_cov = float(attributes.get("glassesCoverage", 0.12 if attributes.get("hasGlasses") else 0.0))
    occluded_fraction = float(np.clip(mask_cov + hat_cov + glasses_cov, 0.0, 1.0))
    return float(np.clip((1.0 - occluded_fraction) * 100.0, 0.0, 100.0))


def compute_background_clutter(image: np.ndarray, face_bbox: np.ndarray | None) -> float:
    gray = _to_gray(image)
    if gray.size == 0:
        return 0.0

    mask = np.ones_like(gray, dtype=bool)
    if face_bbox is not None:
        x1, y1, x2, y2 = np.asarray(face_bbox, dtype=np.float32).tolist()
        h, w = gray.shape[:2]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = int(round(bw * 0.15))
        pad_y = int(round(bh * 0.15))
        bx1 = max(0, int(x1 - pad_x))
        by1 = max(0, int(y1 - pad_y))
        bx2 = min(w, int(x2 + pad_x))
        by2 = min(h, int(y2 + pad_y))
        mask[by1:by2, bx1:bx2] = False

    if int(np.sum(mask)) < 100:
        return 0.0

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges[mask] > 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    texture_strength = float(np.std(laplacian[mask]))

    score = edge_density * 160.0 + min(40.0, texture_strength * 0.8)
    return float(np.clip(score, 0.0, 100.0))


def compute_face_size_ratio(bbox: np.ndarray, image_shape: tuple) -> float:
    """Ratio of face area to image area."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    image_area = w * h
    return float(face_area / max(image_area, 1))

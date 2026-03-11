"""Derived face attributes built from parsing and quality signals."""

from __future__ import annotations

import cv2
import numpy as np


def _clip_score(value: float) -> float:
    return float(np.clip(float(value), 0.0, 100.0))


def _face_crop_gray(face_crop: np.ndarray) -> np.ndarray:
    if face_crop is None or face_crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if face_crop.ndim == 2:
        return face_crop
    return cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)


def _eye_darkness_score(face_crop: np.ndarray, landmarks: np.ndarray | None) -> float:
    gray = _face_crop_gray(face_crop)
    if gray.size <= 1:
        return 0.0

    face_mean = float(np.mean(gray))
    if landmarks is None:
        upper_half = gray[: max(1, gray.shape[0] // 2), :]
        upper_mean = float(np.mean(upper_half))
        return _clip_score((face_mean - upper_mean) * 1.5)

    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0

    patches = []
    eye_dist = max(float(np.linalg.norm(pts[1] - pts[0])), 1.0)
    radius = max(4, int(round(eye_dist * 0.18)))
    for point in pts[:2]:
        cx = int(round(float(point[0])))
        cy = int(round(float(point[1])))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(gray.shape[1], cx + radius)
        y2 = min(gray.shape[0], cy + radius)
        patch = gray[y1:y2, x1:x2]
        if patch.size > 0:
            patches.append(float(np.mean(patch)))

    if not patches:
        return 0.0
    eye_mean = float(np.mean(patches))
    return _clip_score((face_mean - eye_mean) * 2.0)


def infer_sunglasses(face_crop: np.ndarray, landmarks: np.ndarray | None, parser_attributes: dict | None) -> tuple[bool, float]:
    parser_attributes = parser_attributes or {}
    if not parser_attributes.get("hasGlasses"):
        return False, 0.0

    darkness = _eye_darkness_score(face_crop, landmarks)
    glasses_coverage = float(parser_attributes.get("glassesCoverage", 0.0)) * 100.0
    score = _clip_score(darkness * 0.65 + glasses_coverage * 0.35)
    return bool(score >= 45.0), score


def _brightness_label(value: float) -> str:
    if value < 25.0:
        return "UNDEREXPOSED"
    if value > 85.0:
        return "OVEREXPOSED"
    return "GOOD"


def _illumination_label(asymmetry: float) -> str:
    if asymmetry <= 20.0:
        return "GOOD"
    if asymmetry <= 35.0:
        return "MARGINAL"
    return "UNEVEN"


def _sharpness_label(value: float) -> str:
    if value >= 60.0:
        return "SHARP"
    if value >= 35.0:
        return "SOFT"
    return "BLURRY"


def build_attribute_report(
    face_crop: np.ndarray,
    landmarks: np.ndarray | None,
    parser_attributes: dict | None,
    quality_payload: dict | None,
) -> dict:
    parser_attributes = parser_attributes or {}
    quality_payload = quality_payload or {}

    sunglasses, sunglasses_score = infer_sunglasses(face_crop, landmarks, parser_attributes)
    mask = bool(parser_attributes.get("hasMask"))
    hat = bool(parser_attributes.get("hasHat"))
    glasses = bool(parser_attributes.get("hasGlasses")) and not sunglasses
    face_visible_ratio = float(parser_attributes.get("faceVisibleRatio", 1.0))

    occlusion_ratio = max(
        1.0 - face_visible_ratio,
        float(parser_attributes.get("maskCoverage", 0.0)),
        float(parser_attributes.get("hatCoverage", 0.0)),
    )
    if sunglasses:
        occlusion_ratio = max(occlusion_ratio, sunglasses_score / 100.0 * 0.35)

    occlusion_reasons: list[str] = []
    if mask:
        occlusion_reasons.append("mask")
    if hat:
        occlusion_reasons.append("hat")
    if sunglasses:
        occlusion_reasons.append("sunglasses")
    if glasses:
        occlusion_reasons.append("eyeglasses")

    brightness = float(quality_payload.get("brightness", 0.0))
    illumination = dict(quality_payload.get("illumination", {}) or {})
    asymmetry = float(illumination.get("asymmetry", 0.0))
    sharpness = float(quality_payload.get("sharpness", 0.0))

    return {
        "eyeglasses": glasses,
        "sunglasses": sunglasses,
        "mask": mask,
        "hatCap": hat,
        "facialHair": bool(parser_attributes.get("hasBeard")),
        "majorOcclusion": bool(occlusion_ratio >= 0.35),
        "occlusion": {
            "score": _clip_score((1.0 - occlusion_ratio) * 100.0),
            "visibleRatio": round(float(np.clip(face_visible_ratio, 0.0, 1.0)), 4),
            "reasons": occlusion_reasons,
        },
        "brightness": {
            "score": round(_clip_score(brightness), 2),
            "label": _brightness_label(brightness),
        },
        "illumination": {
            "asymmetry": round(_clip_score(asymmetry), 2),
            "isUniform": bool(illumination.get("isUniform", asymmetry <= 25.0)),
            "label": _illumination_label(asymmetry),
        },
        "blurSharpness": {
            "score": round(_clip_score(sharpness), 2),
            "label": _sharpness_label(sharpness),
        },
        "raw": {
            "parser": parser_attributes,
            "sunglassesScore": round(_clip_score(sunglasses_score), 2),
        },
    }

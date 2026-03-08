"""Centralized face post-processing (alignment + model-specific crops)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# ArcFace canonical 5-point template for 112x112.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # mouth left
        [70.7299, 92.2041],  # mouth right
    ],
    dtype=np.float32,
)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Estimate 2x3 similarity transform with Umeyama SVD method."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 2 or src.shape[1] != 2:
        return None

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    var_src = float(np.sum(src_demean**2) / src.shape[0])
    if var_src <= 1e-12:
        return None

    cov = (dst_demean.T @ src_demean) / src.shape[0]
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None

    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        s[-1] *= -1.0
        rotation = u @ vt

    scale = float(np.sum(s) / var_src)
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = (scale * rotation[:2, :2]).astype(np.float32)
    matrix[:, 2] = translation[:2].astype(np.float32)
    return matrix


def normalize_lighting(face_crop: np.ndarray) -> np.ndarray:
    """CLAHE normalization for low-light photos (e.g. Indonesian indoor selfies).

    Only applies when the L-channel mean is below 100 (genuinely dark image).
    Returns a copy — never modifies the input array.
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    try:
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        if float(l_channel.mean()) < 100:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return face_crop.copy()


def normalize_lighting_for_arcface(face_crop: np.ndarray) -> np.ndarray:
    """CLAHE normalization specifically for ArcFace embeddings.

    Uses configurable L-threshold (default 110, higher than deepfake's 100)
    to cover more indoor selfies common in SEA environments.
    Returns a copy — never modifies the input array.
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    if not settings.ARCFACE_CLAHE_ENABLED:
        return face_crop.copy()
    try:
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        if float(l_channel.mean()) < float(settings.ARCFACE_CLAHE_L_THRESHOLD):
            grid = int(settings.ARCFACE_CLAHE_GRID_SIZE)
            clahe = cv2.createCLAHE(
                clipLimit=float(settings.ARCFACE_CLAHE_CLIP_LIMIT),
                tileGridSize=(grid, grid),
            )
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return face_crop.copy()


def estimate_jpeg_quality_from_array(image: np.ndarray) -> int:
    """Estimate JPEG compression quality from pixel statistics.

    Uses a heuristic based on 8x8 block boundary discontinuity — a
    hallmark of JPEG compression.  Returns 0-100 where lower = heavier
    compression.  Works purely on the decoded numpy array (no access to
    the original file bytes).
    """
    if image is None or image.size == 0:
        return 95  # assume high quality if we can't estimate

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        if h < 16 or w < 16:
            return 95

        gray_f = gray.astype(np.float32)

        # Measure discontinuity at 8-pixel boundaries vs interior
        h_boundary = np.abs(gray_f[:, 8::8] - gray_f[:, 7::8]) if w > 16 else np.array([0.0])
        v_boundary = np.abs(gray_f[8::8, :] - gray_f[7::8, :]) if h > 16 else np.array([0.0])

        h_interior = np.abs(gray_f[:, 1::8] - gray_f[:, ::8]) if w > 16 else np.array([1.0])
        v_interior = np.abs(gray_f[1::8, :] - gray_f[::8, :]) if h > 16 else np.array([1.0])

        boundary_mean = float((h_boundary.mean() + v_boundary.mean()) * 0.5)
        interior_mean = float((h_interior.mean() + v_interior.mean()) * 0.5)

        if interior_mean < 1e-6:
            return 95

        # Ratio > 1.0 indicates JPEG blocking artifacts.
        # Higher ratio → lower quality.
        ratio = boundary_mean / interior_mean
        if ratio <= 1.0:
            return 95
        elif ratio <= 1.1:
            return 85
        elif ratio <= 1.3:
            return 75
        elif ratio <= 1.6:
            return 65
        elif ratio <= 2.0:
            return 55
        else:
            return 45
    except Exception:
        return 95


@dataclass
class FaceProcessor:
    """Prepare one detected face for all downstream models."""

    image: np.ndarray
    face: object
    debug: bool = settings.DEBUG

    def __post_init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def _debug_log(self, message: str, *args):
        if self.debug:
            logger.info(message, *args)

    def _raw_crop(self) -> np.ndarray:
        if "raw_crop" not in self._cache:
            crop = self.face.crop_face(self.image)
            if crop is None or crop.size == 0:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)
            self._cache["raw_crop"] = crop
            self._debug_log("FaceProcessor raw crop shape=%s", crop.shape)
        return self._cache["raw_crop"]

    def _get_landmarks(self) -> np.ndarray | None:
        landmarks = getattr(self.face, "landmarks", None)
        if landmarks is None:
            return None
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 5:
            return None
        return pts[:5]

    def _bbox_xyxy(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = np.asarray(self.face.bbox, dtype=np.float32).tolist()
        return float(x1), float(y1), float(x2), float(y2)

    def aligned_face(self, size: int = 112) -> np.ndarray:
        cache_key = f"aligned_{size}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        landmarks = self._get_landmarks()
        if landmarks is None:
            aligned = cv2.resize(self._raw_crop(), (size, size))
            self._debug_log("FaceProcessor alignment fallback (no landmarks), size=%d", size)
        else:
            dst = ARCFACE_TEMPLATE_112 * (size / 112.0)
            x1, y1, x2, y2 = self._bbox_xyxy()
            face_w = max(1.0, x2 - x1)
            face_h = max(1.0, y2 - y1)
            min_side = min(face_w, face_h)

            # Upscale source image before alignment when face region is very small (e.g. ID card photo).
            upsample = 1.0
            if min_side < 96.0:
                upsample = min(4.0, 96.0 / max(min_side, 1.0))
            elif min_side < 140.0:
                upsample = 1.8

            src_image = self.image
            src_landmarks = landmarks
            if upsample > 1.0:
                src_image = cv2.resize(
                    self.image,
                    None,
                    fx=upsample,
                    fy=upsample,
                    interpolation=cv2.INTER_CUBIC,
                )
                src_landmarks = landmarks * upsample
                self._debug_log(
                    "FaceProcessor tiny-face upsample=%.2f min_side=%.1f",
                    upsample,
                    min_side,
                )

            matrix = _umeyama_similarity(src_landmarks, dst)
            if matrix is None:
                aligned = cv2.resize(self._raw_crop(), (size, size))
                self._debug_log("FaceProcessor alignment fallback (affine failed), size=%d", size)
            else:
                aligned = cv2.warpAffine(
                    src_image,
                    matrix,
                    (size, size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                self._debug_log("FaceProcessor aligned face size=%d shape=%s", size, aligned.shape)

        self._cache[cache_key] = aligned
        return aligned

    def expanded_bbox_crop(self, scale: float = 2.7) -> np.ndarray:
        cache_key = f"expanded_{scale:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = np.asarray(self.face.bbox, dtype=np.float32).tolist()

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        new_w = bw * scale
        new_h = bh * scale

        crop_x1 = max(0, int(cx - new_w * 0.5))
        crop_y1 = max(0, int(cy - new_h * 0.5))
        crop_x2 = min(w, int(cx + new_w * 0.5))
        crop_y2 = min(h, int(cy + new_h * 0.5))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            crop = self._raw_crop()
        else:
            crop = self.image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop is None or crop.size == 0:
                crop = self._raw_crop()

        self._cache[cache_key] = crop
        self._debug_log(
            "FaceProcessor expanded crop scale=%.2f bbox=(%d,%d,%d,%d) shape=%s",
            scale,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            crop.shape,
        )
        return crop

    def centered_crop(self, size: int = 96, scale: float = 1.5) -> np.ndarray:
        cache_key = f"centered_{size}_{scale:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = self._bbox_xyxy()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        side = max(bw, bh) * scale

        crop_x1 = max(0, int(cx - side * 0.5))
        crop_y1 = max(0, int(cy - side * 0.5))
        crop_x2 = min(w, int(cx + side * 0.5))
        crop_y2 = min(h, int(cy + side * 0.5))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            crop = self._raw_crop()
        else:
            crop = self.image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop is None or crop.size == 0:
                crop = self._raw_crop()

        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        self._cache[cache_key] = crop
        self._debug_log(
            "FaceProcessor centered crop size=%d scale=%.2f bbox=(%d,%d,%d,%d) shape=%s",
            size,
            scale,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            crop.shape,
        )
        return crop

    def for_ai_detection(self, target_size: int = 224) -> np.ndarray:
        """Loose bbox crop for AI-generated face detection.

        Uses a configurable expansion scale (default 1.3x) around the face
        bounding box — NO ArcFace warpAffine alignment.  This preserves:
        - JPEG blocking artifacts (8x8 grid patterns)
        - GAN checkerboard artifacts in frequency domain
        - Skin texture micro-patterns
        - Compression noise profiles

        Uses INTER_LINEAR (not INTER_AREA) to preserve high-frequency
        artifact signatures that AI detectors rely on.
        """
        crop_scale = float(settings.AI_DETECT_CROP_SCALE)
        cache_key = f"ai_detect_{target_size}_{crop_scale:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = self._bbox_xyxy()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        side = max(bw, bh) * crop_scale

        crop_x1 = max(0, int(cx - side * 0.5))
        crop_y1 = max(0, int(cy - side * 0.5))
        crop_x2 = min(w, int(cx + side * 0.5))
        crop_y2 = min(h, int(cy + side * 0.5))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            crop = self._raw_crop()
        else:
            crop = self.image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop is None or crop.size == 0:
                crop = self._raw_crop()

        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        self._cache[cache_key] = crop
        self._debug_log(
            "FaceProcessor AI detection crop size=%d scale=%.2f bbox=(%d,%d,%d,%d) shape=%s",
            target_size,
            crop_scale,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            crop.shape,
        )
        return crop

    def for_arcface(self) -> np.ndarray:
        """112x112 aligned face with optional CLAHE normalization for ArcFace."""
        cache_key = "arcface_clahe"
        if cache_key in self._cache:
            return self._cache[cache_key]
        aligned = self.aligned_face(112)
        result = normalize_lighting_for_arcface(aligned)
        self._cache[cache_key] = result
        return result

    def for_age_gender(self) -> np.ndarray:
        return self.aligned_face(96)

    def for_age_gender_hires(self) -> np.ndarray:
        """224x224 aligned face for ViT/FairFace age estimation (avoids 96→224 upsampling loss)."""
        return self.aligned_face(224)

    def for_liveness(self) -> np.ndarray:
        return self.expanded_bbox_crop(scale=2.7)

    def for_deepfake(self) -> np.ndarray:
        # Detector preprocess will resize per model (224/384/etc).
        return self.aligned_face(224)

    def for_attributes(self) -> np.ndarray:
        return self._raw_crop()

    def for_quality(self) -> np.ndarray:
        return self._raw_crop()

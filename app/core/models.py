"""Model registry and ONNX model wrappers for pure inference service."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

try:
    import resource  # Unix only
except ImportError:  # pragma: no cover
    resource = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None

logger = logging.getLogger(__name__)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    denom = np.clip(np.sum(exp, axis=-1, keepdims=True), 1e-9, None)
    return exp / denom


def _build_session_options():
    if ort is None:
        return None
    from app.core.config import settings

    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    if settings.ONNX_INTRA_OP_THREADS > 0:
        options.intra_op_num_threads = int(settings.ONNX_INTRA_OP_THREADS)
    if settings.ONNX_INTER_OP_THREADS > 0:
        options.inter_op_num_threads = int(settings.ONNX_INTER_OP_THREADS)

    opt = settings.ONNX_OPT_LEVEL.strip().lower()
    if opt in ("all", "enable_all", "99"):
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif opt in ("extended", "enable_extended"):
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif opt in ("basic", "enable_basic"):
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt in ("disable", "disabled", "none"):
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    mode = settings.ONNX_EXECUTION_MODE.strip().lower()
    if mode == "parallel":
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return options


def _create_session(model_path: str, providers: Optional[list] = None):
    sess_options = _build_session_options()
    try:
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
    except Exception:
        if model_path.endswith("_int8.onnx"):
            fp32_path = model_path[:-10] + ".onnx"
            if os.path.exists(fp32_path):
                logger.warning(
                    "Failed loading INT8 model (%s), fallback to FP32 (%s)",
                    model_path,
                    fp32_path,
                )
                return ort.InferenceSession(
                    fp32_path,
                    sess_options=sess_options,
                    providers=providers or ["CPUExecutionProvider"],
                )
        raise


def _canonical_model_path(path: str) -> str:
    if not path:
        return ""
    return os.path.normcase(os.path.realpath(path))


def _resolve_model_path(model_dir: str, filename: str, prefer_int8: bool = True) -> str:
    if not filename:
        return ""

    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = Path(model_dir) / candidate
    base_candidate = candidate

    candidates: list[Path] = []
    if prefer_int8 and candidate.suffix.lower() == ".onnx" and not candidate.stem.endswith("_int8"):
        candidates.append(candidate.with_name(f"{candidate.stem}_int8{candidate.suffix}"))
    candidates.append(base_candidate)

    for item in candidates:
        if item.exists():
            return str(item)
    return str(base_candidate)


class ModelRegistry:
    """Central registry holding loaded inference models."""

    def __init__(self):
        self.face_detector: Optional[SCRFDDetector] = None
        self.face_recognizer: Optional[ArcFaceRecognizer] = None
        self.face_recognizer_extra: Optional[ArcFaceRecognizer] = None
        self.adaface_recognizer: Optional[AdaFaceRecognizer] = None
        self.liveness_checker: Optional[LivenessChecker] = None
        self.deepfake_detector: Optional[DeepfakeDetector] = None
        self.ai_face_detector: Optional[AIFaceDetector] = None
        self.ai_face_detector_extra: Optional[AIFaceDetector] = None
        self.npr_detector: Optional[NPRDetector] = None
        self.clip_fake_detector: Optional[CLIPFakeDetector] = None
        self.deepfake_vit_v2: Optional[DeepfakeVitV2Detector] = None
        self.cdcn_liveness: Optional[CDCNLiveness] = None
        self.face_parser: Optional[FaceParser] = None
        self.age_gender: Optional[AgeGenderEstimator] = None
        self._ready = False

    async def load_all(self):
        from app.core.config import settings

        if ort is None:
            raise RuntimeError("onnxruntime is required but not installed")

        model_dir = settings.MODEL_DIR
        providers = settings.ONNX_PROVIDERS

        scrfd_path = _resolve_model_path(model_dir, settings.SCRFD_MODEL, settings.PREFER_INT8_MODELS)
        self.face_detector = SCRFDDetector(
            model_path=scrfd_path,
            providers=providers,
            score_threshold=settings.SCRFD_SCORE_THRESHOLD,
            input_size=(settings.SCRFD_INPUT_SIZE, settings.SCRFD_INPUT_SIZE),
        )

        arcface_path = ""
        for candidate in [settings.ARCFACE_MODEL, "w600k_r50.onnx", "w600k_mbf.onnx"]:
            path = _resolve_model_path(model_dir, candidate, settings.PREFER_INT8_MODELS)
            if os.path.exists(path):
                arcface_path = path
                if candidate != settings.ARCFACE_MODEL:
                    logger.warning(
                        "ArcFace primary '%s' not found, fallback to %s",
                        settings.ARCFACE_MODEL,
                        candidate,
                    )
                break
        if not arcface_path:
            arcface_path = _resolve_model_path(model_dir, settings.ARCFACE_MODEL, settings.PREFER_INT8_MODELS)
        self.face_recognizer = ArcFaceRecognizer(
            arcface_path,
            providers=providers,
            use_flip_aug=settings.ARCFACE_FLIP_AUG,
        )
        extra_candidate = settings.ARCFACE_EXTRA_MODEL.strip()
        self.face_recognizer_extra = None
        if extra_candidate:
            extra_path = _resolve_model_path(model_dir, extra_candidate, settings.PREFER_INT8_MODELS)
            primary_name = os.path.basename(arcface_path).lower()
            if os.path.exists(extra_path) and os.path.basename(extra_path).lower() != primary_name:
                self.face_recognizer_extra = ArcFaceRecognizer(
                    extra_path,
                    providers=providers,
                    use_flip_aug=settings.ARCFACE_FLIP_AUG,
                )
                logger.info("ArcFace extra model loaded for embedding ensemble: %s", extra_path)
            elif os.path.basename(extra_path).lower() == primary_name:
                logger.info("ArcFace extra model skipped (same as primary): %s", extra_candidate)
            else:
                logger.warning("ArcFace extra model not found: %s", extra_path)

        liveness_paths = [
            _resolve_model_path(model_dir, name, settings.PREFER_INT8_MODELS) for name in settings.liveness_model_list
        ]
        self.liveness_checker = LivenessChecker(
            model_paths=liveness_paths,
            providers=providers,
            threshold=settings.LIVENESS_THRESHOLD,
        )

        deepfake_paths = [
            _resolve_model_path(model_dir, name, settings.PREFER_INT8_MODELS) for name in settings.deepfake_model_list
        ]
        self.deepfake_detector = DeepfakeDetector(
            model_paths=deepfake_paths,
            providers=providers,
            threshold=settings.DEEPFAKE_THRESHOLD,
        )
        ai_face_model_path = _resolve_model_path(model_dir, settings.AI_FACE_DETECTOR_MODEL, settings.PREFER_INT8_MODELS)
        if not os.path.exists(ai_face_model_path):
            fallback_ai_model = _resolve_model_path(
                model_dir,
                "deepfake_efficientnet_b0.onnx",
                settings.PREFER_INT8_MODELS,
            )
            if os.path.exists(fallback_ai_model):
                logger.warning(
                    "AI face detector model '%s' not found, fallback to deepfake_efficientnet_b0.onnx",
                    settings.AI_FACE_DETECTOR_MODEL,
                )
                ai_face_model_path = fallback_ai_model
        deepfake_session_by_path: dict[str, object] = {}
        if self.deepfake_detector is not None:
            for item in self.deepfake_detector.models:
                canonical = _canonical_model_path(str(item.get("path", "")))
                session = item.get("session")
                if canonical and session is not None:
                    deepfake_session_by_path[canonical] = session

        shared_session = deepfake_session_by_path.get(_canonical_model_path(ai_face_model_path))
        if shared_session is not None:
            logger.info("AI face detector sharing deepfake session: %s", ai_face_model_path)
        self.ai_face_detector = AIFaceDetector(
            model_path=ai_face_model_path,
            providers=providers,
            threshold_percent=settings.AI_FACE_THRESHOLD,
            shared_session=shared_session,
        )
        extra_ai_model_path = _resolve_model_path(
            model_dir,
            settings.AI_FACE_EXTRA_DETECTOR_MODEL,
            settings.PREFER_INT8_MODELS,
        )
        extra_shared_session = deepfake_session_by_path.get(_canonical_model_path(extra_ai_model_path))
        if extra_shared_session is not None:
            logger.info("AI face extra detector sharing deepfake session: %s", extra_ai_model_path)
        self.ai_face_detector_extra = AIFaceDetector(
            model_path=extra_ai_model_path,
            providers=providers,
            threshold_percent=settings.AI_FACE_THRESHOLD,
            positive_indices=[0, 1],
            shared_session=extra_shared_session,
        )

        # --- AdaFace (Phase 3, disabled by default) ---
        self.adaface_recognizer = None
        if settings.ADAFACE_ENABLED:
            adaface_path = _resolve_model_path(model_dir, settings.ADAFACE_MODEL, settings.PREFER_INT8_MODELS)
            try:
                self.adaface_recognizer = AdaFaceRecognizer(adaface_path, providers=providers)
                if not self.adaface_recognizer.is_loaded:
                    logger.info("AdaFace model not available (model not found), skipping")
                    self.adaface_recognizer = None
                else:
                    logger.info("AdaFace loaded for embedding fusion: %s", adaface_path)
            except Exception:
                logger.warning("Failed to load AdaFace model, skipping", exc_info=True)
                self.adaface_recognizer = None

        # --- Optional enhanced models (graceful skip if not found) ---
        npr_path = _resolve_model_path(model_dir, settings.NPR_MODEL, settings.PREFER_INT8_MODELS)
        try:
            self.npr_detector = NPRDetector(npr_path, providers=providers)
            if not self.npr_detector.is_loaded:
                logger.info("NPR detector not available (model not found), skipping")
                self.npr_detector = None
        except Exception:
            logger.warning("Failed to load NPR detector, skipping", exc_info=True)
            self.npr_detector = None

        clip_fake_path = _resolve_model_path(model_dir, settings.CLIP_FAKE_MODEL, settings.PREFER_INT8_MODELS)
        try:
            self.clip_fake_detector = CLIPFakeDetector(clip_fake_path, providers=providers)
            if not self.clip_fake_detector.is_loaded:
                logger.info("CLIP fake detector not available (model not found), skipping")
                self.clip_fake_detector = None
        except Exception:
            logger.warning("Failed to load CLIP fake detector, skipping", exc_info=True)
            self.clip_fake_detector = None

        self.deepfake_vit_v2 = None
        if settings.DEEPFAKE_VIT_V2_ENABLED:
            deepfake_vit_v2_path = _resolve_model_path(
                model_dir,
                settings.DEEPFAKE_VIT_V2_MODEL,
                settings.PREFER_INT8_MODELS,
            )
            deepfake_vit_v2_shared_session = deepfake_session_by_path.get(
                _canonical_model_path(deepfake_vit_v2_path)
            )
            if deepfake_vit_v2_shared_session is not None:
                logger.info("Deepfake ViT v2 sharing deepfake session: %s", deepfake_vit_v2_path)
            try:
                self.deepfake_vit_v2 = DeepfakeVitV2Detector(
                    deepfake_vit_v2_path,
                    providers=providers,
                    threshold_percent=settings.DEEPFAKE_VIT_V2_THRESHOLD,
                    shared_session=deepfake_vit_v2_shared_session,
                )
                if not self.deepfake_vit_v2.is_loaded:
                    logger.info("Deepfake ViT v2 not available (model not found), skipping")
                    self.deepfake_vit_v2 = None
                else:
                    logger.info("Deepfake ViT v2 loaded for deepfake fusion: %s", deepfake_vit_v2_path)
            except Exception:
                logger.warning("Failed to load Deepfake ViT v2, skipping", exc_info=True)
                self.deepfake_vit_v2 = None

        cdcn_path = _resolve_model_path(model_dir, settings.CDCN_MODEL, settings.PREFER_INT8_MODELS)
        try:
            self.cdcn_liveness = CDCNLiveness(cdcn_path, providers=providers)
            if not self.cdcn_liveness.is_loaded:
                logger.info("CDCN liveness not available (model not found), skipping")
                self.cdcn_liveness = None
        except Exception:
            logger.warning("Failed to load CDCN liveness, skipping", exc_info=True)
            self.cdcn_liveness = None

        self.face_parser = FaceParser(
            model_path=_resolve_model_path(model_dir, settings.FACE_PARSING_MODEL, settings.PREFER_INT8_MODELS),
            providers=providers,
        )

        self.age_gender = AgeGenderEstimator(
            model_path=_resolve_model_path(model_dir, settings.AGE_GENDER_MODEL, settings.PREFER_INT8_MODELS),
            providers=providers,
        )

        # --- MiVOLO (Phase 3, disabled by default) ---
        if settings.MIVOLO_ENABLED and self.age_gender is not None:
            mivolo_path = _resolve_model_path(model_dir, settings.MIVOLO_MODEL, settings.PREFER_INT8_MODELS)
            try:
                mivolo = MiVOLOEstimator(mivolo_path, providers=providers)
                if mivolo.is_loaded:
                    self.age_gender.mivolo_estimator = mivolo
                    self.age_gender.mivolo_weight = max(0.0, float(settings.MIVOLO_WEIGHT))
                    logger.info("MiVOLO loaded for age/gender fusion: %s", mivolo_path)
                else:
                    logger.info("MiVOLO model not available (model not found), skipping")
            except Exception:
                logger.warning("Failed to load MiVOLO model, skipping", exc_info=True)

        if self.face_detector is None or self.face_detector.session is None:
            raise RuntimeError("Required model SCRFD is not loaded")
        if self.face_recognizer is None or self.face_recognizer.session is None:
            raise RuntimeError("Required model ArcFace is not loaded")
        if self.liveness_checker is None or self.liveness_checker.model_count < 1:
            raise RuntimeError("At least one liveness model is required")
        if self.deepfake_detector is None or self.deepfake_detector.model_count < 1:
            raise RuntimeError("At least one deepfake model is required")
        if self.face_parser is None or self.face_parser.session is None:
            raise RuntimeError("Required model FaceParser is not loaded")
        if self.age_gender is None or not self.age_gender.is_loaded:
            raise RuntimeError("Required model AgeGender is not loaded")

        self._ready = True

    def unload_all(self):
        self.face_detector = None
        self.face_recognizer = None
        self.face_recognizer_extra = None
        self.adaface_recognizer = None
        self.liveness_checker = None
        self.deepfake_detector = None
        self.ai_face_detector = None
        self.ai_face_detector_extra = None
        self.npr_detector = None
        self.clip_fake_detector = None
        self.deepfake_vit_v2 = None
        self.cdcn_liveness = None
        self.face_parser = None
        self.age_gender = None
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def get_memory_usage_mb(self) -> float:
        if resource is None:
            return 0.0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0

    def get_status(self) -> dict:
        return {
            "ready": self._ready,
            "faceDetector": bool(self.face_detector and self.face_detector.session is not None),
            "faceRecognizer": bool(self.face_recognizer and self.face_recognizer.session is not None),
            "faceRecognizerExtra": bool(
                self.face_recognizer_extra and self.face_recognizer_extra.session is not None
            ),
            "livenessModels": int(self.liveness_checker.model_count if self.liveness_checker else 0),
            "deepfakeModels": int(self.deepfake_detector.model_count if self.deepfake_detector else 0),
            "aiFaceDetector": bool(self.ai_face_detector and self.ai_face_detector.session is not None),
            "aiFaceDetectorExtra": bool(
                self.ai_face_detector_extra and self.ai_face_detector_extra.session is not None
            ),
            "nprDetector": bool(self.npr_detector and self.npr_detector.is_loaded),
            "clipFakeDetector": bool(self.clip_fake_detector and self.clip_fake_detector.is_loaded),
            "deepfakeVitV2": bool(self.deepfake_vit_v2 and self.deepfake_vit_v2.is_loaded),
            "cdcnLiveness": bool(self.cdcn_liveness and self.cdcn_liveness.is_loaded),
            "faceParser": bool(self.face_parser and self.face_parser.session is not None),
            "ageGender": bool(self.age_gender and self.age_gender.is_loaded),
            "fairFace": bool(
                self.age_gender
                and hasattr(self.age_gender, "fairface_estimator")
                and self.age_gender.fairface_estimator.is_loaded
            ),
            "adaFace": bool(self.adaface_recognizer and self.adaface_recognizer.is_loaded),
            "miVOLO": bool(
                self.age_gender
                and hasattr(self.age_gender, "mivolo_estimator")
                and self.age_gender.mivolo_estimator is not None
                and self.age_gender.mivolo_estimator.is_loaded
            ),
        }


class FaceDetectionResult:
    """Result item from face detector."""

    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        landmarks: Optional[np.ndarray] = None,
        image_width: int = 0,
        image_height: int = 0,
    ):
        self.bbox = bbox
        self.score = score
        self.landmarks = landmarks
        self.image_width = image_width
        self.image_height = image_height

    @property
    def face_detected(self) -> bool:
        return self.score > 0.0

    @property
    def bounding_box_relative(self) -> Optional[dict]:
        if self.image_width <= 0 or self.image_height <= 0:
            return None
        x1, y1, x2, y2 = self.bbox
        return {
            "width": round(float((x2 - x1) / self.image_width), 2),
            "height": round(float((y2 - y1) / self.image_height), 2),
            "left": round(float(x1 / self.image_width), 2),
            "top": round(float(y1 / self.image_height), 2),
        }

    @property
    def landmarks_relative(self) -> Optional[list]:
        if self.landmarks is None or self.image_width <= 0 or self.image_height <= 0:
            return None
        pts = self.landmarks
        mapping = [
            ("leftEye", pts[0]),
            ("rightEye", pts[1]),
            ("leftPupil", pts[0]),
            ("rightPupil", pts[1]),
            ("noseTip", pts[2]),
            ("mouthLeft", pts[3]),
            ("mouthRight", pts[4]),
        ]
        result = []
        for name, pt in mapping:
            result.append(
                {
                    "name": name,
                    "x": f"{float(pt[0] / self.image_width):.2f}",
                    "y": f"{float(pt[1] / self.image_height):.2f}",
                }
            )
        return result

    def crop_face(self, image: np.ndarray, margin: float = 0.1) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.bbox
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        x1 = max(0, int(x1 - fw * margin))
        y1 = max(0, int(y1 - fh * margin))
        x2 = min(w, int(x2 + fw * margin))
        y2 = min(h, int(y2 + fh * margin))
        if y2 <= y1 or x2 <= x1:
            return image
        return image[y1:y2, x1:x2]


class SCRFDDetector:
    """SCRFD detector with decode over strides 8, 16, 32."""

    INPUT_SIZE = (640, 640)
    NMS_THRESHOLD = 0.4

    def __init__(
        self,
        model_path: str,
        providers: Optional[list] = None,
        score_threshold: float = 0.5,
        input_size: Optional[tuple[int, int]] = None,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = self.NMS_THRESHOLD
        if input_size is None:
            self.input_size = self.INPUT_SIZE
        else:
            iw = max(64, int(input_size[0]))
            ih = max(64, int(input_size[1]))
            self.input_size = (iw, ih)
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.session = None
        self.input_name = None
        self.output_names: list[str] = []
        self.batched = False
        self.fmc = 3
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2
        self.use_kps = True
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}

        if ort is None:
            logger.warning("onnxruntime is not available; SCRFD disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("SCRFD model not found: %s", model_path)
            return

        self.session = _create_session(model_path, providers=providers)
        self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
        self._init_runtime_meta()
        logger.info("SCRFD loaded: %s", model_path)

    def _init_runtime_meta(self):
        if self.session is None:
            return
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.batched = bool(outputs and len(outputs[0].shape) == 3)

        output_count = len(outputs)
        if output_count == 6:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = False
        elif output_count == 9:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = True
        elif output_count == 10:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = False
        elif output_count == 15:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = True

    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    @staticmethod
    def _nms(dets: np.ndarray, thresh: float) -> list[int]:
        if dets.shape[0] == 0:
            return []
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1.0)
            h = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter

            iou = np.zeros_like(inter)
            valid = union > 0
            iou[valid] = inter[valid] / union[valid]

            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _get_anchor_centers(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        key = (feat_h, feat_w, stride)
        cached = self.center_cache.get(key)
        if cached is not None:
            return cached

        centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))

        if len(self.center_cache) < 128:
            self.center_cache[key] = centers
        return centers

    def detect(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        faces = self.detect_all(image)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def detect_all(self, image: np.ndarray, score_threshold: Optional[float] = None) -> list[FaceDetectionResult]:
        if self.session is None or self.input_name is None:
            return []
        if image is None or image.ndim != 3:
            return []
        threshold = float(self.score_threshold if score_threshold is None else score_threshold)

        image_h, image_w = image.shape[:2]
        input_w, input_h = self.input_size

        im_ratio = float(image_h) / float(max(image_w, 1))
        model_ratio = float(input_h) / float(input_w)
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = input_w
            new_h = max(1, int(new_w * im_ratio))

        det_scale = float(new_h) / float(max(image_h, 1))
        resized = cv2.resize(image, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128.0,
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )

        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        if not net_outs:
            return []

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(self.feat_stride_fpn):
            if idx >= self.fmc:
                break

            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                kps_preds = net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                kps_preds = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None

            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            bbox_preds = np.asarray(bbox_preds, dtype=np.float32).reshape(-1, 4)
            if self.use_kps and kps_preds is not None:
                kps_preds = np.asarray(kps_preds, dtype=np.float32).reshape(-1, 10)

            feat_h = input_h // stride
            feat_w = input_w // stride
            anchor_centers = self._get_anchor_centers(feat_h, feat_w, stride)

            count = min(len(scores), len(bbox_preds), len(anchor_centers))
            if self.use_kps and kps_preds is not None:
                count = min(count, len(kps_preds))
            if count <= 0:
                continue

            scores = scores[:count]
            bbox_preds = bbox_preds[:count]
            anchor_centers = anchor_centers[:count]
            if self.use_kps and kps_preds is not None:
                kps_preds = kps_preds[:count]

            pos_inds = np.where(scores >= threshold)[0]
            if pos_inds.size == 0:
                continue

            decoded_boxes = self._distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds][:, np.newaxis])
            bboxes_list.append(decoded_boxes[pos_inds])

            if self.use_kps and kps_preds is not None:
                decoded_kps = self._distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(decoded_kps[pos_inds])

        if not scores_list:
            return []

        scores = np.vstack(scores_list).reshape(-1)
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if kpss_list else None

        order = scores.argsort()[::-1]
        det = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        det = det[order]
        if kpss is not None:
            kpss = kpss[order]

        keep = self._nms(det, self.nms_threshold)
        det = det[keep]
        if kpss is not None:
            kpss = kpss[keep]

        results: list[FaceDetectionResult] = []
        for index, row in enumerate(det):
            x1, y1, x2, y2, score = row
            x1 = float(np.clip(x1, 0, image_w - 1))
            y1 = float(np.clip(y1, 0, image_h - 1))
            x2 = float(np.clip(x2, 0, image_w - 1))
            y2 = float(np.clip(y2, 0, image_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            landmarks = None
            if kpss is not None and index < len(kpss):
                landmarks = kpss[index].astype(np.float32)
                landmarks[:, 0] = np.clip(landmarks[:, 0], 0, image_w - 1)
                landmarks[:, 1] = np.clip(landmarks[:, 1], 0, image_h - 1)

            results.append(
                FaceDetectionResult(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=float(score),
                    landmarks=landmarks,
                    image_width=image_w,
                    image_height=image_h,
                )
            )
        return results

    def check_single_face(self, image: np.ndarray) -> tuple[Optional[FaceDetectionResult], Optional[str]]:
        faces = self.detect_all(image)
        if len(faces) == 0:
            return None, "FACE_NOT_DETECTED"
        if len(faces) > 1:
            return None, "MULTIPLE_FACES"
        return faces[0], None


class ArcFaceRecognizer:
    """ArcFace embedding model wrapper."""

    EMBEDDING_DIM = 512

    def __init__(self, model_path: str, providers: Optional[list] = None, use_flip_aug: bool = True):
        from app.core.config import settings as _cfg

        self.session = None
        self.use_flip_aug = bool(use_flip_aug)
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        # INTER_CUBIC preserves more detail when upsampling small faces to 112x112.
        interp_str = _cfg.ARCFACE_RESIZE_INTERPOLATION.strip().lower()
        if interp_str == "cubic":
            self._resize_interpolation = cv2.INTER_CUBIC
        elif interp_str == "area":
            self._resize_interpolation = cv2.INTER_AREA
        elif interp_str == "lanczos":
            self._resize_interpolation = cv2.INTER_LANCZOS4
        else:
            self._resize_interpolation = cv2.INTER_LINEAR

        if ort is None:
            logger.warning("onnxruntime is not available; ArcFace disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("ArcFace model not found: %s", model_path)
            return
        self.session = _create_session(model_path, providers=providers)
        self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
        logger.info("ArcFace loaded: %s (flip_aug=%s interp=%s)", model_path, self.use_flip_aug, interp_str)

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        aligned = cv2.resize(face_crop, (112, 112), interpolation=self._resize_interpolation)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Official InsightFace ArcFace preprocessing: (pixel - 127.5) / 127.5
        aligned = (aligned - 127.5) / 127.5
        return aligned.transpose(2, 0, 1)[np.newaxis, ...]

    def _infer_embedding(self, blob: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: blob})[0]
        return np.asarray(output[0], dtype=np.float32).reshape(-1)

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("ArcFace model not loaded")
        if face_crop is None or face_crop.size == 0:
            raise ValueError("face_crop is empty")

        blob = self._preprocess(face_crop)
        embedding = self._infer_embedding(blob)

        # Flip-test augmentation improves robustness on pose/quality variance (e.g. selfie vs ID card).
        if self.use_flip_aug:
            flipped = cv2.flip(face_crop, 1)
            flip_blob = self._preprocess(flipped)
            flip_emb = self._infer_embedding(flip_blob)
            embedding = (embedding + flip_emb) * 0.5

        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        a = np.asarray(emb_a, dtype=np.float32).reshape(-1)
        b = np.asarray(emb_b, dtype=np.float32).reshape(-1)
        if a.size == 0 or b.size == 0:
            return 0.0
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return max(-1.0, min(1.0, sim))

    @staticmethod
    def similarity_to_percent(cosine_sim: float) -> float:
        from app.core.config import settings

        sim = max(-1.0, min(1.0, float(cosine_sim)))
        # Keep compatibility when calibration is disabled.
        if not settings.SIMILARITY_CALIBRATION_ENABLED:
            percent = ((sim + 1.0) * 0.5) * 100.0
            return round(float(np.clip(percent, 0.0, 100.0)), 2)

        cos01 = max(0.0, sim)

        # SEA calibration: shifts breakpoints lower to accommodate lower cosine
        # scores typical of SEA same-person pairs (training data bias + image quality).
        if settings.SIMILARITY_SEA_CALIBRATION_ENABLED:
            low = float(settings.SIMILARITY_SEA_COS_THRESHOLD_LOW)    # default 0.15
            mid = float(settings.SIMILARITY_SEA_COS_THRESHOLD_MID)    # default 0.28
            high = float(settings.SIMILARITY_SEA_COS_THRESHOLD_HIGH)  # default 0.42
            vhigh = float(settings.SIMILARITY_SEA_COS_THRESHOLD_VERY_HIGH)  # default 0.55

            if cos01 <= low:
                percent = (cos01 / max(low, 1e-9)) * 50.0
            elif cos01 <= mid:
                percent = 50.0 + ((cos01 - low) / max(mid - low, 1e-9)) * 24.0
            elif cos01 <= high:
                percent = 74.0 + ((cos01 - mid) / max(high - mid, 1e-9)) * 16.0
            elif cos01 <= vhigh:
                percent = 90.0 + ((cos01 - high) / max(vhigh - high, 1e-9)) * 7.0
            else:
                t = min(1.0, (cos01 - vhigh) / max(1.0 - vhigh, 1e-9))
                percent = 97.0 + (t ** 0.3) * 2.9
        else:
            # Original calibration (Western-tuned breakpoints).
            # Piecewise calibration on cosine [0..1] domain.
            # Calibrated to produce AWS-comparable scores:
            #   cos 0.0  -> 0%     (completely different)
            #   cos 0.2  -> 50%    (different person)
            #   cos 0.35 -> 74%    (threshold boundary)
            #   cos 0.5  -> 90%    (likely same person)
            #   cos 0.6  -> 97%    (same person, different photo)
            #   cos 0.7+ -> 99%+   (same person, high confidence)
            if cos01 <= 0.2:
                percent = (cos01 / 0.2) * 50.0
            elif cos01 <= 0.35:
                percent = 50.0 + ((cos01 - 0.2) / 0.15) * 24.0
            elif cos01 <= 0.5:
                percent = 74.0 + ((cos01 - 0.35) / 0.15) * 16.0
            elif cos01 <= 0.6:
                percent = 90.0 + ((cos01 - 0.5) / 0.1) * 7.0
            else:
                # Strong same-person zone (cos > 0.6): map to 97-99.9
                # Use power curve to compress into narrow 97-99.9 band
                t = min(1.0, (cos01 - 0.6) / 0.4)
                percent = 97.0 + (t ** 0.3) * 2.9

        cap = float(np.clip(settings.SIMILARITY_CALIBRATION_CAP, 0.0, 100.0))
        return round(float(np.clip(percent, 0.0, cap)), 2)


class AdaFaceRecognizer:
    """AdaFace embedding model wrapper for optional embedding fusion."""

    EMBEDDING_DIM = 512

    def __init__(self, model_path: str, providers: Optional[list] = None, use_flip_aug: Optional[bool] = None):
        from app.core.config import settings as _cfg

        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.weight = max(0.0, float(_cfg.ADAFACE_WEIGHT))
        self.use_flip_aug = bool(_cfg.ARCFACE_FLIP_AUG if use_flip_aug is None else use_flip_aug)

        interp_str = _cfg.ARCFACE_RESIZE_INTERPOLATION.strip().lower()
        if interp_str == "cubic":
            self._resize_interpolation = cv2.INTER_CUBIC
        elif interp_str == "area":
            self._resize_interpolation = cv2.INTER_AREA
        elif interp_str == "lanczos":
            self._resize_interpolation = cv2.INTER_LANCZOS4
        else:
            self._resize_interpolation = cv2.INTER_LINEAR

        if ort is None:
            logger.warning("onnxruntime is not available; AdaFace disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("AdaFace model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
            logger.info(
                "AdaFace loaded: %s (flip_aug=%s interp=%s weight=%.3f)",
                model_path,
                self.use_flip_aug,
                interp_str,
                self.weight,
            )
        except Exception:
            logger.exception("Failed loading AdaFace model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        aligned = cv2.resize(face_crop, (112, 112), interpolation=self._resize_interpolation)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        aligned = (aligned - 127.5) / 127.5
        return aligned.transpose(2, 0, 1)[np.newaxis, ...]

    def _infer_embedding(self, blob: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]
        embedding = np.asarray(output[0], dtype=np.float32).reshape(-1)
        if embedding.size != self.EMBEDDING_DIM:
            raise ValueError(
                f"AdaFace embedding dimension mismatch: expected {self.EMBEDDING_DIM}, got {embedding.size}"
            )
        return embedding

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("AdaFace model not loaded")
        if face_crop is None or face_crop.size == 0:
            raise ValueError("face_crop is empty")

        embedding = self._infer_embedding(self._preprocess(face_crop))
        if self.use_flip_aug:
            flip_emb = self._infer_embedding(self._preprocess(cv2.flip(face_crop, 1)))
            embedding = (embedding + flip_emb) * 0.5

        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


class LivenessChecker:
    """Ensemble liveness checker with MiniFASNet models."""

    def __init__(self, model_paths: list[str], providers: Optional[list] = None, threshold: float = 0.5):
        from app.core.config import settings

        self.threshold = threshold
        self.providers = providers or ["CPUExecutionProvider"]
        self.debug = settings.DEBUG
        self.models: list[dict] = []

        if ort is None:
            logger.warning("onnxruntime is not available; liveness disabled")
            return

        for path in model_paths:
            if os.path.exists(path):
                session = _create_session(path, providers=self.providers)
                input_meta = session.get_inputs()[0]
                output_meta = session.get_outputs()[0]
                input_shape = list(input_meta.shape)
                output_shape = list(output_meta.shape)

                input_h, input_w = 80, 80
                if len(input_shape) == 4:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                    if isinstance(raw_h, int) and raw_h > 0:
                        input_h = raw_h
                    if isinstance(raw_w, int) and raw_w > 0:
                        input_w = raw_w

                real_index = 1
                if len(output_shape) >= 2 and isinstance(output_shape[-1], int):
                    class_count = output_shape[-1]
                    if class_count >= 3:
                        real_index = 2

                self.models.append(
                    {
                        "path": path,
                        "session": session,
                        "input_size": (input_w, input_h),
                        "real_index": real_index,
                    }
                )
                logger.info(
                    "Liveness model loaded: %s input=%s output=%s real_index=%d",
                    path,
                    input_shape,
                    output_shape,
                    real_index,
                )
            else:
                logger.warning("Liveness model not found: %s", path)

        if not self.models:
            logger.warning("No liveness models loaded")

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list[str]:
        return [os.path.basename(str(item.get("path", ""))) for item in self.models]

    @property
    def sessions(self) -> list:
        return [item["session"] for item in self.models]

    def _log_debug(self, message: str, *args):
        if self.debug:
            logger.info(message, *args)

    def _infer_single(self, model_info: dict, face_crop: np.ndarray) -> float:
        session = model_info["session"]
        input_w, input_h = model_info["input_size"]
        real_index = int(model_info["real_index"])
        path = model_info["path"]

        resized = cv2.resize(face_crop, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: blob})[0]
        probs = np.asarray(output, dtype=np.float32).reshape(-1)

        if probs.size == 1:
            logit = float(probs[0])
            real_prob = float(1.0 / (1.0 + np.exp(-logit)))
            self._log_debug(
                "Liveness raw output model=%s input_shape=%s single_logit=%.6f real_prob=%.4f",
                path,
                blob.shape,
                logit,
                real_prob,
            )
            return real_prob

        if np.any(probs < 0.0) or np.sum(probs) < 0.99 or np.sum(probs) > 1.01:
            probs = _softmax(probs.reshape(1, -1))[0]

        if probs.size >= 3:
            real_prob = float(probs[2])
        elif probs.size > real_index:
            real_prob = float(probs[real_index])
        else:
            real_prob = float(probs[-1])

        self._log_debug(
            "Liveness raw output model=%s input_shape=%s probs=%s real_prob=%.4f",
            path,
            blob.shape,
            probs.tolist(),
            real_prob,
        )
        return real_prob

    def predict(self, face_crop: np.ndarray) -> tuple[float, bool]:
        if face_crop is None or face_crop.size == 0:
            return 0.0, False

        scores = []
        for model_info in self.models:
            try:
                scores.append(self._infer_single(model_info, face_crop))
            except Exception:
                logger.exception("Failed liveness inference on one ensemble model")

        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        avg_score = max(0.0, min(1.0, avg_score))
        score_percent = round(avg_score * 100.0, 2)
        is_live = avg_score > self.threshold
        return score_percent, is_live


class DeepfakeDetector:
    """Ensemble deepfake detector with graceful degradation."""

    def __init__(self, model_paths: list[str], providers: Optional[list] = None, threshold: float = 0.5):
        from app.core.config import settings

        self.threshold = threshold
        self.providers = providers or ["CPUExecutionProvider"]
        self.debug = settings.DEBUG
        self.models: list[dict] = []

        if ort is None:
            logger.warning("onnxruntime is not available; deepfake detector disabled")
            return

        for path in model_paths:
            if not os.path.exists(path):
                logger.warning("Deepfake model not found: %s", path)
                continue
            session = _create_session(path, providers=self.providers)
            preprocess = self._detect_preprocess(session, path)
            fake_index = self._resolve_fake_index(path)
            self.models.append(
                {
                    "session": session,
                    "preprocess": preprocess,
                    "path": path,
                    "fake_index": fake_index,
                }
            )
            logger.info("Deepfake model loaded: %s (fake_index=%d)", path, fake_index)

        if not self.models:
            logger.warning("No deepfake models loaded")

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def model_names(self) -> list[str]:
        return [os.path.basename(str(item.get("path", ""))) for item in self.models]

    @staticmethod
    def _resolve_fake_index(path: str) -> int:
        model_name = os.path.basename(path).lower()
        # These ONNX exports use index 0 as synthetic/fake class in practice.
        if (
            "community_forensics" in model_name
            or "deep_fake_detector_v2" in model_name
            or "deepfake_vit_v2" in model_name
            or "deepfake_efficientnet_b0" in model_name
        ):
            return 0
        if "deep-fake-detector-v2" in model_name:
            return 0
        return 1

    def _detect_preprocess(self, session, path: str) -> Callable[[np.ndarray], np.ndarray]:
        input_meta = session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        input_name = input_meta.name
        model_name = os.path.basename(path).lower()

        is_nhwc = len(input_shape) == 4 and input_shape[-1] == 3
        target_h, target_w = 224, 224
        if len(input_shape) == 4:
            if is_nhwc:
                raw_h, raw_w = input_shape[1], input_shape[2]
            else:
                raw_h, raw_w = input_shape[2], input_shape[3]
            if isinstance(raw_h, int) and raw_h > 0:
                target_h = raw_h
            if isinstance(raw_w, int) and raw_w > 0:
                target_w = raw_w

        # CommunityForensics ViT expects CLIP normalization with 440 resize + center crop 384.
        if "community_forensics" in model_name or "deepfakedet-vit" in model_name:
            target_h, target_w = 440, 440

        logger.info(
            "Deepfake input detected: model=%s input=%s shape=%s layout=%s",
            path,
            input_name,
            input_shape,
            "NHWC" if is_nhwc else "NCHW",
        )
        if (
            "deep_fake_detector_v2" in model_name
            or "deep-fake-detector-v2" in model_name
            or "deepfake_vit_v2" in model_name
            or "deepfake_efficientnet_b0" in model_name
        ):
            if is_nhwc:
                return lambda face_crop: self._preprocess_with_stats_nhwc(
                    face_crop,
                    224,
                    224,
                    np.array([0.5, 0.5, 0.5], dtype=np.float32),
                    np.array([0.5, 0.5, 0.5], dtype=np.float32),
                )
            return lambda face_crop: self._preprocess_with_stats_nchw(
                face_crop,
                224,
                224,
                np.array([0.5, 0.5, 0.5], dtype=np.float32),
                np.array([0.5, 0.5, 0.5], dtype=np.float32),
            )
        if "community_forensics" in model_name or "deepfakedet-vit" in model_name:
            if is_nhwc:
                return self._preprocess_clip_440_center384_nhwc
            return self._preprocess_clip_440_center384_nchw
        if is_nhwc:
            return lambda face_crop: self._preprocess_imagenet_nhwc(face_crop, target_w, target_h)
        return lambda face_crop: self._preprocess_imagenet_nchw(face_crop, target_w, target_h)

    def _preprocess_with_stats_nchw(
        self,
        face_crop: np.ndarray,
        width: int,
        height: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        resized = cv2.resize(face_crop, (width, height))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = (resized - mean) / std
        return resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def _preprocess_with_stats_nhwc(
        self,
        face_crop: np.ndarray,
        width: int,
        height: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        resized = cv2.resize(face_crop, (width, height))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = (resized - mean) / std
        return resized[np.newaxis, ...].astype(np.float32)

    def _preprocess_clip_440_center384_nchw(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (440, 440), interpolation=cv2.INTER_CUBIC)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        start = (440 - 384) // 2
        cropped = resized[start : start + 384, start : start + 384]
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        cropped = (cropped - mean) / std
        return cropped.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def _preprocess_clip_440_center384_nhwc(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (440, 440), interpolation=cv2.INTER_CUBIC)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        start = (440 - 384) // 2
        cropped = resized[start : start + 384, start : start + 384]
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        cropped = (cropped - mean) / std
        return cropped[np.newaxis, ...].astype(np.float32)

    def _preprocess_imagenet_nchw(self, face_crop: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
        return self._preprocess_with_stats_nchw(
            face_crop,
            width,
            height,
            np.array([0.485, 0.456, 0.406], dtype=np.float32),
            np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )

    def _preprocess_imagenet_nhwc(self, face_crop: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
        return self._preprocess_with_stats_nhwc(
            face_crop,
            width,
            height,
            np.array([0.485, 0.456, 0.406], dtype=np.float32),
            np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )

    @staticmethod
    def _extract_fake_probability(output_tensor: np.ndarray, fake_index: int) -> float:
        output = np.asarray(output_tensor, dtype=np.float32)
        if output.size == 0:
            raise ValueError("Deepfake model output is empty")

        if output.ndim >= 2 and output.shape[0] == 1:
            output = output[0]
        output = output.reshape(-1)

        if output.size == 1:
            return float(1.0 / (1.0 + np.exp(-output[0])))

        probs = output[:2]
        if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
            probs = _softmax(probs.reshape(1, -1))[0]
        idx = 0 if fake_index <= 0 else 1
        return float(probs[idx])

    def _infer_single(self, model_info: dict, face_crop: np.ndarray) -> float:
        session = model_info["session"]
        preprocess_fn = model_info["preprocess"]
        fake_index = int(model_info["fake_index"])
        model_path = str(model_info["path"])
        blob = preprocess_fn(face_crop)
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: blob})
        if not raw_outputs:
            raise ValueError("Deepfake model returned no outputs")

        for raw in raw_outputs:
            try:
                fake_prob = self._extract_fake_probability(raw, fake_index=fake_index)
                if self.debug:
                    logger.info(
                        "Deepfake raw output model=%s fake_index=%d input=%s shape=%s parsed_fake_prob=%.4f",
                        model_path,
                        fake_index,
                        input_name,
                        blob.shape,
                        fake_prob,
                    )
                return max(0.0, min(1.0, fake_prob))
            except Exception:
                continue

        raise ValueError("Unable to parse deepfake model outputs")

    def predict(self, face_crop: np.ndarray) -> dict:
        if face_crop is None or face_crop.size == 0:
            return {
                "isDeepfake": False,
                "attackRiskLevel": "LOW_RISK",
                "attackTypes": [],
                "score": 0.0,
            }

        fake_scores = []
        per_model_scores = []
        for model_info in self.models:
            path = str(model_info["path"])
            try:
                score = self._infer_single(model_info, face_crop)
                fake_scores.append(score)
                per_model_scores.append((path, score))
            except Exception:
                logger.exception("Failed deepfake inference on one ensemble model")
                per_model_scores.append((path, None))

        avg_fake = float(sum(fake_scores) / len(fake_scores)) if fake_scores else 0.0
        avg_fake = max(0.0, min(1.0, avg_fake))

        if self.debug:
            logger.info("Deepfake ensemble member scores: %s avg_fake=%.4f", per_model_scores, avg_fake)

        is_deepfake = avg_fake > self.threshold
        if not is_deepfake:
            risk = "LOW_RISK"
        else:
            high_risk_threshold = max(0.85, self.threshold + 0.15)
            risk = "HIGH_RISK" if avg_fake >= high_risk_threshold else "MEDIUM_RISK"

        attack_types = ["SYNTHETIC_IMAGE"] if is_deepfake else []
        return {
            "isDeepfake": is_deepfake,
            "attackRiskLevel": risk,
            "attackTypes": attack_types,
            "score": round(max(0.0, avg_fake) * 100.0, 2),
        }


class AIFaceDetector:
    """Dedicated detector for AI-generated faces."""

    def __init__(
        self,
        model_path: str,
        providers: Optional[list] = None,
        threshold_percent: float = 50.0,
        positive_indices: Optional[Sequence[int]] = None,
        shared_session=None,
    ):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.threshold_percent = max(0.0, min(100.0, float(threshold_percent)))
        self.threshold = self.threshold_percent / 100.0
        self.input_name = "input"
        self.input_width = 224
        self.input_height = 224
        self.is_nhwc = False
        self.positive_indices = [int(i) for i in positive_indices] if positive_indices else None

        if ort is None:
            logger.warning("onnxruntime is not available; AI face detector disabled")
            return
        if shared_session is None and not os.path.exists(model_path):
            logger.warning("AI face detector model not found: %s", model_path)
            return

        if shared_session is not None:
            self.session = shared_session
        else:
            try:
                self.session = _create_session(model_path, providers=self.providers)
            except Exception:
                logger.warning("Failed to load AI face detector model: %s", model_path, exc_info=True)
                self.session = None
                return
        self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        input_shape = list(input_meta.shape)
        if len(input_shape) == 4:
            self.is_nhwc = input_shape[-1] == 3
            if self.is_nhwc:
                raw_h, raw_w = input_shape[1], input_shape[2]
            else:
                raw_h, raw_w = input_shape[2], input_shape[3]
            if isinstance(raw_h, int) and raw_h > 0:
                self.input_height = raw_h
            if isinstance(raw_w, int) and raw_w > 0:
                self.input_width = raw_w

        logger.info(
            "AI face detector loaded: %s input=%s threshold=%.2f%% positive_indices=%s",
            model_path,
            input_shape,
            self.threshold_percent,
            self.positive_indices,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        resized = (resized - mean) / std
        if self.is_nhwc:
            return resized[np.newaxis, ...].astype(np.float32)
        return resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _extract_probabilities(output_tensor: np.ndarray) -> np.ndarray:
        output = np.asarray(output_tensor, dtype=np.float32)
        if output.size == 0:
            return np.array([], dtype=np.float32)
        if output.ndim >= 2 and output.shape[0] == 1:
            output = output[0]
        output = output.reshape(-1)
        if output.size == 1:
            prob = float(1.0 / (1.0 + np.exp(-output[0])))
            return np.array([1.0 - prob, prob], dtype=np.float32)
        probs = output.astype(np.float32)
        if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
            probs = _softmax(probs.reshape(1, -1))[0]
        return probs

    def _resolve_positive_indices(self, class_count: int) -> list[int]:
        if self.positive_indices:
            return [idx for idx in self.positive_indices if 0 <= idx < class_count]

        model_name = self.model_name.lower()
        if class_count >= 3 and "ai_vs_deepfake_vs_real" in model_name:
            # [Artificial, Deepfake, Real]
            return [0, 1]
        # Binary default: index 0 is "fake/synthetic" for currently used detectors.
        return [0]

    def predict(self, image_bgr: np.ndarray) -> dict:
        if self.session is None or image_bgr is None or image_bgr.size == 0:
            return {"isAIGenerated": False, "aiScore": 0.0}
        try:
            blob = self._preprocess(image_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            fake_prob = 0.0
            for output in outputs:
                try:
                    probs = self._extract_probabilities(output)
                    if probs.size == 0:
                        continue
                    positive_indices = self._resolve_positive_indices(int(probs.size))
                    if not positive_indices:
                        continue
                    fake_prob = float(max(probs[idx] for idx in positive_indices))
                    break
                except Exception:
                    continue
            fake_prob = max(0.0, min(1.0, float(fake_prob)))
            ai_score = round(fake_prob * 100.0, 2)
            return {
                "isAIGenerated": fake_prob >= self.threshold,
                "aiScore": ai_score,
            }
        except Exception:
            logger.exception("AI face detector inference failed")
            return {"isAIGenerated": False, "aiScore": 0.0}


class DeepfakeVitV2Detector:
    """Dedicated wrapper for the DeepFake ViT v2 ONNX classifier."""

    def __init__(
        self,
        model_path: str,
        providers: Optional[list] = None,
        threshold_percent: float = 60.0,
        shared_session=None,
    ):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self.threshold_percent = max(0.0, min(100.0, float(threshold_percent)))
        self.threshold = self.threshold_percent / 100.0
        self.input_name = "input"
        self.input_width = 224
        self.input_height = 224
        self.is_nhwc = False

        if ort is None:
            logger.warning("onnxruntime is not available; Deepfake ViT v2 disabled")
            return
        if shared_session is None and not os.path.exists(model_path):
            logger.warning("Deepfake ViT v2 model not found: %s", model_path)
            return

        if shared_session is not None:
            self.session = shared_session
        else:
            try:
                self.session = _create_session(model_path, providers=self.providers)
            except Exception:
                logger.warning("Failed to load Deepfake ViT v2 model: %s", model_path, exc_info=True)
                self.session = None
                return

        self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        input_shape = list(input_meta.shape)
        if len(input_shape) == 4:
            self.is_nhwc = input_shape[-1] == 3
            if self.is_nhwc:
                raw_h, raw_w = input_shape[1], input_shape[2]
            else:
                raw_h, raw_w = input_shape[2], input_shape[3]
            if isinstance(raw_h, int) and raw_h > 0:
                self.input_height = raw_h
            if isinstance(raw_w, int) and raw_w > 0:
                self.input_width = raw_w

        logger.info(
            "Deepfake ViT v2 loaded: %s input=%s threshold=%.2f%%",
            model_path,
            input_shape,
            self.threshold_percent,
        )

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        resized = (resized - mean) / std
        if self.is_nhwc:
            return resized[np.newaxis, ...].astype(np.float32)
        return resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _extract_fake_probability(output_tensor: np.ndarray) -> float:
        output = np.asarray(output_tensor, dtype=np.float32)
        if output.size == 0:
            raise ValueError("Deepfake ViT v2 output is empty")
        if output.ndim >= 2 and output.shape[0] == 1:
            output = output[0]
        output = output.reshape(-1)
        if output.size == 1:
            return float(1.0 / (1.0 + np.exp(-output[0])))
        probs = _softmax(output[:2].reshape(1, -1))[0]
        return float(probs[0])

    def predict(self, image_bgr: np.ndarray) -> dict:
        if self.session is None or image_bgr is None or image_bgr.size == 0:
            return {"isAIGenerated": False, "aiScore": 0.0}

        try:
            blob = self._preprocess(image_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            fake_prob = 0.0
            for output in outputs:
                try:
                    fake_prob = self._extract_fake_probability(output)
                    break
                except Exception:
                    continue
            fake_prob = max(0.0, min(1.0, float(fake_prob)))
            return {
                "isAIGenerated": fake_prob >= self.threshold,
                "aiScore": round(fake_prob * 100.0, 2),
            }
        except Exception:
            logger.exception("Deepfake ViT v2 inference failed")
            return {"isAIGenerated": False, "aiScore": 0.0}


class NPRDetector:
    """Neighboring Pixel Relationships detector for AI-generated image detection.

    Computes 4-directional pixel difference maps and feeds them through a
    ResNet-18 classifier.  Very fast (~15ms on CPU) and robust to JPEG
    compression because it operates on *relative* pixel differences rather
    than absolute values.

    Reference: "Rethinking the Up-Sampling Operations in CNN-based
    Generative Network for Generalizable Deepfake Detection" (CVPR 2024).
    """

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.input_name = "input"
        self.input_h = 224
        self.input_w = 224
        self.is_nhwc = False

        if ort is None:
            logger.warning("onnxruntime is not available; NPR detector disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("NPR detector model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                self.is_nhwc = input_shape[-1] in (3, 12)
                if self.is_nhwc:
                    raw_h, raw_w = input_shape[1], input_shape[2]
                else:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            logger.info("NPR detector loaded: %s input=%s", model_path, input_shape)
        except Exception:
            logger.exception("Failed loading NPR detector model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    @staticmethod
    def compute_npr_features(image_bgr: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
        """Compute 4-directional pixel difference maps.

        Returns a (target_h, target_w, 12) float32 array: 4 directions x 3 channels.
        Directions: right, down, down-right diagonal, down-left diagonal.
        """
        resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32)

        # Right: pixel[y, x+1] - pixel[y, x]
        h_diff = np.zeros_like(img)
        h_diff[:, :-1, :] = img[:, 1:, :] - img[:, :-1, :]

        # Down: pixel[y+1, x] - pixel[y, x]
        v_diff = np.zeros_like(img)
        v_diff[:-1, :, :] = img[1:, :, :] - img[:-1, :, :]

        # Down-right diagonal
        dr_diff = np.zeros_like(img)
        dr_diff[:-1, :-1, :] = img[1:, 1:, :] - img[:-1, :-1, :]

        # Down-left diagonal
        dl_diff = np.zeros_like(img)
        dl_diff[:-1, 1:, :] = img[1:, :-1, :] - img[:-1, 1:, :]

        # Stack: (H, W, 12)
        features = np.concatenate([h_diff, v_diff, dr_diff, dl_diff], axis=2)
        return features

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Build input tensor from NPR features."""
        features = self.compute_npr_features(image_bgr, self.input_h, self.input_w)

        # Determine what the model actually expects
        input_meta = self.session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        n_channels = None
        if len(input_shape) == 4:
            if self.is_nhwc:
                n_channels = input_shape[-1]
            else:
                n_channels = input_shape[1]

        # If model expects 3 channels (standard ResNet trained on RGB NPR visualization)
        # convert the 12-channel NPR map to a 3-channel representation.
        if isinstance(n_channels, int) and n_channels == 3:
            # Compute per-direction magnitude and use as 3-channel image.
            # h_diff mag, v_diff mag, diagonal_avg mag
            h_mag = np.sqrt(np.sum(features[:, :, 0:3] ** 2, axis=2, keepdims=True))
            v_mag = np.sqrt(np.sum(features[:, :, 3:6] ** 2, axis=2, keepdims=True))
            d_mag = np.sqrt(np.sum(features[:, :, 6:12] ** 2, axis=2, keepdims=True)) * 0.5
            vis = np.concatenate([h_mag, v_mag, d_mag], axis=2)
            # Normalize to 0-1 range
            max_val = vis.max()
            if max_val > 1e-6:
                vis = vis / max_val
            # Standard ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            vis = (vis - mean) / std
            if self.is_nhwc:
                return vis[np.newaxis, ...].astype(np.float32)
            return vis.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # Model expects raw 12-channel NPR features
        features = features / 255.0
        if self.is_nhwc:
            return features[np.newaxis, ...].astype(np.float32)
        return features.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bgr: np.ndarray) -> dict:
        """Returns {"isFake": bool, "fakeScore": float 0-100}."""
        if self.session is None or image_bgr is None or image_bgr.size == 0:
            return {"isFake": False, "fakeScore": 0.0}

        try:
            blob = self._preprocess(image_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            output = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            if output.size == 1:
                fake_prob = float(1.0 / (1.0 + np.exp(-output[0])))
            elif output.size >= 2:
                probs = output[:2]
                if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
                    probs = _softmax(probs.reshape(1, -1))[0]
                fake_prob = float(probs[0])  # index 0 = fake for NPR
            else:
                fake_prob = 0.0

            fake_prob = max(0.0, min(1.0, fake_prob))
            from app.core.config import settings
            threshold = float(settings.NPR_THRESHOLD)
            return {
                "isFake": fake_prob >= threshold,
                "fakeScore": round(fake_prob * 100.0, 2),
            }
        except Exception:
            logger.exception("NPR detector inference failed")
            return {"isFake": False, "fakeScore": 0.0}


class CLIPFakeDetector:
    """UniversalFakeDetect — CLIP ViT-B/16 feature extractor + linear probe.

    Best open-source model for detecting AI-generated images across generators
    (Midjourney, Stable Diffusion, DALL-E, Gemini/Imagen).  Operates in CLIP
    feature space so it generalizes to unseen generators.

    Reference: "Towards Universal Fake Image Detectors that Generalize Across
    Generative Models" (CVPR 2023).
    """

    # Official CLIP normalization constants
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.input_name = "input"
        self.input_h = 224
        self.input_w = 224
        self.is_nhwc = False

        if ort is None:
            logger.warning("onnxruntime is not available; CLIP fake detector disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("CLIP fake detector model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                self.is_nhwc = input_shape[-1] == 3
                if self.is_nhwc:
                    raw_h, raw_w = input_shape[1], input_shape[2]
                else:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            logger.info("CLIP fake detector loaded: %s input=%s", model_path, input_shape)
        except Exception:
            logger.exception("Failed loading CLIP fake detector model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """CLIP preprocessing: resize to 224, center crop, normalize."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Resize shortest edge to 224, then center crop
        if h < w:
            new_h = self.input_h
            new_w = max(self.input_w, int(w * self.input_h / max(h, 1)))
        else:
            new_w = self.input_w
            new_h = max(self.input_h, int(h * self.input_w / max(w, 1)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Center crop
        start_y = (new_h - self.input_h) // 2
        start_x = (new_w - self.input_w) // 2
        cropped = resized[start_y:start_y + self.input_h, start_x:start_x + self.input_w]

        # Normalize
        normalized = cropped.astype(np.float32) / 255.0
        normalized = (normalized - self.CLIP_MEAN) / self.CLIP_STD

        if self.is_nhwc:
            return normalized[np.newaxis, ...].astype(np.float32)
        return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bgr: np.ndarray) -> dict:
        """Returns {"isAIGenerated": bool, "aiScore": float 0-100}."""
        if self.session is None or image_bgr is None or image_bgr.size == 0:
            return {"isAIGenerated": False, "aiScore": 0.0}

        try:
            blob = self._preprocess(image_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            output = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            if output.size == 1:
                # Single logit — positive = fake
                fake_prob = float(1.0 / (1.0 + np.exp(-output[0])))
            elif output.size >= 2:
                probs = output[:2]
                if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
                    probs = _softmax(probs.reshape(1, -1))[0]
                # Convention: index 1 = fake for CLIP-based detectors
                fake_prob = float(probs[1]) if probs.size > 1 else float(probs[0])
            else:
                fake_prob = 0.0

            fake_prob = max(0.0, min(1.0, fake_prob))
            from app.core.config import settings
            threshold = float(settings.CLIP_FAKE_THRESHOLD)
            return {
                "isAIGenerated": fake_prob >= threshold,
                "aiScore": round(fake_prob * 100.0, 2),
            }
        except Exception:
            logger.exception("CLIP fake detector inference failed")
            return {"isAIGenerated": False, "aiScore": 0.0}


class CDCNLiveness:
    """Central Difference Convolutional Network for face anti-spoofing.

    Uses central difference convolutions that capture micro-texture patterns
    (moiré, print dots, screen pixels) more effectively than standard CNNs.
    Consistent across skin tones and ethnicities.

    Reference: "Searching Central Difference Convolutional Networks for
    Face Anti-Spoofing" (CVPR 2020).
    """

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.input_name = "input"
        self.input_h = 256
        self.input_w = 256
        self.is_nhwc = False

        if ort is None:
            logger.warning("onnxruntime is not available; CDCN liveness disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("CDCN liveness model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                self.is_nhwc = input_shape[-1] == 3
                if self.is_nhwc:
                    raw_h, raw_w = input_shape[1], input_shape[2]
                else:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            logger.info("CDCN liveness loaded: %s input=%s", model_path, input_shape)
        except Exception:
            logger.exception("Failed loading CDCN liveness model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        if self.is_nhwc:
            return normalized[np.newaxis, ...].astype(np.float32)
        return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, face_crop: np.ndarray) -> tuple[float, bool]:
        """Returns (score_percent, is_live)."""
        if self.session is None or face_crop is None or face_crop.size == 0:
            return 0.0, False

        try:
            blob = self._preprocess(face_crop)
            outputs = self.session.run(None, {self.input_name: blob})
            output = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            if output.size == 1:
                # CDCN outputs a depth map mean or single score
                raw = float(output[0])
                if 0.0 <= raw <= 1.0:
                    live_prob = raw
                else:
                    live_prob = float(1.0 / (1.0 + np.exp(-raw)))
            elif output.size >= 2:
                probs = output[:2]
                if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
                    probs = _softmax(probs.reshape(1, -1))[0]
                live_prob = float(probs[1])  # index 1 = live
            else:
                # Depth map output — mean value indicates liveness
                depth_map = np.asarray(outputs[0], dtype=np.float32)
                live_prob = float(np.clip(depth_map.mean(), 0.0, 1.0))

            live_prob = max(0.0, min(1.0, live_prob))
            from app.core.config import settings
            threshold = float(settings.CDCN_THRESHOLD)
            score_percent = round(live_prob * 100.0, 2)
            return score_percent, live_prob >= threshold
        except Exception:
            logger.exception("CDCN liveness inference failed")
            return 0.0, False


class FaceParser:
    """BiSeNet face parsing for mask/hat/glasses attributes."""

    GLASSES_CLASSES = {6}
    HAT_CLASSES = {18}

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        if ort is None:
            logger.warning("onnxruntime is not available; face parsing disabled")
            return
        if not os.path.exists(model_path):
            logger.warning("Face parsing model not found: %s", model_path)
            return
        self.session = _create_session(model_path, providers=providers)
        self.model_name = os.path.basename(getattr(self.session, "_model_path", model_path))
        logger.info("Face parsing model loaded: %s", model_path)

    def predict_attributes(self, face_crop: np.ndarray) -> dict:
        if self.session is None:
            return {"hasMask": False, "hasHat": False, "hasGlasses": False}
        if face_crop is None or face_crop.size == 0:
            return {"hasMask": False, "hasHat": False, "hasGlasses": False}
        try:
            resized = cv2.resize(face_crop, (512, 512)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = ((resized - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: tensor})[0]
            seg_map = np.argmax(output[0], axis=0)
            classes = set(seg_map.flatten().tolist())

            has_glasses = bool(self.GLASSES_CLASSES & classes)
            has_hat = bool(self.HAT_CLASSES & classes)
            nose_pixels = int(np.sum(seg_map == 10))
            mouth_pixels = int(np.sum((seg_map == 11) | (seg_map == 12) | (seg_map == 13)))
            total_face = int(np.sum(seg_map > 0))
            visible_ratio = (nose_pixels + mouth_pixels) / max(total_face, 1)
            if total_face < 2500:
                has_mask = False
            else:
                has_mask = visible_ratio < 0.08

            return {
                "hasMask": has_mask,
                "hasHat": has_hat,
                "hasGlasses": has_glasses,
            }
        except Exception:
            logger.exception("Face parsing inference failed")
            return {"hasMask": False, "hasHat": False, "hasGlasses": False}


class AgeGenderVitEstimator:
    """Secondary age/gender estimator (ViT) for ensemble fusion."""

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.input_name = "input"
        self.input_w = 224
        self.input_h = 224
        self.is_nhwc = False

        if ort is None:
            return
        if not os.path.exists(model_path):
            logger.warning("Age/gender ViT model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                self.is_nhwc = input_shape[-1] == 3
                if self.is_nhwc:
                    raw_h, raw_w = input_shape[1], input_shape[2]
                else:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            logger.info("Age/gender ViT loaded: %s input=%s", model_path, input_shape)
        except Exception:
            logger.exception("Failed loading age/gender ViT model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        if self.is_nhwc:
            return norm[np.newaxis, ...].astype(np.float32)
        return norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    def predict(self, face_bgr: np.ndarray) -> Optional[dict]:
        if self.session is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            blob = self._preprocess(face_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            if not outputs:
                return None
            vec = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if vec.size < 2:
                return None

            age_raw = float(vec[0])
            gender_raw = float(vec[1])  # model card: >=0.5 => female

            age_value = age_raw * 100.0 if 0.0 <= age_raw <= 1.2 else age_raw
            age_value = float(np.clip(age_value, 0.0, 100.0))

            if 0.0 <= gender_raw <= 1.0:
                female_prob = gender_raw
            else:
                female_prob = self._sigmoid(gender_raw)
            male_prob = 1.0 - female_prob

            gender = "MAN" if male_prob >= female_prob else "WOMAN"
            confidence = max(male_prob, female_prob) * 100.0
            age = int(round(age_value))
            return {
                "gender": gender,
                "genderConfidence": round(confidence, 2),
                "ageRange": {"low": max(0, age - 4), "high": min(100, age + 4)},
                "maleProb": float(np.clip(male_prob, 0.0, 1.0)),
                "ageValue": age_value,
            }
        except Exception:
            logger.exception("Age/gender ViT inference failed")
            return None


class FairFaceEstimator:
    """FairFace age/gender/race estimator — balanced demographics, good for Asian faces.

    Output heads: Race (7 classes), Gender (2 classes), Age (9 buckets).
    Age buckets: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    Race classes: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern
    """

    AGE_BUCKET_CENTERS = [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 77.0]
    RACE_LABELS = [
        "WHITE", "BLACK", "LATINO_HISPANIC",
        "EAST_ASIAN", "SOUTHEAST_ASIAN", "INDIAN", "MIDDLE_EASTERN",
    ]

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.input_name = "input"
        self.input_w = 224
        self.input_h = 224

        if ort is None:
            return
        if not os.path.exists(model_path):
            logger.warning("FairFace model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            logger.info("FairFace loaded: %s input=%s", model_path, input_shape)
        except Exception:
            logger.exception("Failed loading FairFace model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        return norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, face_bgr: np.ndarray) -> Optional[dict]:
        if self.session is None or face_bgr is None or face_bgr.size == 0:
            return None
        blob = self._preprocess(face_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        if not outputs or len(outputs) < 3:
            return None

        race_logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        gender_logits = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        age_logits = np.asarray(outputs[2], dtype=np.float32).reshape(-1)

        # Gender: softmax → male probability
        gender_probs = _softmax(gender_logits.reshape(1, -1))[0]
        male_prob = float(gender_probs[0]) if gender_probs.size >= 2 else 0.5

        # Age: softmax over 9 buckets → weighted sum of bucket centers
        age_probs = _softmax(age_logits.reshape(1, -1))[0]
        age_value = float(np.dot(age_probs[:len(self.AGE_BUCKET_CENTERS)],
                                 self.AGE_BUCKET_CENTERS[:len(age_probs)]))
        age_value = float(np.clip(age_value, 0.0, 100.0))

        # Race: argmax
        race_idx = int(np.argmax(race_logits))
        race_label = self.RACE_LABELS[race_idx] if race_idx < len(self.RACE_LABELS) else "UNKNOWN"

        gender = "MAN" if male_prob >= 0.5 else "WOMAN"
        confidence = max(male_prob, 1.0 - male_prob) * 100.0
        age = int(round(age_value))

        return {
            "gender": gender,
            "genderConfidence": round(confidence, 2),
            "ageRange": {"low": max(0, age - 5), "high": min(100, age + 5)},
            "maleProb": float(np.clip(male_prob, 0.0, 1.0)),
            "ageValue": age_value,
            "race": race_label,
        }

    def predict(self, face_bgr: np.ndarray) -> Optional[dict]:
        if self.session is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            blob = self._preprocess(face_bgr)
            outputs = self.session.run(None, {self.input_name: blob})
            if not outputs or len(outputs) < 3:
                return None

            race_logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            gender_logits = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
            age_logits = np.asarray(outputs[2], dtype=np.float32).reshape(-1)

            gender_probs = _softmax(gender_logits.reshape(1, -1))[0]
            male_prob = float(gender_probs[0]) if gender_probs.size >= 2 else 0.5

            age_probs = _softmax(age_logits.reshape(1, -1))[0]
            age_value = float(
                np.dot(
                    age_probs[:len(self.AGE_BUCKET_CENTERS)],
                    self.AGE_BUCKET_CENTERS[:len(age_probs)],
                )
            )
            age_value = float(np.clip(age_value, 0.0, 100.0))

            race_idx = int(np.argmax(race_logits))
            race_label = self.RACE_LABELS[race_idx] if race_idx < len(self.RACE_LABELS) else "UNKNOWN"

            gender = "MAN" if male_prob >= 0.5 else "WOMAN"
            confidence = max(male_prob, 1.0 - male_prob) * 100.0
            age = int(round(age_value))

            return {
                "gender": gender,
                "genderConfidence": round(confidence, 2),
                "ageRange": {"low": max(0, age - 5), "high": min(100, age + 5)},
                "maleProb": float(np.clip(male_prob, 0.0, 1.0)),
                "ageValue": age_value,
                "race": race_label,
            }
        except Exception:
            logger.exception("FairFace inference failed")
            return None


class MiVOLOEstimator:
    """MiVOLO age/gender estimator for optional fusion."""

    def __init__(self, model_path: str, providers: Optional[list] = None):
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.input_name = "input"
        self.input_w = 224
        self.input_h = 224
        self.is_nhwc = False
        self.output_names: list[str] = []

        if ort is None:
            return
        if not os.path.exists(model_path):
            logger.warning("MiVOLO model not found: %s", model_path)
            return

        try:
            self.session = _create_session(model_path, providers=providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                self.is_nhwc = input_shape[-1] == 3
                if self.is_nhwc:
                    raw_h, raw_w = input_shape[1], input_shape[2]
                else:
                    raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0:
                    self.input_h = raw_h
                if isinstance(raw_w, int) and raw_w > 0:
                    self.input_w = raw_w
            self.output_names = [meta.name for meta in self.session.get_outputs()]
            logger.info(
                "MiVOLO loaded: %s input=%s outputs=%s",
                model_path,
                input_shape,
                self.output_names,
            )
        except Exception:
            logger.exception("Failed loading MiVOLO model")
            self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        if self.is_nhwc:
            return norm[np.newaxis, ...].astype(np.float32)
        return norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    @staticmethod
    def _normalize_age(age_raw: float) -> float:
        age_value = float(age_raw)
        if 0.0 <= age_value <= 1.5:
            age_value *= 100.0
        return float(np.clip(age_value, 0.0, 100.0))

    def _parse_outputs(self, outputs: list[np.ndarray]) -> Optional[tuple[float, float]]:
        vectors = [
            (name.lower(), np.asarray(output, dtype=np.float32).reshape(-1))
            for name, output in zip(self.output_names or ["" for _ in outputs], outputs)
            if np.asarray(output).size > 0
        ]
        if not vectors:
            return None

        age_raw: Optional[float] = None
        gender_vec: Optional[np.ndarray] = None

        for name, vec in vectors:
            if vec.size == 0:
                continue
            if age_raw is None and "age" in name:
                age_raw = float(vec[-1])
            if gender_vec is None and ("gender" in name or "sex" in name):
                gender_vec = vec

        if len(vectors) == 1:
            vec = vectors[0][1]
            if vec.size >= 3:
                if gender_vec is None:
                    gender_vec = vec[:2]
                if age_raw is None:
                    age_raw = float(vec[2])
            elif vec.size == 2 and gender_vec is None:
                if abs(float(vec[0])) > 1.5 and abs(float(vec[1])) <= 8.0:
                    age_raw = float(vec[0])
                    gender_vec = vec[1:2]
                elif abs(float(vec[1])) > 1.5 and abs(float(vec[0])) <= 8.0:
                    age_raw = float(vec[1])
                    gender_vec = vec[:1]
                else:
                    gender_vec = vec
            elif vec.size == 1 and age_raw is None:
                age_raw = float(vec[0])
        else:
            if age_raw is None:
                for _, vec in vectors:
                    if vec.size == 1:
                        age_raw = float(vec[0])
                        break
            if gender_vec is None:
                for _, vec in vectors:
                    if vec.size >= 2:
                        gender_vec = vec[:2]
                        break

        if age_raw is None or gender_vec is None or gender_vec.size == 0:
            return None

        if gender_vec.size == 1:
            raw = float(gender_vec[0])
            male_prob = raw if 0.0 <= raw <= 1.0 else self._sigmoid(raw)
        else:
            probs = np.asarray(gender_vec[:2], dtype=np.float32)
            if np.any(probs < 0.0) or abs(float(np.sum(probs)) - 1.0) > 0.1:
                probs = _softmax(probs.reshape(1, -1))[0]
            male_prob = float(probs[0])

        return self._normalize_age(age_raw), float(np.clip(male_prob, 0.0, 1.0))

    def predict(self, face_bgr: np.ndarray) -> Optional[dict]:
        if self.session is None or face_bgr is None or face_bgr.size == 0:
            return None

        parsed = self._parse_outputs(self.session.run(None, {self.input_name: self._preprocess(face_bgr)}))
        if parsed is None:
            return None

        age_value, male_prob = parsed
        gender = "MAN" if male_prob >= 0.5 else "WOMAN"
        age = int(round(age_value))
        confidence = max(male_prob, 1.0 - male_prob) * 100.0
        return {
            "gender": gender,
            "genderConfidence": round(confidence, 2),
            "ageRange": {"low": max(0, age - 4), "high": min(100, age + 4)},
            "maleProb": male_prob,
            "ageValue": age_value,
        }

    def predict(self, face_bgr: np.ndarray) -> Optional[dict]:
        if self.session is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            parsed = self._parse_outputs(self.session.run(None, {self.input_name: self._preprocess(face_bgr)}))
            if parsed is None:
                return None

            age_value, male_prob = parsed
            gender = "MAN" if male_prob >= 0.5 else "WOMAN"
            age = int(round(age_value))
            confidence = max(male_prob, 1.0 - male_prob) * 100.0
            return {
                "gender": gender,
                "genderConfidence": round(confidence, 2),
                "ageRange": {"low": max(0, age - 4), "high": min(100, age + 4)},
                "maleProb": male_prob,
                "ageValue": age_value,
            }
        except Exception:
            logger.exception("MiVOLO inference failed")
            return None


class AgeGenderEstimator:
    """Age/gender estimator using InsightFace/ONNX + optional ViT + FairFace ensemble."""

    def __init__(self, model_path: str, providers: Optional[list] = None):
        from app.core.config import settings

        self.debug = settings.DEBUG
        self.providers = providers or ["CPUExecutionProvider"]
        self.model_path = model_path
        self.model_dir = str(Path(model_path).resolve().parent)
        self.model_name = os.path.basename(model_path)
        self.primary_weight = max(0.0, float(settings.AGE_GENDER_PRIMARY_WEIGHT))
        self.vit_weight = max(0.0, float(settings.AGE_GENDER_VIT_WEIGHT))
        self.fairface_weight = max(0.0, float(settings.AGE_GENDER_FAIRFACE_WEIGHT))
        self.male_threshold = float(np.clip(settings.AGE_GENDER_MALE_THRESHOLD, 0.0, 1.0))

        self.app = None
        self.session = None
        self.input_name = "input"
        self.input_size = (96, 96)
        self.backend = "none"
        self.is_loaded = False

        self._init_onnx_fallback()
        self._init_insightface_primary()

        vit_path = _resolve_model_path(self.model_dir, settings.AGE_GENDER_VIT_MODEL, settings.PREFER_INT8_MODELS)
        self.vit_estimator = AgeGenderVitEstimator(vit_path, providers=self.providers)

        fairface_path = _resolve_model_path(self.model_dir, settings.AGE_GENDER_FAIRFACE_MODEL, settings.PREFER_INT8_MODELS)
        self.fairface_estimator = FairFaceEstimator(fairface_path, providers=self.providers)

        # MiVOLO is injected externally by ModelRegistry.load_all() when enabled.
        self.mivolo_estimator: Optional[MiVOLOEstimator] = None
        self.mivolo_weight: float = 0.0

        if self.app is not None:
            self.backend = "insightface"
            self.is_loaded = True
        elif self.session is not None:
            self.backend = "onnx_fallback"
            self.is_loaded = True
        elif self.vit_estimator.is_loaded:
            self.backend = "vit_only"
            self.is_loaded = True
        elif self.fairface_estimator.is_loaded:
            self.backend = "fairface_only"
            self.is_loaded = True

        if not self.is_loaded:
            logger.warning("Age/gender estimator failed to initialize")
        else:
            logger.info(
                "Age/gender backend active: %s (vit_loaded=%s, fairface_loaded=%s)",
                self.backend,
                self.vit_estimator.is_loaded,
                self.fairface_estimator.is_loaded,
            )

    @staticmethod
    def _format_result(
        gender: str,
        male_prob: float,
        age_value: float,
        span: int = 4,
        male_threshold: float = 0.5,
    ) -> dict:
        male_prob = float(np.clip(male_prob, 0.0, 1.0))
        age_value = float(np.clip(age_value, 0.0, 100.0))
        age = int(round(age_value))
        if gender not in ("MAN", "WOMAN"):
            gender = "MAN" if male_prob >= male_threshold else "WOMAN"
        return {
            "gender": gender,
            "genderConfidence": round(max(male_prob, 1.0 - male_prob) * 100.0, 2),
            "ageRange": {"low": max(0, age - span), "high": min(100, age + span)},
            "maleProb": male_prob,
            "ageValue": age_value,
        }

    def _init_onnx_fallback(self):
        if ort is None:
            return
        if not os.path.exists(self.model_path):
            logger.warning("Age/gender model not found: %s", self.model_path)
            return
        try:
            self.session = _create_session(self.model_path, providers=self.providers)
            input_meta = self.session.get_inputs()[0]
            output_meta = self.session.get_outputs()
            self.input_name = input_meta.name
            input_shape = list(input_meta.shape)
            if len(input_shape) == 4:
                raw_h, raw_w = input_shape[2], input_shape[3]
                if isinstance(raw_h, int) and raw_h > 0 and isinstance(raw_w, int) and raw_w > 0:
                    self.input_size = (raw_w, raw_h)
            logger.info(
                "Age/gender ONNX fallback loaded: %s input=%s output=%s",
                self.model_path,
                input_shape,
                [list(meta.shape) for meta in output_meta],
            )
        except Exception:
            logger.exception("Failed loading ONNX age/gender fallback model")
            self.session = None

    def _prepare_insightface_local_models(self):
        root = Path(self.model_dir)
        buffalo_dir = root / "models" / "buffalo_l"
        buffalo_dir.mkdir(parents=True, exist_ok=True)

        src_gender = Path(self.model_path)
        dst_gender = buffalo_dir / "genderage.onnx"
        if src_gender.exists() and (not dst_gender.exists() or dst_gender.stat().st_size != src_gender.stat().st_size):
            shutil.copy2(src_gender, dst_gender)

        src_det = root / "scrfd_10g_bnkps.onnx"
        dst_det = buffalo_dir / "det_10g.onnx"
        if src_det.exists() and (not dst_det.exists() or dst_det.stat().st_size != src_det.stat().st_size):
            shutil.copy2(src_det, dst_det)

    def _init_insightface_primary(self):
        if FaceAnalysis is None:
            logger.warning("insightface is not installed; using ONNX fallback for age/gender")
            return

        try:
            self._prepare_insightface_local_models()
            self.app = FaceAnalysis(
                name="buffalo_l",
                root=self.model_dir,
                providers=self.providers,
                allowed_modules=["detection", "genderage"],
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            logger.exception("Failed to initialize insightface FaceAnalysis for age/gender")
            self.app = None

    @staticmethod
    def _select_primary_face(faces) -> Optional[object]:
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        )

    def _predict_with_insightface(self, image_bgr: np.ndarray) -> Optional[dict]:
        if self.app is None:
            return None

        faces = self.app.get(image_bgr)
        if not faces:
            h, w = image_bgr.shape[:2]
            if h < 300 and w < 300:
                pad = max(64, int(max(h, w) * 0.5))
                padded = cv2.copyMakeBorder(
                    image_bgr,
                    pad,
                    pad,
                    pad,
                    pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(127, 127, 127),
                )
                faces = self.app.get(padded)

        face = self._select_primary_face(faces)
        if face is None:
            return None

        sex = getattr(face, "sex", None)
        age_attr = getattr(face, "age", None)
        gender_attr = getattr(face, "gender", None)

        if isinstance(sex, str):
            male_prob = 0.99 if sex.upper().startswith("M") else 0.01
            gender = "MAN" if male_prob >= 0.5 else "WOMAN"
        elif gender_attr is not None:
            gender_val = float(np.asarray(gender_attr).reshape(-1)[0])
            male_prob = 1.0 if int(round(gender_val)) == 1 else 0.0
            gender = "MAN" if male_prob >= 0.5 else "WOMAN"
        else:
            return None

        if age_attr is None:
            return None
        age = max(0.0, min(100.0, float(age_attr)))

        if self.debug:
            logger.info(
                "AgeGender insightface output: sex=%s gender=%s age=%.2f bbox=%s",
                sex,
                gender,
                age,
                np.asarray(face.bbox).tolist() if hasattr(face, "bbox") else None,
            )
        return self._format_result(
            gender=gender,
            male_prob=male_prob,
            age_value=age,
            span=3,
            male_threshold=self.male_threshold,
        )

    def _predict_with_onnx(self, face_crop: np.ndarray) -> Optional[dict]:
        if self.session is None:
            return None

        input_w, input_h = self.input_size
        aligned = cv2.resize(face_crop, (input_w, input_h))
        tensor = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0,
            size=(input_w, input_h),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
        ).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: tensor})
        primary = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        if self.debug:
            logger.info(
                "AgeGender ONNX fallback input_shape=%s raw_outputs=%s primary=%s",
                tensor.shape,
                [np.asarray(out).shape for out in outputs],
                primary.tolist(),
            )

        gender_logits = np.array([0.5, 0.5], dtype=np.float32)
        age_raw = 25.0
        age_from_buckets: Optional[int] = None

        if primary.size >= 202:
            gender_logits = primary[:2]
            age_block = primary[2:202].reshape((100, 2))
            age_from_buckets = int(round(float(np.argmax(age_block, axis=1).sum())))
        elif primary.size >= 3:
            gender_logits = primary[:2]
            age_raw = float(primary[2])
        elif primary.size == 2:
            gender_logits = primary[:2]
            if len(outputs) > 1:
                age_arr = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
                if age_arr.size > 0:
                    age_raw = float(age_arr[0])

        if np.any(gender_logits < 0.0) or abs(float(np.sum(gender_logits)) - 1.0) > 0.1:
            gender_probs = _softmax(gender_logits.reshape(1, -1))[0]
        else:
            total = max(1e-9, float(np.sum(gender_logits)))
            gender_probs = gender_logits / total

        male_prob = float(gender_probs[1]) if gender_probs.size > 1 else 0.5
        age_value = float(age_from_buckets) if age_from_buckets is not None else float(age_raw)
        if 0.0 <= age_value <= 1.5:
            age_value *= 100.0
        age_value = float(np.clip(age_value, 0.0, 100.0))

        gender = "MAN" if male_prob >= self.male_threshold else "WOMAN"
        return self._format_result(
            gender=gender,
            male_prob=male_prob,
            age_value=age_value,
            span=4,
            male_threshold=self.male_threshold,
        )

    def _fuse_results(
        self,
        primary: dict,
        secondary: Optional[dict],
        tertiary: Optional[dict] = None,
        quaternary: Optional[dict] = None,
    ) -> dict:
        from app.core.config import settings as _cfg

        sources: list[tuple[dict, float]] = []
        if primary:
            sources.append((primary, self.primary_weight))
        if secondary:
            sources.append((secondary, self.vit_weight))
        if tertiary:
            sources.append((tertiary, self.fairface_weight))
        if quaternary:
            mivolo_w = getattr(self, "mivolo_weight", 0.35)
            sources.append((quaternary, mivolo_w))

        if not sources:
            return primary or secondary or tertiary

        total_weight = sum(w for _, w in sources)
        if total_weight <= 0.0:
            equal_w = 1.0 / len(sources)
            sources = [(d, equal_w) for d, _ in sources]
            total_weight = 1.0

        fused_male = 0.0
        fused_age = 0.0
        for src, w in sources:
            nw = w / total_weight
            fused_male += float(src.get("maleProb", 0.5)) * nw
            fused_age += float(src.get("ageValue", 30.0)) * nw

        fused_male = float(np.clip(fused_male, 0.0, 1.0))
        fused_age = float(np.clip(fused_age, 0.0, 100.0))

        # Stretch higher ages to reduce systematic underestimation on elderly samples.
        if fused_age > 40.0:
            fused_age = float(np.clip(40.0 + (fused_age - 40.0) * 1.15, 0.0, 100.0))

        # Race-aware age correction using FairFace's race output.
        race = None
        if tertiary and "race" in tertiary:
            race = str(tertiary["race"]).upper()
        if race:
            sea_max = float(_cfg.AGE_CORRECTION_SEA_YOUNG_MAX_AGE)
            sea_offset = max(0.0, float(_cfg.AGE_CORRECTION_SEA_YOUNG_OFFSET))
            if race == "SOUTHEAST_ASIAN":
                if fused_age <= 22.0:
                    fused_age += sea_offset
                elif fused_age <= sea_max:
                    fused_age += min(sea_offset, 0.5)
            elif race == "EAST_ASIAN" and fused_age <= sea_max:
                fused_age += float(_cfg.AGE_CORRECTION_EAST_ASIAN_YOUNG_OFFSET)
            elif race == "INDIAN" and fused_age > 50.0:
                fused_age = 50.0 + (fused_age - 50.0) * float(_cfg.AGE_CORRECTION_INDIAN_ELDERLY_SCALE)
            fused_age = float(np.clip(fused_age, 0.0, 100.0))

        fused_gender = "MAN" if fused_male >= self.male_threshold else "WOMAN"

        if self.debug:
            parts = []
            labels = ["primary", "vit", "fairface", "mivolo"]
            for i, (src, _) in enumerate(sources):
                lbl = labels[i] if i < len(labels) else f"src{i}"
                parts.append(f"{lbl}(male={src.get('maleProb', '?'):.3f} age={src.get('ageValue', '?'):.2f})")
            logger.info(
                "AgeGender fusion %s -> fused(male=%.3f age=%.2f race=%s)",
                " ".join(parts),
                fused_male,
                fused_age,
                race,
            )

        result = self._format_result(
            gender=fused_gender,
            male_prob=fused_male,
            age_value=fused_age,
            span=4,
            male_threshold=self.male_threshold,
        )
        # Propagate race from FairFace if available.
        if tertiary and "race" in tertiary:
            result["race"] = tertiary["race"]
        return result

    @staticmethod
    def _public_payload(result: dict) -> dict:
        payload = {
            "gender": result["gender"],
            "genderConfidence": result["genderConfidence"],
            "ageRange": result["ageRange"],
        }
        if "race" in result:
            payload["race"] = result["race"]
        return payload

    def predict(
        self,
        image_bgr: np.ndarray,
        face_crop: Optional[np.ndarray] = None,
        face_crop_hires: Optional[np.ndarray] = None,
    ) -> dict:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("input image is empty")

        crop_input = face_crop if face_crop is not None and face_crop.size > 0 else image_bgr
        # Use hires crop (224x224) for ViT/FairFace when available (avoids 96→224 upsample loss).
        hires_input = face_crop_hires if face_crop_hires is not None and face_crop_hires.size > 0 else crop_input

        primary = None
        try:
            primary = self._predict_with_insightface(image_bgr)
        except Exception:
            logger.exception("Age/gender primary insightface inference failed")
        if primary is None:
            try:
                primary = self._predict_with_onnx(crop_input)
            except Exception:
                logger.exception("Age/gender ONNX fallback inference failed")

        secondary = None
        if self.vit_estimator.is_loaded:
            try:
                secondary = self.vit_estimator.predict(hires_input)
            except Exception:
                logger.exception("Age/gender ViT ensemble inference failed")

        tertiary = None
        if self.fairface_estimator.is_loaded:
            try:
                tertiary = self.fairface_estimator.predict(hires_input)
            except Exception:
                logger.exception("FairFace ensemble inference failed")

        quaternary = None
        mivolo = getattr(self, "mivolo_estimator", None)
        if mivolo is not None and mivolo.is_loaded:
            try:
                quaternary = mivolo.predict(hires_input)
            except Exception:
                logger.exception("MiVOLO ensemble inference failed")

        available = [r for r in (primary, secondary, tertiary, quaternary) if r is not None]
        if not available:
            raise RuntimeError("No age/gender backend available")
        if len(available) == 1:
            return self._public_payload(available[0])
        return self._public_payload(self._fuse_results(primary, secondary, tertiary, quaternary))

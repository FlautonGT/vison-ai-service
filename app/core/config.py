"""Application configuration for pure inference mode."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_model_dir() -> str:
    """Resolve model directory with local-dev fallback on Windows/non-container runs."""
    configured = os.getenv("MODEL_DIR", "/opt/models").strip()
    configured_path = Path(configured).expanduser()

    def _has_onnx(path: Path) -> bool:
        return path.exists() and path.is_dir() and any(path.glob("*.onnx"))

    if _has_onnx(configured_path):
        return str(configured_path)

    repo_models = Path(__file__).resolve().parents[2] / "models"
    if _has_onnx(repo_models):
        return str(repo_models)

    if configured_path.exists():
        return str(configured_path)
    return configured


@dataclass
class Settings:
    # Server
    HOST: str = os.getenv("FACE_AI_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("FACE_AI_PORT", "8000"))
    WORKERS: int = int(os.getenv("FACE_AI_WORKERS", "2"))
    DEBUG: bool = os.getenv("FACE_AI_DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Model paths
    MODEL_DIR: str = _resolve_model_dir()

    # Face detection - Tier 3
    SCRFD_MODEL: str = os.getenv("SCRFD_MODEL", "scrfd_10g_bnkps.onnx")
    SCRFD_SCORE_THRESHOLD: float = float(os.getenv("SCRFD_SCORE_THRESHOLD", "0.5"))
    SCRFD_INPUT_SIZE: int = int(os.getenv("SCRFD_INPUT_SIZE", "640"))
    PREFER_INT8_MODELS: bool = os.getenv("PREFER_INT8_MODELS", "true").lower() == "true"

    # Face recognition - Tier 3
    ARCFACE_MODEL: str = os.getenv("ARCFACE_MODEL", "glintr100.onnx")
    ARCFACE_EXTRA_MODEL: str = os.getenv("ARCFACE_EXTRA_MODEL", "w600k_r50.onnx")
    ARCFACE_PRIMARY_WEIGHT: float = float(os.getenv("ARCFACE_PRIMARY_WEIGHT", "0.65"))
    ARCFACE_EXTRA_WEIGHT: float = float(os.getenv("ARCFACE_EXTRA_WEIGHT", "0.35"))
    COMPARE_THRESHOLD_DEFAULT: float = float(os.getenv("COMPARE_THRESHOLD_DEFAULT", "74.0"))
    ARCFACE_ADAPTIVE_COMPARE: bool = os.getenv("ARCFACE_ADAPTIVE_COMPARE", "true").lower() == "true"
    ARCFACE_ADAPTIVE_LOW: float = float(os.getenv("ARCFACE_ADAPTIVE_LOW", "70"))
    ARCFACE_ADAPTIVE_HIGH: float = float(os.getenv("ARCFACE_ADAPTIVE_HIGH", "90"))
    ARCFACE_FLIP_AUG: bool = os.getenv("ARCFACE_FLIP_AUG", "true").lower() == "true"
    COMPARE_PARALLEL_EMBEDDING: bool = (
        os.getenv("COMPARE_PARALLEL_EMBEDDING", "true").lower() == "true"
    )

    # Similarity score calibration (reporting scale closer to commercial APIs)
    SIMILARITY_CALIBRATION_ENABLED: bool = (
        os.getenv("SIMILARITY_CALIBRATION_ENABLED", "true").lower() == "true"
    )
    SIMILARITY_CALIBRATION_START: float = float(os.getenv("SIMILARITY_CALIBRATION_START", "75"))
    SIMILARITY_CALIBRATION_GAIN: float = float(os.getenv("SIMILARITY_CALIBRATION_GAIN", "1.6"))
    SIMILARITY_CALIBRATION_POWER: float = float(os.getenv("SIMILARITY_CALIBRATION_POWER", "0.75"))
    SIMILARITY_CALIBRATION_CAP: float = float(os.getenv("SIMILARITY_CALIBRATION_CAP", "99.99"))

    # Liveness - ensemble
    LIVENESS_MODELS: str = os.getenv("LIVENESS_MODELS", "MiniFASNetV2.onnx,MiniFASNetV1SE.onnx")
    LIVENESS_THRESHOLD: float = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))

    # Deepfake - ensemble
    DEEPFAKE_MODELS: str = os.getenv(
        "DEEPFAKE_MODELS",
        "deepfake_efficientnet_b0.onnx,community_forensics_vit.onnx",
    )
    DEEPFAKE_THRESHOLD: float = float(os.getenv("DEEPFAKE_THRESHOLD", "0.55"))
    DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD: float = float(
        os.getenv("DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD", "95")
    )
    AI_FACE_DETECTOR_MODEL: str = os.getenv("AI_FACE_DETECTOR_MODEL", "deepfake_efficientnet_b0.onnx")
    AI_FACE_EXTRA_DETECTOR_MODEL: str = os.getenv(
        "AI_FACE_EXTRA_DETECTOR_MODEL",
        "ai_vs_deepfake_vs_real.onnx",
    )
    AI_FACE_PRIMARY_WEIGHT: float = float(os.getenv("AI_FACE_PRIMARY_WEIGHT", "0.7"))
    AI_FACE_EXTRA_WEIGHT: float = float(os.getenv("AI_FACE_EXTRA_WEIGHT", "0.3"))
    AI_FACE_CALIBRATION_ALPHA: float = float(os.getenv("AI_FACE_CALIBRATION_ALPHA", "1.0"))
    AI_FACE_CALIBRATION_BETA: float = float(os.getenv("AI_FACE_CALIBRATION_BETA", "0.0"))
    AI_FACE_CONSENSUS_AI_THRESHOLD: float = float(os.getenv("AI_FACE_CONSENSUS_AI_THRESHOLD", "80"))
    AI_FACE_CONSENSUS_FACE_SWAP_THRESHOLD: float = float(
        os.getenv("AI_FACE_CONSENSUS_FACE_SWAP_THRESHOLD", "55")
    )
    AI_FACE_REAL_SUPPRESS_ENABLED: bool = (
        os.getenv("AI_FACE_REAL_SUPPRESS_ENABLED", "true").lower() == "true"
    )
    AI_FACE_REAL_SUPPRESS_FACE_CONF: float = float(os.getenv("AI_FACE_REAL_SUPPRESS_FACE_CONF", "72"))
    AI_FACE_REAL_SUPPRESS_AI_MAX: float = float(os.getenv("AI_FACE_REAL_SUPPRESS_AI_MAX", "88"))
    AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX: float = float(
        os.getenv("AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX", "62")
    )
    AI_FACE_ALWAYS_CROP_CHECK: bool = os.getenv("AI_FACE_ALWAYS_CROP_CHECK", "true").lower() == "true"
    AI_FACE_HARD_BLOCK_THRESHOLD: float = float(os.getenv("AI_FACE_HARD_BLOCK_THRESHOLD", "85"))
    AI_FACE_VOTE_THRESHOLD: float = float(os.getenv("AI_FACE_VOTE_THRESHOLD", "55"))
    AI_FACE_VOTE_MIN_COUNT: int = int(os.getenv("AI_FACE_VOTE_MIN_COUNT", "2"))
    AI_FACE_ANY_TRIGGER_THRESHOLD: float = float(os.getenv("AI_FACE_ANY_TRIGGER_THRESHOLD", "42"))
    AI_FACE_THRESHOLD: float = float(os.getenv("AI_FACE_THRESHOLD", "68"))
    AI_FACE_LOW_CONF_THRESHOLD: float = float(os.getenv("AI_FACE_LOW_CONF_THRESHOLD", "12"))
    AI_FACE_LOW_CONF_FACE_CONF: float = float(os.getenv("AI_FACE_LOW_CONF_FACE_CONF", "70"))

    # Face parsing
    FACE_PARSING_MODEL: str = os.getenv("FACE_PARSING_MODEL", "bisenet_face_parsing.onnx")

    # Age/Gender
    AGE_GENDER_MODEL: str = os.getenv("AGE_GENDER_MODEL", "genderage.onnx")
    AGE_GENDER_VIT_MODEL: str = os.getenv("AGE_GENDER_VIT_MODEL", "age_gender_vit.onnx")
    AGE_GENDER_FAIRFACE_MODEL: str = os.getenv("AGE_GENDER_FAIRFACE_MODEL", "fairface.onnx")
    AGE_GENDER_PRIMARY_WEIGHT: float = float(os.getenv("AGE_GENDER_PRIMARY_WEIGHT", "0.35"))
    AGE_GENDER_VIT_WEIGHT: float = float(os.getenv("AGE_GENDER_VIT_WEIGHT", "0.40"))
    AGE_GENDER_FAIRFACE_WEIGHT: float = float(os.getenv("AGE_GENDER_FAIRFACE_WEIGHT", "0.25"))
    AGE_GENDER_MALE_THRESHOLD: float = float(os.getenv("AGE_GENDER_MALE_THRESHOLD", "0.40"))

    # Quality validation thresholds (match Go defaults)
    QUALITY_MIN_SCORE: float = float(os.getenv("QUALITY_MIN_SCORE", "60.0"))
    QUALITY_MIN_SHARPNESS: int = int(os.getenv("QUALITY_MIN_SHARPNESS", "30"))
    QUALITY_MIN_BRIGHTNESS: int = int(os.getenv("QUALITY_MIN_BRIGHTNESS", "30"))
    QUALITY_MAX_BRIGHTNESS: int = int(os.getenv("QUALITY_MAX_BRIGHTNESS", "90"))
    QUALITY_POSE_MAX_ABS_DEG: float = float(os.getenv("QUALITY_POSE_MAX_ABS_DEG", "15"))
    QUALITY_MIN_INTER_EYE_PX: float = float(os.getenv("QUALITY_MIN_INTER_EYE_PX", "60"))
    QUALITY_MAX_ILLUM_ASYMMETRY: float = float(os.getenv("QUALITY_MAX_ILLUM_ASYMMETRY", "30"))
    QUALITY_MIN_CONTRAST: float = float(os.getenv("QUALITY_MIN_CONTRAST", "18"))
    FACE_MIN_AREA_RATIO: float = float(os.getenv("FACE_MIN_AREA_RATIO", "0.05"))
    FACE_MIN_AREA_RATIO_HARD: float = float(os.getenv("FACE_MIN_AREA_RATIO_HARD", "0.003"))
    FACE_MIN_PIXELS_HARD: int = int(os.getenv("FACE_MIN_PIXELS_HARD", "20"))
    ALLOW_SMALL_FACE_AUTOCROP: bool = os.getenv("ALLOW_SMALL_FACE_AUTOCROP", "true").lower() == "true"
    PRE_CROPPED_MIN_DIM: int = int(os.getenv("PRE_CROPPED_MIN_DIM", "40"))
    PRE_CROPPED_ASPECT_MIN: float = float(os.getenv("PRE_CROPPED_ASPECT_MIN", "0.5"))
    PRE_CROPPED_ASPECT_MAX: float = float(os.getenv("PRE_CROPPED_ASPECT_MAX", "2.0"))

    # Max image size
    MAX_IMAGE_BYTES: int = 5 * 1024 * 1024

    # ONNX Runtime
    ONNX_PROVIDERS: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider"))
    )
    ONNX_INTRA_OP_THREADS: int = int(os.getenv("ONNX_INTRA_OP_THREADS", "2"))
    ONNX_INTER_OP_THREADS: int = int(os.getenv("ONNX_INTER_OP_THREADS", "1"))
    ONNX_OPT_LEVEL: str = os.getenv("ONNX_OPT_LEVEL", "all")
    ONNX_EXECUTION_MODE: str = os.getenv("ONNX_EXECUTION_MODE", "parallel")

    # Internal auth
    AI_SERVICE_SECRET: str = os.getenv("AI_SERVICE_SECRET", "")
    ALLOWED_IPS: str = os.getenv("ALLOWED_IPS", "")
    RATE_LIMIT_RPS: int = int(os.getenv("RATE_LIMIT_RPS", "100"))

    @property
    def liveness_model_list(self) -> list[str]:
        return _split_csv(self.LIVENESS_MODELS)

    @property
    def deepfake_model_list(self) -> list[str]:
        return _split_csv(self.DEEPFAKE_MODELS)


settings = Settings()

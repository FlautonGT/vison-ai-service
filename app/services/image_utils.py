"""
Image utilities — read multipart files, validate format, decode to numpy.
Matches Go validation: readFaceImage(), detectFaceImageFormat(), size limits.
"""

import io

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import settings

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


class ImageValidationError(Exception):
    """Raised when image validation fails. Maps to Go error codes."""

    def __init__(self, code: str, message: str, detail: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


async def read_image(file: UploadFile, field_name: str = "image") -> np.ndarray:
    """
    Read and validate image from multipart upload.
    Returns: BGR numpy array (OpenCV format).

    Matches Go flow:
    1. readFaceImage() — read bytes, check size
    2. validateFaceImageForAWS() / validateFaceImageForTencentIAI() — format check
    3. Decode to image
    """
    contents = await file.read()

    # Check empty
    if not contents or len(contents) == 0:
        raise ImageValidationError(
            code="INVALID_MANDATORY_FIELD",
            message="Invalid Mandatory Field",
            detail=f"{field_name} is required",
        )

    # Check size (match Go: maxMultipartImageBytes = 5MB)
    if len(contents) > settings.MAX_IMAGE_BYTES:
        raise ImageValidationError(
            code="IMAGE_TOO_LARGE",
            message="Payload Too Large",
            detail="Maximum allowed image size is 5MB",
            status_code=413,
        )

    # Check format (match Go: detectFaceImageFormat)
    fmt = _detect_image_format(contents)
    if fmt not in ("jpeg", "png"):
        raise ImageValidationError(
            code="UNSUPPORTED_FORMAT",
            message="Unsupported Media Type",
            detail="Image must be JPEG or PNG",
            status_code=415,
        )

    img = _decode_image(contents)

    if img is None:
        raise ImageValidationError(
            code="IMAGE_CORRUPT",
            message="Invalid Image",
            detail="Failed to decode image",
            status_code=422,
        )

    return img


def read_image_sync(data: bytes, field_name: str = "image") -> np.ndarray:
    """Synchronous version for use in non-async contexts."""
    if not data or len(data) == 0:
        raise ImageValidationError(
            code="INVALID_MANDATORY_FIELD",
            message="Invalid Mandatory Field",
            detail=f"{field_name} is required",
        )

    if len(data) > settings.MAX_IMAGE_BYTES:
        raise ImageValidationError(
            code="IMAGE_TOO_LARGE",
            message="Payload Too Large",
            detail="Maximum allowed image size is 5MB",
            status_code=413,
        )

    fmt = _detect_image_format(data)
    if fmt not in ("jpeg", "png"):
        raise ImageValidationError(
            code="UNSUPPORTED_FORMAT",
            message="Unsupported Media Type",
            detail="Image must be JPEG or PNG",
            status_code=415,
        )

    img = _decode_image(data)
    if img is None:
        raise ImageValidationError(
            code="IMAGE_CORRUPT",
            message="Invalid Image",
            detail="Failed to decode image",
            status_code=422,
        )
    return img


def _detect_image_format(data: bytes) -> str:
    """
    Detect image format from magic bytes.
    Matches Go: detectFaceImageFormat()
    """
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        return "jpeg"
    if (
        len(data) >= 8
        and data[0] == 0x89
        and data[1] == 0x50  # P
        and data[2] == 0x4E  # N
        and data[3] == 0x47  # G
        and data[4] == 0x0D
        and data[5] == 0x0A
        and data[6] == 0x1A
        and data[7] == 0x0A
    ):
        return "png"
    if len(data) >= 2 and data[0] == 0x42 and data[1] == 0x4D:
        return "bmp"
    return "unknown"


def _decode_image(data: bytes) -> np.ndarray | None:
    """
    Decode image bytes and apply EXIF orientation normalization when available.
    """
    if Image is not None and ImageOps is not None:
        try:
            with Image.open(io.BytesIO(data)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                rgb = np.asarray(pil_img)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_image_dimensions(image: np.ndarray) -> tuple:
    """Returns (width, height) of OpenCV image."""
    h, w = image.shape[:2]
    return w, h

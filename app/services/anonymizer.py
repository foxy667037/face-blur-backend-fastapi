"""Face-region anonymization using Gaussian blur or pixelation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Literal

import cv2
import numpy as np

from app.services.face_detector import FaceDetection, FaceDetector


AnonymizationMethod = Literal["blur", "pixelate"]
MIN_STRENGTH = 1
MAX_STRENGTH = 50
SUPPORTED_METHODS = ("blur", "pixelate")


def validate_anonymization_options(
    method: str,
    strength: int,
    margin: float,
) -> None:
    """Validate anonymization options shared by the image and video flows."""
    if method not in SUPPORTED_METHODS:
        raise ValueError("method must be 'blur' or 'pixelate'")
    if isinstance(strength, bool) or not isinstance(strength, int):
        raise ValueError("strength must be an integer")
    minimum = 2 if method == "pixelate" else MIN_STRENGTH
    if not minimum <= strength <= MAX_STRENGTH:
        raise ValueError(
            f"strength for {method} must be between {minimum} and {MAX_STRENGTH}"
        )
    if not isinstance(margin, (int, float)) or not math.isfinite(margin):
        raise ValueError("margin must be a finite number between 0 and 1")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be a finite number between 0 and 1")


@dataclass(frozen=True)
class AnonymizationMetadata:
    method: AnonymizationMethod
    strength: int
    margin: float
    face_count: int
    processed_face_count: int

    @property
    def anonymized(self) -> bool:
        return self.processed_face_count > 0


class FaceAnonymizer:
    """Anonymize detected face regions without modifying the source image."""

    def __init__(self, detector: FaceDetector | None = None) -> None:
        self._detector = detector if detector is not None else FaceDetector()

    def anonymize_image(
        self,
        image: np.ndarray,
        method: AnonymizationMethod = "blur",
        strength: int = 15,
        margin: float = 0.0,
    ) -> tuple[np.ndarray, AnonymizationMetadata]:
        """Return an anonymized copy and metadata describing applied regions.

        ``strength`` is a blur radius mapped to an odd Gaussian kernel, or a
        pixelation block size. ``margin`` expands each face box by a fraction
        of its width and height.
        """
        self._validate_options(method, strength, margin)
        detection_image = self._prepare_detection_image(image)

        output = image.copy()
        detections = self._detector.detect(detection_image)
        processed_face_count = 0

        for detection in detections:
            bounds = self._clamped_bounds(detection, image.shape, margin)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            region = output[y1:y2, x1:x2]
            if image.ndim == 3 and image.shape[2] == 4:
                region_colors = region[:, :, :3]
                processed = self._process_region(region_colors, method, strength)
                output[y1:y2, x1:x2, :3] = processed
                processed_face_count += 1
                continue
            processed = self._process_region(region, method, strength)
            output[y1:y2, x1:x2] = processed
            processed_face_count += 1

        metadata = AnonymizationMetadata(
            method=method,
            strength=strength,
            margin=margin,
            face_count=len(detections),
            processed_face_count=processed_face_count,
        )
        return output, metadata

    @staticmethod
    def _validate_options(
        method: str,
        strength: int,
        margin: float,
    ) -> None:
        validate_anonymization_options(method, strength, margin)

    @staticmethod
    def _prepare_detection_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError(
                "image must be a non-empty uint8 grayscale, BGR, or BGRA image"
            )
        if image.ndim == 2:
            height, width = image.shape
            if height < 1 or width < 1:
                raise ValueError("image dimensions must be non-empty")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError(
                "image must be a non-empty uint8 grayscale, BGR, or BGRA image"
            )
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(
            "image must be a non-empty uint8 grayscale, BGR, or BGRA image"
        )

    @classmethod
    def _process_region(
        cls,
        region: np.ndarray,
        method: AnonymizationMethod,
        strength: int,
    ) -> np.ndarray:
        if method == "blur":
            return cls._blur(region, strength)
        return cls._pixelate(region, strength)

    @staticmethod
    def _clamped_bounds(
        detection: FaceDetection,
        image_shape: tuple[int, ...],
        margin: float,
    ) -> tuple[int, int, int, int] | None:
        height, width = image_shape[:2]
        box = detection.bounding_box
        values = (box.x, box.y, box.width, box.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detector returned non-finite bounding-box coordinates")
        if box.width <= 0 or box.height <= 0:
            raise ValueError("detector returned a non-positive bounding box")

        expand_x = box.width * margin
        expand_y = box.height * margin
        x1 = max(0, math.floor(box.x - expand_x))
        y1 = max(0, math.floor(box.y - expand_y))
        x2 = min(width, math.ceil(box.x + box.width + expand_x))
        y2 = min(height, math.ceil(box.y + box.height + expand_y))
        if x1 >= x2 or y1 >= y2:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _blur(region: np.ndarray, strength: int) -> np.ndarray:
        kernel_size = 2 * strength + 1
        return cv2.GaussianBlur(
            region,
            (kernel_size, kernel_size),
            sigmaX=0,
            sigmaY=0,
        )

    @staticmethod
    def _pixelate(region: np.ndarray, strength: int) -> np.ndarray:
        height, width = region.shape[:2]
        reduced_size = (max(1, width // strength), max(1, height // strength))
        reduced = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
        return cv2.resize(
            reduced,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )


_default_anonymizer: FaceAnonymizer | None = None
_default_anonymizer_lock = threading.Lock()


def anonymize_image(
    image: np.ndarray,
    method: AnonymizationMethod = "blur",
    strength: int = 15,
    margin: float = 0.0,
) -> tuple[np.ndarray, AnonymizationMetadata]:
    """Anonymize an image using a lazily created, reusable detector."""
    global _default_anonymizer
    if _default_anonymizer is None:
        with _default_anonymizer_lock:
            if _default_anonymizer is None:
                _default_anonymizer = FaceAnonymizer()
    return _default_anonymizer.anonymize_image(
        image,
        method=method,
        strength=strength,
        margin=margin,
    )

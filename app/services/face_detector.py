"""Reusable YuNet face detection backed by OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading

import cv2
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "face_detection_yunet"
    / "face_detection_yunet_2023mar.onnx"
)


@dataclass(frozen=True)
class BoundingBox:
    """A face rectangle in image pixels, represented as x, y, width, height."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class FaceDetection:
    """A detected face's geometry, confidence, and five facial landmarks."""

    bounding_box: BoundingBox
    confidence: float
    landmarks: tuple[tuple[float, float], ...]


class FaceDetector:
    """Detect faces with a single reusable OpenCV YuNet detector instance."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self._validate_options(confidence_threshold, nms_threshold, top_k)
        self._lock = threading.Lock()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"YuNet model file not found: {self.model_path}")

        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),
                score_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
            )
        except cv2.error as exc:
            raise RuntimeError(
                f"Failed to load YuNet model at {self.model_path}: {exc}"
            ) from exc

    @staticmethod
    def _validate_options(
        confidence_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> None:
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("nms_threshold", nms_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value between 0 and 1")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

    @staticmethod
    def _validate_image(image: np.ndarray) -> tuple[int, int]:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a non-empty three-channel BGR image")
        height, width, _ = image.shape
        if height < 1 or width < 1:
            raise ValueError("image must be a non-empty three-channel BGR image")
        if image.dtype != np.uint8:
            raise ValueError("image must use the uint8 data type")
        return width, height

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Return detections for a non-empty uint8 BGR image."""
        width, height = self._validate_image(image)

        # setInputSize mutates the shared OpenCV detector; keep sizing and
        # inference in one critical section when this instance is reused.
        with self._lock:
            try:
                self._detector.setInputSize((width, height))
                status, faces = self._detector.detect(image)
            except cv2.error as exc:
                raise RuntimeError(f"YuNet face detection failed: {exc}") from exc

        if not status:
            raise RuntimeError("YuNet face detection failed")
        if faces is None:
            return []

        results: list[FaceDetection] = []
        for row in faces:
            if len(row) < 15:
                raise RuntimeError(
                    "YuNet returned an invalid detection row; expected "
                    "bounding box, five landmarks, and confidence"
                )
            results.append(
                FaceDetection(
                    bounding_box=BoundingBox(*(float(value) for value in row[:4])),
                    confidence=float(row[-1]),
                    landmarks=tuple(
                        (float(row[index]), float(row[index + 1]))
                        for index in range(4, 14, 2)
                    ),
                )
            )
        return results

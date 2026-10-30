"""Single-frame processing for privacy-preserving video pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.anonymizer import (
    AnonymizationMetadata,
    AnonymizationMethod,
    FaceAnonymizer,
)


@dataclass(frozen=True)
class VideoFrameResult:
    """A processed BGR frame and metadata for one video frame."""

    frame: np.ndarray
    metadata: AnonymizationMetadata


class VideoFrameProcessor:
    """Process decoded BGR frames with a reusable face anonymizer.

    Processing errors are raised to the caller; source frames are never
    returned as a fallback after detection or anonymization fails.
    """

    def __init__(self, anonymizer: FaceAnonymizer | None = None) -> None:
        self._anonymizer = anonymizer if anonymizer is not None else FaceAnonymizer()

    def process_frame(
        self,
        frame: np.ndarray,
        method: AnonymizationMethod = "blur",
        strength: int = 15,
        margin: float = 0.0,
    ) -> VideoFrameResult:
        """Return an anonymized frame without modifying the input frame."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] < 1
            or frame.shape[1] < 1
        ):
            raise ValueError("frame must be a non-empty uint8 three-channel BGR image")

        processed, metadata = self._anonymizer.anonymize_image(
            frame,
            method=method,
            strength=strength,
            margin=margin,
        )
        if processed.shape != frame.shape or processed.dtype != frame.dtype:
            raise RuntimeError("frame processor returned an incompatible frame")
        return VideoFrameResult(frame=processed, metadata=metadata)

"""File-based video anonymization reusing the image frame pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from app.services.anonymizer import (
    AnonymizationMethod,
    validate_anonymization_options,
)
from app.services.video_processor import VideoFrameProcessor


DEFAULT_FRAME_RATE = 25.0
MIN_FRAME_RATE = 1.0
MAX_FRAME_RATE = 120.0
# Candidate containers are tried in order; the first writer that opens wins.
# WebM/VP8 is preferred because it is broadly playable in browsers.
_VIDEO_WRITER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("VP80", ".webm"),
    ("VP90", ".webm"),
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
)
_MEDIA_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
}


class VideoDecodeError(ValueError):
    """Raised when an uploaded video cannot be opened or decoded."""


class VideoTooLongError(ValueError):
    """Raised when a video exceeds the configured frame limit."""


@dataclass(frozen=True)
class VideoAnonymizationResult:
    """Encoded output bytes plus a summary of what was processed."""

    content: bytes
    filename: str
    media_type: str
    frame_count: int
    processed_frame_count: int
    face_count: int
    method: AnonymizationMethod
    strength: int
    margin: float
    width: int
    height: int
    fps: float

    @property
    def anonymized(self) -> bool:
        return self.processed_frame_count > 0


class VideoAnonymizer:
    """Anonymize every frame of a video file with a reusable frame processor."""

    def __init__(self, video_processor: VideoFrameProcessor | None = None) -> None:
        self._processor = (
            video_processor if video_processor is not None else VideoFrameProcessor()
        )

    def anonymize_video(
        self,
        input_path: str | Path,
        output_directory: str | Path,
        *,
        method: AnonymizationMethod = "blur",
        strength: int = 15,
        margin: float = 0.0,
        max_frames: int = 1500,
        max_dimension: int = 1280,
    ) -> VideoAnonymizationResult:
        """Return encoded anonymized video bytes and processing metadata.

        Decoding and re-encoding require real files, so ``input_path`` must
        already exist and ``output_directory`` must be writable. Callers are
        responsible for the lifecycle of the input file; this method owns and
        deletes the intermediate output file it creates.
        """
        validate_anonymization_options(method, strength, margin)
        self._validate_limits(max_frames, max_dimension)

        input_path = Path(input_path)
        output_directory = Path(output_directory)
        if not input_path.is_file():
            raise ValueError("input video file does not exist")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise VideoDecodeError("uploaded video could not be opened")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = DEFAULT_FRAME_RATE
            fps = min(max(fps, MIN_FRAME_RATE), MAX_FRAME_RATE)

            reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if reported_frames > max_frames:
                raise VideoTooLongError(
                    "video is longer than the configured limit"
                )

            success, frame = capture.read()
            if not success or frame is None:
                raise VideoDecodeError(
                    "uploaded video contains no readable frames"
                )
            source_height, source_width = frame.shape[:2]

            target_width, target_height = self._target_dimensions(
                source_width,
                source_height,
                max_dimension,
            )
            frame = self._resize_frame(frame, target_width, target_height)

            output_path = output_directory / "anonymized-output"
            writer, _, extension = self._create_writer(
                output_path,
                fps,
                target_width,
                target_height,
            )
            output_path = output_path.with_suffix(extension)
            try:
                frame_count = 0
                processed_frame_count = 0
                face_count = 0
                try:
                    while True:
                        if frame_count >= max_frames:
                            raise VideoTooLongError(
                                "video is longer than the configured limit"
                            )
                        result = self._processor.process_frame(
                            frame,
                            method=method,
                            strength=strength,
                            margin=margin,
                        )
                        writer.write(result.frame)
                        frame_count += 1
                        processed_frame_count += result.metadata.processed_face_count
                        face_count += result.metadata.face_count

                        success, frame = capture.read()
                        if not success or frame is None:
                            break
                        frame = self._resize_frame(
                            frame,
                            target_width,
                            target_height,
                        )

                    if frame_count < 1:
                        raise VideoDecodeError(
                            "uploaded video contains no readable frames"
                        )
                finally:
                    writer.release()
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
        finally:
            capture.release()

        try:
            content = output_path.read_bytes()
        finally:
            output_path.unlink(missing_ok=True)
        if not content:
            raise RuntimeError("anonymized video encoding produced no data")

        return VideoAnonymizationResult(
            content=content,
            filename=f"anonymized-video{extension}",
            media_type=_MEDIA_TYPES.get(extension, "application/octet-stream"),
            frame_count=frame_count,
            processed_frame_count=processed_frame_count,
            face_count=face_count,
            method=method,
            strength=strength,
            margin=margin,
            width=target_width,
            height=target_height,
            fps=fps,
        )

    @staticmethod
    def _validate_limits(max_frames: int, max_dimension: int) -> None:
        for name, value in (("max_frames", max_frames), ("max_dimension", max_dimension)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _target_dimensions(
        width: int,
        height: int,
        max_dimension: int,
    ) -> tuple[int, int]:
        scale = min(1.0, max_dimension / max(width, height))
        target_width = max(2, int(round(width * scale)))
        target_height = max(2, int(round(height * scale)))
        # Video codecs are unreliable with odd frame dimensions.
        return target_width - (target_width % 2), target_height - (target_height % 2)

    @staticmethod
    def _resize_frame(
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        if frame.shape[0] == height and frame.shape[1] == width:
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _create_writer(
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> tuple[cv2.VideoWriter, str, str]:
        last_error: Exception | None = None
        for fourcc_name, extension in _VIDEO_WRITER_CANDIDATES:
            candidate_path = output_path.with_suffix(extension)
            writer = cv2.VideoWriter(
                str(candidate_path),
                cv2.VideoWriter_fourcc(*fourcc_name),
                fps,
                (width, height),
            )
            if writer.isOpened():
                return writer, fourcc_name, extension
            writer.release()
            candidate_path.unlink(missing_ok=True)
            last_error = RuntimeError(
                f"OpenCV video writer could not encode {extension}"
            )
        raise last_error or RuntimeError("no supported video encoder is available")

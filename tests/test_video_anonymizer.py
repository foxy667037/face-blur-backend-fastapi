"""Unit tests for the file-based VideoAnonymizer service."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection
from app.services.video_anonymizer import (
    VideoAnonymizer,
    VideoDecodeError,
    VideoTooLongError,
)
from app.services.video_processor import VideoFrameProcessor


def make_detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        BoundingBox(x, y, width, height),
        0.9,
        ((1.0, 1.0),) * 5,
    )


def patterned_frame(height: int = 120, width: int = 160) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=-1,
    ).astype(np.uint8)


def write_test_video(
    directory: Path,
    filename: str = "input.webm",
    frames: int = 12,
    fps: float = 12.0,
    width: int = 160,
    height: int = 120,
) -> Path:
    path = directory / filename
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("test fixture writer could not open")
    for index in range(frames):
        frame = patterned_frame(height, width)
        frame[:] = ((index * 8) % 256, 60, 120)
        writer.write(frame)
    writer.release()
    if not path.is_file():
        raise RuntimeError("test fixture video was not written")
    return path


class VideoAnonymizerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = Mock()
        self.detector.detect.return_value = []
        self.processor = VideoFrameProcessor(FaceAnonymizer(self.detector))
        self.anonymizer = VideoAnonymizer(self.processor)
        self.tmpdir = Path(tempfile.mkdtemp())

    def _make_source(self) -> Path:
        return write_test_video(self.tmpdir)

    def tearDown(self) -> None:
        for path in self.tmpdir.iterdir():
            path.unlink()
        self.tmpdir.rmdir()

    def test_processes_every_frame_and_counts_faces(self) -> None:
        self.detector.detect.return_value = [make_detection(10, 10, 40, 40)]
        source = self._make_source()
        result = self.anonymizer.anonymize_video(
            source,
            self.tmpdir,
            method="pixelate",
            strength=8,
        )

        self.assertEqual(result.frame_count, 12)
        self.assertEqual(result.processed_frame_count, 12)
        self.assertEqual(result.face_count, 12)
        self.assertTrue(result.anonymized)
        self.assertEqual(result.method, "pixelate")
        self.assertEqual(result.strength, 8)
        self.assertEqual(result.width, 160)
        self.assertEqual(result.height, 120)
        self.assertEqual(result.media_type, "video/webm")
        self.assertEqual(result.filename, "anonymized-video.webm")
        self.assertGreater(len(result.content), 0)
        self.assertEqual(self.detector.detect.call_count, 12)

    def test_output_is_decodable_and_preserves_frame_count_and_size(self) -> None:
        self.detector.detect.return_value = [make_detection(10, 10, 40, 40)]
        source = self._make_source()
        result = self.anonymizer.anonymize_video(source, self.tmpdir)

        output_path = self.tmpdir / result.filename
        output_path.write_bytes(result.content)
        capture = cv2.VideoCapture(str(output_path))
        read_frames = 0
        first_shape = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            read_frames += 1
            if first_shape is None:
                first_shape = frame.shape
        capture.release()
        self.assertEqual(read_frames, 12)
        self.assertEqual(first_shape, (120, 160, 3))
        output_path.unlink()

    def test_no_face_frames_still_return_valid_output(self) -> None:
        self.detector.detect.return_value = []
        source = self._make_source()
        result = self.anonymizer.anonymize_video(source, self.tmpdir)

        self.assertEqual(result.frame_count, 12)
        self.assertEqual(result.processed_frame_count, 0)
        self.assertEqual(result.face_count, 0)
        self.assertFalse(result.anonymized)
        self.assertGreater(len(result.content), 0)

    def test_strength_and_margin_are_preserved_in_metadata(self) -> None:
        source = self._make_source()
        result = self.anonymizer.anonymize_video(
            source,
            self.tmpdir,
            method="blur",
            strength=23,
            margin=0.25,
        )
        self.assertEqual(result.method, "blur")
        self.assertEqual(result.strength, 23)
        self.assertEqual(result.margin, 0.25)

    def test_exceeding_max_frames_is_rejected(self) -> None:
        source = write_test_video(self.tmpdir, frames=20, fps=12)
        with self.assertRaises(VideoTooLongError):
            self.anonymizer.anonymize_video(source, self.tmpdir, max_frames=10)

    def test_undecodable_input_is_rejected(self) -> None:
        bad = self.tmpdir / "not-a-video.webm"
        bad.write_bytes(b"\xff\xd8\xff not a video")
        with self.assertRaises(VideoDecodeError):
            self.anonymizer.anonymize_video(bad, self.tmpdir)

    def test_missing_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.anonymizer.anonymize_video(self.tmpdir / "missing.webm", self.tmpdir)

    def test_invalid_options_raise_before_processing(self) -> None:
        source = self._make_source()
        invalid_cases = (
            {"method": "mosaic"},
            {"strength": 0},
            {"strength": 51},
            {"method": "pixelate", "strength": 1},
            {"strength": 1.5},
            {"margin": -0.1},
            {"margin": float("nan")},
        )
        for options in invalid_cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.anonymizer.anonymize_video(source, self.tmpdir, **options)

    def test_frame_limit_is_enforced_during_decode(self) -> None:
        self.detector.detect.return_value = []
        source = write_test_video(self.tmpdir, frames=20)
        with self.assertRaises(VideoTooLongError):
            self.anonymizer.anonymize_video(source, self.tmpdir, max_frames=12)

    def test_temp_output_is_deleted_after_processing(self) -> None:
        source = self._make_source()
        result = self.anonymizer.anonymize_video(source, self.tmpdir)
        # The intermediate output file must not linger on disk.
        self.assertFalse((self.tmpdir / result.filename).is_file())


if __name__ == "__main__":
    unittest.main()

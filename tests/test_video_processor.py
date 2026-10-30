from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from app.services.anonymizer import AnonymizationMetadata, FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection
from app.services.video_processor import VideoFrameProcessor


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "tests" / "output" / "largest_selfie_before.jpg"


def detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        bounding_box=BoundingBox(x, y, width, height),
        confidence=0.9,
        landmarks=((0.0, 0.0),) * 5,
    )


def patterned_frame(height: int = 64, width: int = 80) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=-1,
    ).astype(np.uint8)


class VideoFrameProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = Mock()
        self.detector.detect.return_value = []
        self.processor = VideoFrameProcessor(FaceAnonymizer(self.detector))

    def test_no_faces_returns_copy_and_zero_metadata(self) -> None:
        frame = patterned_frame()
        original = frame.copy()

        result = self.processor.process_frame(frame)

        self.assertIsNot(result.frame, frame)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(result.frame, frame)
        self.assertEqual(result.metadata.face_count, 0)
        self.assertEqual(result.metadata.processed_face_count, 0)
        self.assertFalse(result.metadata.anonymized)

    def test_multiple_faces_are_processed_and_dimensions_preserved(self) -> None:
        self.detector.detect.return_value = [
            detection(4, 5, 18, 18),
            detection(50, 35, 20, 20),
        ]
        frame = patterned_frame()
        original = frame.copy()

        result = self.processor.process_frame(frame, method="pixelate", strength=8)

        self.assertEqual(result.frame.shape, frame.shape)
        self.assertEqual(result.frame.dtype, frame.dtype)
        self.assertEqual(result.metadata.face_count, 2)
        self.assertEqual(result.metadata.processed_face_count, 2)
        self.assertFalse(np.array_equal(result.frame[5:23, 4:22], frame[5:23, 4:22]))
        self.assertFalse(np.array_equal(result.frame[35:55, 50:70], frame[35:55, 50:70]))
        np.testing.assert_array_equal(frame, original)

    def test_selected_processing_settings_are_preserved_in_metadata(self) -> None:
        frame = patterned_frame()

        result = self.processor.process_frame(
            frame,
            method="pixelate",
            strength=23,
            margin=0.25,
        )

        self.detector.detect.assert_called_once()
        self.assertEqual(result.metadata.method, "pixelate")
        self.assertEqual(result.metadata.strength, 23)
        self.assertEqual(result.metadata.margin, 0.25)

    def test_invalid_frames_are_rejected_before_detection(self) -> None:
        for frame in (
            None,
            np.zeros((20, 20), dtype=np.uint8),
            np.zeros((20, 20, 4), dtype=np.uint8),
            np.zeros((0, 20, 3), dtype=np.uint8),
            np.zeros((20, 20, 3), dtype=np.float32),
        ):
            with self.subTest(frame_type=type(frame).__name__):
                with self.assertRaises(ValueError):
                    self.processor.process_frame(frame)
        self.detector.detect.assert_not_called()

    def test_detector_failure_is_propagated_without_source_fallback(self) -> None:
        self.detector.detect.side_effect = RuntimeError("YuNet failed")
        frame = patterned_frame()
        original = frame.copy()

        with self.assertRaisesRegex(RuntimeError, "YuNet failed"):
            self.processor.process_frame(frame)

        np.testing.assert_array_equal(frame, original)

    def test_incompatible_anonymizer_result_is_rejected(self) -> None:
        anonymizer = Mock()
        anonymizer.anonymize_image.return_value = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            AnonymizationMetadata("blur", 15, 0.0, 0, 0),
        )
        processor = VideoFrameProcessor(anonymizer)

        with self.assertRaisesRegex(RuntimeError, "incompatible frame"):
            processor.process_frame(patterned_frame())


@unittest.skipUnless(SAMPLE_IMAGE.is_file(), "Real YuNet sample image is unavailable")
class VideoFrameProcessorIntegrationTests(unittest.TestCase):
    def test_real_yunet_processes_one_video_frame(self) -> None:
        frame = cv2.imread(str(SAMPLE_IMAGE))
        self.assertIsNotNone(frame)
        original = frame.copy()
        result = VideoFrameProcessor().process_frame(
            frame,
            method="blur",
            strength=15,
        )

        self.assertEqual(result.frame.shape, frame.shape)
        self.assertEqual(result.metadata.face_count, 140)
        self.assertEqual(result.metadata.processed_face_count, 140)
        self.assertTrue(result.metadata.anonymized)
        np.testing.assert_array_equal(frame, original)

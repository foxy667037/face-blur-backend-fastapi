from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app.services.face_detector import (
    DEFAULT_MODEL_PATH,
    BoundingBox,
    FaceDetector,
)


ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_SAMPLE_IMAGE = (
    ROOT
    / "app"
    / "model"
    / "face_detection_yunet"
    / "example_outputs"
    / "largest_selfie.jpg"
)
SAMPLE_IMAGE = (
    _BUNDLED_SAMPLE_IMAGE
    if _BUNDLED_SAMPLE_IMAGE.is_file()
    else ROOT / "tests" / "output" / "largest_selfie_before.jpg"
)


def make_fake_detector(faces: np.ndarray | None, status: int = 1) -> Mock:
    detector = Mock()
    detector.detect.return_value = (status, faces)
    return detector


class FaceDetectorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = make_fake_detector(None)
        self.create_patcher = patch(
            "app.services.face_detector.cv2.FaceDetectorYN.create",
            return_value=self.detector,
        )
        self.create_model = self.create_patcher.start()
        self.addCleanup(self.create_patcher.stop)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.model_path = Path(temp_dir.name) / "fake-yunet.onnx"
        self.model_path.touch()

    def test_missing_model_raises_clear_file_error(self) -> None:
        missing = self.model_path.parent / "missing-yunet.onnx"
        with self.assertRaisesRegex(FileNotFoundError, "YuNet model file not found"):
            FaceDetector(missing)
        self.create_model.assert_not_called()

    def test_model_loading_error_has_context(self) -> None:
        self.create_model.side_effect = cv2.error("invalid ONNX")
        with self.assertRaisesRegex(RuntimeError, "Failed to load YuNet model"):
            FaceDetector(self.model_path)

    def test_invalid_image_input_is_rejected(self) -> None:
        detector = FaceDetector(self.model_path)
        for image in (
            None,
            np.zeros((20, 20), dtype=np.uint8),
            np.zeros((20, 20, 4), dtype=np.uint8),
            np.zeros((0, 20, 3), dtype=np.uint8),
            np.zeros((20, 20, 3), dtype=np.float32),
        ):
            with self.subTest(image_type=type(image).__name__):
                with self.assertRaises(ValueError):
                    detector.detect(image)
        self.detector.detect.assert_not_called()

    def test_no_faces_returns_empty_list_and_uses_image_dimensions(self) -> None:
        detector = FaceDetector(self.model_path)
        result = detector.detect(np.zeros((48, 96, 3), dtype=np.uint8))
        self.assertEqual(result, [])
        self.detector.setInputSize.assert_called_once_with((96, 48))
        self.detector.detect.assert_called_once()

    def test_bbox_confidence_and_landmarks_keep_float_coordinates(self) -> None:
        row = np.array(
            [[10.25, 20.5, 30.75, 40.125, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6,
              7.7, 8.8, 9.9, 10.1, 0.875]],
            dtype=np.float32,
        )
        self.detector.detect.return_value = (1, row)
        detector = FaceDetector(self.model_path)
        result = detector.detect(np.zeros((48, 96, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].bounding_box,
            BoundingBox(10.25, 20.5, 30.75, 40.125),
        )
        self.assertAlmostEqual(result[0].confidence, 0.875)
        np.testing.assert_allclose(
            result[0].landmarks,
            ((1.1, 2.2), (3.3, 4.4), (5.5, 6.6), (7.7, 8.8), (9.9, 10.1)),
        )

    def test_multiple_detections_are_returned(self) -> None:
        rows = np.zeros((2, 15), dtype=np.float32)
        rows[:, -1] = (0.9, 0.8)
        self.detector.detect.return_value = (1, rows)
        detector = FaceDetector(self.model_path)

        self.assertEqual(len(detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))), 2)

    def test_configuration_is_forwarded_to_opencv(self) -> None:
        FaceDetector(
            self.model_path,
            confidence_threshold=0.72,
            nms_threshold=0.25,
            top_k=100,
        )
        self.create_model.assert_called_once_with(
            model=str(self.model_path.resolve()),
            config="",
            input_size=(320, 320),
            score_threshold=0.72,
            nms_threshold=0.25,
            top_k=100,
        )


@unittest.skipUnless(
    DEFAULT_MODEL_PATH.is_file() and SAMPLE_IMAGE.is_file(),
    "Bundled YuNet model or real sample image is unavailable",
)
class FaceDetectorIntegrationTests(unittest.TestCase):
    def test_bundled_model_detects_faces_in_sample(self) -> None:
        detector = FaceDetector()
        image = cv2.imread(str(SAMPLE_IMAGE))
        self.assertIsNotNone(image)
        height, width = image.shape[:2]

        results = detector.detect(image)

        self.assertGreaterEqual(len(results), 1)
        for result in results:
            box = result.bounding_box
            self.assertTrue(np.isfinite([box.x, box.y, box.width, box.height]).all())
            self.assertGreater(box.width, 0)
            self.assertGreater(box.height, 0)
            self.assertTrue(0.0 <= result.confidence <= 1.0)
            self.assertEqual(len(result.landmarks), 5)
        self.assertEqual((width, height), (2048, 1150))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "app"
    / "model"
    / "face_detection_yunet"
    / "face_detection_yunet_2023mar.onnx"
)
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


def detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        bounding_box=BoundingBox(x, y, width, height),
        confidence=0.9,
        landmarks=((0.0, 0.0),) * 5,
    )


def patterned_image(height: int = 80, width: int = 100) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=-1,
    ).astype(np.uint8)


class FaceAnonymizerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = Mock()
        self.detector.detect.return_value = []
        self.anonymizer = FaceAnonymizer(self.detector)

    def test_no_faces_returns_copy_and_zero_count(self) -> None:
        image = patterned_image()
        output, metadata = self.anonymizer.anonymize_image(image)

        self.assertIsNot(output, image)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(metadata.face_count, 0)
        self.assertEqual(metadata.processed_face_count, 0)
        self.assertFalse(metadata.anonymized)

    def test_blur_alters_only_face_region_and_preserves_shape(self) -> None:
        self.detector.detect.return_value = [detection(20, 15, 30, 25)]
        image = patterned_image()
        original = image.copy()

        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="blur",
            strength=8,
        )

        self.assertEqual(output.shape, image.shape)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(output[15:40, 20:50], image[15:40, 20:50]))
        np.testing.assert_array_equal(output[:15], image[:15])
        np.testing.assert_array_equal(output[40:], image[40:])
        np.testing.assert_array_equal(output[15:40, :20], image[15:40, :20])
        np.testing.assert_array_equal(output[15:40, 50:], image[15:40, 50:])
        self.assertEqual(metadata.face_count, 1)
        self.assertEqual(metadata.processed_face_count, 1)
        self.assertTrue(metadata.anonymized)

    def test_pixelation_alters_face_region_and_preserves_shape(self) -> None:
        self.detector.detect.return_value = [detection(20, 15, 30, 25)]
        image = patterned_image()

        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="pixelate",
            strength=8,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertFalse(np.array_equal(output[15:40, 20:50], image[15:40, 20:50]))
        np.testing.assert_array_equal(output[:15], image[:15])
        self.assertEqual(metadata.method, "pixelate")

    def test_strength_changes_blur_and_pixelation_outputs(self) -> None:
        self.detector.detect.return_value = [detection(20, 15, 30, 25)]
        image = patterned_image()

        for method, strengths in (("blur", (1, 15)), ("pixelate", (2, 20))):
            with self.subTest(method=method):
                subtle, _ = self.anonymizer.anonymize_image(
                    image,
                    method=method,
                    strength=strengths[0],
                )
                strong, _ = self.anonymizer.anonymize_image(
                    image,
                    method=method,
                    strength=strengths[1],
                )
                self.assertFalse(np.array_equal(subtle, strong))

    def test_multiple_faces_are_each_processed(self) -> None:
        self.detector.detect.return_value = [
            detection(4, 5, 18, 18),
            detection(68, 50, 20, 20),
        ]
        image = patterned_image()
        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="pixelate",
            strength=6,
        )

        self.assertEqual(metadata.face_count, 2)
        self.assertEqual(metadata.processed_face_count, 2)
        self.assertFalse(np.array_equal(output[5:23, 4:22], image[5:23, 4:22]))
        self.assertFalse(np.array_equal(output[50:70, 68:88], image[50:70, 68:88]))

    def test_bounds_touching_image_edges_are_clamped(self) -> None:
        self.detector.detect.return_value = [detection(-5, -4, 20, 20)]
        image = patterned_image(24, 24)
        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="blur",
            strength=3,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(metadata.processed_face_count, 1)
        self.assertFalse(np.array_equal(output[:16, :15], image[:16, :15]))

    def test_bounds_touching_bottom_right_are_clamped(self) -> None:
        self.detector.detect.return_value = [detection(20, 20, 12, 12)]
        image = patterned_image(24, 24)
        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="blur",
            strength=3,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(metadata.processed_face_count, 1)
        self.assertFalse(np.array_equal(output[20:, 20:], image[20:, 20:]))

    def test_tiny_bounding_box_is_handled(self) -> None:
        self.detector.detect.return_value = [detection(10.2, 11.1, 1.1, 1.2)]
        image = patterned_image(24, 24)
        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="pixelate",
            strength=2,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(metadata.processed_face_count, 1)
        self.assertFalse(np.array_equal(output[11:13, 10:12], image[11:13, 10:12]))

    def test_margin_expands_target_region(self) -> None:
        self.detector.detect.return_value = [detection(30, 20, 10, 10)]
        image = patterned_image()
        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="blur",
            strength=4,
            margin=0.5,
        )

        self.assertFalse(np.array_equal(output[15:20, 30:40], image[15:20, 30:40]))
        self.assertEqual(metadata.margin, 0.5)

    def test_invalid_methods_strengths_margin_and_images_raise(self) -> None:
        image = patterned_image()
        invalid_options = (
            {"method": "mosaic"},
            {"strength": 0},
            {"strength": 51},
            {"method": "pixelate", "strength": 1},
            {"strength": 1.5},
            {"margin": -0.1},
            {"margin": float("nan")},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.anonymizer.anonymize_image(image, **options)

        for invalid in (
            None,
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 2), dtype=np.uint8),
            np.zeros((10, 10, 1), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        ):
            with self.subTest(image_type=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    self.anonymizer.anonymize_image(invalid)

        self.detector.detect.assert_not_called()

    def test_different_image_sizes_are_preserved(self) -> None:
        for height, width in ((10, 15), (64, 90), (151, 37)):
            image = patterned_image(height, width)
            output, _ = self.anonymizer.anonymize_image(image)
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, image.dtype)

    def test_grayscale_input_preserves_layout_and_is_anonymized(self) -> None:
        self.detector.detect.return_value = [detection(20, 15, 30, 25)]
        image = patterned_image().mean(axis=2).astype(np.uint8)

        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="blur",
            strength=8,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.ndim, 2)
        self.assertFalse(np.array_equal(output[15:40, 20:50], image[15:40, 20:50]))
        detection_image = self.detector.detect.call_args.args[0]
        self.assertEqual(detection_image.shape, (*image.shape, 3))
        self.assertEqual(metadata.processed_face_count, 1)

    def test_bgra_input_preserves_alpha_and_channel_layout(self) -> None:
        self.detector.detect.return_value = [detection(20, 15, 30, 25)]
        colors = patterned_image()
        alpha = np.full((*colors.shape[:2], 1), 173, dtype=np.uint8)
        image = np.concatenate((colors, alpha), axis=2)

        output, metadata = self.anonymizer.anonymize_image(
            image,
            method="pixelate",
            strength=8,
        )

        self.assertEqual(output.shape, image.shape)
        self.assertFalse(
            np.array_equal(output[15:40, 20:50, :3], image[15:40, 20:50, :3])
        )
        np.testing.assert_array_equal(output[:, :, 3], image[:, :, 3])
        self.assertEqual(metadata.processed_face_count, 1)

    def test_face_box_completely_outside_image_is_reported_unprocessed(self) -> None:
        self.detector.detect.return_value = [detection(200, 200, 10, 10)]
        image = patterned_image()
        output, metadata = self.anonymizer.anonymize_image(image)
        np.testing.assert_array_equal(output, image)
        self.assertEqual(metadata.face_count, 1)
        self.assertEqual(metadata.processed_face_count, 0)
        self.assertFalse(metadata.anonymized)


@unittest.skipUnless(
    MODEL_PATH.is_file() and SAMPLE_IMAGE.is_file(),
    "Bundled YuNet model or real sample image is unavailable",
)
class FaceAnonymizerIntegrationTests(unittest.TestCase):
    def test_real_model_blur_and_pixelation_process_sample_faces(self) -> None:
        image = cv2.imread(str(SAMPLE_IMAGE))
        self.assertIsNotNone(image)
        original = image.copy()
        anonymizer = FaceAnonymizer()

        for method in ("blur", "pixelate"):
            with self.subTest(method=method):
                output, metadata = anonymizer.anonymize_image(
                    image,
                    method=method,
                    strength=15,
                )
                self.assertEqual(output.shape, image.shape)
                self.assertGreater(metadata.face_count, 0)
                self.assertEqual(
                    metadata.processed_face_count,
                    metadata.face_count,
                )
                self.assertTrue(metadata.anonymized)
                self.assertFalse(np.array_equal(output, image))

        np.testing.assert_array_equal(image, original)


if __name__ == "__main__":
    unittest.main()
